"""
immunize_batch.py

Batch driver for gaussian_ce_immunization.py. Immunizes every clean image
with the H_max and/or H_min Gaussian cross-entropy attack at each epsilon,
with and/or without a mask, and writes the results in the folder layout
batch_edit.py and visualize_latent_batch.py read.

Reads from:
  inputs/clean/<category>/          -- clean source images (animal / human / object)
  masks/mask_<stem>.png             -- per-image masks (human only, matched by stem)

Writes to:
  outputs/<method>_gaussian/<epsilon>/<mask_type>/<category>/<filename>

  <method>    hmax or hmin
  <epsilon>   4 / 8 / 16 / 32  (pixel units out of 255)
  <mask_type> no_mask, or mask (human category only)

Existing outputs are skipped, so an interrupted run resumes where it left off.

Usage:
  python immunize_batch.py --method both --mask both
  python immunize_batch.py --method hmax --mask no_mask --epsilon 8 16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image

from gaussian_ce_immunization import (
    IMAGE_EXTENSIONS,
    attack_hmax,
    attack_hmin,
    cleanup_memory,
    get_device,
    load_model,
    save_image,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
INPUT_DIR  = BASE_DIR / "inputs" / "clean"
MASKS_DIR  = BASE_DIR / "masks"
OUTPUT_DIR = BASE_DIR / "outputs"

# ---------------------------------------------------------------------------
# Run settings  (edit these to tune the attack)
# ---------------------------------------------------------------------------
CATEGORIES    = ["animal", "human", "object"]
EPSILONS      = [4, 8, 16, 32]          # pixel-space; divided by 255 internally
MAX_SIZE      = 512                      # longest side; 0 = no resize
PGD_STEPS     = 100
PGD_STEP_SIZE = 1 / 255
KAPPA         = 1.0                      # variance of H_min's N(0, kappa I) target
SEED          = 42

METHODS = ["hmax", "hmin"]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def prepare_image(path: Path) -> Image.Image:
    """Load RGB, cap the longest side at MAX_SIZE, round H and W down to multiples of 8."""
    img = Image.open(path).convert("RGB")
    if MAX_SIZE > 0:
        longest = max(img.size)
        if longest > MAX_SIZE:
            scale = MAX_SIZE / longest
            img = img.resize(
                (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
            )
    w = (img.width  // 8) * 8
    h = (img.height // 8) * 8
    if (w, h) != (img.width, img.height):
        img = img.resize((w, h), Image.LANCZOS)
    return img


def to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL -> [1, 3, H, W] float32 tensor in [0, 1]."""
    array = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def find_mask(image_path: Path, ref_tensor: torch.Tensor) -> torch.Tensor | None:
    """
    Looks for masks/mask_<stem>.png. Returns a [1, 3, H, W] tensor in [0, 1]
    resized to match ref_tensor, or None if the file is missing.
    """
    candidate = MASKS_DIR / f"mask_{image_path.stem}.png"
    if not candidate.is_file():
        print(f"  warn  mask not found: {candidate.name} -- skipping this image for mask run")
        return None

    h, w = ref_tensor.shape[-2], ref_tensor.shape[-1]
    mask_pil = Image.open(candidate).convert("L").resize((w, h), Image.NEAREST)
    mask_t   = torch.from_numpy(np.asarray(mask_pil, dtype=np.float32) / 255.0)  # [H, W]
    return mask_t.expand(1, 3, h, w).contiguous().to(ref_tensor.device)


# ---------------------------------------------------------------------------
# Output path convention
# ---------------------------------------------------------------------------
def output_dir_for(method: str, eps: int, mask_type: str, category: str) -> Path:
    return OUTPUT_DIR / f"{method}_gaussian" / str(eps) / mask_type / category


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Immunize every clean image with the H_max / H_min Gaussian "
                    "cross-entropy attack from gaussian_ce_immunization.py."
    )
    parser.add_argument("--method", choices=METHODS + ["both"], required=True,
                        help="Which attack to run.")
    parser.add_argument("--mask", choices=["no_mask", "mask", "both"], required=True,
                        help="Which mask type to run (mask runs are human category only).")
    parser.add_argument("--epsilon", type=int, nargs="+", choices=EPSILONS, default=EPSILONS,
                        metavar="EPS",
                        help=f"Epsilon(s) in pixel units out of 255 (default: all of {EPSILONS}).")
    parser.add_argument("--model", choices=["flux", "instruct_pix2pix", "sd15"], default="flux",
                        help="Whose VAE to attack (default: flux, the model batch_edit.py uses).")
    parser.add_argument("--input", type=Path, default=INPUT_DIR,
                        help="Folder holding <category>/ clean images (default: ./inputs/clean).")
    parser.add_argument("--masks", type=Path, default=MASKS_DIR,
                        help="Folder holding mask_<stem>.png files (default: ./masks).")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help="Where immunized images are written (default: ./outputs).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Work-plan builder
# ---------------------------------------------------------------------------
def collect_work(
    methods:    list[str],
    epsilons:   list[int],
    mask_types: list[str],
) -> tuple[list[tuple], int]:
    """
    Returns (work_items, n_skipped).
    Each work item: (img_path, out_path, method, eps, mask_type, seed)
    """
    work: list[tuple] = []
    skipped = 0

    for category in CATEGORIES:
        cat_dir = INPUT_DIR / category
        if not cat_dir.is_dir():
            print(f"  warn  missing input folder: {cat_dir}")
            continue

        images = sorted(
            [p for p in cat_dir.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: p.name,
        )
        if not images:
            print(f"  warn  no images found in {cat_dir}")
            continue

        for mask_type in mask_types:
            if mask_type == "mask" and category != "human":
                continue  # mask runs are human-only

            for eps in epsilons:
                for method in methods:
                    out_dir = output_dir_for(method, eps, mask_type, category)
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # Seed depends only on the image's position in its
                    # category, so reruns and partial runs are reproducible.
                    for index, img_path in enumerate(images):
                        out_path = out_dir / img_path.name
                        if out_path.exists():
                            skipped += 1
                        else:
                            work.append((img_path, out_path, method, eps, mask_type, SEED + index))

    return work, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    global INPUT_DIR, MASKS_DIR, OUTPUT_DIR

    args = parse_args()
    INPUT_DIR, MASKS_DIR, OUTPUT_DIR = args.input, args.masks, args.output

    if not INPUT_DIR.is_dir():
        print(f"error: clean input folder not found: {INPUT_DIR}", file=sys.stderr)
        return 1

    methods    = METHODS if args.method == "both" else [args.method]
    mask_types = ["no_mask", "mask"] if args.mask == "both" else [args.mask]
    epsilons   = sorted(set(args.epsilon))

    print("\nScanning input folders...")
    work, skipped = collect_work(methods, epsilons, mask_types)

    total = len(work) + skipped
    print(f"\n{total} total | {len(work)} to process | {skipped} already done")
    if not work:
        print("Nothing to do.")
        return 0

    device = get_device()
    if device.type != "cuda":
        print("warn  no CUDA GPU detected -- running on CPU will be very slow.")
    _, vae = load_model(model_name=args.model, device=device)

    succeeded = failed = 0
    for i, (img_path, out_path, method, eps, mask_type, seed) in enumerate(work):
        rel = img_path.relative_to(INPUT_DIR)
        print(f"\n  [{i + 1}/{len(work)}] {rel} | {method} | eps={eps} | {mask_type}")

        start = time.perf_counter()
        try:
            image = to_tensor(prepare_image(img_path), device)

            mask = None
            if mask_type == "mask":
                mask = find_mask(img_path, image)
                if mask is None:
                    failed += 1
                    continue

            common = dict(image=image, vae=vae, steps=PGD_STEPS, step_size=PGD_STEP_SIZE,
                          epsilon=eps / 255.0, seed=seed, mask=mask)
            if method == "hmax":
                result, objective, _ = attack_hmax(**common)
            else:
                result, objective, _ = attack_hmin(kappa=KAPPA, **common)

            save_image(result, out_path)
            elapsed = time.perf_counter() - start
            print(f"  ok    -> {out_path.relative_to(OUTPUT_DIR)} "
                  f"(objective {objective:.4f}, {elapsed:.1f}s)")
            succeeded += 1

        except Exception as exc:
            elapsed = time.perf_counter() - start
            print(f"  FAIL  {img_path.name}: {exc} ({elapsed:.1f}s)", file=sys.stderr)
            failed += 1

        finally:
            cleanup_memory()

    print(f"\nDone. {succeeded} succeeded | {failed} failed | {skipped} skipped.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
