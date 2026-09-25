#!/usr/bin/env python3
"""
compare_clip_embeddings.py

Compares images using plain image-to-image CLIP similarity, matched by
filename across a reference folder and one or more comparison folders --
same matching convention as compare_image_metrics.py, so this tool
composes naturally with it (same --reference/--comparison folder
structure, same per-image + summary CSV output pattern).

Assumes ALL images already exist on disk (clean, immunized, edited-clean,
edited-immunized, ...) -- this script never runs any editing or
immunization itself, it only reads images you already have.

similarity = cosine( CLIP_image(reference), CLIP_image(comparison) )

A semantic similarity, not a pixel similarity -- answers "do these two
images depict roughly the same thing," regardless of exact pixel values.
Higher = more semantically similar.

Typical use -- two SEPARATE runs, matching the standard evaluation
protocol used in the image immunization literature (e.g. DiffVax,
Universal Image Immunization): imperceptibility and disruption are always
reported as two distinct pairwise comparisons, never combined into one
metric.

  Run 1 -- imperceptibility check (clean vs immunized, unedited):
      python compare_clip_embeddings.py \
          --reference ./clean \
          --comparison ./immunized \
          --output ./clip_imperceptibility.csv
      Expect HIGH similarity -- immunization should look like the same
      content as the original.

  Run 2 -- edit-outcome check (edited-clean vs edited-immunized):
      python compare_clip_embeddings.py \
          --reference ./edited_clean \
          --comparison ./edited_immunized \
          --output ./clip_edit_outcome.csv
      Expect LOW similarity if immunization is working -- meaning editing
      the protected image produced a meaningfully different result than
      editing the unprotected one.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from transformers import CLIPModel, CLIPProcessor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ======================================================================
# Matching -- identical convention to compare_image_metrics.py
# ======================================================================

def find_images(folder: Path) -> dict:
    """Returns {stem_lower: Path} for every supported image directly inside
    `folder` (non-recursive)."""
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            result[path.stem.lower()] = path
    return result


def match_comparison_file(reference_path: Path, comparison_folder: Path) -> Optional[Path]:
    """Find the file in comparison_folder corresponding to reference_path,
    by filename stem (exact match, or comparison stem starting with the
    reference stem followed by a non-alphanumeric separator)."""
    ref_stem = reference_path.stem.lower()
    candidates = find_images(comparison_folder)

    if ref_stem in candidates:
        return candidates[ref_stem]

    prefix_matches = sorted(
        path for stem, path in candidates.items()
        if stem.startswith(ref_stem)
        and (len(stem) == len(ref_stem) or not stem[len(ref_stem)].isalnum())
    )
    if prefix_matches:
        return prefix_matches[0]

    return None


def verify_exact_matches(reference_dir: Path, comparison_dirs: list) -> None:
    """
    Pre-flight check, run BEFORE any CLIP computation: confirms every
    reference image has an EXACT filename match (same name, same
    extension) in every comparison folder. Prints a clear per-folder
    report and raises if anything is missing or only matched via the
    fuzzy prefix fallback in match_comparison_file, rather than silently
    proceeding on a possibly-wrong pairing.

    Use this when clean/immunized/edited images are all expected to share
    the exact same filename (e.g. "cat.jpg" in every folder) -- as
    opposed to a suffix convention like "cat_immunized.jpg", where exact
    matching is expected to fail and the fuzzy fallback is normal.
    """
    reference_images = find_images(reference_dir)
    problems = []

    for comparison_dir in comparison_dirs:
        comparison_images = find_images(comparison_dir)
        missing = []
        fuzzy_only = []

        for ref_stem, ref_path in reference_images.items():
            if ref_stem in comparison_images:
                continue
            fallback = match_comparison_file(ref_path, comparison_dir)
            if fallback is not None:
                fuzzy_only.append((ref_path.name, fallback.name))
            else:
                missing.append(ref_path.name)

        n_exact = len(reference_images) - len(missing) - len(fuzzy_only)
        print(f"[{comparison_dir.name}] exact matches: {n_exact}/{len(reference_images)}")

        if fuzzy_only:
            print(f"  WARNING: {len(fuzzy_only)} image(s) only matched via fuzzy "
                  f"prefix fallback, NOT an exact filename match:")
            for ref_name, fallback_name in fuzzy_only:
                print(f"    '{ref_name}' -> '{fallback_name}' (not identical)")
            problems.append(comparison_dir)

        if missing:
            print(f"  MISSING: {len(missing)} reference image(s) have NO match at all "
                  f"in {comparison_dir}:")
            for name in missing:
                print(f"    {name}")
            problems.append(comparison_dir)

    if problems:
        raise RuntimeError(
            "Exact-filename check failed for one or more comparison folders (see warnings "
            "above). Fix the mismatched/missing files, or if filenames are genuinely not "
            "meant to be identical across folders (e.g. a '_suffix' naming convention), this "
            "check does not apply -- the normal fuzzy matching in match_comparison_file "
            "already handles that case without this stricter verification."
        )

    print("All reference images have an exact filename match in every comparison folder.")


# ======================================================================
# CLIP embeddings
# ======================================================================

def load_clip(model_name: str, device: torch.device):
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


def clip_image_embedding(model, processor, image: Image.Image, device: torch.device) -> torch.Tensor:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    # transformers >= 5 returns a model-output object here, not a tensor
    if not torch.is_tensor(features):
        features = (features.image_embeds if hasattr(features, "image_embeds")
                    else features.pooler_output)
    return torch.nn.functional.normalize(features, dim=-1).squeeze(0)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.dot(a, b).item()


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare images using plain image-to-image CLIP similarity."
    )
    parser.add_argument("--reference", required=True, type=str,
                         help="Folder of reference images.")
    parser.add_argument("--comparison", required=True, type=str, nargs="+",
                         help="One or more folders to compare against the reference. "
                              "Each is scored independently.")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32",
                         help="HuggingFace CLIP model identifier.")
    parser.add_argument("--output", type=str, default="./clip_report.csv",
                         help="Path to write the per-image CSV report.")
    parser.add_argument("--no-exact-match-check", action="store_true", default=False,
                         help="Skip the pre-flight check that verifies every reference image "
                              "has an EXACT filename match (not just a fuzzy prefix match) in "
                              "every comparison folder. The check is ON by default since clean/"
                              "immunized/edited images are commonly expected to share identical "
                              "filenames; disable it only if your folders intentionally use "
                              "different naming conventions per folder (e.g. 'cat_immunized.jpg').")

    args = parser.parse_args()

    reference_dir = Path(args.reference)
    if not reference_dir.is_dir():
        raise NotADirectoryError(f"--reference is not a directory: {reference_dir}")

    comparison_dirs = [Path(p) for p in args.comparison]
    for d in comparison_dirs:
        if not d.is_dir():
            raise NotADirectoryError(f"--comparison path is not a directory: {d}")

    if not args.no_exact_match_check:
        print("Verifying exact filename matches before running any comparison...")
        verify_exact_matches(reference_dir, comparison_dirs)
        print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading CLIP model: {args.clip_model}")
    model, processor = load_clip(args.clip_model, device)

    reference_images = find_images(reference_dir)
    if not reference_images:
        raise RuntimeError(f"No supported images found in --reference: {reference_dir}")
    print(f"Reference images found: {len(reference_images)}")

    # Cache reference image embeddings -- each reference image is reused
    # across every comparison folder, no need to re-embed it each time.
    reference_embeddings = {}
    for ref_stem, ref_path in reference_images.items():
        img = Image.open(ref_path).convert("RGB")
        reference_embeddings[ref_stem] = clip_image_embedding(model, processor, img, device)

    rows = []

    for ref_stem, ref_path in reference_images.items():
        ref_embedding = reference_embeddings[ref_stem]

        for comparison_dir in comparison_dirs:
            match_path = match_comparison_file(ref_path, comparison_dir)

            if match_path is None:
                print(f"[SKIP] no match for '{ref_path.name}' in {comparison_dir}")
                continue

            print(f"Comparing: {ref_path.name}  <->  {comparison_dir.name}/{match_path.name}")

            comparison_img = Image.open(match_path).convert("RGB")
            comparison_embedding = clip_image_embedding(model, processor, comparison_img, device)

            image_similarity = cosine_similarity(ref_embedding, comparison_embedding)

            rows.append({
                "reference_image": ref_path.name,
                "comparison_folder": comparison_dir.name,
                "comparison_image": match_path.name,
                "clip_image_similarity": image_similarity,
            })

    if not rows:
        raise RuntimeError("No matched image pairs were found across any comparison folder.")

    fieldnames = ["reference_image", "comparison_folder", "comparison_image", "clip_image_similarity"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print(f"Wrote {len(rows)} rows to {output_path.resolve()}")
    print("=" * 70)

    per_folder: dict = {}
    for row in rows:
        per_folder.setdefault(row["comparison_folder"], []).append(row)

    summary_rows = []
    for folder_name, folder_rows in per_folder.items():
        img_sims = [r["clip_image_similarity"] for r in folder_rows]
        summary_rows.append({
            "comparison_folder": folder_name,
            "n_images": len(folder_rows),
            "mean_clip_image_similarity": np.mean(img_sims),
        })

    summary_path = output_path.with_name(output_path.stem + "_summary" + output_path.suffix)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {len(summary_rows)} summary row(s) to {summary_path.resolve()}")
    print("=" * 70)
    print()
    print("Mean CLIP image similarity per comparison folder (higher = more similar):")
    print()
    for s in summary_rows:
        print(f"[{s['comparison_folder']}]  (n={s['n_images']})  "
              f"similarity: {s['mean_clip_image_similarity']:.4f}")


if __name__ == "__main__":
    main()
