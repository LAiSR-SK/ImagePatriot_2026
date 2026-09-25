import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import FluxKontextPipeline
from PIL import Image

INPUT_DIR   = Path(__file__).parent / "inputs"
OUTPUT_DIR  = Path(__file__).parent / "outputs"
TRACKING_CSV_MAP = {
    "animal":                 Path(__file__).parent / "./../prompts/run_log_animal.csv",
    "object":                 Path(__file__).parent / "./../prompts/run_log_object.csv",
    "human":                  Path(__file__).parent / "./../prompts/run_log_human.csv"
}

# --method choice -> input folder name. "clean" edits the un-immunized
# originals (clean/<category>/, no epsilon or mask levels); those edits are
# the reference compare_batch.py scores immunized edits against.
METHOD_DIRS      = {"hmax": "hmax_gaussian", "hmin": "hmin_gaussian", "clean": "clean"}
IMMUNIZED_METHODS = ["hmax", "hmin"]
VALID_MASK_TYPES = {"mask", "no_mask"}
VALID_CATEGORIES = set(TRACKING_CSV_MAP.keys())

MAX_SIZE = 512
QUANT_VRAM_THRESHOLD_GB = 40.0
MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ATTACK_SUFFIXES = ["_attacked", "_multistep", "_onestep"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FLUX.1-Kontext edits over every method/mask/category "
                    "folder under inputs/, writing results to outputs/."
    )
    parser.add_argument("--method", choices=list(METHOD_DIRS) + ["both"], required=True,
                        help="hmax / hmin: immunized images (both = hmax and hmin). "
                             "clean: the un-immunized originals, for the reference edits.")
    parser.add_argument("--mode", choices=["standard", "attack"], required=True,
                        help="standard: edit the images named in the run log. "
                             "attack: edit their _attacked/_multistep/_onestep variants.")
    parser.add_argument("--mask", choices=sorted(VALID_MASK_TYPES) + ["both"],
                        help="Which mask-type folders to process. Required for hmax/hmin; "
                             "not used with --method clean.")
    parser.add_argument("--input", type=Path, default=INPUT_DIR,
                        help="Root folder holding <method>/.../<mask>/<category>/ "
                             "(or clean/<category>/) (default: ./inputs). "
                             "Images not in the run log are skipped.")
    args = parser.parse_args()
    if args.method != "clean" and args.mask is None:
        parser.error("--mask is required unless --method is clean")
    return args


def load_run_log(category: str) -> list[tuple[str, str]]:
    log_path = TRACKING_CSV_MAP[category]
    if not log_path.is_file():
        raise FileNotFoundError(f"Run log not found: {log_path}")
    pairs = []
    with open(log_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().lower() == "image":
                continue
            pairs.append((row[0].strip(), row[1].strip()))
    if not pairs:
        raise ValueError(f"Run log is empty: {log_path}")
    return pairs


def get_category(folder_name: str) -> str | None:
    if folder_name in VALID_CATEGORIES:
        return folder_name
    for cat in sorted(VALID_CATEGORIES, key=len, reverse=True):
        if folder_name.endswith(f"_{cat}"):
            return cat
    return None


def find_category_dirs(parent: Path, mask_selection: set[str]) -> list[tuple[Path, str]]:
    results = []
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        if child.name in VALID_MASK_TYPES:
            if child.name in mask_selection:
                for cat_dir in sorted(child.iterdir()):
                    if cat_dir.is_dir():
                        category = get_category(cat_dir.name)
                        if category is not None:
                            results.append((cat_dir, category))
        else:
            results.extend(find_category_dirs(child, mask_selection))
    return results


def split_attack_suffix(path: Path) -> tuple[str, str | None]:
    """image_0_attacked.png -> ("image_0", "_attacked"); image_0.png -> ("image_0", None)."""
    for suffix in ATTACK_SUFFIXES:
        if path.stem.endswith(suffix):
            return path.stem[: -len(suffix)], suffix
    return path.stem, None


def lookup_prompt(base_stem: str, ext: str, prompts: dict[str, str]) -> str | None:
    """Exact filename first. Otherwise fall back to the stem, but only when it is
    unambiguous -- some run logs reuse a stem across extensions (image0003.jpg / .jpeg)."""
    exact = prompts.get(f"{base_stem}{ext}")
    if exact is not None:
        return exact
    matches = [p for name, p in prompts.items() if Path(name).stem == base_stem]
    return matches[0] if len(matches) == 1 else None


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def prepare_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if MAX_SIZE > 0:
        longest = max(image.size)
        if longest > MAX_SIZE:
            scale = MAX_SIZE / longest
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return image


def load_pipeline():
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("No CUDA GPU detected. This script requires a GPU.")
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        gpu_name = torch.cuda.get_device_name(0)
    except RuntimeError as e:
        raise RuntimeError(f"CUDA GPU found but failed to query device properties: {e}") from e
    print(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GiB")

    if vram_gb >= QUANT_VRAM_THRESHOLD_GB:
        print("Strategy: bf16 + CPU offload")
        pipe = FluxKontextPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        return pipe

    print("Strategy: NF4 4-bit quantization")
    from diffusers import BitsAndBytesConfig as DiffusersBnbConfig, FluxTransformer2DModel
    from transformers import BitsAndBytesConfig as TransformersBnbConfig, T5EncoderModel

    bnb_cfg = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID, subfolder="transformer",
        quantization_config=DiffusersBnbConfig(**bnb_cfg), torch_dtype=torch.bfloat16,
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder_2",
        quantization_config=TransformersBnbConfig(**bnb_cfg), torch_dtype=torch.bfloat16,
    )
    pipe = FluxKontextPipeline.from_pretrained(
        MODEL_ID, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def main() -> int:
    global INPUT_DIR

    args = parse_args()
    INPUT_DIR = args.input
    if not INPUT_DIR.is_dir():
        print(f"error: --input is not a directory: {INPUT_DIR}", file=sys.stderr)
        return 2
    mode = args.mode
    mask_selection = set(VALID_MASK_TYPES) if args.mask == "both" else {args.mask}
    methods = IMMUNIZED_METHODS if args.method == "both" else [args.method]
    method_names = {METHOD_DIRS[m] for m in methods}

    planned, skipped = [], 0

    for method_dir in sorted(INPUT_DIR.iterdir()):
        if not method_dir.is_dir() or method_dir.name not in method_names:
            continue
        if method_dir.name == METHOD_DIRS["clean"]:
            # clean/<category>/ -- no epsilon or mask levels to walk
            category_dirs = [(d, get_category(d.name)) for d in sorted(method_dir.iterdir())
                             if d.is_dir() and get_category(d.name) is not None]
        else:
            category_dirs = find_category_dirs(method_dir, mask_selection)
        for category_dir, category in category_dirs:
            out_dir  = OUTPUT_DIR / category_dir.relative_to(INPUT_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                run_log = load_run_log(category)
            except (FileNotFoundError, ValueError) as e:
                print(f"  warn  {category_dir.relative_to(INPUT_DIR)}: {e}")
                continue

            print(f"\n  Scanning {category_dir.relative_to(INPUT_DIR)} ...")

            # Driven by what is in the folder, not by the run log, so a folder
            # holding any subset of the dataset is processed without noise.
            prompts = dict(run_log)
            for src in list_images(category_dir):
                base_stem, attack_suffix = split_attack_suffix(src)
                if (mode == "attack") != (attack_suffix is not None):
                    continue  # standard wants clean images, attack wants variants
                prompt = lookup_prompt(base_stem, src.suffix, prompts)
                if prompt is None:
                    print(f"  skip  {src.name} (no unambiguous match in run log for '{category}')")
                    skipped += 1
                    continue
                dst = out_dir / src.name
                if dst.exists():
                    print(f"  skip  {src.name} (output exists)")
                    skipped += 1
                else:
                    planned.append((src, dst, prompt))

    print(f"\n{len(planned)} to process | {skipped} skipped")
    if not planned:
        return 0

    print(f"\nLoading pipeline... (first run downloads ~24 GB)")
    pipe = load_pipeline()

    succeeded = failed = 0
    for i, (src, dst, prompt) in enumerate(planned):
        print(f"\n  [{i + 1}/{len(planned)}] {src.name}")
        print(f'  prompt: "{prompt}"')
        start = time.perf_counter()
        try:
            result = pipe(image=prepare_image(src), prompt=prompt, guidance_scale=5.0, num_inference_steps=28).images[0]
            result.save(dst)
            print(f"  ok    -> {dst.name} ({time.perf_counter() - start:.1f}s)")
            succeeded += 1
        except Exception as exc:
            print(f"  FAIL  {src.name}: {exc} ({time.perf_counter() - start:.1f}s)", file=sys.stderr)
            failed += 1
        finally:
            torch.cuda.empty_cache()

    print(f"\nDone. {succeeded} succeeded, {failed} failed, {skipped} skipped.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
