#!/usr/bin/env python3
"""
compare_image_metrics.py

Computes eight full-reference image quality metrics between a folder of
reference (original) images and one or more folders of comparison images
(e.g. immunized outputs, edited outputs) -- matched by filename.

    DSS      - DCT Subband Similarity
    GMSD     - Gradient Magnitude Similarity Deviation
    HaarPSI  - Haar wavelet-based Perceptual Similarity
    LPIPS    - Learned Perceptual Image Patch Similarity
    PSNR     - Peak Signal-to-Noise Ratio
    SR-SIM   - Spectral Residual based Similarity
    VIFp     - Visual Information Fidelity (pixel domain)
    VSI      - Visual Saliency-based Index

All metrics are computed via the `piq` library, so every metric shares
the same [0,1]-range, [B,C,H,W] tensor convention -- no per-metric ad hoc
preprocessing.

MATCHING
--------
For each image in --reference, a matching file is looked for in each
--comparison folder by filename STEM, allowing the comparison filename to
have extra suffix text (e.g. reference "cat.jpg" matches comparison
"cat_v1.jpg", "cat-edited.png", or exactly "cat.png"). A reference image
with no match in a given comparison folder is skipped for that folder,
reported, and does not stop the run.

Different resolutions between a reference/comparison pair are handled by
resizing the comparison image to the reference's resolution before
scoring (with a printed note), since all eight metrics require matching
shapes.

Usage
-----
    python compare_image_metrics.py \
        --reference ./images \
        --comparison ./results/sd15/variant_a ./results/sd15/variant_b \
        --output ./metrics_report.csv

Directions on interpreting each metric -- HIGHER is more similar
(better) for: HaarPSI, PSNR, SR-SIM, VIFp, VSI.
LOWER is more similar (better) for: DSS is higher=more similar too (it
is a similarity index like the others above); GMSD and LPIPS are
DISTANCES -- lower means more similar. This distinction is printed in
the summary so results aren't misread.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

import piq

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# name -> (piq callable, higher_is_more_similar)
# LPIPS is handled separately below since it's a class, not a function.
METRICS = {
    "DSS": (piq.dss, True),
    "GMSD": (piq.gmsd, False),
    "HaarPSI": (piq.haarpsi, True),
    "PSNR": (piq.psnr, True),
    "SRSIM": (piq.srsim, True),
    "VIFp": (piq.vif_p, True),
    "VSI": (piq.vsi, True),
}
# LPIPS is a distance: lower = more similar.
LPIPS_HIGHER_IS_BETTER = False


def find_images(folder: Path) -> dict:
    """Returns {stem_lower: Path} for every supported image directly inside
    `folder` (non-recursive)."""
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            result[path.stem.lower()] = path
    return result


def match_comparison_file(reference_path: Path, comparison_folder: Path) -> Optional[Path]:
    """
    Find the file in comparison_folder that corresponds to reference_path,
    by filename stem. Tries, in order:
      1. exact stem match (any extension)
      2. a comparison file whose stem STARTS WITH the reference's stem
         followed by a non-alphanumeric separator (e.g. "cat" matches
         "cat_v1", "cat-edited", but not "catalog")
    Returns None if nothing matches.
    """
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


def load_as_tensor(path: Path) -> torch.Tensor:
    """Load an RGB image as a [1,3,H,W] float tensor in [0,1]."""
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def align_shapes(reference: torch.Tensor, comparison: torch.Tensor, name: str) -> torch.Tensor:
    """Resize `comparison` to match `reference`'s spatial size if they
    differ, since every metric here requires matching shapes."""
    if comparison.shape[-2:] != reference.shape[-2:]:
        print(f"  NOTE: resizing '{name}' from {tuple(comparison.shape[-2:])} "
              f"to {tuple(reference.shape[-2:])} to match reference")
        comparison = torch.nn.functional.interpolate(
            comparison, size=reference.shape[-2:], mode="bilinear", align_corners=False
        )
    return comparison


def compute_all_metrics(reference: torch.Tensor, comparison: torch.Tensor, lpips_fn, device: torch.device) -> dict:
    """Compute all 8 metrics for one reference/comparison pair. Both
    tensors must already be the same shape, in [0,1], on `device`."""
    reference = reference.to(device)
    comparison = comparison.to(device)

    scores = {}
    with torch.no_grad():
        for metric_name, (fn, _) in METRICS.items():
            try:
                scores[metric_name] = fn(comparison, reference, data_range=1.0).item()
            except Exception as exc:
                print(f"    WARNING: {metric_name} failed ({type(exc).__name__}: {exc}); recording NaN")
                scores[metric_name] = float("nan")

        try:
            scores["LPIPS"] = lpips_fn(comparison, reference).item()
        except Exception as exc:
            print(f"    WARNING: LPIPS failed ({type(exc).__name__}: {exc}); recording NaN")
            scores["LPIPS"] = float("nan")

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Compare images against a reference folder using DSS, GMSD, HaarPSI, "
                    "LPIPS, PSNR, SR-SIM, VIFp, and VSI."
    )
    parser.add_argument("--reference", required=True, type=str,
                         help="Folder of original/reference images.")
    parser.add_argument("--comparison", required=True, type=str, nargs="+",
                         help="One or more folders of images to compare against the reference "
                              "(e.g. immunized outputs, edited outputs). Each is scored separately.")
    parser.add_argument("--output", type=str, default="./metrics_report.csv",
                         help="Path to write the per-image CSV report.")

    args = parser.parse_args()

    reference_dir = Path(args.reference)
    if not reference_dir.is_dir():
        raise NotADirectoryError(f"--reference is not a directory: {reference_dir}")

    comparison_dirs = [Path(p) for p in args.comparison]
    for d in comparison_dirs:
        if not d.is_dir():
            raise NotADirectoryError(f"--comparison path is not a directory: {d}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    lpips_fn = piq.LPIPS().to(device)

    reference_images = find_images(reference_dir)
    if not reference_images:
        raise RuntimeError(f"No supported images found in --reference: {reference_dir}")
    print(f"Reference images found: {len(reference_images)}")

    metric_names = list(METRICS.keys()) + ["LPIPS"]
    rows = []

    for ref_stem, ref_path in reference_images.items():
        reference_tensor = load_as_tensor(ref_path)

        for comparison_dir in comparison_dirs:
            match_path = match_comparison_file(ref_path, comparison_dir)

            if match_path is None:
                print(f"[SKIP] no match for '{ref_path.name}' in {comparison_dir}")
                continue

            print(f"Comparing: {ref_path.name}  <->  {comparison_dir.name}/{match_path.name}")

            comparison_tensor = load_as_tensor(match_path)
            comparison_tensor = align_shapes(reference_tensor, comparison_tensor, match_path.name)

            scores = compute_all_metrics(reference_tensor, comparison_tensor, lpips_fn, device)

            row = {
                "reference_image": ref_path.name,
                "comparison_folder": comparison_dir.name,
                "comparison_image": match_path.name,
            }
            row.update(scores)
            rows.append(row)

    if not rows:
        raise RuntimeError("No matched image pairs were found across any comparison folder.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["reference_image", "comparison_folder", "comparison_image"] + metric_names
        )
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
        summary_row = {"comparison_folder": folder_name, "n_images": len(folder_rows)}
        for metric_name in metric_names:
            values = [r[metric_name] for r in folder_rows if not np.isnan(r[metric_name])]
            summary_row[metric_name] = np.mean(values) if values else float("nan")
        summary_rows.append(summary_row)

    summary_path = output_path.with_name(output_path.stem + "_summary" + output_path.suffix)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["comparison_folder", "n_images"] + metric_names)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {len(summary_rows)} summary row(s) to {summary_path.resolve()}")
    print("=" * 70)

    print()
    print("Mean scores per comparison folder:")
    print("(higher = more similar for DSS, HaarPSI, PSNR, SRSIM, VIFp, VSI)")
    print("(lower  = more similar for GMSD, LPIPS -- these are distances)")
    print()

    for summary_row in summary_rows:
        print(f"[{summary_row['comparison_folder']}]  (n={summary_row['n_images']})")
        for metric_name in metric_names:
            print(f"    {metric_name:8s}: {summary_row[metric_name]:.4f}")
        print()


if __name__ == "__main__":
    main()
