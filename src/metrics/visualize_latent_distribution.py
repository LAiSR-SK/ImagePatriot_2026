#!/usr/bin/env python3
"""
visualize_latent_distribution.py

Visualizes the shift in the VAE's predicted latent distribution caused by
immunization, ONE FIGURE PER ATTACK (H_max and/or H_min), each figure
showing TWO complementary views:

  1. Marginal histograms of mean and variance, pooled across every latent
     position and every image -- the real, unprojected quantities
     diagonal_gaussian_cross_entropy() actually operates on.

  2. A 2D PCA scatter: each latent SPATIAL POSITION (one point per
     position, per image, pooled across the whole set) is treated as a
     vector over its channels, PCA-reduced to 2D, and colored by
     clean/immunized.

Assumes clean, hmax, and hmin folders ALREADY CONTAIN the images you want
compared -- this script never runs any attack itself, only reads and
visualizes existing images (--hmax/--hmin point at gaussian_ce_immunization
.py's own output folders, e.g. results/<model>/hmax and results/<model>/hmin).

The VAE is loaded ONCE and reused for both attacks' figures, since both
use the same underlying model.

Usage
-----
    python visualize_latent_distribution.py \
        --clean ./clean \
        --hmax ./results/instruct_pix2pix/hmax \
        --hmin ./results/instruct_pix2pix/hmin \
        --model instruct_pix2pix \
        --output-dir ./latent_figures

    # or just one attack:
    python visualize_latent_distribution.py \
        --clean ./clean --hmax ./results/instruct_pix2pix/hmax \
        --model instruct_pix2pix --output-dir ./latent_figures

Produces, per attack given: <output-dir>/latent_distribution_hmax.png
and/or <output-dir>/latent_distribution_hmin.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
from scipy import stats as scipy_stats
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaussian_ce_immunization import (
    load_model,
    load_image,
    encode_raw_distribution,
    get_device,
)


# ======================================================================
# PCA from scratch (numpy SVD) -- no sklearn dependency
# ======================================================================

def fit_pca_2d(data: np.ndarray):
    mean = data.mean(axis=0, keepdims=True)
    centered = data - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return mean, components


def apply_pca_2d(data: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (data - mean) @ components


# ======================================================================
# Matching + encoding
# ======================================================================

def match_exact_filenames(clean_dir: Path, other_dir: Path, label: str) -> dict:
    """Exact-filename matching between clean_dir and other_dir. Returns
    {stem: filename}, warns about any clean image missing in other_dir."""
    clean_images = {p.stem.lower(): p.name for p in sorted(clean_dir.iterdir()) if p.is_file()}
    other_images = {p.stem.lower(): p.name for p in sorted(other_dir.iterdir()) if p.is_file()}
    common = sorted(set(clean_images) & set(other_images))

    missing = sorted(set(clean_images) - set(other_images))
    if missing:
        print(f"WARNING [{label}]: {len(missing)} clean image(s) have no exact-name match "
              f"in {other_dir}, skipped: {missing}")

    if not common:
        raise RuntimeError(f"No matched clean/{label} filename pairs found in {other_dir}.")

    return {s: other_images[s] for s in common}, {s: clean_images[s] for s in common}


def encode_folder(image_dir: Path, filenames: dict, vae, device: torch.device, image_size=None):
    """Returns (pooled_means, pooled_vars, pooled_positions) -- see module
    docstring for what each represents."""
    all_means, all_vars, all_position_vectors = [], [], []

    for stem, filename in filenames.items():
        path = image_dir / filename
        img = load_image(path, image_size=image_size).to(device)

        with torch.no_grad():
            dist = encode_raw_distribution(vae, img)

        mean = dist.mean.detach().cpu().numpy()
        var = dist.var.detach().cpu().numpy()

        all_means.append(mean.ravel())
        all_vars.append(var.ravel())

        c = mean.shape[1]
        all_position_vectors.append(mean.reshape(c, -1).T)

    return (
        np.concatenate(all_means),
        np.concatenate(all_vars),
        np.concatenate(all_position_vectors, axis=0),
    )


# ======================================================================
# One figure, for one attack
# ======================================================================

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized effect size between two samples, using the pooled
    standard deviation. |d| ~ 0.2 small, ~0.5 medium, ~0.8 large -- the
    standard interpretation bands in the stats literature."""
    pooled_std = np.sqrt((a.var() + b.var()) / 2)
    if pooled_std == 0:
        return 0.0
    return (b.mean() - a.mean()) / pooled_std


def annotate_histogram_panel(
    ax, clean_data: np.ndarray, immunized_data: np.ndarray, attack_label: str,
    kde_x: np.ndarray, add_kde: bool = True, log_x: bool = False,
) -> None:
    """
    Adds, on top of the existing histogram bars:
      - a dashed vertical line at each distribution's mean, so the shift
        is visible even when the bars overlap heavily
      - a smooth KDE curve overlay for each distribution
      - a text box reporting Cohen's d (effect size) and a two-sample
        Kolmogorov-Smirnov test p-value, so the panel states
        quantitatively whether the shift is statistically real, not just
        "looks different"

    kde_x are the x-positions the KDE curves are evaluated at (same range
    as the panel's bin edges, so the curve spans the visible axis).

    log_x: if True (use for the variance panel), the KDE is fit on
    log10-transformed data and evaluated/plotted back on the linear
    variance axis (with the axis itself set to log scale by the caller).
    Fitting the KDE directly on raw, heavily right-skewed variance values
    would pick a bandwidth dominated by the few large outliers, producing
    an almost-flat, uninformative curve on a log-x view -- fitting in log
    space matches the log-spaced bins already used for this panel's bars.
    """
    clean_mean = clean_data.mean()
    immunized_mean = immunized_data.mean()

    ax.axvline(clean_mean, color="#1D9E75", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(immunized_mean, color="#D85A30", linestyle="--", linewidth=1.5, alpha=0.8)

    if add_kde:
        for data, color in [(clean_data, "#0F6E56"), (immunized_data, "#A8391F")]:
            if data.size > 1 and data.std() > 0:
                if log_x:
                    log_data = np.log10(data)
                    if log_data.std() > 0:
                        kde = scipy_stats.gaussian_kde(log_data)
                        log_kde_x = np.log10(kde_x)
                        ax.plot(kde_x, kde(log_kde_x), color=color, linewidth=1.8)
                else:
                    kde = scipy_stats.gaussian_kde(data)
                    ax.plot(kde_x, kde(kde_x), color=color, linewidth=1.8)

    d = cohens_d(clean_data, immunized_data)
    # KS test on a subsample if the pooled data is very large -- ks_2samp's
    # p-value becomes numerically unstable/uninformative at extreme sample
    # sizes (trivially significant), and this keeps the test responsive to
    # the effect size rather than just sample count.
    max_ks_n = 5000
    rng = np.random.default_rng(0)
    clean_for_ks = clean_data if clean_data.size <= max_ks_n else rng.choice(clean_data, max_ks_n, replace=False)
    immunized_for_ks = immunized_data if immunized_data.size <= max_ks_n else rng.choice(immunized_data, max_ks_n, replace=False)
    ks_stat, ks_p = scipy_stats.ks_2samp(clean_for_ks, immunized_for_ks)

    ax.text(
        0.02, 0.98,
        f"Cohen's d = {d:.3f}\nKS p = {ks_p:.2e}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )


def generate_attack_figure(
    clean_dir: Path,
    immunized_dir: Path,
    attack_label: str,
    vae,
    device: torch.device,
    model_name: str,
    output_path: Path,
    image_size: Optional[int],
    bins: int,
    max_scatter_points: int,
    seed: int,
) -> None:
    print()
    print("=" * 70)
    print(f"Attack: {attack_label}")
    print("=" * 70)

    immunized_match, clean_match = match_exact_filenames(clean_dir, immunized_dir, attack_label)
    print(f"Matched {len(clean_match)} image pair(s) for {attack_label}.")

    print(f"Encoding clean images ({attack_label} pairing)...")
    clean_means, clean_vars, clean_positions = encode_folder(
        clean_dir, clean_match, vae, device, image_size
    )
    print(f"Encoding {attack_label} images...")
    immunized_means, immunized_vars, immunized_positions = encode_folder(
        immunized_dir, immunized_match, vae, device, image_size
    )

    print(f"  clean      mean:  mean={clean_means.mean():.4f}  std={clean_means.std():.4f}")
    print(f"  {attack_label:10s} mean:  mean={immunized_means.mean():.4f}  std={immunized_means.std():.4f}")
    print(f"  clean      var:   mean={clean_vars.mean():.4f}  std={clean_vars.std():.4f}")
    print(f"  {attack_label:10s} var:   mean={immunized_vars.mean():.4f}  std={immunized_vars.std():.4f}")

    combined_positions = np.concatenate([clean_positions, immunized_positions], axis=0)
    pca_mean, pca_components = fit_pca_2d(combined_positions)
    clean_2d = apply_pca_2d(clean_positions, pca_mean, pca_components)
    immunized_2d = apply_pca_2d(immunized_positions, pca_mean, pca_components)

    rng = np.random.default_rng(seed)

    def subsample(points_2d, cap):
        if points_2d.shape[0] <= cap:
            return points_2d
        idx = rng.choice(points_2d.shape[0], size=cap, replace=False)
        return points_2d[idx]

    clean_2d_plot = subsample(clean_2d, max_scatter_points)
    immunized_2d_plot = subsample(immunized_2d, max_scatter_points)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Identical bin edges for clean vs immunized in each panel -- computed
    # from the COMBINED range of both, so the two histograms are directly
    # comparable bin-for-bin. Without this, hist() with an integer bin
    # COUNT computes edges independently per call from that call's own
    # data range, which can silently give clean and immunized different
    # edges if their value ranges differ even slightly.
    var_floor = 1e-8  # variance is strictly positive; guards log(0) below
    clean_vars_safe = np.clip(clean_vars, var_floor, None)
    immunized_vars_safe = np.clip(immunized_vars, var_floor, None)
    combined_vars_safe = np.concatenate([clean_vars_safe, immunized_vars_safe])

    mean_edges = np.histogram_bin_edges(
        np.concatenate([clean_means, immunized_means]), bins=bins
    )
    # Log-spaced bins for variance: with a handful of large outlier values,
    # linear bins crush almost all the real data into a single spike near
    # zero (this is exactly what happened in the eps32/no-mask/object run --
    # the panel showed one tall spike at ~0 and an unreadable empty stretch
    # out to 20+). Log-spaced bins + a log x-axis make the full range of
    # variance values readable instead.
    var_edges = np.logspace(
        np.log10(combined_vars_safe.min()), np.log10(combined_vars_safe.max()), bins + 1
    )

    axes[0].hist(clean_means, bins=mean_edges, density=True, label="clean",
                 histtype="step", linewidth=1.5, color="#1D9E75")
    axes[0].hist(immunized_means, bins=mean_edges, density=True, label=attack_label,
                 histtype="step", linewidth=1.5, color="#D85A30")
    annotate_histogram_panel(
        axes[0], clean_means, immunized_means, attack_label,
        kde_x=np.linspace(mean_edges[0], mean_edges[-1], 300),
    )
    axes[0].set_title("Latent mean\n(pooled across all positions/images)")
    axes[0].set_xlabel("mean value")
    axes[0].set_ylabel("density")
    axes[0].legend()

    axes[1].hist(clean_vars_safe, bins=var_edges, density=True, label="clean",
                 histtype="step", linewidth=1.5, color="#1D9E75")
    axes[1].hist(immunized_vars_safe, bins=var_edges, density=True, label=attack_label,
                 histtype="step", linewidth=1.5, color="#D85A30")
    axes[1].set_xscale("log")
    annotate_histogram_panel(
        axes[1], clean_vars_safe, immunized_vars_safe, attack_label,
        kde_x=np.logspace(np.log10(var_edges[0]), np.log10(var_edges[-1]), 300),
        log_x=True,
    )
    axes[1].set_title("Latent variance (log scale)\n(pooled across all positions/images)")
    axes[1].set_xlabel("variance value (log scale)")
    axes[1].set_ylabel("density")
    axes[1].legend()

    axes[2].scatter(clean_2d_plot[:, 0], clean_2d_plot[:, 1], s=4, alpha=0.35,
                     label="clean", color="#1D9E75")
    axes[2].scatter(immunized_2d_plot[:, 0], immunized_2d_plot[:, 1], s=4, alpha=0.35,
                     label=attack_label, color="#D85A30")
    axes[2].set_title("PCA of per-position channel vectors\n(2D projection of latent means)")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].legend(markerscale=3)

    fig.suptitle(f"VAE latent distribution: clean vs. {attack_label} "
                 f"({model_name}, n={len(clean_match)} images)")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {output_path.resolve()}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize the VAE latent distribution shift for H_max and/or H_min "
                    "immunization, one figure per attack given."
    )
    parser.add_argument("--clean", required=True, type=str, help="Folder of clean images.")
    parser.add_argument("--hmax", type=str, default=None,
                         help="Folder of H_max-immunized images (same filenames as --clean). "
                              "Omit to skip the H_max figure.")
    parser.add_argument("--hmin", type=str, default=None,
                         help="Folder of H_min-immunized images (same filenames as --clean). "
                              "Omit to skip the H_min figure.")
    parser.add_argument("--model", default="instruct_pix2pix",
                         choices=["flux", "instruct_pix2pix", "sd15"],
                         help="Model whose VAE to use for encoding (loaded once, shared "
                              "across both figures).")
    parser.add_argument("--image-size", type=int, default=None,
                         help="Optional square resize before encoding.")
    parser.add_argument("--output-dir", type=str, default="./latent_figures",
                         help="Directory to save the figure(s) into. Filenames are "
                              "latent_distribution_hmax.png / latent_distribution_hmin.png.")
    parser.add_argument("--bins", type=int, default=80, help="Histogram bin count.")
    parser.add_argument("--max-scatter-points", type=int, default=4000,
                         help="Cap on points plotted per condition in the PCA scatter.")
    parser.add_argument("--seed", type=int, default=0,
                         help="Random seed for scatter-plot subsampling.")

    args = parser.parse_args()

    if not args.hmax and not args.hmin:
        raise ValueError("At least one of --hmax or --hmin must be given.")

    clean_dir = Path(args.clean)
    if not clean_dir.is_dir():
        raise NotADirectoryError(f"--clean is not a directory: {clean_dir}")

    output_dir = Path(args.output_dir)

    device = get_device()
    print(f"Device: {device}")
    _, vae = load_model(model_name=args.model, device=device)

    if args.hmax:
        hmax_dir = Path(args.hmax)
        if not hmax_dir.is_dir():
            raise NotADirectoryError(f"--hmax is not a directory: {hmax_dir}")
        generate_attack_figure(
            clean_dir, hmax_dir, "hmax", vae, device, args.model,
            output_dir / "latent_distribution_hmax.png",
            args.image_size, args.bins, args.max_scatter_points, args.seed,
        )

    if args.hmin:
        hmin_dir = Path(args.hmin)
        if not hmin_dir.is_dir():
            raise NotADirectoryError(f"--hmin is not a directory: {hmin_dir}")
        generate_attack_figure(
            clean_dir, hmin_dir, "hmin", vae, device, args.model,
            output_dir / "latent_distribution_hmin.png",
            args.image_size, args.bins, args.max_scatter_points, args.seed,
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
