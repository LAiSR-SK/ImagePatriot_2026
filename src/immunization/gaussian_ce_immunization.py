#!/usr/bin/env python3
"""
gaussian_ce_immunization.py

Encoder-space image immunization using two distributional attacks:

    H_min:
        delta* = argmin_{||delta||_inf <= epsilon}
                 H( N(0, kappa I),
                    N(E(x + delta)) )

    H_max:
        delta* = argmax_{||delta||_inf <= epsilon}
                 H( N(E(x)),
                    N(E(x + delta)) )

Both attacks are run through ONE shared PGD routine (_run_pgd). Direction
(ascent vs descent) and which iterate counts as "best" (max vs min) are
both derived from a single `maximize` flag, so a caller cannot pair the
wrong PGD direction with its own objective.

Supported models:

    FLUX.1-Kontext-dev:
        black-forest-labs/FLUX.1-Kontext-dev

    InstructPix2Pix:
        timbrooks/instruct-pix2pix

    Stable Diffusion 1.5:
        stable-diffusion-v1-5/stable-diffusion-v1-5

IMPORTANT: image / VAE input range
-----------------------------------
Images are loaded, perturbed, saved, and epsilon-bounded entirely in
[0, 1] pixel units. Diffusers' AutoencoderKL (used by all three models
above) is trained on inputs normalized to [-1, 1] -- the same conversion
VaeImageProcessor.preprocess() applies internally. To keep delta/epsilon
meaningful in the original [0,1] pixel units while still feeding the
encoder what it expects, the [0,1] -> [-1,1] rescale happens ONLY at the
point of calling vae.encode() (see `_to_vae_input`), not throughout the
rest of the optimization.

IMPORTANT: mask
---------------
mask is optional, same shape as the image, values expected in [0, 1]
(binary {0,1} or a soft/graded mask). It is applied at TWO points, both
inside the single shared PGD routine:

    1. the per-step update:      delta = delta + direction*alpha*grad.sign()*mask
    2. the epsilon projection:   delta = clamp(delta, -epsilon*mask, epsilon*mask)

Scaling the projection by mask (not just the per-step update) matters for
a soft/graded mask: without it, a partially-masked pixel could still drift
up to the FULL epsilon budget over many iterations even though each
individual step was scaled down. For a strictly binary {0,1} mask this is
equivalent to the unscaled projection, since masked-out pixels are already
pinned at exactly 0 (both the random init and every update step are
themselves multiplied by mask).

PGD
---
H_max:
    gradient ascent
        delta <- delta + alpha * sign(grad) * mask

H_min:
    gradient descent
        delta <- delta - alpha * sign(grad) * mask

Step-size scheduler (shared by both attacks):

    first 25%:   8 * step_size
    25%-50%:     4 * step_size
    50%-75%:     1 * step_size
    final 25%:   2 * step_size

The best perturbation is stored BEFORE the next PGD update, so the saved
perturbation and saved objective always correspond to the same iterate.

Example
-------

    python gaussian_ce_immunization.py \
        --model sd15 \
        --input ./images \
        --output ./results \
        --attack both \
        --steps 50 \
        --step-size 0.0039215686 \
        --epsilon 0.031372549 \
        --mask ./masks/region.png
"""

from __future__ import annotations

import argparse
import gc
import math
import random
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import Tensor


# ======================================================================
# Model definitions
# ======================================================================

MODEL_IDS = {
    "flux": "black-forest-labs/FLUX.1-Kontext-dev",
    "instruct_pix2pix": "timbrooks/instruct-pix2pix",
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
}


# ======================================================================
# Random seed
# ======================================================================

def set_seed(seed: int) -> None:
    """Set all relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# Cleanup
# ======================================================================

def cleanup_memory() -> None:
    """Release unused Python and CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ======================================================================
# Device
# ======================================================================

def get_device() -> torch.device:
    """Select CUDA when available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ======================================================================
# Image loading
# ======================================================================

def find_images(input_path: Path) -> list[Path]:
    """Return all supported images from a file or directory."""
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {input_path.suffix}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not input_path.is_dir():
        raise ValueError(f"Input path is neither a file nor directory: {input_path}")

    images = sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError(f"No supported images found in: {input_path}")

    return images


def load_image(path: Path, image_size: Optional[int] = None) -> Tensor:
    """Load an RGB image and convert it to [1, 3, H, W] with values in [0,1]."""
    image = Image.open(path).convert("RGB")

    if image_size is not None:
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)

    array = np.asarray(image, dtype=np.float32)
    array /= 255.0

    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)

    return tensor


def load_mask(path: Path, image_shape: torch.Size, image_size: Optional[int] = None) -> Tensor:
    """
    Load a mask image and convert it to [1, C, H, W] in [0, 1], matching
    image_shape. The mask is loaded as grayscale and broadcast across all
    channels, so a single-channel mask image applies uniformly to R, G, B.
    """
    mask_image = Image.open(path).convert("L")

    if image_size is not None:
        mask_image = mask_image.resize((image_size, image_size), Image.Resampling.LANCZOS)

    array = np.asarray(mask_image, dtype=np.float32)
    array /= 255.0

    tensor = torch.from_numpy(array)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    num_channels = image_shape[1]
    tensor = tensor.expand(1, num_channels, tensor.shape[2], tensor.shape[3]).contiguous()

    if tensor.shape[2:] != image_shape[2:]:
        raise ValueError(
            f"Mask spatial shape {tuple(tensor.shape[2:])} does not match "
            f"image spatial shape {tuple(image_shape[2:])}."
        )

    return tensor


def save_image(tensor: Tensor, path: Path) -> None:
    """Save [1,3,H,W] tensor in [0,1] as an RGB image."""
    tensor = tensor.detach().float()
    tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0.0, 1.0)
    tensor = tensor.permute(1, 2, 0)

    array = tensor.cpu().numpy() * 255.0
    array = np.round(array).astype(np.uint8)

    image = Image.fromarray(array, mode="RGB")

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


# ======================================================================
# VAE input rescale: [0,1] pixel space -> [-1,1] encoder input space
# ======================================================================

def _to_vae_input(image_01: Tensor) -> Tensor:
    """
    Diffusers AutoencoderKL (SD1.5 / InstructPix2Pix / FLUX's VAE) expects
    input normalized to [-1, 1], matching VaeImageProcessor.preprocess().
    Images and epsilon/delta throughout this script are defined in [0,1]
    pixel units; rescale ONLY at the point of calling .encode(), so the
    perturbation budget keeps its original [0,1]-space meaning everywhere
    else in the optimization.
    """
    return image_01 * 2.0 - 1.0


# ======================================================================
# Gaussian cross entropy
# ======================================================================

def diagonal_gaussian_cross_entropy(
    ref_mean: Tensor,
    ref_var: Tensor,
    pred_mean: Tensor,
    pred_var: Tensor,
    var_floor: float = 1e-8,
) -> Tensor:
    """
    Cross entropy between two diagonal Gaussian distributions.

        p = N(ref_mean, diag(ref_var))
        q = N(pred_mean, diag(pred_var))

    Delta-dependent part only (constants omitted, do not affect optimum):

        sum[ log(pred_var) + ref_var/pred_var + (ref_mean-pred_mean)^2/pred_var ]

    IMPORTANT: all variance-dependent terms use pred_var, because q is the
    distribution being optimized -- the one that appears inside -E_p[log q(z)].
    """
    safe_pred_var = pred_var.clamp_min(var_floor)

    log_term = torch.sum(torch.log(safe_pred_var))
    trace_term = torch.sum(ref_var / safe_pred_var)
    mean_difference = ref_mean - pred_mean
    mahalanobis_term = torch.sum((mean_difference * mean_difference) / safe_pred_var)

    return log_term + trace_term + mahalanobis_term


# ======================================================================
# Step-size scheduler
# ======================================================================

def step_size_at(step: int, steps: int, step_size: float) -> float:
    """
    Piecewise scheduler.
        first 25%: 8x   25%-50%: 4x   50%-75%: 1x   final 25%: 2x
    """
    if steps <= 0:
        raise ValueError("steps must be greater than zero.")

    fraction = step / steps

    if fraction < 0.25:
        return step_size * 8.0
    elif fraction < 0.50:
        return step_size * 4.0
    elif fraction > 0.75:
        return step_size * 2.0
    else:
        return step_size


# ======================================================================
# Load model
# ======================================================================

def load_model(model_name: str, device: torch.device):
    """
    Load ONLY the VAE encoder for the requested model, not the full
    pipeline. This attack only ever calls vae.encode() -- it never runs
    the transformer/UNet, text encoders, tokenizer, scheduler, or safety
    checker, so loading the full pipeline (as diffusers' *Pipeline
    .from_pretrained() does) wastes a large amount of memory for no
    benefit. This matters most for FLUX.1-Kontext-dev: its full pipeline
    includes a ~12B-parameter transformer and a T5-XXL text encoder
    (tens of GB), while its VAE alone is a small fraction of that size.

    Returns (None, vae) -- None in place of a pipeline object, since none
    is loaded. Kept as a 2-tuple for compatibility with existing callers
    that unpack `pipeline, vae = load_model(...)`.
    """
    from diffusers import AutoencoderKL

    model_id = MODEL_IDS[model_name]

    print()
    print("=" * 70)
    print(f"Loading VAE only from: {model_id}")
    print("=" * 70)

    if model_name == "flux":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    elif model_name in {"instruct_pix2pix", "sd15"}:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Every model here organizes its VAE under a "vae" subfolder of the
    # same repo -- loading just that component skips downloading/
    # allocating memory for the transformer/UNet and text encoder(s)
    # entirely.
    #
    # IMPORTANT: load in the reduced-precision dtype above (halves the
    # download/transfer size), then immediately cast the VAE itself to
    # float32 before use. attack_hmax/attack_hmin/_run_pgd all keep the
    # image tensor in float32 throughout (needed to avoid NaN/overflow in
    # the loss's log(var) and 1/var terms, which have limited dynamic
    # range in fp16). Without this cast, a float32 image tensor hitting a
    # float16/bfloat16 VAE raises a dtype-mismatch error on the very
    # first .encode() call. The VAE is tiny (well under a GB even for
    # FLUX), so keeping it in float32 for the actual attack has a
    # negligible memory cost -- nowhere near what full-pipeline loading
    # cost before the load_model fix above.
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae = vae.to(device).float()
    vae.eval()

    # Freeze parameters. This does NOT prevent gradients with respect to
    # the image -- only weight gradients are disabled.
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    print(f"VAE:      {type(vae).__name__}")
    print(f"Device:   {device}")
    print(f"Download dtype: {dtype}  (VAE itself cast to float32 for the attack, see comment above)")

    return None, vae


# ======================================================================
# Raw encoder distribution
# ======================================================================

def encode_raw_distribution(vae: Any, image_01: Tensor):
    """
    Rescale [0,1] -> [-1,1] and encode. Returns the raw latent_dist,
    exposing .mean / .var / .logvar. No diffusion-model latent scaling
    factor is applied -- the cross-entropy objective operates directly on
    the encoder's own Gaussian.
    """
    encoded = vae.encode(_to_vae_input(image_01))
    return encoded.latent_dist


# ======================================================================
# Shared PGD routine
# ======================================================================

def _total_variation(delta: Tensor) -> Tensor:
    """
    Cheap smoothness penalty on the perturbation itself. Sign-gradient PGD
    tends to produce salt-and-pepper-style high-frequency noise, which is
    more visually noticeable per unit of pixel-intensity change than
    smooth noise of the same magnitude. Lower TV = smoother/less noisy.
    """
    dh = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
    dw = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
    return dh + dw


def _run_pgd(
    image: Tensor,
    vae: Any,
    steps: int,
    step_size: float,
    epsilon: float,
    maximize: bool,
    objective_fn: Callable[[Any], Tensor],
    name: str,
    seed: int,
    mask: Optional[Tensor] = None,
    target_loss: Optional[float] = None,
    tv_weight: float = 0.0,
    clamp_candidate: bool = True,
) -> tuple[Tensor, float, list[float]]:
    """
    Single PGD loop used by BOTH attack_hmax and attack_hmin.

    maximize=True  -> ascent, keep the maximum objective  (H_max, Eq. 4)
    maximize=False -> descent, keep the minimum objective (H_min, Eq. 3)

    objective_fn(candidate_dist) -> scalar Tensor. Everything about which
    reference distribution is fixed and how it's compared lives in
    objective_fn, supplied by the caller (attack_hmax / attack_hmin);
    this function only implements the shared PGD mechanics: gradient,
    sign, direction, step-size schedule, mask, and epsilon projection.

    Quality/imperceptibility controls (all optional, default to the
    original unmodified behavior):

      clamp_candidate: if True (default), candidate_image = image + delta
          is clamped to [0,1] BEFORE encoding at every step, not just on
          the final returned image. Without this, some gradient signal
          driving delta is computed from pixel values that never actually
          appear in the output (they'd be clipped away at the end anyway),
          which wastes perturbation budget and can produce a less
          purposeful, more visible result for the same epsilon.

      target_loss: if set, PGD stops EARLY once the running best objective
          reaches this value, instead of always consuming the full `steps`
          budget. Often the single biggest lever for a cleaner-looking
          image: many images cross a "good enough" divergence well before
          the epsilon ball is maxed out, and continuing past that point
          mostly adds visible noise for little further disruption.

      tv_weight: if > 0, subtracts (H_max) or adds (H_min) a Total
          Variation penalty on delta, scaled by tv_weight, discouraging
          high-frequency salt-and-pepper noise in favor of smoother
          perturbations at the same epsilon budget. Try small values
          (e.g. 0.01-0.1) and tune by eye against your own images.
    """
    if steps <= 0:
        raise ValueError("steps must be greater than zero.")
    if step_size <= 0:
        raise ValueError("step_size must be greater than zero.")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    set_seed(seed)

    image = image.float()

    if mask is None:
        mask = torch.ones_like(image)
    else:
        mask = mask.float().to(image.device)
        if mask.shape != image.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {image.shape}."
            )

    # Random init over the full epsilon ball, scaled by mask so masked-out
    # pixels start at exactly 0 rather than a random value that then gets
    # clamped down on the first projection step.
    delta = ((torch.rand_like(image) * 2.0) - 1.0) * epsilon * mask
    delta = delta.detach()

    best_delta = delta.clone()
    best_objective = -math.inf if maximize else math.inf
    history: list[float] = []

    progress = tqdm(range(steps), desc=name)

    for step in progress:
        delta = delta.detach().requires_grad_(True)

        candidate_image = image + delta
        if clamp_candidate:
            # Clamp BEFORE encoding, not just on the final output, so every
            # gradient step reflects an image that could actually be
            # returned -- see the clamp_candidate docstring above.
            candidate_image = candidate_image.clamp(0.0, 1.0)
        candidate_dist = encode_raw_distribution(vae, candidate_image)

        objective = objective_fn(candidate_dist)

        if tv_weight > 0:
            tv_penalty = _total_variation(delta)
            # H_max wants the adversarial term large -> subtract the
            # penalty so smoother deltas are preferred among otherwise
            # similar objective values. H_min wants the adversarial term
            # small -> add the penalty so it moves the same direction.
            scored_objective = objective - tv_weight * tv_penalty if maximize \
                else objective + tv_weight * tv_penalty
        else:
            scored_objective = objective

        objective_value = objective.item()  # reported/tracked value stays the pure H(p,q)
        history.append(objective_value)

        progress.set_description(f"{name}: {objective_value:.4f}")

        # IMPORTANT: save the CURRENT perturbation, before the update step
        # below reassigns `delta`. objective_value was computed from this
        # exact delta, so best_delta must be saved here, not after the
        # update -- otherwise best_delta and best_objective would refer to
        # two different iterates.
        is_better = (objective_value > best_objective) if maximize else (objective_value < best_objective)
        if is_better:
            best_objective = objective_value
            best_delta = delta.detach().clone()

        if target_loss is not None:
            reached = (best_objective >= target_loss) if maximize else (best_objective <= target_loss)
            if reached:
                print(f"\n[{name}] target_loss={target_loss} reached at step {step} "
                      f"(best_objective={best_objective:.4f}) -- stopping early.")
                break

        gradient = torch.autograd.grad(scored_objective, delta)[0]
        alpha = step_size_at(step, steps, step_size)
        direction = 1.0 if maximize else -1.0

        with torch.no_grad():
            delta = delta + direction * alpha * gradient.sign() * mask

            # Projection scaled explicitly by mask, not just the fixed
            # epsilon range. For a binary {0,1} mask this is equivalent to
            # the unscaled clamp, since masked-out pixels are already
            # pinned at exactly 0 (init and every update step are also
            # multiplied by mask). For a soft mask in (0,1), this caps
            # each pixel's accumulated perturbation at epsilon * mask,
            # instead of letting it drift toward the full epsilon over
            # many steps even though each individual step was scaled down.
            delta = torch.clamp(delta, min=-epsilon * mask, max=epsilon * mask)

    final = (image + best_delta).clamp(0.0, 1.0)

    # The clamp above can trim a pixel whose image+best_delta fell outside
    # [0,1] (possible whenever a pixel starts near 0 or 1), which means
    # best_objective -- measured pre-clamp -- may not exactly describe the
    # returned image on that small subset of pixels. Recompute on the
    # actual returned image so best_objective is always an exact,
    # verifiable description of what's returned, not an approximation.
    with torch.no_grad():
        final_dist = encode_raw_distribution(vae, final)
        best_objective = objective_fn(final_dist).item()

    return final.detach(), best_objective, history


# ======================================================================
# H_max
# ======================================================================

def attack_hmax(
    image: Tensor,
    vae: Any,
    steps: int,
    step_size: float,
    epsilon: float,
    seed: int,
    mask: Optional[Tensor] = None,
    target_loss: Optional[float] = None,
    tv_weight: float = 0.0,
    clamp_candidate: bool = True,
) -> tuple[Tensor, float, list[float]]:
    """
    H_max attack, Eq. 4:
        delta* = argmax H( N(E(x)), N(E(x + delta)) )
    PGD performs gradient ASCENT via the shared _run_pgd routine.

    See _run_pgd's docstring for target_loss / tv_weight / clamp_candidate
    -- quality/imperceptibility controls, all optional and off by default
    except clamp_candidate which defaults to on.
    """
    image = image.float()

    # Fixed reference distribution N(E(x)), computed once from the clean
    # image (rescaled to [-1,1] for the encoder), independent of delta.
    with torch.no_grad():
        original_dist = encode_raw_distribution(vae, image)
        reference_mean = original_dist.mean.detach().float()
        reference_var = original_dist.var.detach().float()

    def objective(candidate_dist: Any) -> Tensor:
        return diagonal_gaussian_cross_entropy(
            ref_mean=reference_mean,
            ref_var=reference_var,
            pred_mean=candidate_dist.mean.float(),
            pred_var=candidate_dist.var.float(),
        )

    return _run_pgd(
        image=image, vae=vae, steps=steps, step_size=step_size, epsilon=epsilon,
        maximize=True, objective_fn=objective, name="H_max", seed=seed, mask=mask,
        target_loss=target_loss, tv_weight=tv_weight, clamp_candidate=clamp_candidate,
    )


# ======================================================================
# H_min
# ======================================================================

def attack_hmin(
    image: Tensor,
    vae: Any,
    steps: int,
    step_size: float,
    epsilon: float,
    kappa: float,
    seed: int,
    mask: Optional[Tensor] = None,
    target_loss: Optional[float] = None,
    tv_weight: float = 0.0,
    clamp_candidate: bool = True,
) -> tuple[Tensor, float, list[float]]:
    """
    H_min attack, Eq. 3:
        delta* = argmin H( N(0, kappa I), N(E(x + delta)) )
    PGD performs gradient DESCENT via the shared _run_pgd routine.

    See _run_pgd's docstring for target_loss / tv_weight / clamp_candidate.
    """
    if kappa <= 0:
        raise ValueError("kappa must be greater than zero.")

    image = image.float()

    def objective(candidate_dist: Any) -> Tensor:
        # Fixed reference N(0, kappa I) -- mean is genuinely zero, NOT the
        # original image's mean; this reference does not depend on image
        # at all, per Eq. 3.
        zero_mean = torch.zeros_like(candidate_dist.mean).float()
        kappa_var = torch.full_like(candidate_dist.var, fill_value=kappa).float()
        return diagonal_gaussian_cross_entropy(
            ref_mean=zero_mean,
            ref_var=kappa_var,
            pred_mean=candidate_dist.mean.float(),
            pred_var=candidate_dist.var.float(),
        )

    return _run_pgd(
        image=image, vae=vae, steps=steps, step_size=step_size, epsilon=epsilon,
        maximize=False, objective_fn=objective, name="H_min", seed=seed, mask=mask,
        target_loss=target_loss, tv_weight=tv_weight, clamp_candidate=clamp_candidate,
    )


# ======================================================================
# Statistics
# ======================================================================

def report_statistics(
    original: Tensor, immunized: Tensor, epsilon: float, attack_name: str, objective: float,
) -> None:
    """Print perturbation statistics."""
    difference = (immunized - original).abs()
    max_difference = difference.max().item()
    mean_difference = difference.mean().item()

    print(f"\n[{attack_name}]")
    print(f"    objective:       {objective:.6f}")
    print(f"    max |delta|:     {max_difference:.6f}")
    print(f"    mean |delta|:    {mean_difference:.6f}")
    print(f"    epsilon:         {epsilon:.6f}")

    if max_difference <= epsilon + 1e-6:
        print("    epsilon check:   PASS")
    else:
        print("    epsilon check:   WARNING")


# ======================================================================
# Process one image
# ======================================================================

def resolve_mask_path(mask_input: Optional[Path], image_path: Path) -> Optional[Path]:
    """
    Resolve the mask to use for a given image.

    mask_input is None:
        No mask -- returns None.

    mask_input is a FILE:
        The same single mask is used for every image in the batch
        (original behavior).

    mask_input is a DIRECTORY:
        Looks for a mask file in that directory matching image_path's
        stem, under either of two names -- the bare stem, or the stem
        with a "mask_" prefix (e.g. "photo.jpg" -> "photo.png" or
        "mask_photo.png"; any supported extension, only the name needs
        to match). The bare stem wins if both exist.

        Returns None if no match is found, meaning that image is
        immunized WITHOUT a mask. This supports datasets where only some
        categories have masks (e.g. masks for human images but not for
        animal or object), so one run can cover a mixed input tree
        instead of failing on every unmasked image.
    """
    if mask_input is None:
        return None

    if mask_input.is_file():
        return mask_input

    if mask_input.is_dir():
        accepted_stems = (image_path.stem, f"mask_{image_path.stem}")

        candidates = sorted(
            (path for path in mask_input.iterdir()
             if path.is_file()
             and path.stem in accepted_stems
             and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: (accepted_stems.index(path.stem), path.name),
        )

        if not candidates:
            return None

        if len(candidates) > 1:
            print(
                f"WARNING: multiple mask candidates found for "
                f"'{image_path.stem}' in {mask_input}, using {candidates[0].name}"
            )

        return candidates[0]

    raise FileNotFoundError(f"Mask path does not exist: {mask_input}")


def process_image(
    image_path: Path,
    output_root: Path,
    model_name: str,
    vae: Any,
    attack: str,
    steps: int,
    step_size: float,
    epsilon: float,
    kappa: float,
    seed: int,
    image_size: Optional[int],
    mask_input: Optional[Path],
    target_loss_hmax: Optional[float] = None,
    target_loss_hmin: Optional[float] = None,
    tv_weight: float = 0.0,
    clamp_candidate: bool = True,
) -> None:
    """Run selected attacks on one image.

    target_loss_hmax / target_loss_hmin: separate early-stop thresholds
        per attack, since H_max and H_min operate on different objective
        scales (H_max's objective typically differs numerically from
        H_min's -- see the printed history from an unbounded run to pick
        sensible values for each).
    tv_weight / clamp_candidate: shared quality controls, see _run_pgd's
        docstring for what each does.
    """
    print()
    print(f"Processing: {image_path}")

    image = load_image(image_path, image_size=image_size)
    device = next(vae.parameters()).device
    image = image.to(device)

    mask: Optional[Tensor] = None
    mask_path = resolve_mask_path(mask_input, image_path)
    if mask_path is not None:
        print(f"Mask:       {mask_path}")
        mask = load_mask(mask_path, image.shape, image_size=image_size)
        mask = mask.to(device)

    if attack in {"hmax", "both"}:
        print()
        print("Running H_max...")

        result, objective, _ = attack_hmax(
            image=image, vae=vae, steps=steps, step_size=step_size,
            epsilon=epsilon, seed=seed, mask=mask,
            target_loss=target_loss_hmax, tv_weight=tv_weight,
            clamp_candidate=clamp_candidate,
        )

        output_path = (
            output_root / model_name / "hmax"
            / image_path.parent.name / image_path.name
        )
        save_image(result, output_path)
        report_statistics(image, result, epsilon, "H_max", objective)

    if attack in {"hmin", "both"}:
        print()
        print("Running H_min...")

        result, objective, _ = attack_hmin(
            image=image, vae=vae, steps=steps, step_size=step_size,
            epsilon=epsilon, kappa=kappa, seed=seed, mask=mask,
            target_loss=target_loss_hmin, tv_weight=tv_weight,
            clamp_candidate=clamp_candidate,
        )

        output_path = (
            output_root / model_name / "hmin"
            / image_path.parent.name / image_path.name
        )
        save_image(result, output_path)
        report_statistics(image, result, epsilon, "H_min", objective)


# ======================================================================
# Command-line arguments
# ======================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Encoder-space Gaussian cross-entropy immunization attack."
    )

    parser.add_argument("--model", required=True, choices=["flux", "instruct_pix2pix", "sd15"],
                         help="Model used for the VAE encoder.")
    parser.add_argument("--input", required=True, type=str,
                         help="Input image or directory containing images.")
    parser.add_argument("--output", default="./results", type=str,
                         help="Directory where immunized images are saved.")
    parser.add_argument("--attack", choices=["hmax", "hmin", "both"], default="both",
                         help="Attack to run.")
    parser.add_argument("--steps", type=int, default=50, help="Number of PGD iterations.")
    parser.add_argument("--step-size", type=float, default=1.0 / 255.0, help="Base PGD step size.")
    parser.add_argument("--epsilon", type=float, default=8.0 / 255.0, help="L-infinity perturbation budget.")
    parser.add_argument("--kappa", type=float, default=1.0, help="Variance of the H_min isotropic Gaussian.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--image-size", type=int, default=None,
                         help="Optional square image size. If omitted, original resolution is preserved.")
    parser.add_argument("--mask", type=str, default=None,
                         help="Optional path to a mask. Either a single grayscale image file "
                              "(applied to every input image), or a DIRECTORY of masks, one per "
                              "input image, matched by filename stem (e.g. input 'photo.jpg' "
                              "looks for a mask named 'photo.<ext>' in the mask directory). "
                              "Mask values in [0,255] are interpreted as [0,1].")
    parser.add_argument("--target-loss-hmax", type=float, default=None,
                         help="Optional early-stop threshold for H_max. PGD stops once the running "
                              "best objective reaches this value, instead of always using all "
                              "--steps. Run once without this flag first to see where H_max's "
                              "objective naturally plateaus (printed each step), then set this "
                              "just past that point on a later run.")
    parser.add_argument("--target-loss-hmin", type=float, default=None,
                         help="Optional early-stop threshold for H_min, same idea as "
                              "--target-loss-hmax but for the H_min run (their objective scales "
                              "differ, hence separate flags).")
    parser.add_argument("--tv-weight", type=float, default=0.0,
                         help="Optional Total Variation penalty weight on the perturbation "
                              "(0 = off). Discourages high-frequency salt-and-pepper noise in "
                              "favor of a smoother-looking perturbation at the same epsilon. "
                              "Try small values, e.g. 0.02-0.1.")
    parser.add_argument("--no-clamp-candidate", action="store_true", default=False,
                         help="Disable per-step clamping of image+delta to [0,1] before encoding "
                              "(clamping is ON by default). Only disable this to reproduce the "
                              "original unclamped behavior; leaving it on avoids wasting gradient "
                              "budget on pixel values that would be clipped away in the final "
                              "output anyway.")

    return parser.parse_args()


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_arguments()

    if args.steps <= 0:
        raise ValueError("--steps must be greater than zero.")
    if args.step_size <= 0:
        raise ValueError("--step-size must be greater than zero.")
    if args.epsilon < 0:
        raise ValueError("--epsilon cannot be negative.")
    if args.kappa <= 0:
        raise ValueError("--kappa must be greater than zero.")

    device = get_device()

    print()
    print("=" * 70)
    print("GAUSSIAN CROSS-ENTROPY IMMUNIZATION")
    print("=" * 70)
    print(f"Model:       {MODEL_IDS[args.model]}")
    print(f"Attack:      {args.attack}")
    print(f"Steps:       {args.steps}")
    print(f"Base step:   {args.step_size}")
    print(f"Epsilon:     {args.epsilon}")
    print(f"Kappa:       {args.kappa}")
    print(f"Mask:        {args.mask if args.mask else '(none)'}")
    print(f"Target loss (H_max): {args.target_loss_hmax if args.target_loss_hmax is not None else '(none, uses full --steps)'}")
    print(f"Target loss (H_min): {args.target_loss_hmin if args.target_loss_hmin is not None else '(none, uses full --steps)'}")
    print(f"TV weight:   {args.tv_weight}")
    print(f"Clamp candidate each step: {not args.no_clamp_candidate}")
    print(f"Device:      {device}")

    input_path = Path(args.input)
    images = find_images(input_path)
    print(f"Images found: {len(images)}")

    mask_path = Path(args.mask) if args.mask else None
    if mask_path is not None and not mask_path.exists():
        raise FileNotFoundError(f"Mask path does not exist: {mask_path}")

    pipeline, vae = load_model(model_name=args.model, device=device)

    output_root = Path(args.output)
    successful = 0
    failed = 0

    for index, image_path in enumerate(images, start=1):
        print()
        print("=" * 70)
        print(f"IMAGE {index}/{len(images)}")
        print("=" * 70)

        try:
            process_image(
                image_path=image_path,
                output_root=output_root,
                model_name=args.model,
                vae=vae,
                attack=args.attack,
                steps=args.steps,
                step_size=args.step_size,
                epsilon=args.epsilon,
                kappa=args.kappa,
                seed=args.seed + index,
                image_size=args.image_size,
                mask_input=mask_path,
                target_loss_hmax=args.target_loss_hmax,
                target_loss_hmin=args.target_loss_hmin,
                tv_weight=args.tv_weight,
                clamp_candidate=not args.no_clamp_candidate,
            )
            successful += 1

        except Exception as exc:
            failed += 1
            print()
            print(f"ERROR: {image_path}")
            print(f"{type(exc).__name__}: {exc}")

        finally:
            cleanup_memory()

    del pipeline
    del vae
    cleanup_memory()

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Results:    {output_root.resolve()}")


if __name__ == "__main__":
    main()
