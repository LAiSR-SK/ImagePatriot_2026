# ImagePatriot 2026

Research code for **image immunization**: adding a small, bounded perturbation
to a photo so that diffusion-based editors have a harder time editing it
convincingly. The perturbation targets the image encoder (the VAE) of
[FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).
The repository contains the full evaluation pipeline: immunize images, edit
them with FLUX.1-Kontext, score the results, and visualize how the latent
distribution shifts.

> **Research prototype.** This code is released to support reproducibility of
> our experiments. Immunization is **not** a guarantee of protection. It is
> evaluated against one editing model under specific settings, and it may not
> survive resizing, compression, screenshots, other editing models, or future
> countermeasures. Do not rely on it to protect sensitive images.

---

## Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
  - [1. Immunize: `immunize_batch.py`](#1-immunize-immunize_batchpy)
  - [2. Edit: `batch_edit.py`](#2-edit-batch_editpy)
  - [3. Score: `compare_batch.py`](#3-score-compare_batchpy)
  - [4. Visualize: `visualize_latent_batch.py`](#4-visualize-visualize_latent_batchpy)
- [Repository layout](#repository-layout)
- [Known limitations](#known-limitations)
- [Responsible use](#responsible-use)
- [Third-party models and licenses](#third-party-models-and-licenses)

---

## Requirements

- **An NVIDIA GPU with CUDA.** Editing with FLUX.1-Kontext requires one.
  `batch_edit.py` loads the model in bf16 when the GPU has ≥ 40 GB of VRAM.
  Otherwise it falls back to 4-bit (NF4) quantization with CPU offload. Smaller
  GPUs may work under 4-bit but are untested. The immunization and
  visualization steps only load the VAE, so they need far less memory.
- **Python 3.12.** The pinned PyTorch build (`2.6.0+cu126`) has no wheels for
  Python 3.14 or newer.
- **Disk space** for the model weights (tens of GB on first download, cached by
  Hugging Face) plus the generated images.
- **A Hugging Face account** with access to FLUX.1-Kontext-dev. See the next
  section.

Setup instructions are given for Linux and Windows. macOS is not supported
because the pinned PyTorch build requires CUDA.

---

## Setup

### 1. Get the code and create a virtual environment

```bash
git clone https://github.com/LAiSR-SK/ImagePatriot_2026.git ImagePatriot_2026
cd ImagePatriot_2026
```

**Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` already points pip at PyTorch's CUDA 12.6 package index.
Linux-only CUDA packages are skipped on other platforms automatically.

Check that PyTorch can see your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. Get access to FLUX.1-Kontext-dev

FLUX.1-Kontext-dev is a gated model with its own license.

1. Sign in at [huggingface.co](https://huggingface.co), open the
   [model page](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev),
   read the license, and accept it if you agree to its terms.
2. Create an access token (Settings → Access Tokens, "Read" is enough).
3. Log in from your terminal:

   ```bash
   hf auth login
   ```

The weights download on first use. The scripts never ask for your token
directly. Hugging Face's own client reads it.

---

## Dataset

The dataset is included in this repository under `images/`:

| Category | Images | Formats | Masks |
|----------|-------:|---------|------:|
| `animal` | 50 | PNG | none |
| `human` | 200 | PNG | 200 (one per image) |
| `object` | 50 | JPEG, PNG | none |

```
images/clean/animal/<images>
images/clean/human/<images>
images/clean/object/<images>
images/masks/mask_<image stem>.png      # human images only
```

Each image has one edit prompt, stored in a CSV of `filename,prompt` rows:

```
src/prompts/run_log_animal.csv
src/prompts/run_log_human.csv
src/prompts/run_log_object.csv
```

Filenames must match the CSVs **exactly, including the extension**. Some
object images share a name and differ only by extension (`image0003.jpg` and
`image0003.jpeg` are different photos with different prompts). Names such as
`0018 (1).png` are separate images, not duplicates. Do not rename them.

<!-- TODO before publishing: document where the images come from and the license they are released under. -->
> **Source and license:** the provenance and license of these images are still
> being confirmed and will be documented here. Until then, please use them only
> to reproduce the experiments in this repository, and do not redistribute them.

### Put the images where the scripts read them

The pipeline scripts read clean images from `src/immunization/inputs/clean/`
and masks from `src/immunization/masks/`. Copy the dataset there before
step 1:

**Linux**

```bash
mkdir -p src/immunization/inputs
cp -r images/clean src/immunization/inputs/clean
cp -r images/masks src/immunization/masks
```

**Windows (PowerShell)**

```powershell
New-Item -ItemType Directory -Force src/immunization/inputs | Out-Null
Copy-Item -Recurse images/clean src/immunization/inputs/clean
Copy-Item -Recurse images/masks src/immunization/masks
```

---

## Pipeline

Run all commands from the repository root. Every script has `--help`. Each
batch script **skips outputs that already exist**, so you can stop a run and
rerun the same command to resume it.

### 1. Immunize: `immunize_batch.py`

This step applies H_max and/or H_min to every clean image, at every ε, with and
without masks.

```bash
# everything: both attacks, all epsilons, masked + unmasked
python src/immunization/immunize_batch.py --method both --mask both

# a subset
python src/immunization/immunize_batch.py --method hmax --mask no_mask --epsilon 8 16
```

| Flag | Values | Default |
|------|--------|---------|
| `--method` *(required)* | `hmax`, `hmin`, `both` | |
| `--mask` *(required)* | `no_mask`, `mask`, `both` (masked runs are `human` only) | |
| `--epsilon` | one or more of `4 8 16 32` | all |
| `--input` | folder of `<category>/` clean images | `src/immunization/inputs/clean` |
| `--masks` | folder of `mask_<stem>.png` files | `src/immunization/masks` |
| `--output` | output root | `src/immunization/outputs` |

In a mask, **white marks where the perturbation is allowed** and black leaves
pixels untouched. Images are downscaled so the longest side is at most 512 px.
The PGD settings (100 steps, base step 1/255, κ = 1) are constants at the top
of the script.

**Output:**

```
src/immunization/outputs/<hmax|hmin>_gaussian/<epsilon>/<mask|no_mask>/<category>/<filename>
```

> **Single folder instead of a batch?** `gaussian_ce_immunization.py` runs the
> same attacks on any image or folder with every setting exposed as a flag
> (`--steps`, `--epsilon` in 0–1 units, `--kappa`, `--tv-weight`,
> `--target-loss-hmax`, …). Its output layout is `results/<model>/<hmax|hmin>/`,
> not the one the later stages expect.
>
> ```bash
> python src/immunization/gaussian_ce_immunization.py --model flux \
>     --input path/to/images --output ./results --attack both --epsilon 0.0314
> ```

### 2. Edit: `batch_edit.py`

This step edits images with FLUX.1-Kontext, using each image's prompt from the
run log. You need to run it **twice**:

- once on the **immunized** images, to measure how much immunization disrupts
  editing
- once on the **clean** images, to produce the reference edits that step 3
  compares against

```bash
# edits of the immunized images
python src/editing/batch_edit.py --method both --mask both \
    --input src/immunization/outputs

# reference edits of the clean images (--mask does not apply here)
python src/editing/batch_edit.py --method clean \
    --input src/immunization/inputs
```

| Flag | Values | Notes |
|------|--------|-------|
| `--method` *(required)* | `hmax`, `hmin`, `both`, `clean` | `both` = `hmax` + `hmin` |
| `--mask` | `mask`, `no_mask`, `both` | required unless `--method clean` |
| `--input` | root holding `<method>/…/<mask>/<category>/` or `clean/<category>/` | default `src/editing/inputs` |

Only images listed in the category's run-log CSV are edited. Anything else in
the folder is skipped with a message, so you can run it on any subset of the
dataset. Editing uses 28 inference steps and guidance scale 5.0.

**Output** mirrors the input path:

```
src/editing/outputs/<hmax|hmin>_gaussian/<epsilon>/<mask>/<category>/<filename>
src/editing/outputs/clean/<category>/<filename>
```

### 3. Score: `compare_batch.py`

This step computes eight full-reference image-quality metrics (DSS, GMSD,
HaarPSI, LPIPS, PSNR, SR-SIM, VIFp, VSI, via
[`piq`](https://github.com/photosynthesis-team/piq)) for two comparisons:

| Stage | Compares | Question it answers |
|-------|----------|---------------------|
| `immune` | immunized image vs. clean image | Is the perturbation imperceptible? (want **high** similarity) |
| `edit` | edit of immunized vs. edit of clean | Did immunization disrupt the edit? (want **low** similarity) |

`compare_batch.py` reads from `src/metrics/comp/` and `src/metrics/ref/`, so
first copy the outputs of steps 1 and 2 into place:

**Linux**

```bash
mkdir -p src/metrics/comp src/metrics/ref
cp -r src/immunization/outputs         src/metrics/comp/immune
cp -r src/immunization/inputs/clean    src/metrics/ref/clean
cp -r src/editing/outputs/clean        src/metrics/ref/edit
mkdir -p src/metrics/comp/edit
cp -r src/editing/outputs/*_gaussian   src/metrics/comp/edit/
```

**Windows (PowerShell)**

```powershell
New-Item -ItemType Directory -Force src/metrics/comp/edit, src/metrics/ref | Out-Null
Copy-Item -Recurse src/immunization/outputs       src/metrics/comp/immune
Copy-Item -Recurse src/immunization/inputs/clean  src/metrics/ref/clean
Copy-Item -Recurse src/editing/outputs/clean      src/metrics/ref/edit
Copy-Item -Recurse src/editing/outputs/*_gaussian src/metrics/comp/edit/
```

The resulting layout:

```
src/metrics/comp/immune/<method>/<epsilon>/<mask>/<category>/   vs   src/metrics/ref/clean/<category>/
src/metrics/comp/edit/<method>/<epsilon>/<mask>/<category>/     vs   src/metrics/ref/edit/<category>/
```

Then run:

```bash
python src/metrics/compare_batch.py --stage both --mask both

# quick smoke test on 5 images per folder
python src/metrics/compare_batch.py --stage both --mask both --limit 5
```

| Flag | Values |
|------|--------|
| `--stage` *(required)* | `immune`, `edit`, `both` |
| `--mask` *(required)* | `mask`, `no_mask`, `both` |
| `--comp`, `--ref`, `--metrics` | override the three folders (defaults under `src/metrics/`) |
| `--limit N` | score at most N images per folder |
| `--report-only` | reprint the summary from existing CSVs without rescoring |

**Output:**

```
src/metrics/metrics/raw/…/*.csv           per-image scores
src/metrics/metrics/summary_detail.csv    one row per folder
src/metrics/metrics/summary_by_method.csv one row per method / epsilon / mask
```

Higher is more similar for DSS, HaarPSI, PSNR, SR-SIM, VIFp, and VSI. GMSD
and LPIPS are distances, so **lower** is more similar. The summary table
labels each direction.

> For semantic (rather than pixel-level) similarity,
> `src/metrics/compare_clip_embeddings.py --reference <dir> --comparison <dir> …`
> computes CLIP image-to-image cosine similarity for any pair of folders.
> `compare_image_metrics.py` is the single-folder tool `compare_batch.py` calls
> internally.

### 4. Visualize: `visualize_latent_batch.py`

This step plots how immunization shifts the VAE's latent distribution. Each
figure compares clean vs. immunized using histograms of the latent mean and
variance (with Cohen's d and a KS test) and a 2-D PCA of latent vectors. It
reads the step 1 output directly, so no copying is needed:

```bash
python src/metrics/visualize_latent_batch.py \
    --inputs-root src/immunization/outputs \
    --clean-root  src/immunization/inputs/clean

# preview what would be drawn
python src/metrics/visualize_latent_batch.py \
    --inputs-root src/immunization/outputs \
    --clean-root  src/immunization/inputs/clean --dry-run
```

| Flag | Values | Default |
|------|--------|---------|
| `--scope` | `leaf` (per folder), `epsilon` (pooled per method/ε), `both` | `both` |
| `--method`, `--epsilon`, `--mask`, `--category` | comma-separated filters, e.g. `--epsilon 4,8` | all |
| `--output-dir` | where figures go | `src/metrics/latent_figures_updated` |
| `--limit N`, `--dry-run`, `--force` | smoke test / preview / redraw existing | |

**Output:**

```
<output-dir>/flux/per_leaf/latent_distribution_<method>_eps<E>_<mask>_<category>.png
<output-dir>/flux/per_epsilon/latent_distribution_<method>_eps<E>.png
```

---

## Repository layout

```
src/
├── immunization/
│   ├── gaussian_ce_immunization.py   H_max / H_min attacks (library + single-folder CLI)
│   └── immunize_batch.py             step 1: batch immunization
├── editing/
│   └── batch_edit.py                 step 2: FLUX.1-Kontext edits
├── metrics/
│   ├── compare_batch.py              step 3: batch image-quality scoring
│   ├── compare_image_metrics.py      single-folder scoring (used by compare_batch)
│   ├── compare_clip_embeddings.py    CLIP similarity between folders
│   ├── visualize_latent_batch.py     step 4: batch latent figures
│   └── visualize_latent_distribution.py  single-figure plotting (used by the batch)
└── prompts/
    └── run_log_<category>.csv        edit prompt per image
requirements.txt
```

---

## Known limitations

- **Saved format matters.** Immunized images keep their original filename and
  extension. Images that started as `.jpg` / `.jpeg` are re-saved as JPEG,
  and JPEG compression can weaken the perturbation. Keep this in mind when
  comparing categories with different source formats.
- **One target model.** The perturbation is optimized against the FLUX VAE.
  We make no claims about other editors or encoders.
- **Images are resized** to at most 512 px on the longest side (dimensions
  rounded down to a multiple of 8) before immunization and editing.
- **Nondeterminism.** Immunization uses fixed per-image seeds. The FLUX editing
  step is not seeded, so rerunning `batch_edit.py` produces different edits.
  GPU kernels and 4-bit quantization can also make results vary between
  hardware and driver versions.

## Responsible use

This project studies a defensive technique. Please:

- use only images you have the right to process, and respect the consent and
  privacy of people who appear in them
- do not present immunization as reliable protection to people whose images
  are at risk
- report results together with their limitations and the exact settings used
- follow the license and acceptable-use terms of every model you download

## Third-party models and licenses

These models download from Hugging Face on first use. Each one is governed by
its **own** license, separate from this repository's code. Review those terms
before use. Some, including FLUX.1-Kontext-dev, restrict commercial use.

| Used by | Model |
|---------|-------|
| immunization, editing, visualization | [`black-forest-labs/FLUX.1-Kontext-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) |
| CLIP similarity | [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) |
| LPIPS metric (via `piq`) | pretrained LPIPS weights fetched by `piq` |
