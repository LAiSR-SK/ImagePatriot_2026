#!/usr/bin/env python3
"""
visualize_latent_batch.py

Batch wrapper around visualize_latent_distribution.py. Walks the
structured immunization tree and produces a latent-distribution figure for
every combination, instead of one flat folder at a time.

visualize_latent_distribution.py is never modified -- it is invoked as a
subprocess once per figure.

TREE
----
    inputs/<method>_gaussian/<epsilon>/<mask|no_mask>/<category>/
    ref/clean/<category>/

where <method> is hmax or hmin, <epsilon> is 4/8/16/32 and <category> is
human/animal/object. Each immunized image is paired with the clean image
of the same filename.

FIGURES
-------
Two sets are produced:

  per_leaf/     one figure per method/epsilon/mask/category  (32)
                clean vs immunized for that exact folder
  per_epsilon/  one figure per method/epsilon                (8)
                both mask types and all categories pooled

mask/human and no_mask/human use the SAME filenames, so the pooled
per-epsilon figures cannot simply point the child at several folders.
Those runs stage a temporary folder of symlinks (hard links or copies
where symlinks are not allowed, e.g. Windows without Developer Mode) named
"<mask>__<category>__<filename>" on both the clean and immunized side, so
the child's exact-filename matching still lines the pairs up correctly.

OUTPUT
------
    <output-dir>/<model>/per_leaf/latent_distribution_<attack>_eps<E>_<mask>_<cat>.png
    <output-dir>/<model>/per_epsilon/latent_distribution_<attack>_eps<E>.png

<model> defaults to flux; pass --model instruct_pix2pix or sd15 to encode
with a different VAE.

Existing figures are reused, so an interrupted run resumes where it left
off. Delete a figure to force it to be redrawn.

Usage
-----
    python visualize_latent_batch.py                    # everything
    python visualize_latent_batch.py --scope epsilon    # just the 8 pooled
    python visualize_latent_batch.py --method hmin --epsilon 4
    python visualize_latent_batch.py --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent
INPUTS_ROOT  = BASE_DIR / "inputs"
CLEAN_ROOT   = BASE_DIR / "ref" / "clean"
OUTPUT_DIR   = BASE_DIR / "latent_figures_updated"
CHILD_SCRIPT = BASE_DIR / "visualize_latent_distribution.py"
# The child imports gaussian_ce_immunization, which lives here, not beside it.
IMMUNIZATION_DIR = BASE_DIR.parent / "immunization"

# folder name -> the flag the child expects, and the label it puts on the figure
METHODS = {
    "hmax_gaussian": ("--hmax", "hmax"),
    "hmin_gaussian": ("--hmin", "hmin"),
}

EPSILONS         = ["4", "8", "16", "32"]
MASK_TYPES       = ["mask", "no_mask"]
MASK_DIR_ALIASES = {"mask": "mask", "no_mask": "no_mask", "nomask": "no_mask"}
VALID_CATEGORIES = ["human", "animal", "object"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class Job:
    """One invocation of visualize_latent_distribution.py."""

    def __init__(self, kind, method, label, epsilon, mask, category,
                 pairs, out_png, direct=None):
        self.kind     = kind          # "leaf" or "epsilon"
        self.method   = method        # hmax_gaussian / hmin_gaussian
        self.label    = label         # hmax / hmin
        self.epsilon  = epsilon
        self.mask     = mask          # None when pooled
        self.category = category      # None when pooled
        # (clean_path, immunized_path, staged_name) for every matched pair
        self.pairs    = pairs
        self.out_png  = out_png
        # (clean_dir, immunized_dir) when the child can read the real
        # folders directly instead of a staged copy
        self.direct   = direct

    @property
    def title(self) -> str:
        where = (f"{self.mask}/{self.category}" if self.kind == "leaf"
                 else "all masks + categories")
        return f"{self.label} | eps={self.epsilon} | {where} | {len(self.pairs)} pairs"

    @property
    def sort_key(self):
        return (0 if self.kind == "leaf" else 1,
                self.label, int(self.epsilon),
                self.mask or "",
                VALID_CATEGORIES.index(self.category) if self.category in VALID_CATEGORIES else -1)


# ======================================================================
# Discovery
# ======================================================================

def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def by_stem(folder: Path, warn=None) -> dict[str, Path]:
    """Images keyed by lowercased filename stem -- the same convention the
    child uses, so pair counts here match what it would compute."""
    result: dict[str, Path] = {}
    collisions: dict[str, list[str]] = {}
    for path in list_images(folder):
        key = path.stem.lower()
        if key in result:
            collisions.setdefault(key, [result[key].name]).append(path.name)
        result[key] = path
    if collisions and warn is not None:
        example = sorted(collisions)[0]
        warn(f"{display(folder)}: {len(collisions)} filename(s) appear with more than "
             f"one extension (e.g. {' and '.join(collisions[example])}); only the last "
             f"is used, matching visualize_latent_distribution.py's own behaviour")
    return result


def pair_folder(clean_dir: Path, imm_dir: Path, prefix: str, warn) -> list[tuple[Path, Path, str]]:
    """Match a leaf against its clean folder by filename stem."""
    clean = by_stem(clean_dir, warn)
    imm   = by_stem(imm_dir, warn)
    common = sorted(set(clean) & set(imm))

    missing = sorted(set(imm) - set(clean))
    if missing:
        warn(f"{display(imm_dir)}: {len(missing)} image(s) have no clean counterpart "
             f"(e.g. {missing[0]})")
    if not common:
        warn(f"{display(imm_dir)}: no filenames match {display(clean_dir)} -- skipped")

    # The staged name must be unique once several leaves are pooled, and
    # identical on the clean and immunized side so the child pairs them.
    return [(clean[s], imm[s], f"{prefix}{imm[s].name}") for s in common]


def collect_jobs(inputs_root: Path, clean_root: Path, output_dir: Path,
                 scope: str, only_method, only_eps, only_mask, only_cat,
                 limit, warn) -> list[Job]:
    jobs: list[Job] = []
    # (method, epsilon) -> every pair under it, for the pooled figures
    pooled: dict[tuple, list] = defaultdict(list)

    for method, (_, label) in METHODS.items():
        if only_method and label not in only_method:
            continue
        method_dir = inputs_root / method
        if not method_dir.is_dir():
            warn(f"missing method folder: {display(method_dir)}")
            continue

        for epsilon in EPSILONS:
            if only_eps and epsilon not in only_eps:
                continue
            eps_dir = method_dir / epsilon
            if not eps_dir.is_dir():
                warn(f"missing epsilon folder: {display(eps_dir)}")
                continue

            for mask_dir in sorted(eps_dir.iterdir()):
                if not mask_dir.is_dir():
                    continue
                mask = MASK_DIR_ALIASES.get(mask_dir.name)
                if mask is None:
                    warn(f"unrecognized folder, skipped: {display(mask_dir)}")
                    continue

                for cat_dir in sorted(mask_dir.iterdir()):
                    if not cat_dir.is_dir():
                        continue
                    category = cat_dir.name
                    if category not in VALID_CATEGORIES:
                        warn(f"unrecognized category, skipped: {display(cat_dir)}")
                        continue

                    clean_dir = clean_root / category
                    if not clean_dir.is_dir():
                        warn(f"no clean folder for category '{category}': {display(clean_dir)}")
                        continue

                    pairs = pair_folder(clean_dir, cat_dir, f"{mask}__{category}__", warn)
                    if not pairs:
                        continue

                    # Every leaf contributes to its epsilon's pooled figure,
                    # even when the per-leaf figures are filtered out.
                    pooled[(method, label, epsilon)] += pairs

                    if scope in {"leaf", "both"}:
                        if only_mask and mask not in only_mask:
                            continue
                        if only_cat and category not in only_cat:
                            continue
                        selected = pairs[:limit] if limit else pairs
                        # No staging needed for a single folder at full size:
                        # the child can read the real directories.
                        direct = (clean_dir, cat_dir) if not limit else None
                        jobs.append(Job(
                            "leaf", method, label, epsilon, mask, category, selected,
                            output_dir / "per_leaf" /
                            f"latent_distribution_{label}_eps{epsilon}_{mask}_{category}.png",
                            direct,
                        ))

    if scope in {"epsilon", "both"}:
        for (method, label, epsilon), pairs in pooled.items():
            selected = pairs[:limit] if limit else pairs
            jobs.append(Job(
                "epsilon", method, label, epsilon, None, None, selected,
                output_dir / "per_epsilon" /
                f"latent_distribution_{label}_eps{epsilon}.png",
            ))

    jobs.sort(key=lambda j: j.sort_key)
    return jobs


def display(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


# ======================================================================
# Execution
# ======================================================================

def link_or_copy(src: Path, dst: Path) -> None:
    """Symlinks need admin rights or Developer Mode on Windows, so fall back
    to a hard link (same volume only), then to a plain copy."""
    try:
        os.symlink(src, dst)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(IMMUNIZATION_DIR), env.get("PYTHONPATH")) if p
    )
    return env


def run_job(job: Job, model: str, passthrough: list[str]) -> tuple[bool, str]:
    flag, label = METHODS[job.method]

    with contextlib.ExitStack() as stack:
        if job.direct is not None:
            clean_arg, imm_arg = job.direct
        else:
            # Pooled (or limited) runs need a flat folder per side, with
            # matching synthetic filenames on both.
            root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="latent_batch_")))
            clean_arg, imm_arg = root / "clean", root / label
            clean_arg.mkdir()
            imm_arg.mkdir()
            for clean_path, imm_path, name in job.pairs:
                stem = Path(name).stem
                link_or_copy(clean_path.resolve(), clean_arg / f"{stem}{clean_path.suffix}")
                link_or_copy(imm_path.resolve(),   imm_arg   / f"{stem}{imm_path.suffix}")

        # The child always writes latent_distribution_<label>.png into
        # --output-dir, so give it a private directory and move the result
        # to this job's own name.
        staging_out = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="latent_out_")))

        result = subprocess.run(
            [sys.executable, str(CHILD_SCRIPT),
             "--clean", str(clean_arg),
             flag,      str(imm_arg),
             "--model", model,
             "--output-dir", str(staging_out)] + passthrough,
            capture_output=True, text=True, env=child_env(),
        )

        if result.returncode != 0:
            tail = (result.stderr or result.stdout).strip().splitlines()
            return False, tail[-1] if tail else f"exit code {result.returncode}"

        produced = staging_out / f"latent_distribution_{label}.png"
        if not produced.is_file():
            return False, "child reported success but wrote no figure"

        job.out_png.parent.mkdir(parents=True, exist_ok=True)
        produced.replace(job.out_png)

    return True, ""


# ======================================================================
# Entry point
# ======================================================================

def csv_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Produce latent-distribution figures for every method/epsilon/"
                    "mask/category, by driving visualize_latent_distribution.py."
    )
    parser.add_argument("--inputs-root", type=Path, default=INPUTS_ROOT,
                        help="Root holding <method>_gaussian/<eps>/<mask>/<category>/ "
                             "(default: ./inputs).")
    parser.add_argument("--clean-root", type=Path, default=CLEAN_ROOT,
                        help="Root holding <category>/ clean images (default: ./ref/clean).")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Where figures are written (default: ./latent_figures_updated).")
    parser.add_argument("--scope", choices=["leaf", "epsilon", "both"], default="both",
                        help="Which figure sets to produce (default: both).")
    parser.add_argument("--method", type=csv_list,
                        help="Limit to these attacks, e.g. hmin or hmax,hmin.")
    parser.add_argument("--epsilon", type=csv_list,
                        help="Limit to these epsilons, e.g. 4,8.")
    parser.add_argument("--mask", type=csv_list,
                        help="Limit per-leaf figures to these mask types.")
    parser.add_argument("--category", type=csv_list,
                        help="Limit per-leaf figures to these categories.")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="Use at most N image pairs per figure. For smoke tests.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the figures that would be produced and stop.")
    parser.add_argument("--force", action="store_true",
                        help="Redraw figures that already exist.")
    # passed straight through to the child
    parser.add_argument("--model", default="flux",
                        choices=["flux", "instruct_pix2pix", "sd15"],
                        help="Whose VAE encodes the images (default: flux, the VAE "
                             "the immunization targets). Figures go under "
                             "<output-dir>/<model>/.")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--bins", type=int, default=None)
    parser.add_argument("--max-scatter-points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not CHILD_SCRIPT.is_file():
        print(f"error: {CHILD_SCRIPT.name} not found at {CHILD_SCRIPT}", file=sys.stderr)
        return 2
    for label, path in (("--inputs-root", args.inputs_root), ("--clean-root", args.clean_root)):
        if not path.is_dir():
            print(f"error: {label} is not a directory: {path}", file=sys.stderr)
            return 2

    passthrough = []
    for flag, value in (("--image-size", args.image_size), ("--bins", args.bins),
                        ("--max-scatter-points", args.max_scatter_points),
                        ("--seed", args.seed)):
        if value is not None:
            passthrough += [flag, str(value)]

    warnings: list[str] = []
    # Per-model subfolder: figures are reused when present, so without it a
    # flux run would skip every figure already drawn with another model.
    output_dir = args.output_dir / args.model
    jobs = collect_jobs(args.inputs_root, args.clean_root, output_dir,
                        args.scope, args.method, args.epsilon, args.mask,
                        args.category, args.limit, warnings.append)

    for message in warnings:
        print(f"  warn  {message}")

    if not jobs:
        print("\nNo figures to produce -- nothing matched the tree or the filters.")
        return 0

    todo = jobs if args.force else [j for j in jobs if not j.out_png.is_file()]
    skipped = len(jobs) - len(todo)

    n_leaf = sum(1 for j in jobs if j.kind == "leaf")
    n_eps  = sum(1 for j in jobs if j.kind == "epsilon")
    print(f"\n{len(jobs)} figure(s): {n_leaf} per-leaf, {n_eps} per-epsilon | "
          f"{len(todo)} to draw | {skipped} already present")

    if args.dry_run:
        print()
        for job in jobs:
            mark = "skip" if job.out_png.is_file() and not args.force else "draw"
            print(f"  {mark}  {job.title}")
            print(f"        -> {display(job.out_png)}")
        print("\nDry run -- nothing was drawn.")
        return 0

    if not todo:
        print("Nothing to do.")
        return 0

    print(f"\nThe VAE is reloaded once per figure, so this takes a while.")

    done = failed = 0
    for i, job in enumerate(todo, start=1):
        print(f"\n  [{i}/{len(todo)}] {job.title}")
        start = time.perf_counter()
        ok, detail = run_job(job, args.model, passthrough)
        elapsed = time.perf_counter() - start
        if ok:
            print(f"  ok    -> {display(job.out_png)} ({elapsed:.1f}s)")
            done += 1
        else:
            print(f"  FAIL  {job.title}: {detail} ({elapsed:.1f}s)", file=sys.stderr)
            failed += 1

    print(f"\nDone. {done} drawn, {failed} failed, {skipped} reused.")
    print(f"Figures in {output_dir.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
