#!/usr/bin/env python3
"""
compare_batch.py

Batch wrapper around compare_image_metrics.py. Walks the `comp/` tree the
same way batch_edit_ip2p.py walks `inputs/`, pairs every comparison folder
with the matching reference folder, and rolls the results up into one
readable report.

compare_image_metrics.py itself is never modified -- it is invoked as a
subprocess once per (method, epsilon, mask type, category) leaf.

PAIRING
-------
    comp/immune/<method>/.../<category>/   <->   ref/clean/<category>/
    comp/edit/<method>/.../<category>/     <->   ref/edit/<category>/

Matching is by filename: image_0.png is scored against image_0.png.

LAYOUTS
-------
Two comparison folder layouts are recognized, detected automatically from
what is actually on disk:

    <method>/<epsilon>/<mask_type>/<category>/    (epsilon methods)
    <method>/<mask_type>/<category>/              (no epsilon)

where <epsilon> is 4/8/16/32 and <mask_type> is mask or no_mask.
Anything else is reported and skipped rather than silently ignored.

VARIANTS
--------
Some methods emit several attacked images per source image (diffProtect
writes image_0_attacked / image_0_multistep / image_0_onestep). Each
variant is scored separately against the same reference and gets its own
row, and an extra combined row ("ALL") averages across every variant.

OUTPUT
------
    metrics/raw/<stage>/<method>/[<eps>/]<mask>/<category>[__variant].csv
        per-image scores, straight from compare_image_metrics.py
    metrics/summary_detail.csv    one row per leaf (+ ALL rows)
    metrics/summary_by_method.csv one row per method/epsilon/mask
        plus both tables printed to the terminal.

Existing raw CSVs are reused, so an interrupted run resumes where it left
off. Delete metrics/raw to force a full rescore.

Usage
-----
    python compare_batch.py --stage both --mask both
    python compare_batch.py --stage edit --mask no_mask
    python compare_batch.py --stage both --mask both --limit 5   # quick smoke test
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

BASE_DIR     = Path(__file__).parent
COMP_DIR     = BASE_DIR / "comp"
REF_DIR      = BASE_DIR / "ref"
METRICS_DIR  = BASE_DIR / "metrics"
CHILD_SCRIPT = BASE_DIR / "compare_image_metrics.py"

# stage -> reference subfolder. comp/immune is scored against the clean
# originals; comp/edit against the edits of those clean originals.
STAGE_REF = {"immune": "clean", "edit": "edit"}

EPSILONS         = ["4", "8", "16", "32"]
VALID_CATEGORIES = ["human", "animal", "object"]
MASK_TYPES       = ["mask", "no_mask"]
# tolerate either spelling of the no_mask folder
MASK_DIR_ALIASES = {"mask": "mask", "no_mask": "no_mask", "nomask": "no_mask"}
# folder names that differ only by case across stages
METHOD_CANONICAL = {"photoguard": "photoGuard"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# The pre-reorganization photoGuard folder names, recognized only so the
# run can say why they were skipped.
_LEGACY_PG = re.compile(r"^photoguard_\d+_(with_mask|no_mask)_", re.IGNORECASE)

# Direction of every metric compare_image_metrics.py reports.
# True  = higher is more similar to the reference.
# False = it is a distance, lower is more similar.
HIGHER_IS_BETTER = {
    "DSS": True, "GMSD": False, "HaarPSI": True, "LPIPS": False,
    "PSNR": True, "SRSIM": True, "VIFp": True, "VSI": True,
}
ID_COLUMNS = ["reference_image", "comparison_folder", "comparison_image"]


# ======================================================================
# Work item
# ======================================================================

class Run:
    """One invocation of compare_image_metrics.py."""

    def __init__(self, stage, method, epsilon, mask_type, category,
                 variant, comp_dir, ref_dir, out_csv, staged_files=None):
        self.stage        = stage
        self.method       = method
        self.epsilon      = epsilon          # None for methods without one
        self.mask_type    = mask_type
        self.category     = category
        self.variant      = variant          # None when filenames match exactly
        self.comp_dir     = comp_dir
        self.ref_dir      = ref_dir
        self.out_csv      = out_csv
        # When set, only these files are exposed to the child (via a
        # temporary folder of symlinks) instead of the whole comp_dir.
        self.staged_files = staged_files

    @property
    def label(self) -> str:
        eps = self.epsilon if self.epsilon else "-"
        var = f" [{self.variant}]" if self.variant else ""
        return f"{self.stage} | {self.method}{var} | eps={eps} | {self.mask_type} | {self.category}"

    @property
    def sort_key(self):
        return (
            list(STAGE_REF).index(self.stage),
            self.method.lower(),
            int(self.epsilon) if self.epsilon else -1,
            self.mask_type,
            VALID_CATEGORIES.index(self.category) if self.category in VALID_CATEGORIES else 99,
            self.variant or "",
        )


# ======================================================================
# Discovery
# ======================================================================

def display(path: Path) -> str:
    """Path relative to the project when possible, absolute otherwise, so
    custom --comp/--ref/--metrics roots still print cleanly."""
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def canonical_method(name: str) -> str:
    return METHOD_CANONICAL.get(name.lower(), name)


def split_variants(comp_files: list[Path], ref_stems: set[str]) -> tuple[dict[str, list[Path]], list[Path]]:
    """
    Group comparison files by the suffix they add to a reference stem.

    "image_0.png" against reference "image_0" yields suffix "" (exact
    match); "image_0_attacked.png" yields "_attacked". Files matching no
    reference stem are returned separately.
    """
    by_suffix: dict[str, list[Path]] = defaultdict(list)
    unmatched: list[Path] = []

    for path in comp_files:
        stem = path.stem
        if stem in ref_stems:
            by_suffix[""].append(path)
            continue
        # longest reference stem that is a prefix ending on a separator,
        # so "cat" matches "cat_v1" but never "catalog"
        best = None
        for ref_stem in ref_stems:
            if (stem.startswith(ref_stem)
                    and not stem[len(ref_stem)].isalnum()
                    and (best is None or len(ref_stem) > len(best))):
                best = ref_stem
        if best is None:
            unmatched.append(path)
        else:
            by_suffix[stem[len(best):]].append(path)

    return dict(by_suffix), unmatched


def build_runs(leaf: Path, stage: str, method: str, epsilon: str | None,
               mask_type: str, category: str, warn) -> list[Run]:
    """Turn one comparison leaf folder into one Run per variant."""
    ref_dir = REF_DIR / STAGE_REF[stage] / category
    if not ref_dir.is_dir():
        warn(f"no reference folder for category '{category}': {ref_dir}")
        return []

    comp_files = list_images(leaf)
    if not comp_files:
        warn(f"no images in {display(leaf)}")
        return []

    ref_stems = {p.stem for p in list_images(ref_dir)}
    by_suffix, unmatched = split_variants(comp_files, ref_stems)
    if unmatched:
        warn(f"{len(unmatched)} file(s) in {display(leaf)} match no "
             f"reference image (e.g. {unmatched[0].name})")
    if not by_suffix:
        return []

    parts = [stage, method] + ([epsilon] if epsilon else []) + [mask_type]
    out_dir = METRICS_DIR.joinpath("raw", *parts)

    runs = []
    single = len(by_suffix) == 1
    for suffix, files in sorted(by_suffix.items()):
        variant = suffix.lstrip("_-. ") or None
        # Only hand the child a filtered folder when this leaf really does
        # hold several variants; otherwise point it straight at the source.
        staged = None if single else files
        name = category if variant is None else f"{category}__{variant}"
        runs.append(Run(stage, method, epsilon, mask_type, category, variant,
                        leaf, ref_dir, out_dir / f"{name}.csv", staged))
    return runs


def collect_runs(stages: list[str], mask_types: list[str]) -> tuple[list[Run], list[str]]:
    runs: list[Run] = []
    warnings: list[str] = []

    def warn(message: str):
        warnings.append(message)

    for stage in stages:
        stage_dir = COMP_DIR / stage
        if not stage_dir.is_dir():
            warn(f"missing comparison folder: {stage_dir}")
            continue

        for method_dir in sorted(stage_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = canonical_method(method_dir.name)
            children = sorted(d for d in method_dir.iterdir() if d.is_dir())
            eps_dirs  = [d for d in children if d.name in EPSILONS]
            mask_dirs = [d for d in children if d.name in MASK_DIR_ALIASES]

            for other in children:
                if other not in eps_dirs and other not in mask_dirs:
                    warn(f"unrecognized folder, skipped: "
                         f"{display(other)}")

            if eps_dirs:
                # <method>/<epsilon>/<mask_type>/<category>/
                for eps_dir in sorted(eps_dirs, key=lambda d: int(d.name)):
                    runs += walk_masks(eps_dir, stage, method, eps_dir.name,
                                       mask_types, warn)
            elif mask_dirs:
                # <method>/<mask_type>/<category>/
                runs += walk_masks(method_dir, stage, method, None,
                                   mask_types, warn)
            elif children:
                warn(f"{display(method_dir)} has neither epsilon "
                     f"nor mask folders, skipped")

    runs.sort(key=lambda r: r.sort_key)
    return runs, warnings


def walk_masks(parent: Path, stage: str, method: str, epsilon: str | None,
               mask_types: list[str], warn) -> list[Run]:
    runs = []
    for mask_dir in sorted(parent.iterdir()):
        if not mask_dir.is_dir():
            continue
        mask_type = MASK_DIR_ALIASES.get(mask_dir.name)
        if mask_type is None:
            warn(f"unrecognized folder, skipped: {display(mask_dir)}")
            continue
        if mask_type not in mask_types:
            continue
        unknown = []
        for cat_dir in sorted(mask_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            if cat_dir.name not in VALID_CATEGORIES:
                unknown.append(cat_dir.name)
                continue
            runs += build_runs(cat_dir, stage, method, epsilon,
                               mask_type, cat_dir.name, warn)
        if unknown:
            # One line for the whole folder rather than one per subfolder.
            legacy = [n for n in unknown if _LEGACY_PG.match(n)]
            where  = display(mask_dir)
            if legacy:
                warn(f"{where}: {len(legacy)} folder(s) still use the old "
                     f"photoGuard layout (e.g. {legacy[0]}) -- skipped. Expected "
                     f"{method}/<epsilon>/<mask_type>/<category>/")
            other = [n for n in unknown if n not in legacy]
            if other:
                warn(f"{where}: skipped {len(other)} folder(s) that are not a "
                     f"category ({', '.join(sorted(other)[:3])}"
                     f"{', ...' if len(other) > 3 else ''})")
    return runs


# ======================================================================
# Execution
# ======================================================================

def execute(run: Run, limit: int | None) -> tuple[bool, str]:
    """Invoke compare_image_metrics.py for one Run. Returns (ok, detail)."""
    run.out_csv.parent.mkdir(parents=True, exist_ok=True)

    with contextlib.ExitStack() as stack:
        files = run.staged_files
        if limit is not None:
            files = (files if files is not None else list_images(run.comp_dir))[:limit]

        if files is None:
            comparison = run.comp_dir
        else:
            # The child records the comparison folder's NAME in every row,
            # so give the staging folder the same name as the real leaf.
            # That keeps the CSV byte-identical to running the child
            # directly on this set of files.
            root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="compare_batch_")))
            staging = root / run.comp_dir.name
            staging.mkdir()
            for path in files:
                os.symlink(path.resolve(), staging / path.name)
            comparison = staging

        result = subprocess.run(
            [sys.executable, str(CHILD_SCRIPT),
             "--reference",  str(run.ref_dir),
             "--comparison", str(comparison),
             "--output",     str(run.out_csv)],
            capture_output=True, text=True,
        )

    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()
        return False, tail[-1] if tail else f"exit code {result.returncode}"
    if not run.out_csv.is_file():
        return False, "child reported success but wrote no CSV"
    return True, ""


# ======================================================================
# Aggregation and reporting
# ======================================================================

def read_scores(csv_path: Path) -> tuple[list[str], list[dict]]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        metric_names = [c for c in (reader.fieldnames or []) if c not in ID_COLUMNS]
        rows = []
        for raw in reader:
            row = {}
            for name in metric_names:
                try:
                    row[name] = float(raw[name])
                except (TypeError, ValueError):
                    row[name] = float("nan")
            rows.append(row)
    return metric_names, rows


def mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")


def format_value(metric: str, value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.2f}" if metric == "PSNR" else f"{value:.4f}"


def header_label(metric: str) -> str:
    arrow = {True: "↑", False: "↓"}.get(HIGHER_IS_BETTER.get(metric), "")
    return metric + arrow


def print_table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    if not rows:
        return
    widths = [max(len(columns[i]), max(len(r[i]) for r in rows)) for i in range(len(columns))]
    line = "  ".join(c.ljust(w) for c, w in zip(columns, widths))
    print()
    print(title)
    print("-" * len(line))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))


def report(runs: list[Run], metrics_dir: Path) -> None:
    """Read every raw CSV that exists and roll it up into both reports."""
    metric_names: list[str] = []
    # (stage, method, eps, mask, category, variant) -> per-image rows
    detail: dict[tuple, list[dict]] = defaultdict(list)

    for run in runs:
        if not run.out_csv.is_file():
            continue
        names, rows = read_scores(run.out_csv)
        if names and not metric_names:
            metric_names = names
        key = (run.stage, run.method, run.epsilon or "", run.mask_type,
               run.category, run.variant or "")
        detail[key] += rows

    if not metric_names:
        print("\nNo scores were produced, so there is nothing to report.")
        return

    # An "ALL" row wherever a leaf produced more than one variant.
    combined: dict[tuple, list[dict]] = defaultdict(list)
    variants_per_leaf: dict[tuple, set[str]] = defaultdict(set)
    for key, rows in detail.items():
        leaf = key[:5]
        variants_per_leaf[leaf].add(key[5])
        combined[leaf] += rows
    for leaf, variants in variants_per_leaf.items():
        if len(variants) > 1:
            detail[leaf + ("ALL",)] = combined[leaf]

    id_columns = ["stage", "method", "epsilon", "mask_type", "category", "variant", "n_images"]

    def sort_key(key):
        stage, method, eps, mask, category, variant = key
        return (list(STAGE_REF).index(stage), method.lower(),
                int(eps) if eps else -1, mask,
                VALID_CATEGORIES.index(category) if category in VALID_CATEGORIES else 99,
                variant != "ALL", variant)

    detail_rows = []
    for key in sorted(detail, key=sort_key):
        rows = detail[key]
        values = [format_value(m, mean([r[m] for r in rows])) for m in metric_names]
        stage, method, eps, mask, category = key[:5]
        detail_rows.append([stage, method, eps or "-", mask, category,
                            key[5] or "-", str(len(rows))] + values)

    # Rolled up across categories (and across variants) per method/eps/mask.
    rollup: dict[tuple, list[dict]] = defaultdict(list)
    for key, rows in detail.items():
        if key[5] == "ALL":
            continue  # already counted through its individual variants
        rollup[(key[0], key[1], key[2], key[3])] += rows

    rollup_rows = []
    for key in sorted(rollup, key=lambda k: (list(STAGE_REF).index(k[0]), k[1].lower(),
                                             int(k[2]) if k[2] else -1, k[3])):
        rows = rollup[key]
        values = [format_value(m, mean([r[m] for r in rows])) for m in metric_names]
        rollup_rows.append([key[0], key[1], key[2] or "-", key[3], str(len(rows))] + values)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = metrics_dir / "summary_detail.csv"
    with open(detail_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(id_columns + metric_names)
        writer.writerows(detail_rows)

    rollup_csv = metrics_dir / "summary_by_method.csv"
    rollup_columns = ["stage", "method", "epsilon", "mask_type", "n_images"]
    with open(rollup_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rollup_columns + metric_names)
        writer.writerows(rollup_rows)

    headers = [header_label(m) for m in metric_names]
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("↑ higher = more similar to the reference"
          "   ↓ lower = more similar (GMSD and LPIPS are distances)")
    print("immune rows compare against ref/clean, edit rows against ref/edit.")

    for stage in STAGE_REF:
        stage_rows = [r for r in rollup_rows if r[0] == stage]
        print_table(f"[{stage}] averaged over categories",
                    rollup_columns[1:] + headers,
                    [r[1:] for r in stage_rows])

    for stage in STAGE_REF:
        stage_rows = [r for r in detail_rows if r[0] == stage]
        print_table(f"[{stage}] per category",
                    id_columns[1:] + headers,
                    [r[1:] for r in stage_rows])

    print()
    print(f"Per-leaf report:   {detail_csv.resolve()}")
    print(f"Per-method report: {rollup_csv.resolve()}")


# ======================================================================
# Entry point
# ======================================================================

def main() -> int:
    global COMP_DIR, REF_DIR, METRICS_DIR

    parser = argparse.ArgumentParser(
        description="Score every folder under comp/ against ref/ using "
                    "compare_image_metrics.py, and summarize the results."
    )
    parser.add_argument("--stage", choices=list(STAGE_REF) + ["both"], required=True,
                        help="immune: comp/immune vs ref/clean. edit: comp/edit vs ref/edit.")
    parser.add_argument("--mask", choices=MASK_TYPES + ["both"], required=True,
                        help="Which mask type to compare.")
    parser.add_argument("--comp", type=Path, default=COMP_DIR,
                        help="Root of the comparison tree (default: ./comp).")
    parser.add_argument("--ref", type=Path, default=REF_DIR,
                        help="Root of the reference tree (default: ./ref).")
    parser.add_argument("--metrics", type=Path, default=METRICS_DIR,
                        help="Where reports are written (default: ./metrics).")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="Score at most N images per folder. For smoke tests.")
    parser.add_argument("--report-only", action="store_true",
                        help="Re-print the report from existing CSVs without scoring.")
    args = parser.parse_args()

    COMP_DIR, REF_DIR, METRICS_DIR = args.comp, args.ref, args.metrics

    if not CHILD_SCRIPT.is_file():
        print(f"error: compare_image_metrics.py not found at {CHILD_SCRIPT}", file=sys.stderr)
        return 2
    for label, path in (("--comp", COMP_DIR), ("--ref", REF_DIR)):
        if not path.is_dir():
            print(f"error: {label} is not a directory: {path}", file=sys.stderr)
            return 2

    stages     = ["immune", "edit"] if args.stage == "both" else [args.stage]
    mask_types = MASK_TYPES if args.mask == "both" else [args.mask]

    runs, warnings = collect_runs(stages, mask_types)

    for message in warnings:
        print(f"  warn  {message}")

    todo    = [r for r in runs if not r.out_csv.is_file()]
    skipped = len(runs) - len(todo)
    if args.report_only:
        todo = []

    print(f"\n{len(runs)} folder pair(s) matched | {len(todo)} to score | "
          f"{skipped} already scored")
    if not runs:
        print("Nothing to do.")
        return 0

    succeeded = failed = 0
    for i, run in enumerate(todo, start=1):
        print(f"\n  [{i}/{len(todo)}] {run.label}")
        start = time.perf_counter()
        ok, detail = execute(run, args.limit)
        elapsed = time.perf_counter() - start
        if ok:
            print(f"  ok    -> {display(run.out_csv)} ({elapsed:.1f}s)")
            succeeded += 1
        else:
            print(f"  FAIL  {run.label}: {detail} ({elapsed:.1f}s)", file=sys.stderr)
            failed += 1

    if todo:
        print(f"\nDone. {succeeded} scored, {failed} failed, {skipped} reused.")

    report(runs, METRICS_DIR)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
