#!/usr/bin/env python3
"""
inventory.py — Classify files in the GCE_12yr_reproduce/ working directory.

Walks the working directory and classifies every file as one of:
  KEEP-INPUT   : Manually placed inputs (NEVER delete)
  KEEP-CODE    : Notebooks, scripts, docs
  RESULT       : Pipeline final outputs (.dat files) — your call
  REGEN        : Auto-regenerable by v3 notebook (safe to delete)
  LEGACY       : Files from previous sessions (v2 port, v8/v9, etc.)
  UNKNOWN      : Doesn't match known patterns — please review manually

Usage:
    cd /home/haebarg/GCE-Chi-square-fitting/GCE_12yr_reproduce/

    # 1) Just report (no changes):
    python inventory.py

    # 2) Move LEGACY files to archive (preserves them, doesn't delete):
    python inventory.py --archive _archive_v2/

    # 3) Move REGEN intermediate files to backup (forces fresh re-run):
    python inventory.py --reset-pipeline

    # 4) Both archive + reset, plus also move RESULT to backup
    #    (truly clean slate; preserves previous results for comparison):
    python inventory.py --archive _archive_v2/ --reset-pipeline --include-results

    # 5) Dry-run: show what would happen without moving anything:
    python inventory.py --archive _archive_v2/ --reset-pipeline --dry-run

Notes:
  - All --archive and --reset-pipeline operations MOVE files (don't delete).
    You can verify the archive then `rm -rf` it if you want.
  - KEEP-INPUT and KEEP-CODE files are never moved.
  - UNKNOWN files are reported but never moved automatically.
"""
import os
import sys
import argparse
import shutil
import re
from pathlib import Path
from datetime import datetime

# ============================================================================
# Classification patterns
# ============================================================================

class C:
    KEEP_INPUT  = "KEEP-INPUT"
    KEEP_CODE   = "KEEP-CODE"
    RESULT      = "RESULT"
    REGEN       = "REGEN"
    LEGACY      = "LEGACY"
    UNKNOWN     = "UNKNOWN"

# Order matters: first matching pattern wins.
# Each entry: (regex matching the path relative to the working dir, category, short description)
PATTERNS = [
    # ----- KEEP-INPUT (never delete; manually prepared or downloaded) -----
    (r'^GCE_template_NFW2\.fits$',
        C.KEEP_INPUT, 'NFW² spatial template (haebarg, 600×600)'),
    (r'^Fermi_Bubbles_template\.fits$',
        C.KEEP_INPUT, 'Fermi bubble spatial template'),
    (r'^isotropic_spectrum_ff\.txt$',
        C.KEEP_INPUT, 'IGRB FileFunction spectrum'),
    (r'^fermi_bubble_spectrum\.txt$',
        C.KEEP_INPUT, 'Bubble FileFunction spectrum'),
    (r'^MapCubes/(bremss|ics|pion)_mapcube_model[A-Z]+\.fits$',
        C.KEEP_INPUT, 'Cholis Zenodo GDE MapCube'),
    (r'^GC_analysis_sanghwan/Model/bubble_constraints\.txt$',
        C.KEEP_INPUT, 'Bubble constraints (1407.7905)'),
    (r'^GC_analysis_sanghwan/Model/iso_constraints_full_err\.txt$',
        C.KEEP_INPUT, 'Isotropic constraints (1410.3696)'),

    # ----- KEEP-CODE (current session deliverables) -----
    (r'^GC_analysis-60x60-models_12yr_haebarg(_v[0-9]+(\.[0-9]+)?)?\.ipynb$',
        C.KEEP_CODE, 'Current haebarg-port notebook'),
    (r'^preflight_check\.py$',
        C.KEEP_CODE, 'Preflight check script'),
    (r'^inventory\.py$',
        C.KEEP_CODE, 'This inventory script'),
    (r'^PORT_v[0-9]+(\.[0-9]+)?_SUMMARY\.md$',
        C.KEEP_CODE, 'Port summary doc'),
    (r'^CLEANUP_GUIDE\.md$',
        C.KEEP_CODE, 'Cleanup guide'),
    (r'^REF_.*\.md$',
        C.KEEP_CODE, 'Reference notes'),

    # ----- RESULT (final pipeline outputs — keep or delete is your call) -----
    (r'^GCE_model_[A-Z]+_12yr_cholis\.dat$',
        C.RESULT, 'Per-model final GCE flux'),
    (r'^GCE_model_[A-Z]+_12yr_cholis_likelihood_value$',
        C.RESULT, 'Per-model final likelihood'),

    # ----- REGEN (auto-regenerable by v3 pipeline) -----
    # Cell 6 output
    (r'^photon_data\.txt$',
        C.REGEN, 'Cell 6 ls output (instant regen)'),
    # Cell 8 outputs
    (r'^GC_analysis_sanghwan/bin_definitions\.(txt|fits)$',
        C.REGEN, 'Cell 8 bin definitions (instant regen)'),
    # Cell 13 outputs (data prep — heavy)
    (r'^GC_analysis_sanghwan/Allsky_select_12yr_front_clean\.fits$',
        C.REGEN, 'Cell 13 stage 1 gtselect (~30 min, ~10 GB)'),
    (r'^GC_analysis_sanghwan/Allsky_gti_12yr_front_clean\.fits$',
        C.REGEN, 'Cell 13 stage 2 gtmktime (~5 min)'),
    (r'^GC_analysis_sanghwan/GC_ccube_12yr_front_clean\.fits$',
        C.REGEN, 'Cell 13 stage 3 gtbin CCUBE (~5 min)'),
    # Cell 16-18 outputs
    (r'^GC_analysis_sanghwan/Allsky_ltcube_12yr_front_clean\.fits$',
        C.REGEN, 'Cell 16 ltcube (~30 min, ~1 GB)'),
    (r'^GC_analysis_sanghwan/GC_expcube_center_12yr_front_clean\.fits$',
        C.REGEN, 'Cell 17 expcube center (~10 min)'),
    (r'^GC_analysis_sanghwan/Allsky_expcube_edge_12yr_front_clean\.fits$',
        C.REGEN, 'Cell 18 expcube edge (~30 min)'),
    # Cell 19, 20, 32 — XMLs
    (r'^GC_analysis_sanghwan/Model/(GC_model_DR2|GC_psc_model_DR2|empty_model)\.xml$',
        C.REGEN, 'Cells 19/20/32 base XML (~30 sec)'),
    # Cell 29-31 — masks
    (r'^GC_analysis_sanghwan/Model/GC_(disk_)?mask_60x60_definitions(_DR2)?\.npy$',
        C.REGEN, 'Cells 29-31 mask .npy (~15 min)'),
    # Cell 35 — per-model XML / srcmap / component maps
    (r'^GC_analysis_sanghwan/Model/GC_(model|Extended_model)[A-Z]+_test\.xml$',
        C.REGEN, 'Cell 35 per-model XML (instant regen)'),
    (r'^GC_analysis_sanghwan/Model/GC_(pion|bremss|ics)_model[A-Z]+_test\.xml$',
        C.REGEN, 'Cell 35 component XML'),
    (r'^GC_analysis_sanghwan/Model/GC_(GCE|fermi_bubble|isotropic)_singlecomp\.xml$',
        C.REGEN, 'Cell 35 v3 single-comp XML'),
    (r'^GC_analysis_sanghwan/GC_Extended_srcmap_12yr_front_clean_model_[A-Z]+(_no_convol)?\.fits$',
        C.REGEN, 'Cell 35 gtsrcmaps output (~5 min/model)'),
    (r'^GC_analysis_sanghwan/GC_(pion|bremss|ics)_model[A-Z]+_12yr_front_clean(_no_convol)?\.fits$',
        C.REGEN, 'Cell 35 gtmodel output (per model)'),
    (r'^GC_analysis_sanghwan/GC_(GCE|fermi_bubble|isotropic)_model_12yr_front_clean(_no_convol)?\.fits$',
        C.REGEN, 'Cell 35 v3 GCE/bubble/iso gtmodel'),
    # Auxiliary subprocess logs/par files
    (r'^GC_analysis_sanghwan/.*\.par$',
        C.REGEN, 'Fermi tools par file'),
    (r'^GC_analysis_sanghwan/corner_.*\.png$',
        C.REGEN, 'Cell 35 corner plot'),

    # ----- LEGACY (from previous sessions/attempts, not used by v3) -----
    (r'^GCE_results_v8.*\.pkl$',
        C.LEGACY, 'v8.x pickle (haebarg early pipeline)'),
    (r'^GCE_results_v9.*\.pkl$',
        C.LEGACY, 'v9.x pickle (haebarg pipeline)'),
    (r'^bin_definitions(_extended)?\.(txt|fits)$',
        C.LEGACY, 'v9.x extended bin definition (top-level)'),
    (r'^GC_analysis_12yr_haebarg.*\.ipynb$',
        C.LEGACY, 'v2-port notebook (older)'),
    (r'^GCE_chi_square_fitting.*\.ipynb$',
        C.LEGACY, 'haebarg legacy notebook'),
    (r'^run_main_loop\.py$',
        C.LEGACY, 'v2-port main loop script'),
    (r'^run_gtmaps_only\.py$',
        C.LEGACY, 'v2-port stage-1 script'),
    (r'^run_mcmc_only\.py$',
        C.LEGACY, 'v2-port stage-2 script'),
    (r'^phase[12]_.*\.py$',
        C.LEGACY, 'v9.x phase script'),
    (r'^cleanup\.py$',
        C.LEGACY, 'v9.x cleanup script'),
    (r'^monitor\.py$',
        C.LEGACY, 'v9.x monitor script'),
    (r'^compare_model_X\.py$',
        C.LEGACY, 'v2-port compare script'),
    (r'^extend_spectrum_files\.py$',
        C.LEGACY, 'v2-port spectrum extender'),
    (r'^generate_nfw2_fast\.py$',
        C.LEGACY, 'v2-port NFW² generator'),
    (r'^generate_cholis_mask\.py$',
        C.LEGACY, 'v9.x Cholis mask generator'),
    (r'^main_loop\.log$',
        C.LEGACY, 'v2-port log'),
    (r'^GCE_template_NFW2_sanghwan\.fits$',
        C.LEGACY, 'Sanghwan-style 2400×1800 NFW² (alt; not used by v3)'),
    (r'^GC_analysis_sanghwan/Model/source_classification\.pkl$',
        C.LEGACY, 'v2-port source classification cache'),
    (r'^claude_try/.*$',
        C.LEGACY, 'v2-port output dir'),
    (r'^haebarg_v_claude/.*$',
        C.LEGACY, 'v2-port comparison files'),
    (r'^v9_.*\.ipynb$',
        C.LEGACY, 'haebarg v9.x notebook'),
    (r'^GCE_template_N.*\.fits$',
        C.LEGACY, 'Truncated/leftover NFW template?'),

    # Catch-all for any file inside the previous Sanghwan-port output dir
    # that didn't match a more specific REGEN pattern above:
    (r'^GC_analysis_sanghwan/.*\.fits$',
        C.REGEN, 'Inside GC_analysis_sanghwan/ — assumed regenerable'),
    (r'^GC_analysis_sanghwan/.*\.xml$',
        C.REGEN, 'Inside GC_analysis_sanghwan/ — assumed regenerable'),
    (r'^GC_analysis_sanghwan/.*\.npy$',
        C.REGEN, 'Inside GC_analysis_sanghwan/ — assumed regenerable'),
]


# ============================================================================
# Helpers
# ============================================================================

def classify_file(rel_path: str):
    for pattern, cat, desc in PATTERNS:
        if re.match(pattern, rel_path):
            return cat, desc
    return C.UNKNOWN, ""


def human_size(n):
    f = float(n)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if f < 1024:
            return f"{f:7.1f} {unit:<2}"
        f /= 1024
    return f"{f:7.1f} PB"


def walk_files(root: Path, skip_dirs):
    """Yield (relative_path, abs_path, size) for each file under root."""
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        # skip files inside dirs the user already chose as archive/backup
        if any(part in skip_dirs or part.startswith("_archive_")
               or part.startswith("_backup_") for part in rel.parts):
            continue
        yield str(rel).replace(os.sep, "/"), p, p.stat().st_size


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Classify and optionally archive files in the GCE_12yr_reproduce/ workdir.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", default=".",
                   help="Working directory (default: current).")
    p.add_argument("--archive", metavar="DIR", default=None,
                   help="Move LEGACY files into DIR (preserves them, doesn't delete).")
    p.add_argument("--reset-pipeline", action="store_true",
                   help="Move REGEN files into _backup_v3_<timestamp>/ to force fresh re-run.")
    p.add_argument("--include-results", action="store_true",
                   help="Together with --reset-pipeline: also move RESULT (.dat) files. "
                        "Useful when you want to re-run from scratch and compare new vs old.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions without actually moving anything.")
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    # We never walk into these
    skip_dirs = set()
    if args.archive:
        skip_dirs.add(Path(args.archive).name)

    # Walk and classify
    by_cat = {}
    by_cat_sizes = {}
    for rel, abs_path, size in walk_files(root, skip_dirs):
        cat, desc = classify_file(rel)
        by_cat.setdefault(cat, []).append((rel, abs_path, size, desc))
        by_cat_sizes[cat] = by_cat_sizes.get(cat, 0) + size

    # ----- Report -----
    print(f"\n{'='*78}")
    print(f"  Inventory of {root}")
    print(f"{'='*78}\n")
    print(f"  Summary:")
    print(f"  {'Category':<14} {'Count':>6}  {'Total size':>12}")
    print(f"  {'-'*14} {'-'*6}  {'-'*12}")
    for cat in [C.KEEP_INPUT, C.KEEP_CODE, C.RESULT, C.REGEN, C.LEGACY, C.UNKNOWN]:
        files = by_cat.get(cat, [])
        if not files:
            continue
        print(f"  {cat:<14} {len(files):>6}  {human_size(by_cat_sizes[cat])}")

    # Detailed listing per category
    for cat in [C.UNKNOWN, C.LEGACY, C.RESULT, C.REGEN, C.KEEP_CODE, C.KEEP_INPUT]:
        files = by_cat.get(cat, [])
        if not files:
            continue
        print(f"\n--- {cat}  ({len(files)} files, {human_size(by_cat_sizes[cat])}) ---")
        files_sorted = sorted(files, key=lambda x: (-x[2], x[0]))  # by size desc
        for rel, _abs, size, desc in files_sorted[:50]:
            print(f"  {human_size(size)}  {rel}")
            if desc:
                print(f"  {' '*10}    └─ {desc}")
        if len(files_sorted) > 50:
            print(f"  ... and {len(files_sorted) - 50} more (run with --root to limit scope)")

    # ----- Actions -----
    moved = 0

    def do_move(srcabs, rel, dst_root, dry):
        nonlocal moved
        dst = dst_root / rel
        if dry:
            print(f"  [DRY-RUN] {rel} -> {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(srcabs), str(dst))
            print(f"  [MOVED]   {rel} -> {dst}")
        moved += 1

    if args.archive:
        archive_dir = (root / args.archive).resolve()
        legacy_files = by_cat.get(C.LEGACY, [])
        if legacy_files:
            print(f"\n=== Moving {len(legacy_files)} LEGACY files to {archive_dir} ===")
            if not args.dry_run:
                archive_dir.mkdir(parents=True, exist_ok=True)
            for rel, abs_path, size, _desc in legacy_files:
                do_move(abs_path, rel, archive_dir, args.dry_run)
        else:
            print(f"\n=== No LEGACY files to archive ===")

    if args.reset_pipeline:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = (root / f"_backup_v3_{timestamp}").resolve()
        regen_files = by_cat.get(C.REGEN, [])
        targets = list(regen_files)
        if args.include_results:
            targets += by_cat.get(C.RESULT, [])
        if targets:
            tag = "REGEN + RESULT" if args.include_results else "REGEN"
            print(f"\n=== Reset pipeline: moving {len(targets)} {tag} files to {backup_dir} ===")
            if not args.dry_run:
                backup_dir.mkdir(parents=True, exist_ok=True)
            for rel, abs_path, size, _desc in targets:
                do_move(abs_path, rel, backup_dir, args.dry_run)
        else:
            print(f"\n=== No REGEN files to reset ===")

    if not (args.archive or args.reset_pipeline):
        print(f"\n  (No actions requested. Re-run with --archive and/or --reset-pipeline to move files.)")

    if moved:
        print(f"\n  {'(would move)' if args.dry_run else 'moved'}: {moved} files")
    print()


if __name__ == "__main__":
    main()
