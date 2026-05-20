#!/usr/bin/env python3
"""Prepare a baseline USINE overlay for standard tau+tau- antiproton spectra.

The baseline BIG grid is a bbbar grid.  To avoid relying on the ordering of
USINE's branching-ratio vector, this overlay keeps the baseline bbbar init files
unchanged and replaces the PPPC kCIRELLI19 b column by the PPPC tau column.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_STANDARD_PPPC = Path("inputs/PPPC4DMID-spectra/2019/AtProduction_antiprotons.dat")
DEFAULT_BASE_MANIFEST = Path("Codex_files/configs/generated/BIG_2D_DM_NFW_bbbar_grid_kCIRELLI19_20260601/manifest.txt")
DEFAULT_OUT_ROOT = Path("Codex_files/generated_outputs/standard_tautau_pbar_limits_20260728")

TAU_KCIRELLI19_COL = 4
BBAR_KCIRELLI19_COL = 7


def make_overlay_links(out_dir: Path) -> None:
    overlay = out_dir / "usine_overlay"
    overlay.mkdir(parents=True, exist_ok=True)
    for name in ["DOC_PBAR"]:
        target = Path.cwd() / name
        link = overlay / name
        if not link.exists():
            os.symlink(target, link)

    inputs_overlay = overlay / "inputs"
    inputs_overlay.mkdir(exist_ok=True)
    for item in (Path.cwd() / "inputs").iterdir():
        if item.name == "PPPC4DMID-spectra":
            continue
        link = inputs_overlay / item.name
        if not link.exists():
            os.symlink(item, link)

    pppc_overlay = inputs_overlay / "PPPC4DMID-spectra"
    pppc_overlay.mkdir(exist_ok=True)
    for item in (Path.cwd() / "inputs" / "PPPC4DMID-spectra").iterdir():
        if item.name == "2019":
            continue
        link = pppc_overlay / item.name
        if not link.exists():
            os.symlink(item, link)


def write_overlay_pppc(source: Path, out_dir: Path) -> dict[str, object]:
    overlay_file = out_dir / "usine_overlay" / "inputs" / "PPPC4DMID-spectra" / "2019" / source.name
    overlay_file.parent.mkdir(parents=True, exist_ok=True)
    copied_rows = 0
    nonzero_tau_rows = 0
    with source.open(encoding="utf-8") as src, overlay_file.open("w", encoding="utf-8") as dst:
        for raw in src:
            if not raw.strip() or raw.lstrip().startswith("#"):
                dst.write(raw)
                continue
            row = [float(item) for item in raw.split()]
            tau_value = row[TAU_KCIRELLI19_COL]
            row[BBAR_KCIRELLI19_COL] = tau_value
            if tau_value > 0.0:
                nonzero_tau_rows += 1
            copied_rows += 1
            dst.write(" ".join(f"{value:.8e}" for value in row) + "\n")
    return {
        "overlay_pppc": str(overlay_file),
        "standard_pppc": str(source),
        "replacement": "kCIRELLI19 b column replaced by kCIRELLI19 tau column",
        "tau_column_index_zero_based": TAU_KCIRELLI19_COL,
        "b_column_index_zero_based": BBAR_KCIRELLI19_COL,
        "rows": copied_rows,
        "nonzero_tau_rows": nonzero_tau_rows,
    }


def write_selected_manifest(base_manifest: Path, out_dir: Path, max_mass_gev: float | None) -> dict[str, object]:
    manifest = out_dir / "manifest_standard_tautau_via_bb_slot.txt"
    rows = []
    masses = set()
    l_values = set()
    with base_manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            mass = float(fields[1])
            if max_mass_gev is not None and mass > max_mass_gev:
                continue
            rows.append(line.rstrip())
            masses.add(mass)
            l_values.add(float(fields[0]))
    if not rows:
        raise RuntimeError(f"No selected rows from {base_manifest}")
    with manifest.open("w", encoding="utf-8") as out:
        out.write("# Standard tau+tau- primary-DM antiproton grid via PPPC overlay\n")
        out.write("# The init files are the tentative-baseline bbbar kCIRELLI19 files.\n")
        out.write("# The overlay replaces the PPPC b column by the PPPC tau column, so only the prompt spectrum changes.\n")
        out.write(f"# source_manifest: {base_manifest}\n")
        if max_mass_gev is not None:
            out.write(f"# max_mass_GeV: {max_mass_gev:g}\n")
        for row in rows:
            out.write(row + "\n")
    return {
        "manifest": str(manifest),
        "source_manifest": str(base_manifest),
        "rows": len(rows),
        "masses": sorted(masses),
        "L_values": sorted(l_values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--standard-pppc", type=Path, default=DEFAULT_STANDARD_PPPC)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--max-mass-gev", type=float, default=1000.0)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    make_overlay_links(args.out_root)
    overlay_info = write_overlay_pppc(args.standard_pppc, args.out_root)
    manifest_info = write_selected_manifest(args.base_manifest, args.out_root, args.max_mass_gev)
    payload = {
        "case": "standard_tautau_via_bb_slot",
        "channel": "tau+tau-",
        "baseline": "calore_cdata_best_branch_20260601 / tentative_baseline",
        "overlay": overlay_info,
        "grid": manifest_info,
    }
    path = args.out_root / "standard_tautau_overlay_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
