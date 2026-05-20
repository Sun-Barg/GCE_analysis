#!/usr/bin/env python3
"""Run C_data-only antiproton limits for on-shell 4b, 4tau, and 2b2tau grids."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_OUT_ROOT = Path("Codex_files/generated_outputs/onshell_30_pbar_limits_20260727")
DEFAULT_SECONDARY_DIR = Path("Codex_files/.archived_generated_outputs_20260603/calore_phi0p732_grid_20260531/BIG_L_grid_phi0p732_known_good_sourcefit")
DEFAULT_SECONDARY_BASENAME = "local_fluxes_1HBAR_R_Model1DKisoVc_SolMod0DFF_phi0_732GV_1.out"
STATES = ["4b", "4tau", "2b2tau"]
R_VALUES = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]


def safe_r_label(value: str) -> str:
    return "r" + value.replace(".", "p")


def case_name(state: str, r_value: str) -> str:
    return f"onshell_{state}_{safe_r_label(r_value)}"


def l_tag(value: float) -> str:
    return f"{value:05.2f}".replace(".", "p")


def mass_tag(value: float) -> str:
    if value >= 1000.0:
        text = f"{value:.0f}"
    elif value >= 100.0:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2f}"
    return text.replace(".", "p")


def run_checked(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def parse_manifest(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        rows.append({"L": float(fields[0]), "mass": float(fields[1]), "init_file": fields[-1]})
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def run_usine_grid(case_dir: Path, primary_dir: Path, manifest: Path, phi_gv: str, force: bool = False) -> None:
    rows = parse_manifest(manifest)
    overlay = (case_dir / "usine_overlay").resolve()
    env = os.environ.copy()
    env["USINE"] = str(overlay)
    completed = 0
    reused = 0
    for row in rows:
        l_value = float(row["L"])
        mass = float(row["mass"])
        out_dir = primary_dir / f"L_{l_tag(l_value)}_kpc" / f"m_{mass_tag(mass)}_GeV"
        expected = out_dir / "local_fluxes_1HBAR_R.C"
        if force and out_dir.exists():
            shutil.rmtree(out_dir)
        if expected.exists():
            reused += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (out_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
            status = subprocess.run(
                [
                    "./bin/usine",
                    "-l",
                    str(row["init_file"]),
                    str(out_dir),
                    f"1H-BAR:kR:{phi_gv}",
                    "0.",
                    "1",
                    "1",
                    "0",
                ],
                env=env,
                stdout=stdout,
                stderr=stderr,
                check=False,
            ).returncode
        if status != 0 and not expected.exists():
            raise RuntimeError(f"USINE failed for L={l_value}, mDM={mass}; see {out_dir}")
        completed += 1
    print(f"USINE grid ready: reused={reused}, newly_completed={completed}, total={len(rows)}", flush=True)


def append_summary(out_root: Path, case: str, state: str, r_value: str, limits_path: Path, manifest_info: dict[str, object]) -> None:
    summary = out_root / "combined_onshell_pbar_limit_summary.tsv"
    write_header = not summary.exists()
    with limits_path.open(newline="", encoding="utf-8") as handle, summary.open("a", newline="", encoding="utf-8") as out:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = ["case", "state", "r_value", "mass_mode", "custom_mass_min_GeV", "custom_mass_max_GeV"] + list(reader.fieldnames or [])
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        overlay = manifest_info["overlay"]
        for row in reader:
            merged = {
                "case": case,
                "state": state,
                "r_value": r_value,
                "mass_mode": manifest_info.get("mass_mode", ""),
                "custom_mass_min_GeV": overlay.get("custom_mass_min_GeV", ""),
                "custom_mass_max_GeV": overlay.get("custom_mass_max_GeV", ""),
            }
            merged.update(row)
            writer.writerow(merged)


def limit_table_has_nan(path: Path) -> bool:
    text = path.read_text()
    return "nan" in text.lower()


def run_case(args: argparse.Namespace, state: str, r_value: str) -> None:
    case = case_name(state, r_value)
    case_dir = args.out_root / case
    primary_dir = case_dir / "primary"
    limits_dir = case_dir / "limits"
    report_dir = case_dir / "report"
    limits_path = limits_dir / "pbar_95cl_upper_limits_data_cov_only.tsv"
    print(f"\n=== {case} ===", flush=True)

    run_checked(
        [
            "python3",
            "Codex_files/codex_codes/prepare_onshell_pbar_usine_overlay.py",
            "--state",
            state,
            "--r-value",
            r_value,
            "--out-root",
            str(args.out_root),
            "--mass-mode",
            args.mass_mode,
        ]
    )
    manifest_info = json.loads((case_dir / "onshell_pbar_overlay_manifest.json").read_text())
    manifest = Path(manifest_info["grid"]["manifest"])
    primary_grid = primary_dir / "plots_primary_curve" / "big_2d_dm_explicit_primary_curve_pbar_flux_grid.tsv"

    if limits_path.exists() and not args.force and not limit_table_has_nan(limits_path):
        print(f"limits already exist, appending summary: {limits_path}", flush=True)
        append_summary(args.out_root, case, state, r_value, limits_path, manifest_info)
        return
    if limits_path.exists() and not args.force:
        print(f"limits contain nan, recomputing with max_sigmav={args.max_sigmav}: {limits_path}", flush=True)

    if not args.skip_usine:
        run_usine_grid(case_dir, primary_dir, manifest, args.phi_gv, force=args.force_usine)

    run_checked(
        [
            "python3",
            "Codex_files/codex_codes/extract_big_2d_dm_primary_curve_grid.py",
            "--manifest",
            str(manifest),
            "--grid-dir",
            str(primary_dir),
            "--out-dir",
            str(primary_grid.parent),
        ]
    )
    run_checked(
        [
            "python3",
            "Codex_files/codex_codes/calculate_pbar_upper_limits_data_cov.py",
            "--secondary-dir",
            str(args.secondary_dir),
            "--secondary-flux-basename",
            args.secondary_flux_basename,
            "--dm-grid-tsv",
            str(primary_grid),
            "--output-dir",
            str(limits_dir),
            "--max-sigmav",
            args.max_sigmav,
        ]
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "case_manifest.json").write_text(json.dumps(manifest_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_summary(args.out_root, case, state, r_value, limits_path, manifest_info)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--states", nargs="+", default=STATES)
    parser.add_argument("--r-values", nargs="+", default=R_VALUES)
    parser.add_argument("--mass-mode", choices=["baseline-plus-edges", "all"], default="baseline-plus-edges")
    parser.add_argument("--phi-gv", default="0.732")
    parser.add_argument("--secondary-dir", type=Path, default=DEFAULT_SECONDARY_DIR)
    parser.add_argument("--secondary-flux-basename", default=DEFAULT_SECONDARY_BASENAME)
    parser.add_argument("--max-sigmav", default="1e-16")
    parser.add_argument("--skip-usine", action="store_true")
    parser.add_argument("--force-usine", action="store_true", help="Rerun existing USINE primary outputs for the selected cases.")
    parser.add_argument("--force", action="store_true", help="Recompute extraction and limits even when a case limit table already exists.")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary = args.out_root / "combined_onshell_pbar_limit_summary.tsv"
    if summary.exists():
        summary.unlink()
    for state in args.states:
        for r_value in args.r_values:
            run_case(args, state, r_value)
    print(f"\nWrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
