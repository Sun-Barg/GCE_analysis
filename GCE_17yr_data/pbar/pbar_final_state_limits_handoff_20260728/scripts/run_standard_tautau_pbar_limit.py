#!/usr/bin/env python3
"""Run the C_data-only antiproton upper limit for standard tau+tau-."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_OUT_ROOT = Path("Codex_files/generated_outputs/standard_tautau_pbar_limits_20260728")
DEFAULT_SECONDARY_DIR = Path("Codex_files/.archived_generated_outputs_20260603/calore_phi0p732_grid_20260531/BIG_L_grid_phi0p732_known_good_sourcefit")
DEFAULT_SECONDARY_BASENAME = "local_fluxes_1HBAR_R_Model1DKisoVc_SolMod0DFF_phi0_732GV_1.out"


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


def run_usine_grid(out_root: Path, primary_dir: Path, manifest: Path, phi_gv: str, force_usine: bool) -> None:
    rows = parse_manifest(manifest)
    env = os.environ.copy()
    env["USINE"] = str((out_root / "usine_overlay").resolve())
    completed = 0
    reused = 0
    for row in rows:
        l_value = float(row["L"])
        mass = float(row["mass"])
        out_dir = primary_dir / f"L_{l_tag(l_value)}_kpc" / f"m_{mass_tag(mass)}_GeV"
        expected = out_dir / "local_fluxes_1HBAR_R.C"
        if force_usine and out_dir.exists():
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--phi-gv", default="0.732")
    parser.add_argument("--max-mass-gev", type=float, default=1000.0)
    parser.add_argument("--secondary-dir", type=Path, default=DEFAULT_SECONDARY_DIR)
    parser.add_argument("--secondary-flux-basename", default=DEFAULT_SECONDARY_BASENAME)
    parser.add_argument("--max-sigmav", default="1e8")
    parser.add_argument("--skip-usine", action="store_true")
    parser.add_argument("--force-usine", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            "python3",
            "Codex_files/codex_codes/prepare_standard_tautau_pbar_usine_overlay.py",
            "--out-root",
            str(args.out_root),
            "--max-mass-gev",
            f"{args.max_mass_gev:g}",
        ]
    )
    manifest_info = json.loads((args.out_root / "standard_tautau_overlay_manifest.json").read_text())
    manifest = Path(manifest_info["grid"]["manifest"])
    primary_dir = args.out_root / "primary"
    primary_grid = primary_dir / "plots_primary_curve" / "big_2d_dm_explicit_primary_curve_pbar_flux_grid.tsv"
    limits_dir = args.out_root / "limits"

    if not args.skip_usine:
        run_usine_grid(args.out_root, primary_dir, manifest, args.phi_gv, args.force_usine)

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
    print(f"Wrote {limits_dir / 'pbar_95cl_upper_limits_data_cov_only.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
