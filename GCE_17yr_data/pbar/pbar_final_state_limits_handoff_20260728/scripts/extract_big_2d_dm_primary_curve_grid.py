#!/usr/bin/env python3
"""Extract explicit __prim_ antiproton curves from BIG 2D DM USINE macros."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


DEFAULT_MANIFEST = Path("Codex_files/configs/generated/BIG_2D_DM_NFW_bbbar_grid/manifest.txt")
DEFAULT_GRID_DIR = Path("Codex_files/generated_outputs/calore_phi0p732_grid_20260531/BIG_2D_DM_NFW_bbbar_grid_phi0p732")
DEFAULT_OUT_DIR = DEFAULT_GRID_DIR / "plots_primary_curve"


VECTOR_RE = re.compile(
    r"std::vector<Double_t>\s+(graph_[xy]_vect\d+)\{(.*?)\n\s*\};",
    re.DOTALL,
)
GRAPH_RE = re.compile(
    r"(?:TGraph \*graph = new TGraph|graph = new TGraph)"
    r"\(\s*\d+\s*,\s*(graph_x_vect\d+)\.data\(\),\s*(graph_y_vect\d+)\.data\(\)\s*\);"
    r"\s*graph->SetName\(\"([^\"]+)\"\)",
    re.DOTALL,
)
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|\b0\b")


def format_tag(value: float) -> str:
    if value >= 1000:
        return f"{value:.0f}".replace(".", "p")
    if value >= 100:
        return f"{value:.1f}".replace(".", "p")
    return f"{value:.2f}".replace(".", "p")


def l_tag(value: float) -> str:
    return f"{value:05.2f}".replace(".", "p")


def parse_manifest(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 15:
            raise RuntimeError(f"Malformed manifest row in {path}: {line}")
        rows.append({"L": float(fields[0]), "mass": float(fields[1]), "init_file": fields[14]})
    if not rows:
        raise RuntimeError(f"No grid rows found in manifest: {path}")
    return rows


def output_dir_for(row: dict[str, float | str], grid_dir: Path) -> Path:
    return grid_dir / f"L_{l_tag(float(row['L']))}_kpc" / f"m_{format_tag(float(row['mass']))}_GeV"


def parse_root_macro_graph(path: Path, component_suffix: str) -> tuple[np.ndarray, np.ndarray, str]:
    text = path.read_text()
    vectors: dict[str, np.ndarray] = {}
    for name, body in VECTOR_RE.findall(text):
        vectors[name] = np.asarray([float(x) for x in NUMBER_RE.findall(body)], dtype=float)

    matches: list[tuple[str, str, str]] = []
    for x_name, y_name, graph_name in GRAPH_RE.findall(text):
        if graph_name.endswith(component_suffix):
            matches.append((x_name, y_name, graph_name))

    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {component_suffix} graph in {path}, found {len(matches)}")

    x_name, y_name, graph_name = matches[0]
    try:
        x_values = vectors[x_name]
        y_values = vectors[y_name]
    except KeyError as exc:
        raise RuntimeError(f"Missing vector {exc} for graph {graph_name} in {path}") from exc
    if len(x_values) != len(y_values) or len(x_values) == 0:
        raise RuntimeError(f"Bad graph lengths in {path}: {len(x_values)} vs {len(y_values)}")
    order = np.argsort(x_values)
    return x_values[order], y_values[order], graph_name


def write_comparison(old_tsv: Path, new_tsv: Path, out_path: Path) -> None:
    old = np.genfromtxt(old_tsv, names=True, dtype=float, delimiter="\t")
    new = np.genfromtxt(new_tsv, names=True, dtype=float, delimiter="\t")
    if old.shape != new.shape:
        raise RuntimeError(f"Shape mismatch: old {old.shape}, new {new.shape}")
    tolerances = {
        "L_kpc": (0.0, 1.0e-10),
        "mDM_GeV": (0.0, 1.0e-8),
        # The old plain .out table stores rounded rigidities, while the ROOT
        # macro keeps full double precision.
        "R_GV": (1.0e-5, 1.0e-10),
    }
    for name, (rtol, atol) in tolerances.items():
        if not np.allclose(old[name], new[name], rtol=rtol, atol=atol):
            raise RuntimeError(f"Grid column mismatch for {name}")
    old_flux = old["flux"]
    new_flux = new["flux"]
    mask = np.isfinite(old_flux) & np.isfinite(new_flux) & (old_flux != 0.0)
    ratio = np.full_like(new_flux, np.nan, dtype=float)
    ratio[mask] = new_flux[mask] / old_flux[mask]
    finite = np.isfinite(ratio)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("metric\tvalue\n")
        handle.write(f"rows\t{len(new_flux)}\n")
        handle.write(f"finite_ratio_rows\t{int(np.count_nonzero(finite))}\n")
        handle.write(f"max_abs_delta\t{float(np.nanmax(np.abs(new_flux - old_flux))):.10e}\n")
        handle.write(f"median_primary_over_old_total\t{float(np.nanmedian(ratio[finite])):.10e}\n")
        handle.write(f"min_primary_over_old_total\t{float(np.nanmin(ratio[finite])):.10e}\n")
        handle.write(f"max_primary_over_old_total\t{float(np.nanmax(ratio[finite])):.10e}\n")
        for mass in (7.0, 34.481147, 89.757737, 123.47204):
            mass_mask = finite & np.isclose(new["mDM_GeV"], mass, rtol=1.0e-7, atol=1.0e-7)
            if np.any(mass_mask):
                handle.write(
                    f"median_primary_over_old_total_mDM_{mass:.6g}\t"
                    f"{float(np.nanmedian(ratio[mass_mask])):.10e}\n"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--component-suffix", default="__prim_")
    parser.add_argument(
        "--divide-by-r-power",
        type=float,
        default=0.0,
        help=(
            "Undo USINE Display@FluxPowIndex for plotted ROOT macro graphs. "
            "For example, use 2.8 when the macro stores R^2.8 * flux."
        ),
    )
    parser.add_argument("--drop-highest-mass", action="store_true")
    parser.add_argument("--old-tsv", type=Path, default=None)
    args = parser.parse_args()

    rows = parse_manifest(args.manifest)
    if args.drop_highest_mass:
        masses = sorted({float(row["mass"]) for row in rows})
        retained = set(masses[:-1])
        rows = [row for row in rows if float(row["mass"]) in retained]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.out_dir / "big_2d_dm_explicit_primary_curve_pbar_flux_grid.tsv"
    provenance_path = args.out_dir / "big_2d_dm_explicit_primary_curve_provenance.tsv"
    missing: list[Path] = []

    with combined_path.open("w", encoding="utf-8") as flux_out, provenance_path.open("w", encoding="utf-8") as prov_out:
        flux_out.write("L_kpc\tmDM_GeV\tR_GV\tflux\n")
        prov_out.write("L_kpc\tmDM_GeV\tmacro\tgraph_name\trows\n")
        for row in rows:
            l_value = float(row["L"])
            mass = float(row["mass"])
            macro_path = output_dir_for(row, args.grid_dir) / "local_fluxes_1HBAR_R.C"
            if not macro_path.exists():
                missing.append(macro_path)
                continue
            rigidity, flux, graph_name = parse_root_macro_graph(macro_path, args.component_suffix)
            if args.divide_by_r_power != 0.0:
                if np.any(rigidity <= 0.0):
                    raise RuntimeError(f"Cannot undo R power for non-positive rigidity in {macro_path}")
                flux = flux / np.power(rigidity, args.divide_by_r_power)
            prov_out.write(f"{l_value:.8g}\t{mass:.8g}\t{macro_path}\t{graph_name}\t{len(rigidity)}\n")
            for r_value, flux_value in zip(rigidity, flux):
                flux_out.write(f"{l_value:.8g}\t{mass:.8g}\t{r_value:.8e}\t{flux_value:.8e}\n")

    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} macro files. First missing files:\n{preview}")

    print(f"Wrote explicit primary curve table: {combined_path}")
    print(f"Wrote provenance: {provenance_path}")
    if args.old_tsv:
        comparison_path = args.out_dir / "explicit_primary_vs_old_total_comparison.tsv"
        write_comparison(args.old_tsv, combined_path, comparison_path)
        print(f"Wrote comparison: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
