#!/usr/bin/env python3
"""Prepare a USINE PPPC overlay for on-shell four-body antiproton spectra."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import re
from pathlib import Path


DEFAULT_TABLE_ROOT = Path("Prompt_spectra/custom_spectrum_tables")
DEFAULT_STANDARD_PPPC = Path("inputs/PPPC4DMID-spectra/2019/AtProduction_antiprotons.dat")
DEFAULT_BASE_MANIFEST = Path("Codex_files/configs/generated/BIG_2D_DM_NFW_bbbar_grid_kCIRELLI19_20260601/manifest.txt")
DEFAULT_OUT_ROOT = Path("Codex_files/generated_outputs/onshell_30_pbar_limits_20260727")

BBAR_KCIRELLI19_COL = 7  # row fields: mDM, log10x, e, mu, tau, q, c, b, ...
CSV_RE = re.compile(r"^MM_mpsi(?P<mass>[0-9.]+)GeV_(?P<state>.+)_antiproton\.csv$")
BASELINE_MASSES = [
    25.065996,
    34.481147,
    47.432764,
    65.249196,
    89.757737,
    123.47204,
    169.84993,
    233.64801,
    321.40958,
    442.13567,
    608.20822,
    836.66003,
    1150.9216,
    1583.2244,
    2177.9064,
    2995.9596,
    4121.2853,
    5669.2996,
    7798.7706,
]


def safe_r_label(value: str) -> str:
    return "r" + value.replace(".", "p")


def case_name(state: str, r_value: str) -> str:
    return f"onshell_{state}_{safe_r_label(r_value)}"


def spectrum_dir(table_root: Path, state: str, r_value: str) -> Path:
    return table_root / f"Spectra_Data_sfdm_{state}_off" / f"Spectra_Data_sfdm_{state}_off_r{r_value}"


def read_standard_pppc(path: Path) -> tuple[str, list[list[float]], list[float], list[float]]:
    header = ""
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            if not header:
                header = line
            continue
        rows.append([float(item) for item in line.split()])
    masses: list[float] = []
    logxs: list[float] = []
    for row in rows:
        if not masses or row[0] > masses[-1]:
            masses.append(row[0])
    first_mass = masses[0]
    for row in rows:
        if row[0] != first_mass:
            break
        logxs.append(row[1])
    return header, rows, masses, logxs


def read_onshell_csv_tables(root: Path, state: str) -> tuple[list[float], dict[float, tuple[list[float], list[float]]]]:
    tables: dict[float, tuple[list[float], list[float]]] = {}
    for path in sorted(root.glob(f"MM_mpsi*GeV_{state}_antiproton.csv")):
        match = CSV_RE.match(path.name)
        if not match:
            continue
        mass = float(match.group("mass"))
        xs: list[float] = []
        dndlogx: list[float] = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"Energy_GeV", "x", "dNdE"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{path}: missing columns {sorted(required - set(reader.fieldnames or []))}")
            for row in reader:
                energy = float(row["Energy_GeV"])
                x_value = float(row["x"])
                dnde = float(row["dNdE"])
                if energy <= 0.0 or x_value <= 0.0 or dnde < 0.0:
                    continue
                xs.append(x_value)
                dndlogx.append(dnde * energy * math.log(10.0))
        order = sorted(range(len(xs)), key=xs.__getitem__)
        xs = [xs[i] for i in order]
        dndlogx = [dndlogx[i] for i in order]
        if xs:
            tables[mass] = (xs, dndlogx)
    masses = sorted(tables)
    if not masses:
        raise FileNotFoundError(f"No {state} antiproton CSV tables found in {root}")
    return masses, tables


def interp_loglog(x: float, xs: list[float], ys: list[float], grid_xs: list[float] | None = None) -> float:
    if len(xs) == 1:
        if grid_xs is None:
            return max(0.0, ys[0]) if x == xs[0] else 0.0
        nearest = min(grid_xs, key=lambda value: abs(math.log(value / xs[0])))
        return max(0.0, ys[0]) if x == nearest else 0.0
    if x <= 0.0 or x < xs[0] or x > xs[-1]:
        return 0.0
    pos = bisect.bisect_left(xs, x)
    if pos < len(xs) and xs[pos] == x:
        return max(0.0, ys[pos])
    if pos == 0 or pos >= len(xs):
        return 0.0
    x0, x1 = xs[pos - 1], xs[pos]
    y0, y1 = max(ys[pos - 1], 0.0), max(ys[pos], 0.0)
    t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    if y0 > 0.0 and y1 > 0.0:
        return math.exp((1.0 - t) * math.log(y0) + t * math.log(y1))
    return (1.0 - t) * y0 + t * y1


def interp_mass_loglog(
    mass: float,
    x: float,
    masses: list[float],
    tables: dict[float, tuple[list[float], list[float]]],
    grid_xs: list[float] | None = None,
) -> float:
    if mass < masses[0] or mass > masses[-1]:
        return 0.0
    pos = bisect.bisect_left(masses, mass)
    if pos < len(masses) and masses[pos] == mass:
        return interp_loglog(x, *tables[masses[pos]], grid_xs=grid_xs)
    if pos == 0 or pos >= len(masses):
        return 0.0
    m0, m1 = masses[pos - 1], masses[pos]
    y0 = interp_loglog(x, *tables[m0], grid_xs=grid_xs)
    y1 = interp_loglog(x, *tables[m1], grid_xs=grid_xs)
    t = (math.log(mass) - math.log(m0)) / (math.log(m1) - math.log(m0))
    if y0 > 0.0 and y1 > 0.0:
        return math.exp((1.0 - t) * math.log(y0) + t * math.log(y1))
    return (1.0 - t) * y0 + t * y1


def nearest_mass(target: float, masses: list[float]) -> float:
    return min(masses, key=lambda value: abs(math.log(value / target)))


def selected_run_masses(custom_masses: list[float], mode: str) -> list[float]:
    if mode == "all":
        return custom_masses
    if mode != "baseline-plus-edges":
        raise ValueError(f"Unknown mass mode {mode}")
    selected = {custom_masses[0], custom_masses[-1]}
    for target in BASELINE_MASSES:
        if custom_masses[0] <= target <= custom_masses[-1]:
            selected.add(nearest_mass(target, custom_masses))
    return sorted(selected)


def mass_tag(value: float) -> str:
    if value >= 1000.0:
        text = f"{value:.0f}"
    elif value >= 100.0:
        text = f"{value:.2f}".rstrip("0").rstrip(".")
    else:
        text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


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


def write_overlay_pppc(args: argparse.Namespace, case_dir: Path, custom_masses: list[float], custom_tables: dict[float, tuple[list[float], list[float]]]) -> dict[str, object]:
    header, rows, pppc_masses, logxs = read_standard_pppc(args.standard_pppc)
    grid_xs = [10.0 ** value for value in logxs]
    overlay_file = case_dir / "usine_overlay" / "inputs" / "PPPC4DMID-spectra" / "2019" / "AtProduction_antiprotons.dat"
    overlay_file.parent.mkdir(parents=True, exist_ok=True)
    with overlay_file.open("w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        for row in rows:
            new_row = row[:]
            new_row[BBAR_KCIRELLI19_COL] = interp_mass_loglog(row[0], 10.0 ** row[1], custom_masses, custom_tables, grid_xs=grid_xs)
            handle.write(" ".join(f"{value:.8e}" for value in new_row) + "\n")
    return {
        "overlay_pppc": str(overlay_file),
        "standard_pppc": str(args.standard_pppc),
        "custom_mass_min_GeV": custom_masses[0],
        "custom_mass_max_GeV": custom_masses[-1],
        "custom_table_count": len(custom_masses),
        "pppc_masses": len(pppc_masses),
        "pppc_log10x": len(logxs),
        "replacement_column": "kCIRELLI19 b column used through bbbar branching slot",
    }


def write_manifest(args: argparse.Namespace, case_dir: Path, run_masses: list[float]) -> dict[str, object]:
    out_config = case_dir / "configs"
    out_config.mkdir(parents=True, exist_ok=True)
    manifest = out_config / "manifest.txt"
    template_by_l: dict[float, list[str]] = {}
    with args.base_manifest.open() as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            l_value = float(parts[0])
            template_by_l.setdefault(l_value, parts)
    with manifest.open("w", encoding="utf-8") as dst:
        dst.write("# BIG 2D primary-DM antiproton grid\n")
        dst.write(f"# channel: {args.state}, r={args.r_value}, model=kCIRELLI19 overlay b column\n")
        dst.write("# Convention: baseline BIG/NFW transport; custom on-shell prompt only.\n")
        for l_value, base_parts in sorted(template_by_l.items()):
            for mass in run_masses:
                old_init = Path(base_parts[-1])
                new_init = out_config / f"init.BIG_2D_DM_NFW_{case_name(args.state, args.r_value)}_L_{str(l_value).replace('.', 'p')}_m_{mass_tag(mass)}.par"
                text = old_init.read_text()
                text = re.sub(
                    r"(DarkMatter\s+@\s+ParticlePhysics\s+@\s+MassDM\s+@\s+M=0\s+@\s+)[^\n]+",
                    rf"\g<1>{mass:.8g}",
                    text,
                )
                text = text.replace(
                    "# channel: bbbar, model=kCIRELLI19",
                    f"# channel: {case_name(args.state, args.r_value)} via overlay b column, model=kCIRELLI19",
                )
                new_init.write_text(text, encoding="utf-8")
                parts = list(base_parts)
                parts[1] = f"{mass:.8g}"
                parts[-1] = str(new_init)
                dst.write(" ".join(parts) + "\n")
    return {"manifest": str(manifest), "rows": len(run_masses) * len(template_by_l), "masses": run_masses}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, choices=["4b", "4tau", "2b2tau"])
    parser.add_argument("--r-value", required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    parser.add_argument("--standard-pppc", type=Path, default=DEFAULT_STANDARD_PPPC)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--mass-mode", choices=["baseline-plus-edges", "all"], default="baseline-plus-edges")
    args = parser.parse_args()

    cname = case_name(args.state, args.r_value)
    case_dir = args.out_root / cname
    case_dir.mkdir(parents=True, exist_ok=True)
    source_dir = spectrum_dir(args.table_root, args.state, args.r_value)
    custom_masses, custom_tables = read_onshell_csv_tables(source_dir, args.state)
    run_masses = selected_run_masses(custom_masses, args.mass_mode)
    make_overlay_links(case_dir)
    overlay_info = write_overlay_pppc(args, case_dir, custom_masses, custom_tables)
    manifest_info = write_manifest(args, case_dir, run_masses)
    payload = {
        "case": cname,
        "state": args.state,
        "r_value": args.r_value,
        "source_dir": str(source_dir),
        "mass_mode": args.mass_mode,
        "overlay": overlay_info,
        "grid": manifest_info,
    }
    (case_dir / "onshell_pbar_overlay_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
