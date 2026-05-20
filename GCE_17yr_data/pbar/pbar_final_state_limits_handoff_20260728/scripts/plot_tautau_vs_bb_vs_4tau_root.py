#!/usr/bin/env python3
"""Plot standard tau+tau-, standard bb, and on-shell 4tau antiproton limits."""

from __future__ import annotations

import argparse
import csv
import math
from array import array
from pathlib import Path

import ROOT


DEFAULT_BB = Path("Codex_files/generated_outputs/calore_cdata_best_branch_20260601/pbar_95cl_upper_limits_data_cov_only.tsv")
DEFAULT_TAUTAU = Path("Codex_files/generated_outputs/standard_tautau_pbar_limits_20260728/limits/pbar_95cl_upper_limits_data_cov_only.tsv")
DEFAULT_4TAU = Path("Codex_files/generated_outputs/onshell_30_pbar_limits_20260727/combined_onshell_pbar_limit_summary_finite_only.tsv")
DEFAULT_OUT = Path("Codex_files/generated_outputs/standard_tautau_pbar_limits_20260728/report/tautau_vs_bb_vs_4tau_pbar_limits.png")


def read_limit_points(path: Path, mass_col: str = "mDM_GeV", limit_col: str = "sigmav95_cm3_s") -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                mass = float(row[mass_col])
                limit = float(row[limit_col])
            except (KeyError, ValueError):
                continue
            if math.isfinite(mass) and math.isfinite(limit) and mass > 0.0 and limit > 0.0:
                points.append((mass, limit))
    return sorted(points)


def read_4tau_points(path: Path, r_value: float) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("state") != "4tau":
                continue
            try:
                if abs(float(row["r_value"]) - r_value) > 1.0e-9:
                    continue
                mass = float(row["mDM_GeV"])
                limit = float(row["sigmav95_cm3_s"])
            except (KeyError, ValueError):
                continue
            if math.isfinite(mass) and math.isfinite(limit) and mass > 0.0 and limit > 0.0:
                points.append((mass, limit))
    return sorted(points)


def clipped(points: list[tuple[float, float]], max_limit: float | None) -> list[tuple[float, float]]:
    if max_limit is None:
        return points
    return [(mass, limit) for mass, limit in points if limit <= max_limit]


def graph(points: list[tuple[float, float]], name: str, color: int, width: int, marker: int) -> ROOT.TGraph:
    xs = array("d", [item[0] for item in points])
    ys = array("d", [item[1] for item in points])
    g = ROOT.TGraph(len(points), xs, ys)
    g.SetName(name)
    g.SetLineColor(color)
    g.SetMarkerColor(color)
    g.SetLineWidth(width)
    g.SetMarkerStyle(marker)
    g.SetMarkerSize(0.65)
    return g


def write_comparison_tsv(bb: list[tuple[float, float]], tautau: list[tuple[float, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("series\tmDM_GeV\tsigmav95_cm3_s\n")
        for mass, limit in bb:
            handle.write(f"bb\t{mass:.10e}\t{limit:.10e}\n")
        for mass, limit in tautau:
            handle.write(f"tautau\t{mass:.10e}\t{limit:.10e}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bb", type=Path, default=DEFAULT_BB)
    parser.add_argument("--tautau", type=Path, default=DEFAULT_TAUTAU)
    parser.add_argument("--four-tau", type=Path, default=DEFAULT_4TAU)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--y-min", type=float, default=1.0e-27)
    parser.add_argument("--y-max", type=float, default=1.0e6)
    parser.add_argument("--max-plotted-limit", type=float, default=None)
    args = parser.parse_args()

    bb_points = clipped(read_limit_points(args.bb), args.max_plotted_limit)
    tautau_points = clipped(read_limit_points(args.tautau), args.max_plotted_limit)
    r_values = [round(0.1 * i, 1) for i in range(1, 11)]

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    canvas = ROOT.TCanvas("c", "tau sanity comparison", 1100, 760)
    canvas.SetLogx()
    canvas.SetLogy()
    canvas.SetGrid(1, 1)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.04)
    canvas.SetBottomMargin(0.11)
    canvas.SetTopMargin(0.06)

    frame = ROOT.TH1F(
        "frame",
        ";m_{#chi} [GeV];#LT#sigma v#GT_{95} [cm^{3} s^{-1}]",
        100,
        6.0,
        1200.0,
    )
    frame.SetMinimum(args.y_min)
    frame.SetMaximum(args.y_max)
    frame.GetXaxis().SetMoreLogLabels(True)
    frame.GetXaxis().SetNoExponent(True)
    frame.GetYaxis().SetTitleOffset(1.35)
    frame.Draw()

    keepalive: list[object] = [frame]
    legend = ROOT.TLegend(0.50, 0.16, 0.93, 0.52)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.028)
    keepalive.append(legend)

    bb_graph = graph(bb_points, "bb", ROOT.kBlack, 4, 20)
    bb_graph.Draw("LP SAME")
    legend.AddEntry(bb_graph, "standard b#bar{b}", "lp")
    keepalive.append(bb_graph)

    tautau_graph = graph(tautau_points, "tautau", ROOT.kRed + 1, 4, 21)
    tautau_graph.Draw("LP SAME")
    legend.AddEntry(tautau_graph, "standard #tau^{+}#tau^{-}", "lp")
    keepalive.append(tautau_graph)

    colors = [
        ROOT.kBlue + 1,
        ROOT.kGreen + 2,
        ROOT.kMagenta + 1,
        ROOT.kOrange + 7,
        ROOT.kCyan + 2,
        ROOT.kViolet + 1,
        ROOT.kTeal + 3,
        ROOT.kGray + 2,
        ROOT.kAzure + 7,
        ROOT.kPink + 7,
    ]
    for color, r_value in zip(colors, r_values):
        points = clipped(read_4tau_points(args.four_tau, r_value), args.max_plotted_limit)
        if not points:
            continue
        g = graph(points, f"4tau_r{r_value:g}", color, 2, 24)
        g.Draw("LP SAME")
        legend.AddEntry(g, f"on-shell 4#tau r={r_value:g}", "lp")
        keepalive.append(g)

    legend.Draw()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(args.output))
    canvas.SaveAs(str(args.output.with_suffix(".pdf")))
    write_comparison_tsv(bb_points, tautau_points, args.output.with_suffix(".standard_2body.tsv"))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.standard_2body.tsv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
