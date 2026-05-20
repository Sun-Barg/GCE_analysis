#!/usr/bin/env python3
"""Render the on-shell 30-configuration antiproton limit matrix with PyROOT."""

from __future__ import annotations

import argparse
import csv
import math
from array import array
from pathlib import Path

import ROOT


DEFAULT_INPUT = Path(
    "Codex_files/generated_outputs/onshell_30_pbar_limits_20260727/"
    "combined_onshell_pbar_limit_summary_finite_only.tsv"
)
DEFAULT_OUTPUT = Path(
    "Codex_files/generated_outputs/onshell_30_pbar_limits_20260727/"
    "onshell_30_pbar_limit_matrix.png"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def graph_for(rows: list[dict[str, str]], state: str, r_value: float) -> ROOT.TGraph | None:
    points: list[tuple[float, float]] = []
    for row in rows:
        if row["state"] != state:
            continue
        if abs(float(row["r_value"]) - r_value) > 1e-9:
            continue
        try:
            mass = float(row["mDM_GeV"])
            limit = float(row["sigmav95_cm3_s"])
        except ValueError:
            continue
        if math.isfinite(mass) and math.isfinite(limit) and mass > 0.0 and limit > 0.0:
            points.append((mass, limit))
    if not points:
        return None
    points.sort()
    xs = array("d", [item[0] for item in points])
    ys = array("d", [item[1] for item in points])
    return ROOT.TGraph(len(points), xs, ys)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = read_rows(args.input)
    states = ["4b", "4tau", "2b2tau"]
    r_values = [round(0.1 * i, 1) for i in range(1, 11)]
    colors = [
        ROOT.kBlack,
        ROOT.kRed + 1,
        ROOT.kBlue + 1,
        ROOT.kGreen + 2,
        ROOT.kMagenta + 1,
        ROOT.kOrange + 7,
        ROOT.kCyan + 2,
        ROOT.kViolet + 1,
        ROOT.kTeal + 3,
        ROOT.kGray + 2,
    ]

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    canvas = ROOT.TCanvas("c", "On-shell antiproton limits", 1500, 520)
    canvas.Divide(3, 1, 0.01, 0.01)
    keepalive: list[object] = []

    for index, state in enumerate(states, start=1):
        pad = canvas.cd(index)
        pad.SetLogx()
        pad.SetLogy()
        pad.SetGrid(1, 1)
        pad.SetLeftMargin(0.14 if index == 1 else 0.09)
        pad.SetRightMargin(0.03)
        pad.SetBottomMargin(0.14)
        pad.SetTopMargin(0.10)

        frame = ROOT.TH1F(f"frame_{state}", f"{state};m_{{#chi}} [GeV];#LT#sigma v#GT_{{95}} [cm^{{3}} s^{{-1}}]", 100, 8.0, 1200.0)
        frame.SetMinimum(5.0e-27)
        frame.SetMaximum(1.0e6 if state == "4tau" else 5.0e-20)
        frame.GetXaxis().SetMoreLogLabels(True)
        frame.GetXaxis().SetNoExponent(True)
        frame.GetXaxis().SetTitleSize(0.045)
        frame.GetYaxis().SetTitleSize(0.045)
        frame.GetYaxis().SetTitleOffset(1.4 if index == 1 else 1.1)
        frame.Draw()
        keepalive.append(frame)

        legend = ROOT.TLegend(0.58, 0.18, 0.94, 0.50)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.030)
        keepalive.append(legend)

        for color, r_value in zip(colors, r_values):
            graph = graph_for(rows, state, r_value)
            if graph is None:
                continue
            graph.SetLineColor(color)
            graph.SetMarkerColor(color)
            graph.SetLineWidth(2)
            graph.SetMarkerStyle(20)
            graph.SetMarkerSize(0.65)
            graph.Draw("LP SAME")
            legend.AddEntry(graph, f"r={r_value:g}", "lp")
            keepalive.append(graph)
        legend.Draw()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(args.output))
    canvas.SaveAs(str(args.output.with_suffix(".pdf")))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
