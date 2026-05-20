#!/usr/bin/env python3
"""Plot our tau+tau- antiproton limit against literature order-of-magnitude guides."""

from __future__ import annotations

import argparse
import csv
import math
from array import array
from pathlib import Path

import ROOT


DEFAULT_TAUTAU = Path("Codex_files/generated_outputs/standard_tautau_pbar_limits_20260728/limits/pbar_95cl_upper_limits_data_cov_only.tsv")
DEFAULT_BB = Path("Codex_files/generated_outputs/calore_cdata_best_branch_20260601/pbar_95cl_upper_limits_data_cov_only.tsv")
DEFAULT_OUT = Path("Codex_files/generated_outputs/standard_tautau_pbar_limits_20260728/report/tautau_antiproton_literature_order_comparison.png")


def read_points(path: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                mass = float(row["mDM_GeV"])
                limit = float(row["sigmav95_cm3_s"])
            except (KeyError, ValueError):
                continue
            if math.isfinite(mass) and math.isfinite(limit) and mass > 0.0 and limit > 0.0:
                points.append((mass, limit))
    return sorted(points)


def make_graph(points: list[tuple[float, float]], name: str, color: int, width: int, marker: int) -> ROOT.TGraph:
    xs = array("d", [point[0] for point in points])
    ys = array("d", [point[1] for point in points])
    graph = ROOT.TGraph(len(points), xs, ys)
    graph.SetName(name)
    graph.SetLineColor(color)
    graph.SetMarkerColor(color)
    graph.SetLineWidth(width)
    graph.SetMarkerStyle(marker)
    graph.SetMarkerSize(0.8)
    return graph


def make_band(name: str, x_min: float, x_max: float, y_min: float, y_max: float, color: int, alpha: float) -> ROOT.TGraph:
    xs = array("d", [x_min, x_max, x_max, x_min, x_min])
    ys = array("d", [y_min, y_min, y_max, y_max, y_min])
    graph = ROOT.TGraph(5, xs, ys)
    graph.SetName(name)
    graph.SetFillColorAlpha(color, alpha)
    graph.SetLineColor(color)
    graph.SetLineStyle(2)
    graph.SetLineWidth(2)
    return graph


def write_guides(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("guide\tx_min_GeV\tx_max_GeV\ty_min_cm3_s\ty_max_cm3_s\tnote\n")
        handle.write(
            "Cirelli_Giesen_2013_PAMELA_mutau_visual\t"
            "1.0e1\t1.0e3\t1.0e-22\t1.0e-21\t"
            "Approximate Fig.2 leptonic mu/tau antiproton exclusion band; not digitized.\n"
        )
        handle.write(
            "Calore_et_al_2022_AMS02_leptonic_expectation\t"
            "1.0e2\t1.0e3\t3.0e-23\t3.0e-21\t"
            "Order guide from statement that leptonic antiproton bounds probe cross sections about 1e3 above hadronic modes.\n"
        )
        handle.write(
            "Fermi_dSph_tau_reference\t"
            "1.0e1\t1.0e2\t1.0e-27\t3.0e-26\t"
            "Gamma-ray dwarf reference scale, not an antiproton bound; shown only to contrast messenger sensitivity.\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tautau", type=Path, default=DEFAULT_TAUTAU)
    parser.add_argument("--bb", type=Path, default=DEFAULT_BB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tau_points = read_points(args.tautau)
    bb_points = read_points(args.bb)

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    canvas = ROOT.TCanvas("c", "tau antiproton literature comparison", 1100, 760)
    canvas.SetLogx()
    canvas.SetLogy()
    canvas.SetGrid(1, 1)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.04)
    canvas.SetBottomMargin(0.12)
    canvas.SetTopMargin(0.06)

    frame = ROOT.TH1F(
        "frame",
        ";m_{#chi} [GeV];#LT#sigma v#GT_{95} [cm^{3} s^{-1}]",
        100,
        7.0,
        1200.0,
    )
    frame.SetMinimum(1.0e-27)
    frame.SetMaximum(3.0e-20)
    frame.GetXaxis().SetMoreLogLabels(True)
    frame.GetXaxis().SetNoExponent(True)
    frame.GetYaxis().SetTitleOffset(1.35)
    frame.Draw()

    keepalive: list[object] = [frame]
    c13_band = make_band("cirelli_giesen_2013", 10.0, 1000.0, 1.0e-22, 1.0e-21, ROOT.kOrange + 7, 0.25)
    calore_band = make_band("calore_2022_leptonic_order", 100.0, 1000.0, 3.0e-23, 3.0e-21, ROOT.kAzure + 1, 0.16)
    fermi_band = make_band("fermi_dsph_tau_reference", 10.0, 100.0, 1.0e-27, 3.0e-26, ROOT.kGreen + 2, 0.18)

    for band in [c13_band, calore_band, fermi_band]:
        band.Draw("F SAME")
        band.Draw("L SAME")
        keepalive.append(band)

    bb_graph = make_graph(bb_points, "our_bb", ROOT.kBlack, 3, 20)
    tau_graph = make_graph(tau_points, "our_tautau", ROOT.kRed + 1, 4, 21)
    bb_graph.Draw("LP SAME")
    tau_graph.Draw("LP SAME")
    keepalive.extend([bb_graph, tau_graph])

    thermal = ROOT.TLine(7.0, 3.0e-26, 1200.0, 3.0e-26)
    thermal.SetLineColor(ROOT.kGray + 2)
    thermal.SetLineStyle(7)
    thermal.SetLineWidth(2)
    thermal.Draw("SAME")
    keepalive.append(thermal)

    legend = ROOT.TLegend(0.39, 0.15, 0.93, 0.47)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.027)
    legend.AddEntry(tau_graph, "this work: standard #tau^{+}#tau^{-} (#bar{p})", "lp")
    legend.AddEntry(bb_graph, "this work: standard b#bar{b} (#bar{p})", "lp")
    legend.AddEntry(c13_band, "Cirelli & Giesen 2013 #mu/#tau #bar{p} scale", "f")
    legend.AddEntry(calore_band, "Calore et al. 2022 leptonic #bar{p} order guide", "f")
    legend.AddEntry(fermi_band, "Fermi dSph #tau scale, #gamma-ray only", "f")
    legend.AddEntry(thermal, "thermal relic scale", "l")
    legend.Draw()
    keepalive.append(legend)

    label = ROOT.TLatex()
    label.SetNDC(True)
    label.SetTextSize(0.026)
    label.DrawLatex(0.14, 0.90, "Literature bands are order-of-magnitude guides, not digitized curves")
    keepalive.append(label)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(args.output))
    canvas.SaveAs(str(args.output.with_suffix(".pdf")))
    write_guides(args.output.with_suffix(".guides.tsv"))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.guides.tsv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
