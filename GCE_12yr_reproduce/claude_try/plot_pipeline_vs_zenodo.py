"""
Plot: Pipeline Model XLIX vs Zenodo Model XLIX GCE flux comparison.

Reads the XLIX pkl file from our calibrated-mask global c_gce fit and
overlays it against Cholis+2022 Zenodo reference.
Analogous to v8.8 Section 12 plot, adapted for Claude-v0 pipeline format.
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR, TEMPLATES_DIR

MODEL = os.environ.get('MODEL', 'XLIX')
SUFFIX = os.environ.get('FIT_SUFFIX', 'globalCgce_calibrated')

ZENODO_FIG = f'{TEMPLATES_DIR}/Figures_12_and_14_GCE_Spectra'
ZENODO_FILE = f'{ZENODO_FIG}/GCE_Model{MODEL}_flux_Inner40x40_masked_disk.dat'
PKL_FILE = f'{OUT_DIR}/GCE_model_{MODEL}_haebarg_v_claude_{SUFFIX}.pkl'


def main():
    os.chdir(WORK_DIR)
    print("=" * 80)
    print(f"Pipeline vs Zenodo comparison : Model {MODEL}")
    print(f"  pkl   : {PKL_FILE}")
    print(f"  zen   : {ZENODO_FILE}")
    print("=" * 80)

    if not os.path.exists(PKL_FILE):
        print(f"ERROR: pkl not found: {PKL_FILE}")
        print("       Check FIT_SUFFIX env (current = '{SUFFIX}')")
        sys.exit(1)
    if not os.path.exists(ZENODO_FILE):
        print(f"ERROR: zenodo not found: {ZENODO_FILE}")
        sys.exit(1)

    with open(PKL_FILE, 'rb') as f:
        d = pickle.load(f)

    E = d['E']
    pl_sed = d['flux_best']
    pl_std = d['flux_std']
    pl_lo = d['flux_lo']
    pl_hi = d['flux_hi']
    c_gce = d['c_gce_global']
    c_gce_std = d['c_gce_global_std']
    n_bins = len(E)

    zen = np.loadtxt(ZENODO_FILE)
    zen_E = zen[:n_bins, 0]
    zen_f = zen[:n_bins, 1]
    zen_lo = zen[:n_bins, 2] if zen.shape[1] > 2 else None
    zen_hi = zen[:n_bins, 3] if zen.shape[1] > 3 else None

    if not np.allclose(E, zen_E, rtol=0.02):
        print(f"WARNING: E axis mismatch. pl[0]={E[0]:.3f}, zen[0]={zen_E[0]:.3f}")

    ratio = pl_sed / np.maximum(zen_f, 1e-30)

    print(f"\n--- Flux comparison (Model {MODEL}) ---")
    print(f"{'bin':>3} {'E[GeV]':>8} {'Zenodo':>12} {'Pipeline':>12}"
          f" {'+/- std':>10} {'ratio':>8}")
    print("-" * 68)
    for ie in range(n_bins):
        print(f"{ie:>3} {E[ie]:>8.3f} {zen_f[ie]:>12.3e} {pl_sed[ie]:>12.3e}"
              f" {pl_std[ie]:>10.3e} {ratio[ie]:>8.3f}")

    low = E < 1.0
    mid = (E >= 1.0) & (E <= 10.0)
    hi = E > 10.0
    print(f"\nratio summary:")
    print(f"  low  (<1 GeV) : mean = {ratio[low].mean():.3f}")
    print(f"  mid  (1-10)   : mean = {ratio[mid].mean():.3f}")
    print(f"  high (>10)    : mean = {ratio[hi].mean():.3f}")
    print(f"  full range    : mean = {ratio.mean():.3f}   median = {np.median(ratio):.3f}")

    print(f"\n  global c_gce  = {c_gce:.4f} +/- {c_gce_std:.4f}")
    print(f"  total -2 logL = {d.get('total_neg2_logL', float('nan')):.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    if zen_lo is not None and zen_hi is not None:
        ax1.errorbar(zen_E, zen_f,
                     yerr=[np.maximum(zen_f-zen_lo, 0), np.maximum(zen_hi-zen_f, 0)],
                     fmt='o-', color='royalblue', markersize=7, linewidth=2,
                     capsize=3, label=f'Zenodo Model {MODEL}')
    else:
        ax1.plot(zen_E, zen_f, 'o-', color='royalblue', markersize=7,
                 linewidth=2, label=f'Zenodo Model {MODEL}')

    ax1.errorbar(E, pl_sed,
                 yerr=[np.maximum(pl_sed-pl_lo, 0), np.maximum(pl_hi-pl_sed, 0)],
                 fmt='D--', color='crimson', markersize=6, linewidth=1.8,
                 capsize=3,
                 label=f'Claude-v0 Model {MODEL} (calibrated, global c_gce)')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$E$ [GeV]', fontsize=13)
    ax1.set_ylabel(r'$E^2\, dN/dE$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]', fontsize=13)
    ax1.set_title(f'GCE SED - Model {MODEL}', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, which='both', alpha=0.3)

    ax2.axhspan(0.9, 1.1, color='gray', alpha=0.15,
                label='Cholis band (0.9-1.1)')
    ax2.plot(E, ratio, 'o-', color='purple', markersize=8, linewidth=2)
    ax2.axhline(1.0, color='k', linestyle='--', alpha=0.6, label='Perfect agreement')
    ax2.axhline(ratio.mean(), color='crimson', linestyle=':', alpha=0.8,
                label=f'Mean = {ratio.mean():.3f}')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$E$ [GeV]', fontsize=13)
    ax2.set_ylabel('Pipeline / Zenodo', fontsize=13)
    ax2.set_title(f'Flux ratio (Model {MODEL})', fontsize=13)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.3, 1.8)

    plt.suptitle(f'Claude-v0 vs Cholis+2022 Zenodo :: Model {MODEL} '
                  f'(suffix={SUFFIX})', fontsize=13, y=1.02)
    plt.tight_layout()
    out_png = f'{OUT_DIR}/plot_pipeline_vs_zenodo_model{MODEL}_{SUFFIX}.png'
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {out_png}")


if __name__ == '__main__':
    main()
