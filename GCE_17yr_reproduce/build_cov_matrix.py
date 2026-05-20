#!/usr/bin/env python3
"""
build_cov_matrix.py — assemble 14x14 systematic covariance matrix from
22 per-ROI cov fit outputs.

Reads:
    results_cov_17yr/GCE_cov_l{ROI}_front_17yr_cholis_fit.npz   (22 files)
Writes:
    results_cov_17yr/GCE_systematic_covariance_matrix_17yr.npz

Formula (memory #5, validated against Sanghwan 16yr):
    GCE_flux[bin]  = fitted_params[2] * GCE_template[bin] * E[bin]^2 / delta_E[bin]
    cov_matrix     = np.cov(GCE_flux_per_roi.T)        # bias-corrected (ddof=1)
    mean_GCE       = GCE_flux_per_roi.mean(axis=0)

Output npz keys:
    cov_matrix         (14, 14)  systematic covariance, GeV^2/cm^4/s^2/sr^2
    mean_GCE           (14,)     mean GCE flux across ROIs
    sigma_sys          (14,)     sqrt(diag(cov)) — per-bin systematic 1-sigma
    rois               (20,)     ROI list (int)
    GCE_flux_per_roi   (20, 14)  per-ROI flux at fitted_params[2]
    E                  (14,)     bin centers, GeV
    delta_E            (14,)     bin widths, GeV
    cond_number        scalar    cov matrix condition number
    inv_cov_matrix     (14, 14)  Moore-Penrose pseudoinverse

Validation expectations (memory #5):
    - condition number ~ 10^5
    - sigma_sys peak at ~0.5 GeV: ~10^-6 GeV/cm^2/s/sr
    - sigma_sys at >10 GeV bins: ~10^-7 GeV/cm^2/s/sr
    - matches Calore+ (1409.0042) pattern

Usage:
    python build_cov_matrix.py [--cov-dir results_cov_17yr] [--plot]

The --plot flag emits a sanity-check figure (cov matrix heatmap + sigma_sys
curve) to results_cov_17yr/cov_matrix_validation.png.

Author: haebarg (2026)

Changes:
  [fb17-cov-v1] (2026-07-28) FB17=1 env -> front+back 17-bin cov variant
      (main pipeline과 동일 패턴): WORK_DIR/FRONT/evtype 전환, 결과는
      results_cov_fb17/ 분리(기존 front 22개 .dat와 카운트 충돌 방지).
      env 미설정 시 기존 fiducial 동작과 동일.
"""

import os
import sys
import argparse
import numpy as np


ALL_ROIS = [r for r in range(-70, 75, 5) if r != 0 and abs(r) >= 20]
assert len(ALL_ROIS) == 22

FRONT = '_front'

# FB17 variant — env switch: front_back 네이밍 + 기본 cov-dir 전환.
FB17 = bool(os.environ.get('FB17', '').strip())
_DEFAULT_COV_DIR = 'results_cov_fb17' if FB17 else 'results_cov_17yr'
if FB17:
    FRONT = '_front_back'
    print(f'[config] FB17=1 -> FRONT={FRONT!r}, '
          f'default cov-dir={_DEFAULT_COV_DIR}', flush=True)


def load_one(cov_dir, roi):
    """Load one per-ROI cov fit npz. Returns (E, delta_E, GCE_template,
    fitted_params_2_chain) or raises FileNotFoundError if missing."""
    path = os.path.join(cov_dir, f'GCE_cov_l{roi}{FRONT}_17yr_cholis_fit.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'missing: {path}')
    d = np.load(path)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cov-dir', type=str, default=_DEFAULT_COV_DIR,
                    help='directory containing per-ROI fit npz files')
    ap.add_argument('--out', type=str, default=None,
                    help='output npz path (default: {cov-dir}/GCE_systematic_covariance_matrix_17yr.npz)')
    ap.add_argument('--rois', type=str, default='',
                    help='comma-separated ROI subset (default: use all available)')
    ap.add_argument('--use-median', action='store_true',
                    help='use fitted_params_median[2] instead of best-fit fitted_params[2] '
                         '(more stable to outliers, but Cholis paper uses best-fit)')
    ap.add_argument('--plot', action='store_true',
                    help='emit cov_matrix_validation.png in cov-dir')
    args = ap.parse_args()

    if not os.path.isdir(args.cov_dir):
        print(f'[FATAL] cov-dir not found: {args.cov_dir}')
        sys.exit(2)
    out_path = args.out or os.path.join(args.cov_dir,
                                        'GCE_systematic_covariance_matrix_17yr.npz')

    if args.rois.strip():
        rois = [int(r.strip()) for r in args.rois.split(',') if r.strip()]
    else:
        rois = ALL_ROIS[:]

    # Load all ROIs
    loaded   = {}
    missing  = []
    for roi in rois:
        try:
            loaded[roi] = load_one(args.cov_dir, roi)
        except FileNotFoundError:
            missing.append(roi)

    if missing:
        print(f'[warn] {len(missing)}/{len(rois)} ROI(s) missing: {missing}')
        print(f'[warn] proceeding with {len(loaded)} ROIs')
    if len(loaded) < 5:
        print(f'[FATAL] too few ROIs ({len(loaded)}) for meaningful covariance. need ≥ 5.')
        sys.exit(2)

    rois_used = sorted(loaded.keys())
    n_roi = len(rois_used)

    # Reference axes from first ROI
    first = loaded[rois_used[0]]
    E       = first['E']
    delta_E = first['delta_E']
    n_bin   = len(E)
    print(f'[info] {n_roi} ROIs loaded, {n_bin} bins')
    print(f'[info] E range: {E[0]:.3f} - {E[-1]:.3f} GeV')

    # Cross-check E/delta_E across ROIs
    for roi in rois_used[1:]:
        d = loaded[roi]
        if not np.allclose(d['E'], E):
            print(f'[FATAL] E mismatch at roi={roi}')
            sys.exit(2)
        if not np.allclose(d['delta_E'], delta_E):
            print(f'[FATAL] delta_E mismatch at roi={roi}')
            sys.exit(2)

    # Build GCE_flux[roi, bin] = c_gce[roi, bin] * GCE_template[roi, bin] * E^2 / dE
    GCE_flux_per_roi = np.zeros((n_roi, n_bin))
    c_gce_per_roi    = np.zeros((n_roi, n_bin))
    GCE_tmpl_per_roi = np.zeros((n_roi, n_bin))

    for i, roi in enumerate(rois_used):
        d = loaded[roi]
        # fitted_params reshape (5, n_bin); index 2 = c_gce
        if args.use_median:
            c_gce = d['fitted_params_median'][2]
        else:
            c_gce = d['fitted_params'][2]
        GCE_template = d['GCE']
        c_gce_per_roi[i]    = c_gce
        GCE_tmpl_per_roi[i] = GCE_template
        GCE_flux_per_roi[i] = c_gce * GCE_template * (E ** 2) / delta_E

    # Cov matrix (bias-corrected, ddof=1)
    cov_matrix = np.cov(GCE_flux_per_roi.T, ddof=1)
    mean_GCE   = GCE_flux_per_roi.mean(axis=0)
    sigma_sys  = np.sqrt(np.diag(cov_matrix))

    # Condition number + pseudoinverse
    cond_number = np.linalg.cond(cov_matrix)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)

    # Diagnostics
    print(f'\n=== cov matrix diagnostics ===')
    print(f'  condition number     : {cond_number:.2e}')
    print(f'  determinant          : {np.linalg.det(cov_matrix):.3e}')
    print(f'  trace                : {np.trace(cov_matrix):.3e}')
    print(f'\n=== sigma_sys per bin (GeV/cm^2/s/sr) ===')
    print(f'  {"bin":>3} {"E[GeV]":>8} {"mean_GCE":>11} {"sigma_sys":>11} {"frac":>7}')
    for i in range(n_bin):
        frac = sigma_sys[i] / abs(mean_GCE[i]) if mean_GCE[i] != 0 else float('inf')
        print(f'  {i:>3} {E[i]:>8.3f} {mean_GCE[i]:>11.3e} {sigma_sys[i]:>11.3e} {frac:>7.2f}')

    # Save
    np.savez(
        out_path,
        cov_matrix       = cov_matrix,
        inv_cov_matrix   = inv_cov_matrix,
        mean_GCE         = mean_GCE,
        sigma_sys        = sigma_sys,
        rois             = np.array(rois_used),
        GCE_flux_per_roi = GCE_flux_per_roi,
        c_gce_per_roi    = c_gce_per_roi,
        GCE_tmpl_per_roi = GCE_tmpl_per_roi,
        E                = E,
        delta_E          = delta_E,
        cond_number      = cond_number,
        n_rois_used      = n_roi,
    )
    print(f'\n[done] saved {out_path}')

    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

            # Cov matrix heatmap
            im = ax1.imshow(cov_matrix, origin='lower', aspect='auto',
                            cmap='RdBu_r',
                            vmin=-np.abs(cov_matrix).max(),
                            vmax=+np.abs(cov_matrix).max())
            ax1.set_xlabel('bin')
            ax1.set_ylabel('bin')
            ax1.set_title(f'cov matrix (cond={cond_number:.1e})')
            fig.colorbar(im, ax=ax1)

            # sigma_sys curve + Calore+ pattern reference
            ax2.loglog(E, sigma_sys, 'o-', label='sigma_sys (this work)')
            ax2.loglog(E, np.abs(mean_GCE), 's--', alpha=0.6, label='|mean_GCE|')
            ax2.set_xlabel('E [GeV]')
            ax2.set_ylabel(r'$E^2 dN/dE$ [GeV/cm$^2$/s/sr]')
            ax2.set_title(f'17yr cov: {n_roi} ROIs')
            ax2.legend()
            ax2.grid(True, which='both', linestyle=':', alpha=0.5)

            plt.tight_layout()
            plot_path = os.path.join(args.cov_dir, 'cov_matrix_validation.png')
            plt.savefig(plot_path, dpi=120)
            print(f'[done] saved validation plot: {plot_path}')
        except Exception as e:
            print(f'[warn] plot failed: {e}')


if __name__ == '__main__':
    main()
