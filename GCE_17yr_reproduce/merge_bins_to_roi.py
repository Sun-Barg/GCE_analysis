#!/usr/bin/env python3
"""merge_bins_to_roi.py — Merge 14 per-bin .npz files into the per-ROI .npz
   format expected by build_cov_matrix.py.

Usage:
    python merge_bins_to_roi.py 25
    python merge_bins_to_roi.py 25 --force   # overwrite existing

Reads:
    GCE_cov_l<ROI>_bin00_front_17yr_cholis.npz  ... bin13...

Produces:
    GCE_cov_l<ROI>_front_17yr_cholis_fit.npz
"""

import argparse
import os
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('roi', type=int)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    out_npz = f'./GCE_cov_l{args.roi}_front_17yr_cholis_fit.npz'
    if os.path.exists(out_npz) and not args.force:
        print(f'[skip] {out_npz} exists (use --force to overwrite)')
        return 0

    n_bins = 14
    n_params = 5
    fitted_params       = np.zeros((n_params, n_bins))
    fitted_params_std   = np.zeros((n_params, n_bins))
    fitted_params_median = np.zeros((n_params, n_bins))
    fitted_params_upper = np.zeros((n_params, n_bins))
    fitted_params_lower = np.zeros((n_params, n_bins))
    max_likelihood      = np.zeros(n_bins)

    missing = []
    for b in range(n_bins):
        path = f'./GCE_cov_l{args.roi}_bin{b:02d}_front_17yr_cholis.npz'
        if not os.path.exists(path):
            missing.append(b)
            continue
        d = np.load(path)
        for j in range(n_params):
            fitted_params[j, b]       = d['fitted_params'][j]
            fitted_params_std[j, b]   = d['fitted_params_std'][j]
            fitted_params_median[j, b] = d['fitted_params_median'][j]
            fitted_params_upper[j, b] = d['fitted_params_upper'][j]
            fitted_params_lower[j, b] = d['fitted_params_lower'][j]
        max_likelihood[b] = d['max_likelihood']

    if missing:
        print(f'[FAIL] missing bins for roi={args.roi}: {missing}')
        print(f'       run them with: python orchestrate_cov_bins.py {args.roi} --bins {" ".join(map(str, missing))}')
        return 1

    # Get E and delta_E from the global ccube
    from astropy.io import fits
    front = '_front'  # match CONFIG
    ccube_path = f'./GC_analysis_FL16Y/GC_ccube_17yr{front}_clean.fits'
    E_bounds = fits.open(ccube_path)[1].data
    E = np.zeros(len(E_bounds))
    delta_E = np.zeros(len(E_bounds))
    for i in range(len(E_bounds)):
        E[i] = np.sqrt(E_bounds[i][2]*E_bounds[i][1]*1e-6)*1e-3
        delta_E[i] = (E_bounds[i][2] - E_bounds[i][1])*1e-6

    np.savez(
        out_npz,
        fitted_params=fitted_params,
        fitted_params_std=fitted_params_std,
        fitted_params_median=fitted_params_median,
        fitted_params_upper=fitted_params_upper,
        fitted_params_lower=fitted_params_lower,
        max_likelihood=max_likelihood,
        E=E,
        delta_E=delta_E,
        roi=args.roi,
    )
    print(f'[OK] merged {n_bins} bins -> {out_npz}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
