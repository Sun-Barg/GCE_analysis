#!/usr/bin/env python3
"""build_cov_matrix.py v2 — Compute systematic cov matrix from 20 control ROIs.

CRITICAL FIX from v1:
  v1 used fitted_params[2] directly as 'GCE' — but that's just the multiplier!
  v2 uses ACTUAL flux: fitted_params[2] * GCE_template * E^2 / delta_E
  (matches main runner formula L1097)
"""

import argparse
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

FRONT_SUFFIX = '_front'
ALL_ROIS = [
    -70, -65, -60, -55, -50, -45, -40, -35, -30, -25,
     25,  30,  35,  40,  45,  50,  55,  60,  65,  70,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='./GCE_systematic_covariance_matrix_17yr.npz')
    args = ap.parse_args()

    available = []
    for roi in ALL_ROIS:
        if os.path.exists(f'./GCE_cov_l{roi}{FRONT_SUFFIX}_17yr_cholis_fit.npz'):
            available.append(roi)
    if len(available) != 20:
        print(f'Missing: {set(ALL_ROIS) - set(available)}')
        return 1
    print(f'Found all {len(available)} ROI fit files')

    sample = np.load(f'./GCE_cov_l{available[0]}{FRONT_SUFFIX}_17yr_cholis_fit.npz')
    n_bins = len(sample['E'])
    E = sample['E']
    delta_E = sample['delta_E']
    print(f'n_bins={n_bins}, E=[{E[0]:.3f}, {E[-1]:.1f}] GeV')

    # disk_mask and steradian_per_pixel computation
    disk_mask = np.load(
        './GC_analysis_FL16Y/Model/GC_disk_mask_60x60_definitions.npy'
    )[100:500, 100:500]

    front = FRONT_SUFFIX
    raw_path = f'./GC_analysis_FL16Y/GC_ccube_17yr{front}_clean.fits'
    raw = fits.open(raw_path)
    wcs_full = WCS(raw[0].header).dropaxis(2)
    width, height = np.shape(raw[0].data[0])
    spp = np.zeros([width, height])
    for i in range(height):
        for j in range(width):
            l, b = wcs_full.wcs_pix2world(j, i, 0)
            cosb = np.cos(np.radians(b))
            spp[i, j] = np.radians(0.1) * np.radians(0.1) * cosb
    raw.close()
    steradian_per_pixel = spp[100:500, 100:500]
    print('Computed steradian_per_pixel array')

    GCE_flux_per_roi = np.zeros((len(available), n_bins))
    for idx, roi in enumerate(available):
        d = np.load(f'./GCE_cov_l{roi}{front}_17yr_cholis_fit.npz')
        gce_param = d['fitted_params'][2]

        exp_path = f'./GC_analysis_FL16Y/GC_expcube_center_17yr{front}_clean_l{roi}.fits'
        exp_full = (fits.open(exp_path)[0].data[:, 100:500, 100:500]
                    * steradian_per_pixel)

        # GCE FITS path: try per-ROI first, fall back to global
        gce_path = f'./GC_analysis_FL16Y/GC_GCE_model_17yr{front}_clean_l{roi}.fits'
        if not os.path.exists(gce_path):
            gce_path = f'./GC_analysis_FL16Y/GC_GCE_model_17yr{front}_clean.fits'
        gce_data = fits.open(gce_path)[0].data[:, 100:500, 100:500]

        GCE_template = np.zeros(n_bins)
        norm = np.sum(disk_mask)
        for b in range(n_bins):
            GCE_template[b] = np.sum(disk_mask * gce_data[b] / exp_full[b]) / norm

        GCE_flux_per_roi[idx] = gce_param * GCE_template * (E**2) / delta_E
        peak = np.max(np.abs(GCE_flux_per_roi[idx]))
        print(f'  roi={roi:+4d}  peak |flux|={peak:.3e} GeV/cm^2/s/sr')

    mean_flux = np.mean(GCE_flux_per_roi, axis=0)
    cov_matrix = np.cov(GCE_flux_per_roi.T)

    print(f'\nDiagonal sigma per bin (sqrt of diag cov):')
    print(f'  E[GeV]    sigma_flux [GeV/cm^2/s/sr]')
    for i, e in enumerate(E):
        print(f'  {e:7.3f}   {np.sqrt(cov_matrix[i,i]):.3e}')

    np.savez(args.out,
             cov_matrix=cov_matrix,
             mean_GCE=mean_flux,
             rois=np.array(available),
             GCE_flux_per_roi=GCE_flux_per_roi,
             E=E, delta_E=delta_E)
    print(f'\n[saved] {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
