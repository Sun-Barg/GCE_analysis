#!/usr/bin/env python3
"""
make_wimp_map_per_roi.py — generate per-ROI NFW^2 line-of-sight integral
templates required by the covariance pipeline.

Each ROI has its spatial template centered on (l=roi, b=0). The NFW^2
integration is identical to cov-notebook cell 6's `wimp_annihilation_map`;
the only addition here is the per-ROI longitude shift and the fits write.

The cov notebook itself (cell 16/17/19/20) references
`./GC_analysis_FL16Y/Model/wimp_map_l{roi}.fits` but never generates these
files — that responsibility lived in an external .py that has been cleaned
up. This script restores it.

Usage:
    python make_wimp_map_per_roi.py [--workers N] [--rois ROI1,ROI2,...]

Output:
    ./GC_analysis_FL16Y/Model/wimp_map_l{ROI}.fits   (one per ROI)

The output is a 2D fits image (600×600 @ 0.1 deg/pixel), normalized so
that the integral over the ROI window equals 1 (same convention as
cov-notebook cell 9). The WCS in the header is copied from the per-ROI
ccube file (`GC_ccube_17yr_front_clean_l{ROI}.fits`).

Approximate runtime: ~3-8 min per ROI on a single core (sparse cutoff
at angle > 120 deg from ROI center). With --workers 20 the whole job
finishes in roughly the same wall time as one ROI.

Author: haebarg (2026)

Changes:
  [fb17-cov-v1] (2026-07-28) FB17=1 env -> front+back 17-bin cov variant
      (main pipeline과 동일 패턴): WORK_DIR/FRONT/evtype 전환, 결과는
      results_cov_fb17/ 분리(기존 front 22개 .dat와 카운트 충돌 방지).
      env 미설정 시 기존 fiducial 동작과 동일.
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import integrate
from multiprocessing import Pool

# ============================================================
# CONFIG
# ============================================================
WORK_DIR = './GC_analysis_FL16Y'
FRONT    = '_front'

# FB17 variant — env switch (prepare_common.py 동일 패턴).
FB17 = bool(os.environ.get('FB17', '').strip())
if FB17:
    WORK_DIR = './GC_analysis_FL16Y_fb17'
    FRONT    = '_front_back'
    print(f'[config] FB17=1 -> WORK_DIR={WORK_DIR}, FRONT={FRONT!r}', flush=True)

# Cov pipeline control ROIs (cov notebook: range(-70, 75, 5), skip |roi|<20 and roi=0)
ALL_ROIS = [r for r in range(-70, 75, 5) if r != 0 and abs(r) >= 20]
assert len(ALL_ROIS) == 22

# Pixel-skip cutoff: do not evaluate NFW^2 LOS for pixels farther than this
# angular distance from ROI center. Matches cov notebook cell 9's 120 deg
# (chosen conservatively; NFW^2 is essentially zero beyond ~60 deg anyway).
CUTOFF_DEG = 120


# ============================================================
# NFW^2 LOS integration (verbatim from cov notebook cell 6)
# ============================================================

def _angle(l_deg, b_deg):
    """Angular distance from (0, 0) in radians, with (l, b) in degrees.
    Matches cov-notebook cell 6 `angle` exactly."""
    return np.arccos(np.cos(np.radians(l_deg)) * np.cos(np.radians(b_deg)))


def _wimp_annihilation_map(l_deg, b_deg):
    """LOS integral of NFW^2 at direction (l, b) from observer at r_0 = 8.5 kpc.
    Identical parameters to cov notebook cell 6:
        rho_s = 0.2710150839697834 GeV/cm^3   (calibrated to local 0.4 GeV/cm^3)
        r_s   = 20 kpc
        gamma = 1.2
    Returns (integral, error_estimate) from scipy.integrate.quad."""
    rho_s = 0.2710150839697834
    r_s   = 20.0
    gamma = 1.2
    r_0   = 8.5
    theta = _angle(l_deg, b_deg)

    def NFW(r):
        return rho_s * ((r / r_s) ** -gamma) / ((1 + r / r_s) ** (3 - gamma))

    def R(s):
        return np.sqrt(r_0 ** 2 + s ** 2 - 2 * r_0 * s * np.cos(theta))

    def rho_squared(s):
        return NFW(R(s)) ** 2

    warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)
    return integrate.quad(rho_squared, 0, np.inf)


def _delta_l(l_pix_deg, roi_deg):
    """Longitude difference l_pix - roi wrapped into [-180, 180]."""
    d = (l_pix_deg - roi_deg + 180.0) % 360.0 - 180.0
    return d


# ============================================================
# Per-ROI worker
# ============================================================

def make_one_roi(roi):
    out_path = f'{WORK_DIR}/Model/wimp_map_l{roi}.fits'
    if os.path.exists(out_path):
        return ('skip', roi, out_path, 0.0)

    ref_path = f'{WORK_DIR}/GC_ccube_17yr{FRONT}_clean_l{roi}.fits'
    if not os.path.exists(ref_path):
        return ('error', roi, f'missing reference {ref_path}', 0.0)

    with fits.open(ref_path) as h:
        wcs2d = WCS(h[0].header).dropaxis(2)
        # 3D ccube: shape (n_energy, ny, nx); we want (ny, nx)
        ny, nx = h[0].data.shape[-2:]
        # Build a clean 2D header from WCS (drops spectral keywords)
        hdr2d = wcs2d.to_header()
        hdr2d['NAXIS']  = 2
        hdr2d['NAXIS1'] = nx
        hdr2d['NAXIS2'] = ny

    wimp_map = np.zeros((ny, nx), dtype=np.float64)
    cutoff_rad = np.radians(CUTOFF_DEG)
    eps_l, eps_b = 0.05, 0.05   # cov cell 9 inner-pixel regularization

    t0 = time.time()
    for i in range(ny):
        # Compute (l, b) for the whole row at once
        # (still loop in j for the integral call, but world coord lookup is vectorized)
        js = np.arange(nx)
        l_row, b_row = wcs2d.wcs_pix2world(js, np.full(nx, i), 0)
        for j in range(nx):
            dl = _delta_l(l_row[j], roi)
            db = b_row[j]
            if _angle(dl, db) >= cutoff_rad:
                continue
            if _angle(dl, db) < _angle(eps_l, eps_b):
                wimp_map[i, j] = _wimp_annihilation_map(eps_l, eps_b)[0]
            else:
                wimp_map[i, j] = _wimp_annihilation_map(dl, db)[0]

    # Normalize so that integral over pixels = 1
    # (cov cell 9: norm = sum * (pi/180)**2 * pixel_size_deg**2)
    norm = np.sum(wimp_map) * (np.pi / 180.0) ** 2 * (0.1) ** 2
    if norm <= 0 or not np.isfinite(norm):
        return ('error', roi, f'invalid normalization {norm}', time.time() - t0)
    wimp_map = wimp_map / norm

    fits.writeto(out_path, wimp_map.astype(np.float32), header=hdr2d, overwrite=True)
    return ('done', roi, out_path, time.time() - t0)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8,
                    help='parallel ROI count (default: 8). NFW^2 quad is CPU-bound and IO-light.')
    ap.add_argument('--rois', type=str, default='',
                    help='comma-separated ROI subset (default: all 22)')
    args = ap.parse_args()

    if args.rois.strip():
        rois = [int(r.strip()) for r in args.rois.split(',') if r.strip()]
        bad = [r for r in rois if r not in ALL_ROIS]
        if bad:
            print(f'[FATAL] invalid ROI(s): {bad}')
            print(f'        valid: {ALL_ROIS}')
            sys.exit(2)
    else:
        rois = ALL_ROIS[:]

    print(f'[start] generating wimp_map for {len(rois)} ROIs, workers={args.workers}')
    print(f'        cutoff angle = {CUTOFF_DEG} deg (pixels beyond → 0)')
    t_total = time.time()

    if args.workers <= 1:
        results = [make_one_roi(roi) for roi in rois]
    else:
        with Pool(processes=args.workers) as pool:
            results = pool.map(make_one_roi, rois)

    n_done = sum(1 for r in results if r[0] == 'done')
    n_skip = sum(1 for r in results if r[0] == 'skip')
    n_err  = sum(1 for r in results if r[0] == 'error')
    for status, roi, info, dt in results:
        tag = {'done': '[done]', 'skip': '[skip]', 'error': '[FAIL]'}[status]
        if status == 'error':
            print(f'  {tag} roi={roi:+4d}  {info}')
        else:
            print(f'  {tag} roi={roi:+4d}  {dt/60:5.1f} min  {info}')

    print(f'[done] total {(time.time()-t_total)/60:.1f} min  '
          f'done={n_done} skip={n_skip} error={n_err}')
    if n_err:
        sys.exit(1)


if __name__ == '__main__':
    main()
