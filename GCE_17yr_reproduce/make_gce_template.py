#!/usr/bin/env python3
"""
make_gce_template.py — main-pipeline GCE spatial template generator.

Produces ./GCE_template_NFW2.fits — the GC-centered NFW^2 line-of-sight
template that the main 80-model pipeline references via WIMP_MAP_PATH
(prepare_common.py L114, run_one_model.py L103) as the GCE SpatialMap.

WHY THIS SCRIPT EXISTS
----------------------
The main pipeline references GCE_template_NFW2.fits but NO code in the
pipeline generated it (verified 2026-05-18: zero writeto/np.save for this
file in any .py or .ipynb). It was made by a now-lost port of the 12yr
"Wimp_map_creation.ipynb". That notebook's normalization line, copied
verbatim into every generating cell, is:

    norm = np.sum(counts_map[0]) * (np.pi/180)**2 * (0.01)**2
    duplicate[0].data = counts_map / norm

This is a BUG, confirmed 2026-05-18:
  (1) `counts_map[0]` is the 0th ROW (top edge, |b| large) of the 2D map,
      where NFW^2 ~ 0. The intent was `np.sum(counts_map)` (whole map).
      Dividing by a single near-zero edge row inflates amplitude wildly.
  (2) `(0.01)**2` is the 0.01-deg highresol pixel solid angle, but the
      delivered GCE_template_NFW2.fits has CDELT 0.1 deg, so the pixel
      term must be `(0.1)**2`.
The delivered (buggy) file has normalization integral
    sum * (pi/180)^2 * 0.01^2 = 0.725858   (should be 1.0)

The correct normalization — matching the cov pipeline's
make_wimp_map_per_roi.py (cov-notebook cell 9, Cholis 2022-correct,
already validated, integral = 1.0) — is:

    norm = np.sum(counts_map) * (np.pi/180)**2 * (0.1)**2
    template = counts_map / norm                      # integral = 1.0

Cross-check (2026-05-18): regenerating the GC-centered template with the
cov NFW^2 core + this correct normalization gives sum 3.282806e+05,
identical to cov wimp_map_l-20.fits sum 3.282807e+05 (agree to ~3e-7).
So the main template and the cov templates are the SAME NFW^2 profile,
differing only by translation (cov shifts to ROI longitude; main stays
GC-centered, per Cholis 2022 L1477 "GCE is the only template translated"
— the main analysis keeps GCE at the GC).

NFW^2 CORE
----------
Verbatim from make_wimp_map_per_roi.py L64-93 (which is itself verbatim
from cov-notebook cell 6). Identical parameters:
    rho_s = 0.2710150839697834 GeV/cm^3  (local 0.4 GeV/cm^3 calibration)
    r_s   = 20 kpc,  gamma = 1.2,  r_0 = 8.5 kpc
    inner-pixel regularization at angle(0.05, 0.05) (Wimp_map cell 22)
    cutoff 120 deg (cov cell 9)

CONSEQUENCE
-----------
The existing 80-model results were produced with the BUGGY template and
are therefore invalid (this is the likely root of the Model X rank
1->27 regression). After regenerating the template with this script,
the full 80-model main fit must be rerun. cov (22 ROI) already used the
correct normalization via make_wimp_map_per_roi.py, so cov itself is
unaffected, but the main<->cov relative scale must be re-checked once
the main fit is rerun.

SAFETY
------
- idempotent: if a VALID (integral≈1.0) GCE_template_NFW2.fits already
  exists, skip unless --force.
- the existing BUGGY file is never silently overwritten: it is moved to
  GCE_template_NFW2.fits.buggy_bak_<timestamp> before writing the fixed
  one (so the old 80-model provenance is preserved).
- verify after write (shape, finite, integral≈1.0).

Usage:
    python make_gce_template.py            # generate if missing/buggy
    python make_gce_template.py --force    # always regenerate
    python make_gce_template.py --check    # only report current file status

Author: haebarg · 2026-05-18
"""

import os
import sys
import time
import shutil
import argparse
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import integrate

# ============================================================
# CONFIG
# ============================================================
OUT_PATH    = './GCE_template_NFW2.fits'
GRID_NX     = 600
GRID_NY     = 600
PIXEL_DEG   = 0.1            # CDELT magnitude (matches delivered file)
CUTOFF_DEG  = 120            # cov cell 9
EPS_L, EPS_B = 0.05, 0.05    # Wimp_map cell 22 inner regularization
NORM_TOL    = 1e-3           # |integral - 1.0| tolerance for "valid"


# ============================================================
# NFW^2 LOS core — verbatim from make_wimp_map_per_roi.py L64-93
# (itself verbatim from cov-notebook cell 6). Do NOT alter.
# ============================================================
def _angle(l_deg, b_deg):
    """Angular distance from (0,0) in radians; (l,b) in degrees."""
    return np.arccos(np.cos(np.radians(l_deg)) * np.cos(np.radians(b_deg)))


def _wimp_annihilation_map(l_deg, b_deg):
    """LOS integral of NFW^2 toward (l,b) from observer at r_0=8.5 kpc.
    Identical parameters to cov notebook cell 6 / make_wimp_map_per_roi.py."""
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


# ============================================================
# helpers
# ============================================================
def _norm_integral(arr):
    """The cov / Cholis normalization integral:
       sum * (pi/180)^2 * PIXEL_DEG^2.  Should be 1.0 for a correctly
       normalized template."""
    return float(np.sum(arr)) * (np.pi / 180.0) ** 2 * (PIXEL_DEG ** 2)


def _status(path):
    """Return (exists, shape, integral, verdict_str)."""
    if not os.path.exists(path):
        return (False, None, None, 'missing')
    d = fits.open(path)[0].data
    integ = _norm_integral(d)
    if abs(integ - 1.0) <= NORM_TOL:
        verdict = 'valid (integral≈1.0)'
    else:
        verdict = f'BUGGY/stale (integral={integ:.6f}, expected 1.0)'
    return (True, d.shape, integ, verdict)


# ============================================================
# generation
# ============================================================
def build_template():
    """GC-centered NFW^2 template, correctly normalized (integral=1.0)."""
    cutoff_rad = np.radians(CUTOFF_DEG)
    eps_ang    = _angle(EPS_L, EPS_B)

    # WCS: reuse the delivered file's header if present (preserves exact
    # CRPIX/CRVAL/CD), else build a clean GC-centered CAR grid identical
    # to it (CRVAL 0,0 ; CRPIX (N+1)/2+0.5 ; CDELT -0.1/0.1).
    if os.path.exists(OUT_PATH):
        hdr = fits.open(OUT_PATH)[0].header.copy()
        w   = WCS(hdr)
        ny, nx = fits.open(OUT_PATH)[0].data.shape
    else:
        nx, ny = GRID_NX, GRID_NY
        w = WCS(naxis=2)
        w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
        w.wcs.cdelt = [-PIXEL_DEG, PIXEL_DEG]
        w.wcs.crval = [0.0, 0.0]
        w.wcs.ctype = ['GLON-CAR', 'GLAT-CAR']
        hdr = w.to_header()
        hdr['NAXIS'] = 2
        hdr['NAXIS1'] = nx
        hdr['NAXIS2'] = ny

    raw = np.zeros((ny, nx), dtype=np.float64)
    t0 = time.time()
    for i in range(ny):
        js = np.arange(nx)
        l_row, b_row = w.wcs_pix2world(js, np.full(nx, i), 0)
        for j in range(nx):
            # GC-centered: no ROI translation. wrap longitude to [-180,180]
            l = (l_row[j] + 180.0) % 360.0 - 180.0
            b = b_row[j]
            a = _angle(l, b)
            if a >= cutoff_rad:
                continue
            if a < eps_ang:
                raw[i, j] = _wimp_annihilation_map(EPS_L, EPS_B)[0]
            else:
                raw[i, j] = _wimp_annihilation_map(l, b)[0]

    # CORRECT normalization (whole-map sum, 0.1-deg pixel) → integral 1.0.
    # This is the cov / Cholis convention; replaces the buggy
    # `np.sum(counts_map[0]) * (pi/180)^2 * 0.01^2`.
    norm = np.sum(raw) * (np.pi / 180.0) ** 2 * (PIXEL_DEG ** 2)
    if norm <= 0 or not np.isfinite(norm):
        raise RuntimeError(f'invalid normalization {norm}')
    template = (raw / norm).astype(np.float32)

    elapsed = time.time() - t0
    return template, hdr, elapsed


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true',
                    help='regenerate even if a valid template exists')
    ap.add_argument('--check', action='store_true',
                    help='only report current file status, do not generate')
    args = ap.parse_args()

    exists, shape, integ, verdict = _status(OUT_PATH)
    print(f'[status] {OUT_PATH}')
    print(f'         exists={exists}  shape={shape}  '
          f'norm_integral={integ if integ is None else f"{integ:.6f}"}')
    print(f'         verdict: {verdict}')

    if args.check:
        return

    if exists and verdict.startswith('valid') and not args.force:
        print('[skip] already valid (integral≈1.0). Use --force to regenerate.')
        return

    # Preserve the buggy/old file (80-model provenance) before overwriting.
    if exists:
        ts  = time.strftime('%Y%m%d_%H%M%S')
        bak = f'{OUT_PATH}.buggy_bak_{ts}'
        shutil.copy2(OUT_PATH, bak)
        print(f'[backup] old file -> {bak}')

    print('[build] generating GC-centered NFW^2 template '
          f'({GRID_NX}x{GRID_NY}, {PIXEL_DEG} deg/pix, correct norm)...')
    template, hdr, elapsed = build_template()

    fits.writeto(OUT_PATH, template, header=hdr, overwrite=True)
    print(f'[done]  written in {elapsed/60:.1f} min')

    # verify
    ok_exists, ok_shape, ok_integ, ok_verdict = _status(OUT_PATH)
    print(f'[verify] shape={ok_shape}  norm_integral={ok_integ:.6f}  '
          f'-> {ok_verdict}')
    d = fits.open(OUT_PATH)[0].data
    print(f'         sum={d.sum():.6e}  max={d.max():.4e}  '
          f'center[{d.shape[0]//2},{d.shape[1]//2}]={d[d.shape[0]//2, d.shape[1]//2]:.4e}  '
          f'finite={np.isfinite(d).all()}')
    if not ok_verdict.startswith('valid'):
        print('[FATAL] post-write verification FAILED — do not use this file.')
        sys.exit(1)
    print('[ok] template valid. NOTE: existing 80-model results were made '
          'with the OLD buggy template and must be rerun.')


if __name__ == '__main__':
    main()
