#!/usr/bin/env python3
"""
cholis_masking.py — shared utilities for the 17yr GCE pipeline.

Provides:
  - masking(): per-energy-bin circular point-source mask (Cholis 2022 Table III)
  - equatorial_to_galactic, galactic_to_equatorial: coordinate transforms
  - integrity check helpers (8 verify_* functions, all returning (ok, msg))

Imported by:
  - prepare_common.py      (data prep stage 1, common; 1x per workdir)
  - prepare_one_roi_cov.py (data prep stage 2, per-ROI; 22x via launcher)
  Workers (run_one_model.py, run_one_roi_cov.py) are NOT modified; they
  continue to use their inline masking definitions. Sharing this module
  is forward-looking: if the workers are ever refactored, they can drop
  the duplicate code and import from here.

Verifier contract: (ok: bool, msg: str). Caller decides:
  - ok=True  -> safe to skip rebuild
  - ok=False -> abort with explicit message (NO silent stale-file reuse)

Author: haebarg (2026)
"""

import os
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.interpolate import interp1d


# ============================================================
# Cholis 2022 (arXiv:2112.09706) Table III — point-source mask radii
#   col 0,1 : bin edges in GeV
#   col 2   : small-mask radius (deg)  — used when Signif_Avg <= 49
#   col 3   : large-mask radius (deg)  — used when Signif_Avg >  49
# ============================================================
CHOLIS_TABLE_III = np.array([
    [0.275, 0.357, 1.125, 3.75 ],
    [0.357, 0.464, 0.975, 3.25 ],
    [0.464, 0.603, 0.788, 2.63 ],
    [0.603, 0.784, 0.600, 2.00 ],
    [0.784, 1.02,  0.450, 1.50 ],
    [1.02,  1.32,  0.375, 1.25 ],
    [1.32,  1.72,  0.300, 1.00 ],
    [1.72,  2.24,  0.225, 0.750],
    [2.24,  2.91,  0.188, 0.625],
    [2.91,  3.78,  0.162, 0.540],
    [3.78,  4.91,  0.125, 0.417],
    [4.91, 10.8,   0.100, 0.333],
    [10.8, 23.7,   0.060, 0.200],
    [23.7, 51.9,   0.053, 0.175],
])

EXPECTED_NEBINS = 14
SEC_PER_YR = 365.25 * 86400.0


# ============================================================
# Coordinate transforms (cov cell 12 verbatim)
# ============================================================

def equatorial_to_galactic(ra, dec):
    """ICRS (RA, Dec) [deg] -> Galactic (l, b) [deg]."""
    c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs').galactic
    return c.l.degree, c.b.degree


def galactic_to_equatorial(l, b):
    """Galactic (l, b) [deg] -> ICRS (RA, Dec) [deg]."""
    c = SkyCoord(l=l * u.degree, b=b * u.degree, frame='galactic').icrs
    return c.ra.degree, c.dec.degree


# ============================================================
# Per-bin circular point-source mask
# (unifies main cell 21 + cov cell 12; mask_scale=1.0 default)
# ============================================================

def masking(significance, locations, energy, image_file, mask_scale=1.0):
    """
    Per-energy-bin circular mask for catalog point sources.

    Args:
        significance : 0 -> small radius (Table III col 2),
                       1 -> large radius (Table III col 3).
                       Selection follows main cell 23 / cov cell 13:
                       Signif_Avg > 49  uses significance=1.
        locations    : list of [name, ra_deg, dec_deg] entries.
        energy       : scalar bin energy in GeV (geometric mean of bin edges).
        image_file   : path to a FITS file whose [0] HDU has a 3D CCUBE
                       (data[0] supplies (ny, nx); header carries the WCS).
        mask_scale   : multiplier on radii. Default 1.0 = Cholis Table III
                       strict. Values != 1.0 have no physical justification
                       and should not be used outside of artificial tests.

    Returns:
        masked : 2D float32 ndarray (ny, nx). 1.0 = unmasked,
                 0.0 = inside any source's circular mask at this energy.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="'datfix' made the change")

    # Interpolate radii at the requested bin energy
    mask_energy = np.sqrt(CHOLIS_TABLE_III[:, 0] * CHOLIS_TABLE_III[:, 1])
    radii_small = CHOLIS_TABLE_III[:, 2] * mask_scale
    radii_large = CHOLIS_TABLE_III[:, 3] * mask_scale
    f_small = interp1d(mask_energy, radii_small, fill_value='extrapolate')
    f_large = interp1d(mask_energy, radii_large, fill_value='extrapolate')

    r_small = max(float(f_small(energy)), 0.0)
    r_large = max(float(f_large(energy)), 0.0)

    if significance == 1:
        radius_deg = r_large
    elif significance == 0:
        radius_deg = r_small
    else:
        raise ValueError(f"significance must be 0 or 1, got {significance!r}")

    # WCS from the CCUBE header
    with fits.open(image_file) as hdul:
        data2d = hdul[0].data[0]                # first energy slice (ny, nx)
        header = hdul[0].header.copy()
    wcs = WCS(header).dropaxis(2)
    pix_dx = wcs.wcs.cdelt[0]
    pix_dy = wcs.wcs.cdelt[1]

    masked = np.ones(data2d.shape, dtype=np.float32)
    y_idx, x_idx = np.ogrid[:data2d.shape[0], :data2d.shape[1]]

    for entry in locations:
        ra  = float(entry[1])
        dec = float(entry[2])
        l_deg, b_deg = equatorial_to_galactic(ra, dec)
        gal = SkyCoord(l=l_deg * u.degree, b=b_deg * u.degree, frame='galactic')
        px = wcs.world_to_pixel(gal)
        x_c = float(np.round(px[0], 0))
        y_c = float(np.round(px[1], 0))

        rpx = radius_deg / pix_dx
        rpy = radius_deg / pix_dy
        radius_pix = min(abs(rpx), abs(rpy))

        circle = (x_idx - x_c) ** 2 + (y_idx - y_c) ** 2 < radius_pix ** 2
        masked[circle] = 0.0

    return masked


# ============================================================
# Integrity-check helpers
# ============================================================

def _file_basic(path):
    if not os.path.exists(path):
        return False, f'missing: {path}'
    sz = os.path.getsize(path)
    if sz == 0:
        return False, f'zero size: {path}'
    return True, f'size={sz/1e6:.1f}MB'


def verify_fits(path):
    """Generic: file exists + opens + fits.verify('exception') passes."""
    ok, msg = _file_basic(path)
    if not ok:
        return False, msg
    try:
        with fits.open(path, memmap=True) as hdul:
            hdul.verify('exception')
    except Exception as e:
        return False, f'open/verify failed: {type(e).__name__}: {str(e)[:120]}'
    return True, msg


def verify_cube(path, expected_nebins=EXPECTED_NEBINS, expected_xy=None,
                allow_nebins_plus_one=False):
    """3D cube (CCUBE / EXPCUBE / gtmodel): shape + finite + nonzero + nonneg."""
    ok, msg = verify_fits(path)
    if not ok:
        return False, msg
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        if data is None or data.ndim != 3:
            return False, f'primary not 3D: shape={None if data is None else data.shape}'
        ne, ny, nx = data.shape
        shape_tuple = data.shape           # capture before context exit
        valid = {expected_nebins}
        if allow_nebins_plus_one:
            valid.add(expected_nebins + 1)
        if ne not in valid:
            return False, f'nebins={ne}, expected one of {sorted(valid)}'
        if expected_xy is not None and (nx, ny) != expected_xy:
            return False, f'(nx,ny)=({nx},{ny}), expected {expected_xy}'
        s = float(np.nansum(data))
        if not np.isfinite(s):
            return False, 'NaN or Inf in data'
        if s == 0:
            return False, 'all zeros'
        mn = float(np.nanmin(data))
        if mn < -1e-6:
            return False, f'has negative values, min={mn:.3g}'
    return True, f'{msg}, shape={shape_tuple}, sum={s:.3e}'


def verify_event_file(path, min_events=1_000_000):
    """gtselect / gtmktime output: EVENTS HDU + row count + GTI exists."""
    ok, msg = verify_fits(path)
    if not ok:
        return False, msg
    with fits.open(path, memmap=True) as hdul:
        names = [h.name for h in hdul]
        if 'EVENTS' not in names:
            return False, 'missing EVENTS HDU'
        n_ev = len(hdul['EVENTS'].data)
        if n_ev < min_events:
            return False, f'EVENTS={n_ev:,} (<{min_events:,} minimum)'
        if 'GTI' not in names:
            return False, 'missing GTI HDU'
    return True, f'{msg}, EVENTS={n_ev:,}'


def verify_sc_merged(path, min_rows=1_000_000, expected_tspan_yr=17.0,
                     tspan_tol_yr=2.0):
    """Merged SC FT2: SC_DATA + row count + time-system keys + 17yr span."""
    ok, msg = verify_fits(path)
    if not ok:
        return False, msg
    with fits.open(path, memmap=True) as hdul:
        names = [h.name for h in hdul]
        if 'SC_DATA' not in names:
            return False, 'missing SC_DATA HDU'
        sc = hdul['SC_DATA']
        n = len(sc.data)
        if n < min_rows:
            return False, f'SC_DATA={n:,} (<{min_rows:,} minimum)'
        for key in ('TIMESYS', 'MJDREFI', 'MJDREFF'):
            if key not in sc.header:
                return False, f'SC_DATA missing key {key}'
        t0 = sc.header.get('TSTART')
        t1 = sc.header.get('TSTOP')
        if t0 is not None and t1 is not None:
            span_yr = (t1 - t0) / SEC_PER_YR
            if not (expected_tspan_yr - tspan_tol_yr < span_yr <
                    expected_tspan_yr + tspan_tol_yr):
                return False, f'span={span_yr:.2f} yr (expected ~{expected_tspan_yr})'
    return True, f'{msg}, SC_DATA={n:,} rows'


def verify_ltcube(path):
    """Livetime cube: EXPOSURE HDU + COSBINS sum > 0."""
    ok, msg = verify_fits(path)
    if not ok:
        return False, msg
    with fits.open(path, memmap=True) as hdul:
        if 'EXPOSURE' not in [h.name for h in hdul]:
            return False, 'missing EXPOSURE HDU'
        try:
            cb = hdul['EXPOSURE'].data['COSBINS']
            s = float(np.nansum(cb))
        except Exception as e:
            return False, f'COSBINS read failed: {e}'
        if not np.isfinite(s) or s <= 0:
            return False, f'COSBINS sum={s}'
    return True, f'{msg}, COSBINS sum={s:.3e}'


def verify_xml(path, min_sources=0):
    """XML model: parses + has at least min_sources <source> elements."""
    ok, msg = _file_basic(path)
    if not ok:
        return False, msg
    try:
        tree = ET.parse(path)
        n = len(tree.getroot().findall('.//source'))
    except Exception as e:
        return False, f'parse failed: {e}'
    if n < min_sources:
        return False, f'only {n} sources (<{min_sources})'
    return True, f'{msg}, sources={n}'


def verify_mask_npy(path, expected_shape):
    """Mask npy: shape + finite + not-all-zero (would mean fully masked)."""
    ok, msg = _file_basic(path)
    if not ok:
        return False, msg
    try:
        a = np.load(path, mmap_mode='r')
    except Exception as e:
        return False, f'load failed: {e}'
    if a.shape != expected_shape:
        return False, f'shape={a.shape}, expected {expected_shape}'
    s = float(np.sum(a))
    if not np.isfinite(s):
        return False, 'NaN or Inf in mask'
    if s == 0:
        return False, 'all zeros (fully masked — pipeline would crash)'
    return True, f'{msg}, shape={a.shape}, unmasked_pix={int(s)}'


def verify_bin_def(path, expected_nebins=EXPECTED_NEBINS):
    """bin_definitions.fits: ENERGYBINS HDU with expected_nebins rows."""
    ok, msg = verify_fits(path)
    if not ok:
        return False, msg
    with fits.open(path) as hdul:
        bd = None
        for hdu in hdul[1:]:
            if hasattr(hdu, 'columns') and any(
                    c in hdu.columns.names for c in ('E_MIN', 'EMIN')):
                bd = hdu
                break
        if bd is None:
            return False, 'no bin-definitions HDU found'
        # Capture n_bins inside the context — astropy BinTableHDU.data is
        # lazy-loaded; accessing it after the file closes raises ValueError.
        n_bins = len(bd.data)
        if n_bins != expected_nebins:
            return False, f'n_bins={n_bins}, expected {expected_nebins}'
    return True, f'{msg}, n_bins={n_bins}'


def verify_dat(path, expected_nbins=EXPECTED_NEBINS):
    """run_one_model.py final .dat: 14 rows × 5 columns, all finite.

    Schema (np.savetxt with columns from run_one_model.py Step 10):
        col 0: E (GeV)
        col 1: best-fit GCE flux (E^2 dN/dE, units: GeV/cm^2/s/sr)
        col 2: std of GCE flux
        col 3: lower 16-percentile of GCE flux
        col 4: upper 84-percentile of GCE flux

    A partial-write (e.g. SIGKILL mid-savetxt) typically yields a file
    with fewer than expected_nbins rows or with NaN/Inf, both of which
    this verifier catches.
    """
    ok, msg = _file_basic(path)
    if not ok:
        return False, msg
    try:
        arr = np.loadtxt(path)
    except Exception as e:
        return False, f'np.loadtxt failed: {type(e).__name__}: {e}'
    if arr.ndim != 2:
        return False, f'ndim={arr.ndim}, expected 2'
    if arr.shape != (expected_nbins, 5):
        return False, f'shape={arr.shape}, expected ({expected_nbins}, 5)'
    if not np.all(np.isfinite(arr)):
        n_bad = int((~np.isfinite(arr)).sum())
        return False, f'{n_bad} non-finite (NaN/Inf) entries'
    return True, f'{msg}, shape={arr.shape}'


def verify_srcmap(path):
    """gtsrcmaps output verifier.

    Extends verify_fits with the 12yr Fix 5 NDSKEYS-header check.
    gtsrcmaps writes the NDSKEYS keyword as its very last step, so
    its absence in a sized .fits file is the canonical signature of
    a mid-run kill (OOM / SIGKILL / etc) that left a partial output.
    Without this guard the next launch's skip-check would treat the
    partial file as healthy, and gtmodel would later fail with
    `Cannot read keyword "NDSKEYS"`.

    Ref: REF_12yr_final_code_for_17yr_SUMMARY.md, Fix 5.
    """
    ok, msg = verify_fits(path)
    if not ok:
        return False, msg
    try:
        with fits.open(path, memmap=True) as hdul:
            if 'NDSKEYS' not in hdul[0].header:
                return False, 'NDSKEYS missing (partial gtsrcmaps output)'
    except Exception as e:
        return False, f'NDSKEYS check failed: {type(e).__name__}: {str(e)[:120]}'
    return True, msg
