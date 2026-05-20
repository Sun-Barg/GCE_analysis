#!/usr/bin/env python3
"""
make_gce_template_profiles.py  -  generalized GCE spatial-template builder
===========================================================================
Builds GC-centered, line-of-sight-integrated rho^2 templates for three DM
density profiles on the EXACT same grid / geometry / normalization as the
validated NFW^2 builder (make_gce_template.py).  The ONLY thing that differs
between profiles is the radial density rho(r).  Everything else is identical:

    - 600 x 600, 0.1 deg, GC-centered GLON/GLAT-CAR grid
    - LOS geometry  R(s) = sqrt(r_0^2 + s^2 - 2 r_0 s cos theta),  r_0 = 8.5 kpc
    - inner-pixel regularization: pixels with angle < angle(0.05, 0.05) use the
      (0.05, 0.05) LOS value          (Wimp_map cell 22)
    - 120 deg angular cutoff           (cov cell 9; never reached on a 60x60 map)
    - whole-map normalization  sum(raw) * (pi/180)^2 * (0.1)^2  ->  integral 1.0
      (the cov / Cholis convention; same as make_wimp_map_per_roi.py)

WHY THIS EXISTS  (thesis 6.3, halo-profile sensitivity)
-------------------------------------------------------
Cholis 2022 (2112.09706) tests GCE "cuspiness" only via the gNFW inner slope
gamma (Fig 13 left: 0.8 <= gamma <= 1.4, gamma=1.2-1.3 preferred, gamma<1
strongly disfavored) and via stellar morphologies (Fig 13 right: Boxy+Nuclear
Bulge, X-shaped Bulge).  It does NOT use Einasto or Burkert.  This script
extends the cuspiness test to two full physical profile families on the same
footing as the production NFW^2 template, so the only variable across the
comparison is the profile shape.

PROFILE PARAMETERS
------------------
NFW^2 : gamma = 1.2, r_s = 20 kpc                          <- Cholis (paper-exact)
Einasto: alpha = 0.17, r_s = 20 kpc                        <- MW canonical
         (Navarro+2010 Aquarius; r_s tied to NFW r_s so only the functional
          form differs, not the scale radius)
Burkert: r_0c = 9 kpc (core radius)                        <- MW (Nesti & Salucci 2013)

These Einasto / Burkert numbers are NOT from Cholis (Cholis does not use them);
they are standard Milky-Way literature values.  Edit PARAMS to vary them.

Each profile amplitude is calibrated to the local density rho(8.5 kpc) =
0.4 GeV/cm^3 (Cholis convention).  For an integral=1.0 SHAPE template the
amplitude cancels exactly; it is fixed only so the raw maps are physically
anchored if a J-factor / <sigma v> conversion is wanted later.

BUILT-IN CORRECTNESS CHECK
--------------------------
The NFW2 branch is byte-for-byte the production core: the calibrated amplitude
is rho_s = 0.2710150839697834 (verified == make_gce_template.py).  Run

    python make_gce_template_profiles.py --profile NFW2 --out /tmp/nfw2_chk.fits
    python -c "from astropy.io import fits, numpy as np; \
a=fits.getdata('/tmp/nfw2_chk.fits'); b=fits.getdata('./GCE_template_NFW2.fits'); \
print('max|diff| =', np.nanmax(np.abs(a-b)))"

max|diff| must be ~0 (raw sum 3.282807e+05, integral 1.0).  If the NFW2 branch
reproduces the validated file, the Einasto/Burkert branches are trustworthy by
construction (same machinery, only rho(r) swapped).

No fermitools dependency -> no fork-unsafe state -> zero SIGKILL risk
(same as make_gce_template.py / make_wimp_map_per_roi.py).  ~3 min / profile.

Usage
-----
    python make_gce_template_profiles.py --profile Einasto2
    python make_gce_template_profiles.py --profile Burkert2
    python make_gce_template_profiles.py --profile all
    python make_gce_template_profiles.py --profile Einasto2 --check
    python make_gce_template_profiles.py --profile NFW2 --out /tmp/nfw2_chk.fits

Author: haebarg
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

# ===========================================================================
# Grid / geometry / normalization  -- IDENTICAL to make_gce_template.py
# ===========================================================================
GRID_NX, GRID_NY = 600, 600
PIXEL_DEG        = 0.1
CUTOFF_DEG       = 120
EPS_L, EPS_B     = 0.05, 0.05
NORM_TOL         = 1e-3
R_0              = 8.5     # observer galactocentric distance [kpc]
RHO_LOCAL        = 0.4     # local DM density at R_0 [GeV/cm^3]  (Cholis)

# ===========================================================================
# Profile registry  -- the ONLY profile-dependent part.
# Shape parameters (NFW gamma/r_s are Cholis; Einasto/Burkert are MW literature)
# ===========================================================================
PARAMS = {
    'NFW2':     dict(r_s=20.0, gamma=1.2),     # Cholis gNFW
    'Einasto2': dict(r_s=20.0, alpha=0.17),    # Navarro+2010 alpha, r_s tied to NFW
    'Burkert2': dict(r_0c=9.0),                # Nesti & Salucci 2013 core radius [kpc]
}


def _shape_unnormalized(profile, r, p):
    """rho(r) up to amplitude.  r in kpc (scalar or array)."""
    if profile == 'NFW2':
        return (r / p['r_s']) ** (-p['gamma']) / (1.0 + r / p['r_s']) ** (3.0 - p['gamma'])
    if profile == 'Einasto2':
        return np.exp(-(2.0 / p['alpha']) * ((r / p['r_s']) ** p['alpha'] - 1.0))
    if profile == 'Burkert2':
        x = r / p['r_0c']
        return 1.0 / ((1.0 + x) * (1.0 + x * x))
    raise ValueError(f'unknown profile {profile!r}')


def _amplitude(profile, p):
    """rho_s (or rho_0) so that rho(R_0) = RHO_LOCAL."""
    return RHO_LOCAL / _shape_unnormalized(profile, R_0, p)


def _make_rho(profile):
    p   = PARAMS[profile]
    amp = _amplitude(profile, p)

    def rho(r):
        return amp * _shape_unnormalized(profile, r, p)

    return rho, p, amp


# ===========================================================================
# LOS integration  -- IDENTICAL machinery to make_gce_template.py,
# only the rho callable is injected.
# ===========================================================================
def _angle(l_deg, b_deg):
    """Angular distance from (0, 0) in radians; (l, b) in degrees."""
    return np.arccos(np.cos(np.radians(l_deg)) * np.cos(np.radians(b_deg)))


def _los_integral(rho, l_deg, b_deg):
    """int_0^inf rho(R(s))^2 ds  toward (l, b) from observer at R_0."""
    theta = _angle(l_deg, b_deg)

    def R(s):
        return np.sqrt(R_0 ** 2 + s ** 2 - 2 * R_0 * s * np.cos(theta))

    def rho_squared(s):
        return rho(R(s)) ** 2

    warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)
    val, _ = integrate.quad(rho_squared, 0, np.inf)
    return val


# ===========================================================================
# Template build
# ===========================================================================
def build_template(profile):
    rho, p, amp = _make_rho(profile)
    cutoff_rad = np.radians(CUTOFF_DEG)
    eps_ang    = _angle(EPS_L, EPS_B)

    nx, ny = GRID_NX, GRID_NY
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]   # 300.5, 300.5
    w.wcs.cdelt = [-PIXEL_DEG, PIXEL_DEG]
    w.wcs.crval = [0.0, 0.0]
    w.wcs.ctype = ['GLON-CAR', 'GLAT-CAR']
    hdr = w.to_header()
    hdr['NAXIS']  = 2
    hdr['NAXIS1'] = nx
    hdr['NAXIS2'] = ny

    raw = np.zeros((ny, nx), dtype=np.float64)
    t0 = time.time()
    for i in range(ny):
        js = np.arange(nx)
        l_row, b_row = w.wcs_pix2world(js, np.full(nx, i), 0)
        for j in range(nx):
            l = (l_row[j] + 180.0) % 360.0 - 180.0   # wrap to [-180, 180]
            b = b_row[j]
            a = _angle(l, b)
            if a >= cutoff_rad:
                continue
            if a < eps_ang:
                raw[i, j] = _los_integral(rho, EPS_L, EPS_B)
            else:
                raw[i, j] = _los_integral(rho, l, b)

    raw_sum = float(np.sum(raw))
    norm    = raw_sum * (np.pi / 180.0) ** 2 * (PIXEL_DEG ** 2)
    if norm <= 0 or not np.isfinite(norm):
        raise RuntimeError(f'invalid normalization {norm}')
    template = (raw / norm).astype(np.float32)
    elapsed = time.time() - t0

    # provenance
    hdr['HISTORY'] = f'GCE rho^2 template  profile={profile}'
    hdr['HISTORY'] = f'params={p}  amp(rho_s/rho_0)={amp:.10g} GeV/cm3 (rho(8.5)=0.4)'
    hdr['HISTORY'] = f'raw_sum={raw_sum:.6e}  norm_integral=1.0  (whole-map, 0.1deg pixel)'
    hdr['HISTORY'] = 'GC-centered, 600x600 0.1deg GLON/GLAT-CAR, LOS quad(0,inf)'
    hdr['HISTORY'] = 'machinery identical to make_gce_template.py (NFW2 branch verified)'
    return template, hdr, raw_sum, norm, elapsed


def _status(path):
    if not os.path.exists(path):
        return False, None, None, 'absent'
    d = fits.getdata(path)
    integ = float(np.sum(d.astype(np.float64)) * (np.pi / 180.0) ** 2 * (PIXEL_DEG ** 2))
    ok = (np.all(np.isfinite(d)) and abs(integ - 1.0) <= NORM_TOL)
    return True, tuple(d.shape), integ, ('valid' if ok else 'INVALID')


def _write_atomic(path, template, hdr):
    tmp = path + '.tmp'
    fits.PrimaryHDU(data=template, header=hdr).writeto(tmp, overwrite=True)
    os.replace(tmp, path)
    # post-write verify
    exists, shape, integ, verdict = _status(path)
    if not (exists and verdict == 'valid' and shape == (GRID_NY, GRID_NX)):
        raise RuntimeError(f'post-write verify FAILED: shape={shape} integral={integ} verdict={verdict}')
    return shape, integ


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', required=True,
                    choices=['NFW2', 'Einasto2', 'Burkert2', 'all'],
                    help="which profile template to build")
    ap.add_argument('--out', default=None,
                    help="output path override (default ./GCE_template_<profile>.fits). "
                         "Use a /tmp path for the NFW2 correctness check so the "
                         "production file is never touched.")
    ap.add_argument('--check', action='store_true',
                    help="report current file status only, do not build")
    ap.add_argument('--force', action='store_true',
                    help="rebuild even if a valid file already exists")
    args = ap.parse_args()

    profiles = ['NFW2', 'Einasto2', 'Burkert2'] if args.profile == 'all' else [args.profile]
    if args.out and len(profiles) > 1:
        sys.exit('--out cannot be combined with --profile all')

    for profile in profiles:
        out = args.out if args.out else f'./GCE_template_{profile}.fits'
        exists, shape, integ, verdict = _status(out)
        print(f'\n=== {profile}  ->  {out} ===')
        print(f'    PARAMS={PARAMS[profile]}  amp={_amplitude(profile, PARAMS[profile]):.10g} GeV/cm3')
        print(f'    current: exists={exists}  shape={shape}  '
              f'integral={integ if integ is None else f"{integ:.6f}"}  verdict={verdict}')

        if args.check:
            continue

        # never silently clobber the validated production NFW2 file
        if profile == 'NFW2' and out == './GCE_template_NFW2.fits' and not args.force:
            print('    [skip] refusing to overwrite production ./GCE_template_NFW2.fits '
                  '(use --out /tmp/... for the check, or --force to regenerate).')
            continue
        if exists and verdict == 'valid' and not args.force:
            print('    [skip] already valid (integral~1.0). Use --force to regenerate.')
            continue

        print('    [build] integrating LOS rho^2 over 600x600 grid ...')
        template, hdr, raw_sum, norm, elapsed = build_template(profile)
        shape, integ = _write_atomic(out, template, hdr)
        print(f'    [done]  raw_sum={raw_sum:.6e}  norm={norm:.6e}  '
              f'integral={integ:.6f}  shape={shape}  ({elapsed:.1f}s)')
        if profile == 'NFW2':
            print(f'    [check] raw_sum should equal 3.282807e+05 '
                  f'(cov wimp_map_l-20 / production NFW2).')


if __name__ == '__main__':
    main()
