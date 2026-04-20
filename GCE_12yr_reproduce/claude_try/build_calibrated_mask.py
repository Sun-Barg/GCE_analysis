"""
Calibrated PSC mask builder

Goal: produce a 14-bin (energy) point-source mask whose per-bin masked fraction
matches Cholis+2022 Table III to within +/- 0.5 percentage points.

Strategy
--------
1. Load 4FGL-DR2 catalog (gll_psc_v23.fit).
2. Restrict to ROI |l|, |b| <= 30 deg (60x60 deg analysis window per paper).
3. Classify sources:
     - bright = TS_Value > 49 strict (paper criterion).
       If TS_Value missing or wildly off: top-25 by Signif_Avg per v9.8 fallback.
     - small  = remaining PSC.
     - extended sources are forced to bright (theta_l) in all bins.
4. For each energy bin (Table III: theta_s, theta_l, target_frac):
     - Draw circular masks with radii (theta_s * scale, theta_l * scale)
       on a full 600x600 grid.
     - Apply |b| < 2 deg disk mask.
     - Slice to 400x400 ROI used in the analysis.
     - Bisection on `scale` to drive achieved fraction to target_frac.
5. Save 14x400x400 mask + per-bin scale metadata.

Pixel coordinate convention: CDELT1 = -0.1 (l decreases with x), so
    px = (-l) / 0.1 + crpix - 1.
This matches Sanghwan / Cholis convention; verified by D1 overlay.
"""
import os, sys, numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR, ROI_SLICE

CATALOG_FILE = '/home/haebarg/GCE-Chi-square-fitting/GCE_12yr_data/gll_psc_v23.fit'

TABLE3 = [
    (1.125, 3.75,  71.8),
    (0.975, 3.25,  62.9),
    (0.788, 2.63,  52.2),
    (0.600, 2.00,  38.5),
    (0.450, 1.50,  29.2),
    (0.375, 1.25,  23.4),
    (0.300, 1.00,  19.0),
    (0.225, 0.750, 16.3),
    (0.188, 0.625, 13.0),
    (0.162, 0.540, 12.9),
    (0.125, 0.417, 11.6),
    (0.100, 0.333, 11.5),
    (0.060, 0.200, 10.3),
    (0.053, 0.175, 10.3),
]

ROI_HALF_DEG = 30.0
DISK_CUT_DEG = 2.0
PIXEL_DEG = 0.1
N_BINS = 14
FULL_NX = 600
FULL_NY = 600
BRIGHT_TS_THRESHOLD = 49.0
TARGET_BRIGHT_COUNT = 25

MASK_CALIBRATED_PATH = os.path.join(OUT_DIR, 'calibrated_full_mask_14bin_400x400.npy')
CALIBRATION_META_PATH = os.path.join(OUT_DIR, 'calibrated_mask_meta.npz')

E_CENTERS = np.array([0.313, 0.407, 0.529, 0.688, 0.894, 1.160, 1.507, 1.963,
                       2.553, 3.317, 4.308, 7.282, 15.999, 35.072])


def load_catalog_sources(verbose=True):
    if not os.path.exists(CATALOG_FILE):
        raise FileNotFoundError(f"catalog not found: {CATALOG_FILE}")

    with fits.open(CATALOG_FILE) as h:
        cat = None; cols = None
        for hdu in h:
            if isinstance(hdu, fits.BinTableHDU):
                cn = [c.name for c in hdu.columns]
                if 'RAJ2000' in cn or 'RA' in cn:
                    cat = hdu.data
                    cols = cn
                    break
        if cat is None:
            raise RuntimeError("no BinTable with source data found")

        ra_col = 'RAJ2000' if 'RAJ2000' in cols else 'RA'
        dec_col = 'DEJ2000' if 'DEJ2000' in cols else 'DEC'
        ra = np.asarray(cat[ra_col], dtype=np.float64)
        dec = np.asarray(cat[dec_col], dtype=np.float64)

        ts_actual = None
        ts_col_used = None
        for name in ('TS_Value', 'TS', 'Test_Statistic', 'Detection_TS'):
            if name in cols:
                ts_actual = np.asarray(cat[name], dtype=np.float64)
                ts_col_used = name
                break

        signif = None
        if 'Signif_Avg' in cols:
            signif = np.asarray(cat['Signif_Avg'], dtype=np.float64)

        if 'Extended_Source_Name' in cols:
            is_ext = np.array([len(str(n).strip()) > 0
                                for n in cat['Extended_Source_Name']], dtype=bool)
        else:
            is_ext = np.zeros(len(ra), dtype=bool)

    sky = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    gal = sky.galactic
    l_deg = np.asarray(gal.l.degree)
    b_deg = np.asarray(gal.b.degree)
    l_deg = np.where(l_deg > 180, l_deg - 360, l_deg)

    in_roi = (np.abs(l_deg) <= ROI_HALF_DEG) & (np.abs(b_deg) <= ROI_HALF_DEG)

    if verbose:
        print(f"  catalog: {os.path.basename(CATALOG_FILE)}, {len(ra)} total sources")
        print(f"  TS column used: {ts_col_used or '(none, will fallback to Signif_Avg)'}")
        print(f"  ROI |l|,|b| <= {ROI_HALF_DEG} deg : "
              f"{int(in_roi.sum())} sources  (paper: 944)")
        print(f"    point sources (non-ext): {int((in_roi & ~is_ext).sum())}  (paper: 918)")
        print(f"    extended sources       : {int((in_roi &  is_ext).sum())}  (paper: 26)")

    return {
        'l': l_deg, 'b': b_deg,
        'ts_actual': ts_actual, 'ts_col_used': ts_col_used,
        'signif': signif, 'is_ext': is_ext, 'in_roi': in_roi,
    }


def classify_bright(srcs, verbose=True):
    in_roi = srcs['in_roi']
    ts = srcs['ts_actual']
    signif = srcs['signif']
    is_ext = srcs['is_ext']

    if ts is not None:
        bright_psc = in_roi & ~is_ext & (ts > BRIGHT_TS_THRESHOLD)
        n_bright_psc = int(bright_psc.sum())
        if verbose:
            print(f"  bright PSC (TS_actual > {BRIGHT_TS_THRESHOLD}): {n_bright_psc}  "
                  f"(paper: {TARGET_BRIGHT_COUNT})")
        if abs(n_bright_psc - TARGET_BRIGHT_COUNT) <= 5:
            return bright_psc, f'TS_actual>{int(BRIGHT_TS_THRESHOLD)}'
        if verbose:
            print(f"  -> off by more than 5 from target {TARGET_BRIGHT_COUNT}, "
                  f"falling back to Top-{TARGET_BRIGHT_COUNT} Signif_Avg ranking")

    if signif is None:
        raise RuntimeError("no usable TS or Signif_Avg column")

    psc_in_roi = in_roi & ~is_ext
    psc_signif = np.where(psc_in_roi, signif, -np.inf)
    order = np.argsort(psc_signif)[::-1]
    bright_psc = np.zeros_like(in_roi)
    for idx in order[:TARGET_BRIGHT_COUNT]:
        if psc_in_roi[idx]:
            bright_psc[idx] = True
    n_bright = int(bright_psc.sum())
    min_signif = float(signif[bright_psc].min()) if n_bright > 0 else float('nan')
    if verbose:
        print(f"  bright PSC (Top-{TARGET_BRIGHT_COUNT} Signif_Avg): "
              f"{n_bright}  min Signif_Avg = {min_signif:.2f}")
    return bright_psc, f'TopN_Signif_N={TARGET_BRIGHT_COUNT}'


def gather_source_pixels(srcs, bright_flag):
    """In-ROI sources -> two pixel lists.
    Extended sources are forced to bright (theta_l)."""
    in_roi = srcs['in_roi']
    is_ext = srcs['is_ext']

    bright_pix = []
    small_pix = []
    crpix_x = FULL_NX / 2.0 + 0.5
    crpix_y = FULL_NY / 2.0 + 0.5
    for i in np.where(in_roi)[0]:
        l = srcs['l'][i]; b = srcs['b'][i]
        px = (-l) / PIXEL_DEG + crpix_x - 1
        py = (b)  / PIXEL_DEG + crpix_y - 1
        if bright_flag[i] or is_ext[i]:
            bright_pix.append((px, py))
        else:
            small_pix.append((px, py))
    return bright_pix, small_pix


def draw_mask_one_bin(bright_pix, small_pix, theta_s, theta_l, scale):
    mask_bin = np.ones((FULL_NY, FULL_NX), dtype=np.uint8)
    Yg, Xg = np.ogrid[:FULL_NY, :FULL_NX]

    r_l_pix = (theta_l * scale) / PIXEL_DEG
    r_s_pix = (theta_s * scale) / PIXEL_DEG
    r_l_sq = r_l_pix * r_l_pix
    r_s_sq = r_s_pix * r_s_pix

    if r_l_pix > 0.5:
        for (px, py) in bright_pix:
            m = (Xg - px) ** 2 + (Yg - py) ** 2 < r_l_sq
            mask_bin[m] = 0
    if r_s_pix > 0.5:
        for (px, py) in small_pix:
            m = (Xg - px) ** 2 + (Yg - py) ** 2 < r_s_sq
            mask_bin[m] = 0
    return mask_bin


def apply_disk_mask(mask_bin):
    crpix_y = FULL_NY / 2.0 + 0.5
    yy = np.arange(FULL_NY)
    b_per_row = (yy - crpix_y + 1) * PIXEL_DEG
    disk_cut_rows = np.abs(b_per_row) < DISK_CUT_DEG
    mask_bin[disk_cut_rows, :] = 0
    return mask_bin


def fraction_in_roi(mask_600):
    sub = mask_600[ROI_SLICE[0], ROI_SLICE[1]]
    return (1.0 - sub.mean()) * 100.0


def calibrate_one_bin(bright_pix, small_pix, theta_s, theta_l, target_frac,
                       scale_min=0.3, scale_max=3.0, tol=0.05, max_iter=30):
    def frac_at(s):
        mb = draw_mask_one_bin(bright_pix, small_pix, theta_s, theta_l, s)
        mb = apply_disk_mask(mb)
        return fraction_in_roi(mb), mb

    flo, mb_lo = frac_at(scale_min)
    fhi, mb_hi = frac_at(scale_max)

    if target_frac < flo - tol:
        return flo, scale_min, mb_lo, 'min_bound'
    if target_frac > fhi + tol:
        return fhi, scale_max, mb_hi, 'max_bound'

    lo = scale_min; hi = scale_max
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid, mb = frac_at(mid)
        if abs(fmid - target_frac) <= tol:
            return fmid, mid, mb, 'converged'
        if fmid < target_frac:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    fmid, mb = frac_at(mid)
    return fmid, mid, mb, 'maxiter'


def build_calibrated_mask(verbose=True):
    print("=== Calibrated PSC mask builder ===")
    print()
    print("[1/3] Loading 4FGL-DR2 catalog...")
    srcs = load_catalog_sources(verbose=verbose)

    print()
    print("[2/3] Classifying bright sources...")
    bright_flag, method = classify_bright(srcs, verbose=verbose)
    bright_pix, small_pix = gather_source_pixels(srcs, bright_flag)
    print(f"  pixel lists: bright(+ext)={len(bright_pix)}, small={len(small_pix)}")
    print(f"  classification method: {method}")

    print()
    print("[3/3] Per-bin radius calibration to match Cholis Table III...")
    print()
    print(f"{'Bin':>3} {'E':>8} {'theta_s':>8} {'theta_l':>8} "
          f"{'target%':>8} {'init%':>8} {'scale':>7} {'achiev%':>8} "
          f"{'delta':>7}  {'status':>10}")
    print('-' * 98)

    full_400 = np.zeros((N_BINS, 400, 400), dtype=np.float32)
    scales = np.zeros(N_BINS)
    fracs_achieved = np.zeros(N_BINS)
    fracs_target = np.zeros(N_BINS)
    statuses = []

    for eb in range(N_BINS):
        theta_s, theta_l, target = TABLE3[eb]
        fracs_target[eb] = target

        mb_init = draw_mask_one_bin(bright_pix, small_pix, theta_s, theta_l, 1.0)
        mb_init = apply_disk_mask(mb_init)
        init_frac = fraction_in_roi(mb_init)

        achieved, scale, mb_cal, status = calibrate_one_bin(
            bright_pix, small_pix, theta_s, theta_l, target,
            scale_min=0.3, scale_max=3.0, tol=0.05)

        fracs_achieved[eb] = achieved
        scales[eb] = scale
        statuses.append(status)
        sub = mb_cal[ROI_SLICE[0], ROI_SLICE[1]]
        full_400[eb] = sub.astype(np.float32)

        delta = achieved - target
        print(f"{eb:>3} {E_CENTERS[eb]:>8.3f} {theta_s:>8.3f} {theta_l:>8.3f} "
              f"{target:>7.1f}% {init_frac:>7.1f}% {scale:>7.3f} "
              f"{achieved:>7.1f}% {delta:>+6.2f}%  {status:>10}")

    return full_400, scales, fracs_achieved, fracs_target, statuses


def main():
    os.chdir(WORK_DIR)
    full_mask, scales, achieved, target, statuses = build_calibrated_mask(verbose=True)

    err = achieved - target
    max_err = float(np.max(np.abs(err)))
    rms_err = float(np.sqrt(np.mean(err ** 2)))

    print()
    print("=== Calibration summary ===")
    print(f"  max |achieved - target| = {max_err:.2f} %p")
    print(f"  rms                     = {rms_err:.2f} %p")
    print(f"  bins at min_bound       = {sum(1 for s in statuses if s=='min_bound')}")
    print(f"  bins at max_bound       = {sum(1 for s in statuses if s=='max_bound')}")
    print(f"  bins converged          = {sum(1 for s in statuses if s=='converged')}")

    np.save(MASK_CALIBRATED_PATH, full_mask)
    np.savez(CALIBRATION_META_PATH,
              scales=scales, achieved=achieved, target=target,
              statuses=np.array(statuses))
    print()
    print(f"Saved: {MASK_CALIBRATED_PATH}")
    print(f"Saved: {CALIBRATION_META_PATH}")


if __name__ == '__main__':
    main()
