# ============================================================
# make_perroi_ccube.py
# ============================================================
#!/usr/bin/env python3
"""make_perroi_ccube.py — per-ROI CCUBE generator (cov pipeline prerequisite).

Formalizes the missing cell 6 of GCE_covariance_marix_calculation_17yr_v13.ipynb
(cov notebook has NO per-ROI ccube gen cell; cell 0 markdown only; cell 8 gtbin
= single GC_cmap; cells 17/22/27 only consume per-ROI ccubes).

Outputs (atomic .tmp + rename):
  GC_analysis_FL16Y/GC_ccube_17yr_front_clean_l{roi}.fits   x 22

Per ROI runs gtbin with spatial sampling byte-reverse-engineered from
existing l-25 header: 600x600, 0.1deg, GAL/CAR, xref=roi, yref=0,
ebinfile=bin_definitions.fits (14 bins). Cholis L1462: ROIs "differ
by the longitude at which they are centered" -> xref=roi.

22 ROIs (Cholis L1637): roi != 0, abs(roi) in [20,70], roi % 5 == 0.

Idempotent: existing valid files skipped; existing invalid -> FATAL
(no silent stale reuse, Phase 2 contract).

CLI:
  python make_perroi_ccube.py                       # all 22
  python make_perroi_ccube.py --rois=25,30,-25      # subset (=-N for negatives)
  python make_perroi_ccube.py --workers 4           # parallel
  python make_perroi_ccube.py --force               # regenerate even if valid
"""
import os
import sys
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from astropy.io import fits

# ---- CONFIG (matches prepare_common.py / run_one_roi_cov.py conventions) ----
WORK_DIR = './GC_analysis_FL16Y'
FRONT    = '_front'

# FB17 variant (front+back, 17 bins) — env switch (prepare_common.py 동일).
FB17   = bool(os.environ.get('FB17', '').strip())
NEBINS = 17 if FB17 else 14
if FB17:
    WORK_DIR = './GC_analysis_FL16Y_fb17'
    FRONT    = '_front_back'
    print(f'[config] FB17=1 -> WORK_DIR={WORK_DIR}, FRONT={FRONT!r}, '
          f'NEBINS={NEBINS}', flush=True)

EVFILE   = f'{WORK_DIR}/Allsky_gti_17yr{FRONT}_clean.fits'
SCFILE   = '../GCE_allsky_data/lat_spacecraft_merged_17yr.fits'
EBINFILE = f'{WORK_DIR}/bin_definitions.fits'
EVTABLE  = 'EVENTS'

# gtbin spatial params (byte-for-byte from l-25 header, documented 17yr constants)
NXPIX, NYPIX = 600, 600
BINSZ        = 0.1
COORDSYS     = 'GAL'
AXISROT      = 0.0
PROJ         = 'CAR'

ALL_ROIS = [r for r in range(-70, 75, 5) if r != 0 and abs(r) >= 20]
assert len(ALL_ROIS) == 22, f'expected 22 ROIs, got {len(ALL_ROIS)}'


def _out_path(roi):
    return f'{WORK_DIR}/GC_ccube_17yr{FRONT}_clean_l{roi}.fits'


def _cube_ok(path, expected_nebins=NEBINS):
    if not os.path.exists(path):
        return False, 'missing'
    try:
        with fits.open(path) as h:
            d = h[0].data
            if d is None or d.ndim != 3:
                return False, f'bad ndim ({None if d is None else d.ndim})'
            ne, ny, nx = d.shape
            if ne != expected_nebins:
                return False, f'NAXIS3={ne} (want {expected_nebins})'
            if (nx, ny) != (NXPIX, NYPIX):
                return False, f'XY=({nx},{ny}) want ({NXPIX},{NYPIX})'
            if not (d == d).all():
                return False, 'NaN present'
    except Exception as e:
        return False, f'exception: {e}'
    return True, 'OK'


def _check_prerequisites():
    missing = []
    for p, label in [(EVFILE, 'event file (gtmktime output)'),
                     (SCFILE, 'spacecraft FT2 merged'),
                     (EBINFILE, 'bin definitions FITS'),
                     (WORK_DIR, 'work dir')]:
        if not os.path.exists(p):
            missing.append(f'{label}: {p}')
    if missing:
        print('[FATAL] missing prerequisites:', file=sys.stderr)
        for m in missing:
            print(f'  {m}', file=sys.stderr)
        print('  -> run prepare_common.py first', file=sys.stderr)
        sys.exit(2)


def build_one(roi, force=False):
    out = _out_path(roi)
    ok, msg = _cube_ok(out)
    if ok and not force:
        return roi, True, f'[skip] {out}  {msg}'
    if not ok and msg != 'missing' and not force:
        return roi, False, f'[FATAL] {out} exists but INVALID ({msg}); use --force to rebuild'

    tmp = out + '.tmp'
    cmd = [
        'gtbin',
        f'evfile={EVFILE}',
        f'scfile={SCFILE}',
        f'outfile={tmp}',
        'algorithm=CCUBE',
        f'nxpix={NXPIX}', f'nypix={NYPIX}', f'binsz={BINSZ}',
        f'coordsys={COORDSYS}',
        f'xref={float(roi)}', 'yref=0.0',
        f'axisrot={AXISROT}', f'proj={PROJ}',
        'ebinalg=FILE', f'ebinfile={EBINFILE}',
        f'evtable={EVTABLE}',
        'chatter=2', 'clobber=yes', 'gui=no', 'mode=ql',
    ]
    print(f'[run ] roi={roi:+d}  gtbin -> {out}', flush=True)
    rc = subprocess.call(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT)
    if rc != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        return roi, False, f'[FATAL] roi={roi}  gtbin rc={rc}'

    ok, msg = _cube_ok(tmp)
    if not ok:
        os.remove(tmp)
        return roi, False, f'[FATAL] roi={roi}  output verify fail: {msg}'

    os.rename(tmp, out)
    return roi, True, f'[done] {out}  {msg}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rois', default=None,
                   help='comma-separated ROI list (use --rois=-20,20 for negatives); default = all 22')
    p.add_argument('--workers', type=int, default=1,
                   help='parallel gtbin subprocesses (default 1; gtbin is light, ~hundreds MB each)')
    p.add_argument('--force', action='store_true',
                   help='regenerate even if existing file passes verify')
    args = p.parse_args()

    _check_prerequisites()

    if args.rois:
        try:
            rois = [int(x) for x in args.rois.split(',') if x.strip()]
        except ValueError:
            print(f'[FATAL] bad --rois value: {args.rois}', file=sys.stderr); sys.exit(2)
        bad = [r for r in rois if r not in ALL_ROIS]
        if bad:
            print(f'[FATAL] invalid ROIs: {bad}', file=sys.stderr)
            print(f'  valid: {ALL_ROIS}', file=sys.stderr)
            sys.exit(2)
    else:
        rois = ALL_ROIS

    print(f'==== make_perroi_ccube  rois={len(rois)}  workers={args.workers}  '
          f'force={args.force} ====', flush=True)

    results = []
    if args.workers == 1:
        for roi in rois:
            results.append(build_one(roi, force=args.force))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(build_one, roi, args.force): roi for roi in rois}
            for f in as_completed(futs):
                results.append(f.result())

    n_done = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_done
    for _, _, msg in sorted(results, key=lambda x: x[0]):
        print(msg, flush=True)
    print(f'==== summary: {n_done}/{len(rois)} ok, {n_fail} failed ====', flush=True)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
