#!/usr/bin/env python3
"""
build_mapcubes.py — Cholis Zenodo raw GALPROP maps → fermitools MapCube

자기완결 변환 script. Sanghwan 의 Converting_into_mapcube_test.ipynb CELL 5 의
logic (raw × 1e-3 / E_GeV², flip axis=2) 을 명시적으로 재구현하고 provenance
+ verification 을 추가.

USAGE:
  python build_mapcubes.py --models X,XIII,II --verify-against-existing
  python build_mapcubes.py --verify-against-existing      # all 80 models
  python build_mapcubes.py --check                        # verify out-dir 만
  python build_mapcubes.py --force                        # rebuild even if exists
"""
import os
import sys
import argparse
import hashlib
from datetime import datetime
import numpy as np
from astropy.io import fits

# -----------------------------------------------------------------
# Defaults (working dir = ~/GCE-Chi-square-fitting/GCE_17yr_reproduce 기준)
# -----------------------------------------------------------------
RAW_DIR_DEFAULT  = '../GCE_TEMPLATES_FILES_v3/GALACTIC_DIFFUSE_EMISSION_MAPS_0p25deg'
NAMING_DEFAULT   = '../GCE_TEMPLATES_FILES_v3/NAMING_CONVENTION_OF_DIFFUSE_EMISSION_MODELS.dat'
OUT_DIR_DEFAULT  = './MapCubes_v2'
EXISTING_DIR_DEF = './MapCubes'

# raw filename: {raw_prefix}_{hash}_Map_flux_E_50-814008_MeV_InnerGalaxy_60x60.fits
# output filename: {out_prefix}_mapcube_model{Roman}.fits
COMPONENTS = [
    ('bremss', 'bremss'),
    ('pi0',    'pion'),    # pi0 raw → pion output (베이스 컨벤션)
    ('ICS',    'ics'),     # 대문자 raw → 소문자 output
]

# Cholis README 의 38-bin Ectr (MeV) — ENERGIES 검증 reference
CHOLIS_README_38_ECTR_MEV = np.array([
    50.0, 64.98283, 84.4553638962, 109.762971093, 142.654169817,
    185.40143332, 240.958196464, 313.162910358, 407.004243322,
    528.965751061, 687.473829541, 893.47989989, 1161.21704886,
    1509.18340158, 1961.42016848, 2549.17266733, 3313.04908164,
    4305.82610508, 5596.09531592, 7273.00221156, 9452.40532607,
    12284.8809679, 15966.1266302, 20750.4818513, 26968.5006912,
    35049.7899155, 45552.6907923, 59202.8552359, 76943.3815462,
    99999.9736528, 129965.625758, 168910.683289, 219525.884347,
    285308.264463, 370802.768944, 481916.265956, 626325.655697,
    814008.272176,
])

# Verification thresholds vs existing MapCube
VERIFY_RATIO_TOL    = 1e-5    # per-bin sum ratio = 1.0 ± 1e-5
VERIFY_MAXREL_TOL   = 1e-6    # max per-pixel relative diff


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
def parse_naming(path):
    """NAMING_CONVENTION → {Roman: hash}"""
    m = {}
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith('#'):
                continue
            tok = s.split()
            if len(tok) >= 2:
                m[tok[0]] = tok[1]
    return m


def file_md5(path, chunksize=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunksize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_raw(raw_path):
    """raw FITS → (data, header, energies_array, energies_ext_hdu_copy)"""
    with fits.open(raw_path) as hd:
        if hd[0].data is None:
            raise ValueError('HDU0 data is None')
        data = np.asarray(hd[0].data, dtype=np.float64)
        h0   = hd[0].header.copy()
        energ_hdu = None
        for ext in hd[1:]:
            if ext.header.get('EXTNAME', '').strip() == 'ENERGIES':
                energ_hdu = ext
                break
        if energ_hdu is None:
            raise ValueError('ENERGIES extension missing')
        names = energ_hdu.data.dtype.names
        colname = 'Energy' if 'Energy' in names else names[0]
        E_MeV = np.asarray(energ_hdu.data[colname], dtype=np.float64)
        if E_MeV.size != data.shape[0]:
            raise ValueError(f'ENERGIES size {E_MeV.size} != NAXIS3 {data.shape[0]}')
        # in-memory copy (raw file closes when context exits)
        energ_copy = fits.BinTableHDU(data=energ_hdu.data.copy(),
                                      header=energ_hdu.header.copy(),
                                      name='ENERGIES')
        return data, h0, E_MeV, energ_copy


def build_one(raw_path, out_path, roman, raw_prefix, dry_run=False):
    """raw → output FITS. Returns (ok, info_dict)."""
    if not os.path.exists(raw_path):
        return False, {'error': f'raw missing: {raw_path}'}

    try:
        raw_data, raw_h0, E_MeV, energ_copy = read_raw(raw_path)
    except Exception as e:
        return False, {'error': f'read_raw: {e}'}

    # Cholis README 38-bin Ectr 검증
    if E_MeV.size != CHOLIS_README_38_ECTR_MEV.size:
        return False, {'error': f'ENERGIES n={E_MeV.size} expected 38'}
    ratio = E_MeV / CHOLIS_README_38_ECTR_MEV
    if not (abs(np.median(ratio) - 1.0) < 1e-4 and abs(ratio.max() - 1.0) < 1e-4):
        return False, {'error': f'ENERGIES ≠ Cholis README 38-bin Ectr '
                                 f'(median ratio {np.median(ratio):.6f})'}

    # Unit conversion (per-bin, model-independent)
    #   raw  : E²·dΦ/dE  [GeV/cm²/s/sr]
    #   out  : dN/dE     [ph/cm²/s/sr/MeV]
    #   out  = raw · (1e-3 / E_GeV²)
    E_GeV = E_MeV * 1e-3
    factor = (1e-3 / (E_GeV ** 2))[:, None, None]
    new_data = raw_data * factor
# NO spatial flip: raw FITS already in paper-faithful orientation
# (CDELT1=-0.25 → 왼쪽=+30°l, 오른쪽=-30°l, galactic 표준).
# CELL 5 의 flip 은 그 노트북의 다른 base template 호환용이며
# raw 변환의 본질이 아님. flip 적용 시 paper convention 깨짐 (mirror image).

    # Output header (raw 의 spatial WCS 그대로, 단위/dtype 정정)
    h0 = fits.Header()
    h0['SIMPLE']  = True
    h0['BITPIX']  = -32
    h0['NAXIS']   = 3
    h0['NAXIS1']  = int(raw_h0['NAXIS1'])
    h0['NAXIS2']  = int(raw_h0['NAXIS2'])
    h0['NAXIS3']  = int(raw_h0['NAXIS3'])
    for k in ('CTYPE1', 'CUNIT1', 'CRVAL1', 'CRPIX1', 'CDELT1',
              'CTYPE2', 'CUNIT2', 'CRVAL2', 'CRPIX2', 'CDELT2'):
        if k in raw_h0:
            h0[k] = raw_h0[k]
    h0['CTYPE3']  = 'photon energy'
    h0['CUNIT3']  = 'MeV'
    h0['CRPIX3']  = 1.0
    h0['CRVAL3']  = float(E_MeV[0])
    h0['CDELT3']  = float(E_MeV[1] - E_MeV[0])      # first-bin width (fermitools 는 ENERGIES 우선)
    h0['BUNIT']   = 'photon/cm2/s/MeV/sr'

    # Provenance
    md5 = file_md5(raw_path)
    h0.add_history('Built by build_mapcubes.py')
    h0.add_history(f'Source raw:  {os.path.basename(raw_path)}')
    h0.add_history(f'Source md5:  {md5}')
    h0.add_history(f'Model:       {roman}  (raw prefix "{raw_prefix}")')
    h0.add_history(f'Build date:  {datetime.utcnow().isoformat()}Z')
    h0.add_history('Conversion:  dN/dE [ph/cm2/s/sr/MeV]')
    h0.add_history('             = raw [GeV/cm2/s/sr] * 1e-3 / E_GeV^2')
    h0.add_history('             applied per-bin (model-independent)')
    h0.add_history('Spatial:     raw orientation preserved (no flip)')
    h0.add_history('             raw CDELT1=-0.25 already paper-faithful WCS')
    h0.add_history('             (left=+30 deg l, right=-30 deg l, standard galactic)')

    primary = fits.PrimaryHDU(data=new_data.astype(np.float32), header=h0)
    hdul    = fits.HDUList([primary, energ_copy])

    info = {
        'raw_path':  raw_path,
        'raw_md5':   md5,
        'raw_sum':   float(raw_data.sum()),
        'new_sum':   float(new_data.sum()),
        'shape':     new_data.shape,
    }
    if not dry_run:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        tmp = out_path + '.tmp'
        hdul.writeto(tmp, overwrite=True)
        os.replace(tmp, out_path)
    return True, info


def verify_against_existing(new_path, existing_path):
    """기존 MapCube 와 새 MapCube 비교. (ok, stats)."""
    if not os.path.exists(existing_path):
        return None, {'note': f'existing missing: {existing_path}'}
    with fits.open(new_path) as h_new, fits.open(existing_path) as h_old:
        d_new = np.asarray(h_new[0].data, dtype=np.float64)
        d_old = np.asarray(h_old[0].data, dtype=np.float64)
    if d_new.shape != d_old.shape:
        return False, {'error': f'shape mismatch {d_new.shape} vs {d_old.shape}'}

    new_pbs = np.array([float(d_new[i].sum()) for i in range(d_new.shape[0])])
    old_pbs = np.array([float(d_old[i].sum()) for i in range(d_old.shape[0])])
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = new_pbs / np.where(old_pbs == 0, np.nan, old_pbs)
    diff_abs = np.abs(d_new - d_old)
    denom = np.where(np.abs(d_old) > 1e-30, np.abs(d_old), np.nan)
    rel = float(np.nanmax(diff_abs / denom))

    stats = {
        'per_bin_sum_ratio_min':    float(np.nanmin(ratio)),
        'per_bin_sum_ratio_max':    float(np.nanmax(ratio)),
        'per_bin_sum_ratio_median': float(np.nanmedian(ratio)),
        'max_abs_diff':             float(np.max(diff_abs)),
        'max_rel_diff_pixel':       rel,
    }
    ok = (abs(stats['per_bin_sum_ratio_median'] - 1.0) < VERIFY_RATIO_TOL and
          stats['per_bin_sum_ratio_min'] > (1.0 - VERIFY_RATIO_TOL) and
          stats['per_bin_sum_ratio_max'] < (1.0 + VERIFY_RATIO_TOL) and
          stats['max_rel_diff_pixel']   < VERIFY_MAXREL_TOL)
    return ok, stats


def check_existing(out_path):
    """Out-dir 의 파일이 형태적으로 valid 한지 (shape, ENERGIES) — verify 없이."""
    if not os.path.exists(out_path):
        return False, 'missing'
    try:
        with fits.open(out_path) as hd:
            if hd[0].data is None:
                return False, 'HDU0 data None'
            if hd[0].data.shape != (38, 240, 240):
                return False, f'shape {hd[0].data.shape}'
            energ = None
            for ext in hd[1:]:
                if ext.header.get('EXTNAME', '').strip() == 'ENERGIES':
                    energ = ext; break
            if energ is None:
                return False, 'ENERGIES missing'
            names = energ.data.dtype.names
            E = np.asarray(energ.data['Energy' if 'Energy' in names else names[0]],
                           dtype=np.float64)
            if E.size != 38:
                return False, f'ENERGIES size {E.size}'
            r = E / CHOLIS_README_38_ECTR_MEV
            if not (abs(np.median(r) - 1.0) < 1e-4):
                return False, f'ENERGIES median ratio {np.median(r):.6f}'
        return True, 'ok'
    except Exception as e:
        return False, f'{e}'


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw-dir',      default=RAW_DIR_DEFAULT)
    ap.add_argument('--naming',       default=NAMING_DEFAULT)
    ap.add_argument('--out-dir',      default=OUT_DIR_DEFAULT)
    ap.add_argument('--existing-dir', default=EXISTING_DIR_DEF)
    ap.add_argument('--models',       default=None,
                    help='Comma-separated Roman list (default: all 80)')
    ap.add_argument('--check',        action='store_true',
                    help='Verify out-dir files (shape, ENERGIES) without building')
    ap.add_argument('--force',        action='store_true',
                    help='Rebuild even if output exists')
    ap.add_argument('--verify-against-existing', action='store_true',
                    help='Compare new output vs existing-dir MapCube (per-bin + per-pixel)')
    ap.add_argument('--dry-run',      action='store_true')
    args = ap.parse_args()

    roman2hash = parse_naming(args.naming)
    if not roman2hash:
        sys.exit(f'No models loaded from {args.naming}')

    if args.models:
        models = [m.strip() for m in args.models.split(',') if m.strip()]
    else:
        # 길이 순 → alphabetical: I, V, X, L, II, III, ... (사람이 읽기 쉬운 순서)
        models = sorted(roman2hash.keys(), key=lambda s: (len(s), s))

    print(f'Models to process: {len(models)}')
    print(f'Raw dir:           {args.raw_dir}')
    print(f'Output dir:        {args.out_dir}')
    print(f'Existing dir:      {args.existing_dir}')
    print(f'Mode:              '
          f'{"check" if args.check else ("force" if args.force else "build-missing")}'
          f'{" + verify-against-existing" if args.verify_against_existing else ""}'
          f'{" [DRY RUN]" if args.dry_run else ""}')
    print()

    n_built = n_skip = n_err = 0
    n_verify_ok = n_verify_fail = n_verify_noref = 0

    for M in models:
        h = roman2hash.get(M)
        if h is None:
            print(f'[ERR]   {M}: not in NAMING'); n_err += 1; continue

        for raw_prefix, out_prefix in COMPONENTS:
            raw_path = os.path.join(
                args.raw_dir,
                f'{raw_prefix}_{h}_Map_flux_E_50-814008_MeV_InnerGalaxy_60x60.fits')
            out_path = os.path.join(
                args.out_dir, f'{out_prefix}_mapcube_model{M}.fits')
            exist_path = os.path.join(
                args.existing_dir, f'{out_prefix}_mapcube_model{M}.fits')

            tag = f'{M}/{out_prefix:6s}'

            # CHECK-only mode
            if args.check:
                ok, msg = check_existing(out_path)
                print(f'[CHK {"OK " if ok else "FAIL"}] {tag}  {msg}')
                if ok: n_skip += 1
                else:  n_err  += 1
                continue

            # build (or skip if exists)
            if os.path.exists(out_path) and not args.force:
                action = 'SKIP'
            else:
                ok, info = build_one(raw_path, out_path, M, raw_prefix,
                                     dry_run=args.dry_run)
                if not ok:
                    print(f'[ERR]   {tag}: {info.get("error")}')
                    n_err += 1
                    continue
                print(f'[BUILT] {tag}  shape={info["shape"]}  '
                      f'raw_sum={info["raw_sum"]:.4e}  new_sum={info["new_sum"]:.4e}')
                n_built += 1
                action = 'BUILT'

            if action == 'SKIP':
                n_skip += 1

            # Verify against existing (if requested and not dry-run)
            if args.verify_against_existing and not args.dry_run:
                v_ok, stats = verify_against_existing(out_path, exist_path)
                if v_ok is None:
                    print(f'  [VERIFY] {tag}  no existing reference at {exist_path}')
                    n_verify_noref += 1
                elif v_ok:
                    print(f'  [VERIFY OK]   {tag}  median_ratio='
                          f'{stats["per_bin_sum_ratio_median"]:.6f}  '
                          f'max_rel={stats["max_rel_diff_pixel"]:.2e}')
                    n_verify_ok += 1
                else:
                    print(f'  [VERIFY FAIL] {tag}  {stats}')
                    n_verify_fail += 1

    print()
    print('=' * 60)
    print(f'Summary: built={n_built}, skipped={n_skip}, errors={n_err}')
    if args.verify_against_existing:
        print(f'         verify ok={n_verify_ok}, fail={n_verify_fail}, '
              f'no_ref={n_verify_noref}')
    if n_err > 0 or n_verify_fail > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
