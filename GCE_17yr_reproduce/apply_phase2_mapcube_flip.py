#!/usr/bin/env python3
"""
apply_phase2_mapcube_flip.py — phase 2 환경의 MapCube axis=2 in-place flip
+ HDU0 HISTORY marker 로 idempotent 보호.

12yr REF_12yr_phase2_signal_normalization_FINAL.md §1.1 의 flip 을
17yr 80 model × 3 component = 240 file 로 확장.

Usage:
    cp -a MapCubes /tmp/MapCubes_17yr_pre_phase2     # backup 먼저 (수동)
    python apply_phase2_mapcube_flip.py              # 실제 적용
    python apply_phase2_mapcube_flip.py --check      # marker 만 점검, flip 안 함
"""
import argparse
import glob
import os
import sys
from datetime import datetime

from astropy.io import fits

MAPCUBE_DIR = './MapCubes'
COMPONENTS  = ['pion', 'bremss', 'ics']
MARKER      = 'PHASE2_AXIS2_FLIP_APPLIED'   # 두 번 실행 보호 marker


def _has_marker(hdul):
    """HDU0 HISTORY 에 MARKER 가 있는지."""
    hdr = hdul[0].header
    for card in hdr.get('HISTORY', []):
        if MARKER in str(card):
            return True
    return False


def _flip_one(path, check_only=False):
    """Return: 'flipped' / 'skipped' / 'failed'."""
    try:
        with fits.open(path, mode='update') as h:
            if _has_marker(h):
                return 'skipped'
            if check_only:
                return 'would_flip'
            if h[0].data is None or h[0].data.ndim != 3:
                print(f'  [FAIL] {path}: data is None or not 3D', flush=True)
                return 'failed'
            h[0].data = h[0].data[:, :, ::-1]
            h[0].header['HISTORY'] = (
                f'{MARKER} {datetime.utcnow().isoformat()}Z '
                f'(axis=2 GLON flip, phase 2 env per Cholis paper convention)'
            )
            h.flush()
        return 'flipped'
    except Exception as e:
        print(f'  [FAIL] {path}: {e}', flush=True)
        return 'failed'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='marker 점검만, flip 적용 안 함')
    args = ap.parse_args()

    # pion glob 으로 model list 추출 (pi0 는 symlink 라 한 쪽만 처리)
    pion_files = sorted(glob.glob(
        os.path.join(MAPCUBE_DIR, 'pion_mapcube_model*.fits')))
    if not pion_files:
        print(f'[ERROR] no pion_mapcube_model*.fits in {MAPCUBE_DIR}',
              file=sys.stderr)
        sys.exit(1)

    models = [os.path.basename(p)
              .replace('pion_mapcube_model', '')
              .replace('.fits', '')
              for p in pion_files]
    print(f'[info ] found {len(models)} models in {MAPCUBE_DIR}', flush=True)

    counts = {'flipped': 0, 'skipped': 0, 'failed': 0,
              'missing': 0, 'would_flip': 0}
    for m in models:
        for comp in COMPONENTS:
            path = os.path.join(MAPCUBE_DIR,
                                f'{comp}_mapcube_model{m}.fits')
            if not os.path.exists(path):
                print(f'  [MISS] {path}', flush=True)
                counts['missing'] += 1
                continue
            res = _flip_one(path, check_only=args.check)
            counts[res] = counts.get(res, 0) + 1
            if res == 'flipped':
                print(f'  [flip] {path}', flush=True)

    print('\n=== summary ===', flush=True)
    for k, v in counts.items():
        print(f'  {k:>11}: {v}', flush=True)
    expected = len(models) * len(COMPONENTS)
    accounted = sum(counts.values())
    print(f'  expected   : {expected}  (= {len(models)} models × '
          f'{len(COMPONENTS)} components)', flush=True)
    print(f'  accounted  : {accounted}', flush=True)

    if counts['failed'] > 0 or counts['missing'] > 0:
        sys.exit(2)
    sys.exit(0)


if __name__ == '__main__':
    main()
