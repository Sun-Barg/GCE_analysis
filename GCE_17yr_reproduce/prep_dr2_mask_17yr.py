#!/usr/bin/env python3
"""
prep_dr2_mask_17yr.py — 17.5yr 데이터에 4FGL-DR2 PSC mask 적용 준비+검증
(catalog-update systematic study, Stage A).

근거(이 세션 코드 확인):
  - MAIN fit에서 PSC는 mask(.npy)로만 들어감. component map은 PSC-free
    6-template XML 산물이라 catalog 무관 → 재사용.
  - mask는 Cholis Table III radius + catalog 위치/유의도 + 공유 grid에만
    의존(데이터/IRF/연수 비의존). 12yr DR2 mask는 grid 동일 시 17yr에 직접 적용.

동작:
  1) 12yr DR2 mask 존재/shape 확인
  2) 12yr↔17yr CCUBE grid(공간 WCS + 에너지 edge) 동일성 검증
  3) 통과 시 DR2 mask를 17yr Model/ 로 복사
  4) bin별 kept-fraction(DR2 vs FL16Y) 출력 (catalog 효과 미리보기)

GCE_17yr_reproduce/ 에서 실행. 경로는 필요시 상단에서 수정.
"""
import os
import shutil
import numpy as np
from astropy.io import fits

# ---- paths ----
DR2_MASK_SRC = '../GCE_12yr_reproduce/GC_analysis_DR2/Model/GC_mask_60x60_definitions_DR2.npy'
CCUBE_12YR   = '../GCE_12yr_reproduce/GC_analysis_DR2/GC_ccube_12yr_front_clean.fits'
CCUBE_17YR   = './GC_analysis_FL16Y/GC_ccube_17yr_front_clean.fits'
FL16Y_MASK   = './GC_analysis_FL16Y/Model/GC_mask_60x60_definitions_FL16Y.npy'
DISK_MASK    = './GC_analysis_FL16Y/Model/GC_disk_mask_60x60_definitions.npy'
DR2_MASK_DST = './GC_analysis_FL16Y/Model/GC_mask_60x60_definitions_DR2.npy'

INNER = slice(100, 500)   # fit이 사용하는 inner 400x400


def _spatial_wcs_keys(hdr):
    keys = ['NAXIS1', 'NAXIS2', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2',
            'CDELT1', 'CDELT2', 'CRPIX1', 'CRPIX2']
    return {k: hdr.get(k) for k in keys}


def _energy_edges_kev(path):
    """CCUBE HDU1(EBOUNDS): E_MIN, E_MAX (keV)."""
    with fits.open(path) as h:
        d = h[1].data
        emin = np.array([row[1] for row in d], dtype=float)
        emax = np.array([row[2] for row in d], dtype=float)
    return emin, emax


def main():
    # 1) DR2 mask 존재/shape
    if not os.path.exists(DR2_MASK_SRC):
        raise SystemExit(
            f'[FATAL] 12yr DR2 mask not found:\n  {DR2_MASK_SRC}\n'
            f'  경로 수정 필요 (REF_directory_info §1: GC_analysis_DR2/Model/).')
    dr2 = np.load(DR2_MASK_SRC)
    print(f'[1] DR2 mask shape={dr2.shape}  dtype={dr2.dtype}  '
          f'(unique={np.unique(dr2)})')
    assert dr2.shape == (14, 600, 600), 'DR2 mask shape != (14,600,600)'

    # 2) grid 동일성 (공간 WCS + 에너지 edge)
    h17 = fits.open(CCUBE_17YR)[0].header
    w17 = _spatial_wcs_keys(h17)
    print(f'[2] 17yr CCUBE spatial WCS: {w17}')

    grid_ok = True
    if os.path.exists(CCUBE_12YR):
        h12 = fits.open(CCUBE_12YR)[0].header
        w12 = _spatial_wcs_keys(h12)
        mismatch = {k: (w12[k], w17[k]) for k in w17 if w12[k] != w17[k]}
        if mismatch:
            grid_ok = False
            print(f'    [WARN] 공간 WCS 불일치 12yr vs 17yr: {mismatch}')
        else:
            print(f'    공간 WCS 동일 ✓')
        e12, E12b = _energy_edges_kev(CCUBE_12YR)
        e17, E17b = _energy_edges_kev(CCUBE_17YR)
        de = max(float(np.max(np.abs(e12 - e17))),
                 float(np.max(np.abs(E12b - E17b))))
        print(f'    energy edge max|Δ|={de:.3e} keV  '
              f'({"동일 ✓" if de < 1.0 else "WARN: 다름"})')
        if de >= 1.0:
            grid_ok = False
    else:
        print(f'    [info] 12yr CCUBE 없음({CCUBE_12YR}); shape-match로만 확인.\n'
              f'           양쪽 모두 Cholis gtbin spec(600x600, binsz=0.1,\n'
              f'           GAL/CAR, ebinfile=Cholis 14-edge) → grid 동일 보장.')

    if not grid_ok:
        raise SystemExit(
            '[FATAL] grid 불일치 — 12yr DR2 mask 직접 재사용 불가. '
            '경로/스펙 점검 또는 fresh build 필요.')

    # 3) 복사
    os.makedirs(os.path.dirname(DR2_MASK_DST), exist_ok=True)
    shutil.copy2(DR2_MASK_SRC, DR2_MASK_DST)
    print(f'[3] DR2 mask 복사 완료 → {DR2_MASK_DST}')

    # 4) masked-fraction 미리보기 (inner 400x400, disk mask 포함)
    fl16y = np.load(FL16Y_MASK)
    assert fl16y.shape == (14, 600, 600), 'FL16Y mask shape != (14,600,600)'
    disk = np.load(DISK_MASK)[INNER, INNER]
    ntot = disk.size
    emin, emax = _energy_edges_kev(CCUBE_17YR)
    E = 1e-6 * np.sqrt(emin * emax)   # keV -> GeV (geometric mean)

    print(f'[4] bin별 kept-fraction (값 클수록 덜 masking). '
          f'full = PSC×disk (fit 적용분):')
    print(f'    {"bin":>3} {"E[GeV]":>8} | {"DR2 psc":>8} {"FL psc":>8} | '
          f'{"DR2 full":>8} {"FL full":>8} | {"Δfull":>8}')
    for i in range(14):
        d_psc = dr2[i][INNER, INNER]
        f_psc = fl16y[i][INNER, INNER]
        d_full = float((d_psc * disk).sum()) / ntot
        f_full = float((f_psc * disk).sum()) / ntot
        print(f'    {i:>3} {E[i]:>8.3f} | '
              f'{d_psc.mean():>8.3f} {f_psc.mean():>8.3f} | '
              f'{d_full:>8.3f} {f_full:>8.3f} | {d_full - f_full:>+8.3f}')

    print('\n[done] DR2 mask 준비 완료. grid 통과·fraction 확인 후 Stage B(refit) 진행.')


if __name__ == '__main__':
    main()
