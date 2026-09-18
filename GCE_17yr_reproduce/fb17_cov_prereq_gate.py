#!/usr/bin/env python3
"""fb17_cov_prereq_gate.py — 블록 2(prereq) 산출물 정량 게이트.

검사: 22 ROI × {expcube center(17층), psc XML, per-ROI mask(17,600,600),
per-ROI ccube(17층·카운트>0), wimp_map(정규화 합=1.0000)}.
전 항목 PASS 시 exit 0 (블록 3 기동 허가), 아니면 exit 1 + 사유 출력.
실행: cd GCE_17yr_reproduce && FB17=1 python3 fb17_cov_prereq_gate.py
"""
import os, sys
import numpy as np
from astropy.io import fits

W = './GC_analysis_FL16Y_fb17'
F = '_front_back'
ROIS = [r for r in range(-70, 75, 5) if r != 0 and abs(r) >= 20]
assert len(ROIS) == 22
ok = True

def fail(msg):
    global ok
    ok = False
    print(f'  FAIL {msg}')

n_exp = n_xml = n_msk = n_ccb = n_wmp = 0
for r in ROIS:
    p = f'{W}/GC_expcube_center_17yr{F}_clean_l{r}.fits'
    if os.path.exists(p):
        ne = fits.open(p, memmap=True)[0].data.shape[0]
        if ne == 17: n_exp += 1
        else: fail(f'expcube l{r}: 층수 {ne} != 17')
    else: fail(f'expcube l{r}: 없음')

    p = f'{W}/Model/GC_psc_model_FL16Y_l{r}.xml'
    if os.path.exists(p) and os.path.getsize(p) > 10_000: n_xml += 1
    else: fail(f'psc xml l{r}: 없음/과소')

    p = f'{W}/Model/GC_mask_60x60_definitions_FL16Y_l{r}.npy'
    if os.path.exists(p):
        sh = np.load(p, mmap_mode='r').shape
        if sh == (17, 600, 600): n_msk += 1
        else: fail(f'mask l{r}: shape {sh}')
    else: fail(f'mask l{r}: 없음')

    p = f'{W}/GC_ccube_17yr{F}_clean_l{r}.fits'
    if os.path.exists(p):
        d = fits.open(p, memmap=True)[0].data
        if d.shape == (17, 600, 600) and d.sum() > 0: n_ccb += 1
        else: fail(f'ccube l{r}: shape {d.shape} / sum {d.sum():.3g}')
    else: fail(f'ccube l{r}: 없음')

    p = f'{W}/Model/wimp_map_l{r}.fits'
    if os.path.exists(p):
        m = fits.open(p)[0].data.astype(float)
        norm = m.sum() * (np.pi / 180.0) ** 2 * 0.1 ** 2
        if abs(norm - 1.0) < 1e-3: n_wmp += 1
        else: fail(f'wimp l{r}: norm {norm:.5f}')
    else: fail(f'wimp l{r}: 없음')

print(f'[gate] expcube {n_exp}/22, xml {n_xml}/22, mask {n_msk}/22, '
      f'ccube {n_ccb}/22, wimp {n_wmp}/22')
print('[gate] ' + ('ALL PASS — 블록 3(watchdog) 기동 가능' if ok
                   else 'FAIL — 위 사유 보고 후 대기'))
sys.exit(0 if ok else 1)
