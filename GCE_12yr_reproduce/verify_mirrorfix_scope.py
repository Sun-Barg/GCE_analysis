#!/usr/bin/env python3
"""verify_mirrorfix_scope.py — 12yr component orientation 검증 (재빌드 전/후 공용).

원본 verify_12yr_flip_scope.py 가 GCE_12yr_reproduce/ 에 부재하여,
HANDOFF_12yr_mirrorflip_2026-07-03.md §1.2 에 기록된 확정 판정 기준을 동일 구현:

  corr  = Pearson( 12yr comp(E축 합산 2D), 17yr comp(E축 합산 2D) )
  corrM = Pearson( 12yr comp,              GLON-flip(17yr comp) )   # 2D [:, ::-1]
  FLIPPED  if corrM > corr   else OK

핸드오프 확정 수치 (도구 교정 기준):
  flipped 38개: corr 0.81-0.94,  corrM 0.996-0.998
  정상   42개: corr ~1.000,      corrM 0.80-0.94

사용:
  (1) step1 실행 전:  python verify_mirrorfix_scope.py
        -> FLIPPED 목록이 핸드오프 BAD38 과 정확히 일치해야 함 (대칭차 ∅).
           일치하지 않으면 step1 진행 중단하고 결과 공유.
  (2) 재빌드 후    :  python verify_mirrorfix_scope.py
        -> FLIPPED 0건이어야 step2(재fit) 진행.

출력: verify_mirrorfix_scope.csv (model,comp,corr,corrM,status,cluster) + 콘솔 요약.
exit code: 0 = FLIPPED 없음, 1 = FLIPPED 존재.
"""
import os
import sys
import csv
import numpy as np
from astropy.io import fits

D12 = 'GC_analysis_DR2'
D17 = '../GCE_17yr_reproduce/GC_analysis_FL16Y'
COMPS = ('pion', 'bremss', 'ics')


def to_roman(n):
    out = []
    for v, s in ((50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'),
                 (5, 'V'), (4, 'IV'), (1, 'I')):
        while n >= v:
            out.append(s)
            n -= v
    return ''.join(out)


ROMANS = [to_roman(i) for i in range(1, 81)]
BAD38 = set(("II III IV V VI VII VIII IX L LI LII LIV LV LVI LVII LVIII LIX LX "
             "LXI LXII LXIII LXIV LXV LXVI LXVII LXVIII LXIX LXX LXXI LXXII LXXIII "
             "LXXIV LXXV LXXVI LXXVII LXXVIII LXXIX LXXX").split())
assert len(BAD38) == 38, f'BAD38 count {len(BAD38)} != 38'


def load2d(path):
    """gtmodel ccube (nE, 600, 600) -> E축 합산 2D (lat, lon)."""
    with fits.open(path) as h:
        d = np.asarray(h[0].data, dtype=np.float64)
    return d.sum(axis=0) if d.ndim == 3 else d


def pearson(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else float('nan')


rows, flipped, missing = [], [], []
for M in ROMANS:
    cluster = 'BAD38' if M in BAD38 else 'good42'
    model_flipped = False
    for c in COMPS:
        p12 = os.path.join(D12, f'GC_{c}_model{M}_12yr_front_clean.fits')
        p17 = os.path.join(D17, f'GC_{c}_model{M}_17yr_front_clean.fits')
        if not (os.path.exists(p12) and os.path.exists(p17)):
            rows.append([M, c, '', '', 'MISSING', cluster])
            missing.append(f'{M}:{c}')
            continue
        a = load2d(p12)
        b = load2d(p17)
        corr = pearson(a, b)
        corrM = pearson(a, b[:, ::-1])           # GLON(마지막 축) mirror
        st = 'FLIPPED' if corrM > corr else 'OK'
        rows.append([M, c, f'{corr:.6f}', f'{corrM:.6f}', st, cluster])
        if st == 'FLIPPED':
            model_flipped = True
    if model_flipped:
        flipped.append(M)

with open('verify_mirrorfix_scope.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['model', 'comp', 'corr', 'corrM', 'status', 'cluster'])
    w.writerows(rows)

flip = set(flipped)
print(f'\n== verify_mirrorfix_scope: FLIPPED {len(flip)}개 모델, '
      f'MISSING {len(missing)}건 ==')
if flip:
    print('  FLIPPED:', ' '.join(sorted(flip, key=ROMANS.index)))
if missing:
    head = ' '.join(missing[:12])
    print(f'  MISSING: {head}{" ..." if len(missing) > 12 else ""}')

only_flip = sorted(flip - BAD38, key=ROMANS.index)
only_bad = sorted(BAD38 - flip, key=ROMANS.index)
print(f'  FLIPPED∖BAD38 = {only_flip if only_flip else "∅"}')
print(f'  BAD38∖FLIPPED = {only_bad if only_bad else "∅"}')
print('  기대: 재빌드 전 = 대칭차 양쪽 ∅ (FLIPPED==BAD38, 도구 교정 통과)')
print('        재빌드 후 = FLIPPED 0건')
print('  상세: verify_mirrorfix_scope.csv')
sys.exit(1 if flip else 0)
