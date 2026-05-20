#!/usr/bin/env python3
"""lnL at chain-median params — 'median을 썼다면' 대조 열 (polish 모듈 재사용)."""
import importlib, os, sys
import numpy as np

for name in ('polish_peak_17yr_v2', 'polish_peak_12yr'):
    if os.path.exists(name + '.py'):
        P = importlib.import_module(name)
        break
else:
    sys.exit('[FATAL] polish 모듈이 현재 디렉토리에 없음')

models = sys.argv[1:] or ['I', 'X', 'XV', 'XLVIII', 'XLIX', 'LIII']
S = P.load_shared()
rows = []
print(f'{"model":>8} {"bin":>3} {"lnL_max(chain)":>16} {"lnL_at_median":>16} {"max−median":>11}')
for m in models:
    p = P.NPZ_PATTERN.format(m=m)
    if not os.path.exists(p):
        print(f'[SKIP] {m}: npz 없음'); continue
    z = np.load(p)
    lnL = z['max_likelihood'] if 'max_likelihood' in z.files else z['max_lhd']
    if 'fitted_params_median' in z.files:
        Pm = z['fitted_params_median']
    else:
        Pm = 0.5 * (z['fitted_params_lower'] + z['fitted_params_upper'])
    tm = td = 0.0
    for j in range(lnL.shape[0]):
        lh = P.Likelihood(m, j, S)
        lmed = -0.5 * lh.likelihood_constrained(Pm[:, j])
        print(f'{m:>8} {j:>3} {lnL[j]:>16.3f} {lmed:>16.3f} {lnL[j]-lmed:>11.3f}')
        tm += lnL[j]; td += lmed
    rows.append((m, tm, td))
print(f'\n{"model":>8} {"Σ lnL_max":>16} {"Σ lnL_median":>16} {"ΔΣ":>11}')
for m, a, b in rows:
    print(f'{m:>8} {a:>16.2f} {b:>16.2f} {a-b:>11.3f}')
