#!/usr/bin/env python3
"""c_GCE 0-경계 접촉 점검 — 기존 npz만 읽음 (도메인 차이 활성 후보 스크리닝)."""
import glob, re, numpy as np

OURS_WORSE  = ['XLIX','L','LX','LIX','LIV','LVIII','XVI','XXXI','XXVI']
OURS_BETTER = ['LXXIV','LXXV','LXXVI','LXXVII','LXXVIII','LXXII','III']

rows = []
for p in sorted(glob.glob('./results_17yr/GCE_model_*_front_17yr_cholis_fit.npz')):
    m = re.search(r'GCE_model_(.+?)_front', p).group(1)
    z = np.load(p)
    cmap  = z['fitted_params'][2]         # (14,) MAP c_GCE
    cmed  = z['fitted_params_median'][2]
    clo   = z['fitted_params_lower'][2]   # 16th pct
    rows.append(dict(m=m,
                     n_map_1e3=int(np.sum(cmap < 1e-3)),
                     n_map_1e2=int(np.sum(cmap < 1e-2)),
                     n_lo16_1e3=int(np.sum(clo < 1e-3)),
                     min_map=float(cmap.min()), min_med=float(cmed.min())))

rows.sort(key=lambda r: (-r['n_map_1e2'], r['min_map']))
tag = lambda m: 'W' if m in OURS_WORSE else ('B' if m in OURS_BETTER else ' ')
print(f'{"model":>8} {"lst":>3} {"MAP<1e-3":>9} {"MAP<1e-2":>9} {"lo16<1e-3":>10} {"min MAP":>10} {"min med":>10}')
hit = 0
for r in rows:
    if r['n_map_1e2'] or r['n_lo16_1e3']:
        hit += 1
        print(f'{r["m"]:>8} {tag(r["m"]):>3} {r["n_map_1e3"]:>9} {r["n_map_1e2"]:>9} '
              f'{r["n_lo16_1e3"]:>10} {r["min_map"]:>10.4f} {r["min_med"]:>10.4f}')
print(f'\n[{hit}/{len(rows)}] 모델에 경계 접촉 bin 존재 (미표시 = 접촉 없음)')
print('W = ours-worse 목록, B = ours-better 목록')
for r in rows:
    if r['m'] in ('XLIX','X','I'):
        print(f'  기준 확인 {r["m"]:>5}: min MAP c_GCE = {r["min_map"]:.4f} (전 bin)')
