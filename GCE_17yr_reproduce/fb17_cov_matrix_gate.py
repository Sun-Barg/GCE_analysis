#!/usr/bin/env python3
"""fb17_cov_matrix_gate.py — 블록 4 조립본(17×17 cov) 검증 게이트.

하드 게이트(하나라도 실패 시 exit 1):
  E=17빈·중심값 일치, n_rois_used=22, 대칭성, diag>0, 최소고유값≥-수치오차,
  |corr|≤1, sigma_sys 유한·양수, inv_cov 유한.
정보 출력(판정 없음):
  cond, 겹침 14빈 sigma_sys 비율(fb/front), 빈별 sigma_sys/|mean| 표.
실행: cd GCE_17yr_reproduce && python3 fb17_cov_matrix_gate.py
"""
import os
import sys
import numpy as np

FB = 'results_cov_fb17/GCE_systematic_covariance_matrix_17yr.npz'
FR_CANDS = ['results_cov_17yr_synthR/GCE_systematic_covariance_matrix_17yr.npz',
            'results_cov_17yr/GCE_systematic_covariance_matrix_17yr.npz']

ok = True
def fail(msg):
    global ok
    ok = False
    print(f'  FAIL {msg}')

def chk(cond, name):
    print(f'  {"PASS" if cond else "FAIL"}  {name}')
    if not cond:
        fail(name + ' (상세 위 참조)')

assert os.path.exists(FB), f'fb17 cov npz 없음: {FB} — build_cov_matrix 먼저'
z = np.load(FB)
C   = z['cov_matrix']
Ci  = z['inv_cov_matrix']
E   = z['E']
sig = z['sigma_sys']
mu  = z['mean_GCE']
nroi = int(z['n_rois_used'])
rois = np.array(z['rois'])

edges = np.array([0.274698, 0.357, 0.464, 0.603, 0.784, 1.02, 1.32, 1.72,
                  2.24, 2.91, 3.78, 4.91, 10.8, 23.7, 51.9312,
                  114.935, 252.809, 556.077])
cen = np.sqrt(edges[:-1] * edges[1:])

print(f'[fb17-cov-gate] {FB}')
chk(C.shape == (17, 17), f'cov shape {C.shape} == (17, 17)')
chk(len(E) == 17 and np.allclose(E, cen, rtol=1e-6), 'E = fb17 17빈 중심값 일치')
chk(nroi == 22 and len(rois) == 22, f'n_rois_used={nroi}, len(rois)={len(rois)} == 22')
chk(bool(np.all(np.isfinite(C))) and bool(np.all(np.isfinite(Ci))),
    'cov·inv_cov 전 원소 유한')
chk(bool(np.allclose(C, C.T, rtol=1e-10, atol=0)), '대칭성 C == C.T')
chk(bool(np.all(np.diag(C) > 0)), 'diag(C) > 0')
chk(bool(np.all(np.isfinite(sig))) and bool(np.all(sig > 0)), 'sigma_sys 유한·양수')

eig = np.linalg.eigvalsh(C)
tol = -1e-10 * eig.max()
chk(bool(eig.min() >= tol), f'최소고유값 {eig.min():.3e} >= 수치허용 {tol:.1e}')

corr = C / np.outer(sig, sig)
chk(bool(np.max(np.abs(corr)) <= 1 + 1e-9),
    f'|corr| max = {np.max(np.abs(corr)):.6f} <= 1')

print(f'\n  [정보] cond = {float(z["cond_number"]):.2e}, '
      f'eig range = [{eig.min():.3e}, {eig.max():.3e}]')
print(f'  [정보] rois = {sorted(rois.tolist())}')

print(f'\n  {"bin":>3} {"E[GeV]":>8} {"mean_GCE":>11} {"sigma_sys":>11} {"frac":>7}')
for i in range(17):
    frac = sig[i] / abs(mu[i]) if mu[i] != 0 else float('inf')
    print(f'  {i:>3} {E[i]:>8.3f} {mu[i]:>11.3e} {sig[i]:>11.3e} {frac:>7.2f}')

fr_path = next((p for p in FR_CANDS if os.path.exists(p)), None)
if fr_path is not None:
    f = np.load(fr_path)
    sf, Ef = f['sigma_sys'], f['E']
    if len(Ef) == 14 and np.allclose(Ef, cen[:14], rtol=1e-6):
        r = sig[:14] / sf
        print(f'\n  [정보] 겹침 14빈 sigma_sys 비율 fb/front ({fr_path}):')
        print('   ' + ' '.join(f'{v:.2f}' for v in r)
              + f'   중앙값 {np.median(r):.3f}')
    else:
        print(f'\n  [정보] front cov 축 불일치({len(Ef)}빈) — 비율 생략')
else:
    print('\n  [정보] front cov npz 미발견 — 비율 생략')

print('\n[fb17-cov-gate] ' + ('ALL PASS' if ok else 'FAIL — 위 항목 보고'))
sys.exit(0 if ok else 1)
