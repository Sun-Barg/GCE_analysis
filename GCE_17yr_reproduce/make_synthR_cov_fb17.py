#!/usr/bin/env python3
# make_synthR_cov_fb17.py — fb17 SYNTH cov: our-sigma(fb raw) + Sanghwan-R(fb).
# v1.1 (2026-07-29): 도너 선택 model_X 최우선(알파벳순 model_I 선점 결함 수정)
#   + 전 후보 R off-diag 통계 출력.
#
# 원 레시피(make_synthR_cov_and_verify.py, 2026-06-04) 평행:
#   C_synth = diag(sig_our_fb) @ R_donor @ diag(sig_our_fb)
# R 도너 우선순위(원 레시피가 approx_covariance를 썼으므로 approx 우선):
#   1) approx_covariance_*front_back*17*.npy  (Sanghwan 운영 상당)
#   2) GDE_covariance_matrix_front_back_17x17_16yr.npy (명시 폴백 — R 구조 상이)
# 출력: results_cov_fb17_synthR/GCE_systematic_covariance_matrix_17yr.npz
# 사용 규칙(원 handoff 동일): 역행렬은 반드시 diag(stat**2) 합산 후 —
#   systematic 단독 역행렬 금지(조건수 폭주).
# 부가 출력: sigma 3원 표 (fb_raw / front_raw_postfix / front_synthR[=pre-widefix σ])
import os
import sys
import glob
import numpy as np

WORK = os.path.expanduser('~/GCE-Chi-square-fitting/GCE_17yr_reproduce')
FB_RAW_NPZ = f'{WORK}/results_cov_fb17/GCE_systematic_covariance_matrix_17yr.npz'
OUT_DIR    = f'{WORK}/results_cov_fb17_synthR'
OUT_NPZ    = f'{OUT_DIR}/GCE_systematic_covariance_matrix_17yr.npz'
FRONT_RAW_NPZ   = f'{WORK}/results_cov_17yr/GCE_systematic_covariance_matrix_17yr.npz'
FRONT_SYNTH_NPZ = f'{WORK}/results_cov_17yr_synthR/GCE_systematic_covariance_matrix_17yr.npz'

DONOR_DIRS = [
    os.path.expanduser('~/GCE-Chi-square-fitting/Cov'),
    '/home/sanghwan/FermiLAT/Sanghwan/GC_analysis/covariance',
    '/home/sanghwan/FermiLAT/Fermi-LAT-GCE-Analysis/Fermi-LAT-Chi-Square/Cov',
]


def _off_stats(C):
    s = np.sqrt(np.diag(C))
    R = C / np.outer(s, s)
    o = R[~np.eye(len(C), dtype=bool)]
    return float(o.min()), float(o.mean()), float(o.max())


def _find_donor():
    """approx fb 17x17 (model_X 최우선) → GDE fb 폴백. 전 후보 R 통계 출력.

    v1.1: glob 알파벳순으로 model_I가 먼저 잡히던 결함 수정 — front synthR이
    model_X approx를 썼으므로 레시피 평행상 model_X 우선."""
    cands = []
    for d in DONOR_DIRS:
        cands += sorted(glob.glob(f'{d}/approx_covariance*front_back*.npy'))
    cands.sort(key=lambda p: ('model_X' not in os.path.basename(p), p))
    print('[donor 후보] (선택 우선순위순)')
    valid = []
    for p in cands:
        try:
            C = np.load(p)
            if C.shape != (17, 17):
                print(f'  - {p}  shape={C.shape} (제외)')
                continue
            mn, me, mx = _off_stats(C)
            print(f'  - {p}  R off-diag {mn:.4f}/{me:.4f}/{mx:.4f}')
            valid.append(p)
        except Exception as e:
            print(f'  - {p}  로드 실패: {e}')
    if valid:
        return valid[0], 'approx'
    for d in DONOR_DIRS:
        p = f'{d}/GDE_covariance_matrix_front_back_17x17_16yr.npy'
        if os.path.exists(p):
            return p, 'GDE(폴백)'
    return None, None


def main():
    assert os.path.exists(FB_RAW_NPZ), f'fb raw cov 없음: {FB_RAW_NPZ}'
    z = np.load(FB_RAW_NPZ)
    sig_fb = np.sqrt(np.diag(z['cov_matrix']))
    E, dE  = z['E'], z['delta_E']
    assert len(E) == 17

    donor, kind = _find_donor()
    assert donor is not None, ('R 도너 없음 — approx_covariance*front_back*17*.npy / '
                               'GDE_covariance_matrix_front_back_17x17_16yr.npy 부재')
    C_s = np.load(donor)
    s_s = np.sqrt(np.diag(C_s))
    R   = C_s / np.outer(s_s, s_s)
    R   = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    off = R[~np.eye(17, dtype=bool)]
    print(f'[donor] {kind}: {donor}')
    print(f'        R off-diag min/mean/max = {off.min():.4f} / '
          f'{off.mean():.4f} / {off.max():.4f}')
    if kind != 'approx':
        print('        [주의] approx fb 부재 → GDE fb R 사용. 원 레시피(approx, '
              '준-균일 0.94)와 상관 구조가 다르므로 결과 해석 시 명시할 것.')

    C_synth = np.outer(sig_fb, sig_fb) * R

    # 구조 검증
    assert np.allclose(np.diag(C_synth), sig_fb ** 2, rtol=1e-12)
    assert np.allclose(C_synth, C_synth.T, rtol=1e-12, atol=0)
    eig = np.linalg.eigvalsh(C_synth)
    print(f'[synth] diag=σ_fb² 보존 OK, 대칭 OK, eig=[{eig.min():.3e}, '
          f'{eig.max():.3e}], cond={np.linalg.cond(C_synth):.2e}')
    print('        (사용 규칙: 역행렬은 반드시 diag(stat²) 합산 후 — 단독 역행렬 금지)')

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        OUT_NPZ,
        cov_matrix = C_synth,
        sigma_sys  = sig_fb,
        E          = E,
        delta_E    = dE,
        rois       = z['rois'],
        n_rois_used= z['n_rois_used'],
        mean_GCE   = z['mean_GCE'],   # fb raw 유래(참고용; SYNTH와 무관한 vestigial)
        provenance = np.array(
            f'SYNTH-fb17: diag(sigma_sys[fb raw 22-ROI]) @ corr({os.path.basename(donor)}'
            f' [{kind}]) @ diag(sigma_sys); 2026-07-29'),
    )
    print(f'[saved] {OUT_NPZ}')

    # sigma 3원 표: fb_raw / front_raw(post-widefix) / front_synthR(σ=pre-widefix)
    have_fr  = os.path.exists(FRONT_RAW_NPZ)
    have_fs  = os.path.exists(FRONT_SYNTH_NPZ)
    s_fr = np.sqrt(np.diag(np.load(FRONT_RAW_NPZ)['cov_matrix'])) if have_fr else None
    s_fs = np.sqrt(np.diag(np.load(FRONT_SYNTH_NPZ)['cov_matrix'])) if have_fs else None
    print('\n[sigma 3원 표] (겹침 14빈; 비율은 fb_raw 기준)')
    hdr = f'  {"bin":>3} {"E[GeV]":>8} {"fb_raw":>10}'
    hdr += f' {"fr_raw":>10} {"fb/fr_raw":>9}' if have_fr else ''
    hdr += f' {"fr_synthR":>10} {"fb/synthR":>9}' if have_fs else ''
    print(hdr)
    for i in range(14):
        row = f'  {i:>3} {E[i]:>8.3f} {sig_fb[i]:>10.3e}'
        if have_fr:
            row += f' {s_fr[i]:>10.3e} {sig_fb[i] / s_fr[i]:>9.2f}'
        if have_fs:
            row += f' {s_fs[i]:>10.3e} {sig_fb[i] / s_fs[i]:>9.2f}'
        print(row)
    if have_fr:
        r = sig_fb[:14] / s_fr
        print(f'  fb_raw/front_raw(post-fix) 중앙값 = {np.median(r):.3f}  '
              f'<- 진짜 raw–raw 대각 비교')
    return 0


if __name__ == '__main__':
    sys.exit(main())
