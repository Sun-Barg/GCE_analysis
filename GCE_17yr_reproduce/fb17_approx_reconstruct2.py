#!/usr/bin/env python3
# ============================================================================
# fb17_approx_reconstruct2.py — approx/GDE 정체 EXACT 재현 검증
#                                        [단일소스 fb17-diag-2, v2 — 2026-07-30]
# v1 → v2 (hunt 리포트 판독 반영 — 대상·입력·후보 전면 교체):
#   확정 레시피(빌더 코드 실증 + Cholis 2112.09706 §VI.A-B, Eq.15-18 대조):
#     full_covariance  = shifted-ROI flux(l{roi} .dat, col1)의 1/N 모집단 공분산
#     approx_covariance = full의 eig → PCA top-k 절단 (rank 지문 3, 일부 4)
#     GDE_covariance   = 80 GDE-모델 center flux의 np.cov(ddof=1)
#   본 스크립트 = 위 레시피의 EXACT 재현 대조 + 규약 미세부(정렬·ROI 부분집합·
#   np.real·k)를 후보 스캔으로 결정. 판정은 리포트 판독 후.
# 절 구성:
#   A. 소스 덤프 — covariance_matrix_calculation.ipynb 전 코드셀 /
#      Codex covariance.py 전문 / χ² 체크포인트의 GDE·combined 셀 전문
#   B. l-ROI .dat 인벤토리 (GC_analysis 트리 워크)
#   C. approx/full 재구성 대조 (16yr front·front_back × I·X, 12yr 레거시)
#   D. GDE 재구성 대조 (center 80모델 세트 발견 시)
#   E. 우리 fb raw(22-ROI)의 고유스펙트럼·top-3 절단 미리보기 (§3-3 후보 D 예비)
# read-only(Sanghwan 트리) / 산출 = ./fb17_diag/approx_reconstruct2_report_*.txt
# 실행: GCE_17yr_reproduce 에서  python3 fb17_approx_reconstruct2.py
# ============================================================================
import os
import re
import json
import glob
import time

import numpy as np

SW = '/home/sanghwan/FermiLAT'
CDIR = SW + '/Sanghwan/GC_analysis/covariance'
GCA = SW + '/Sanghwan/GC_analysis'
CODEX = (SW + '/Codex_files/codex_codes/GCE/gce12_return_templates_pipeline'
         '/gce12_return_templates/covariance.py')
CHI_NB = (SW + '/Fermi-LAT-GCE-Analysis/Fermi-LAT-Chi-Square'
          '/.ipynb_checkpoints/GCE_chi_square_fitting-checkpoint.ipynb')
CALC_NB = CDIR + '/covariance_matrix_calculation.ipynb'
OUR_FB_RAW = './results_cov_fb17/GCE_systematic_covariance_matrix_17yr.npz'
OUR_FR_DIR = './results_cov_17yr/GCE_systematic_covariance_matrix_17yr.npz'

OUT_DIR = './fb17_diag'
os.makedirs(OUT_DIR, exist_ok=True)
_ts = time.strftime('%Y%m%d_%H%M')
REP_PATH = OUT_DIR + '/approx_reconstruct2_report_' + _ts + '.txt'
_rep = open(REP_PATH, 'w')


def W(*a):
    print(*a, file=_rep)


W('=' * 78)
W('fb17 approx/GDE reconstruct v2 — %s' % time.strftime('%Y-%m-%d %H:%M'))
W('레시피 전제(검증 대상): full=1/N 모집단(Eq.15) | approx=PCA top-k(Eq.16-17)')
W('  | GDE=np.cov(80모델 center, ddof=1). EXACT 재현으로 확정한다.')
W('=' * 78)

# ============================== A. 소스 덤프 ==============================
W('\n' + '=' * 78)
W('== A. 빌더 소스 덤프 ==')


def dump_nb(path, title, marker_any=None, cap_cell=6000, cap_cells=40):
    W('\n-- %s --' % title)
    W('   %s' % path)
    try:
        with open(path) as f:
            nb = json.load(f)
    except Exception as e:
        W('   [열기 실패] %s' % str(e)[:80])
        return
    n_out = 0
    for ci, c in enumerate(nb.get('cells', [])):
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c.get('source', []))
        if marker_any is not None and not any(m in src for m in marker_any):
            continue
        n_out += 1
        if n_out > cap_cells:
            W('   [셀 %d개 초과 — 절단]' % cap_cells)
            break
        W('   == cell %d (exec_count=%s) ==' % (ci, str(c.get('execution_count'))))
        body = src if len(src) <= cap_cell else src[:cap_cell] + '\n... [절단]'
        for ln in body.split('\n'):
            W('   | %s' % ln[:220])


dump_nb(CALC_NB, 'A1. covariance_matrix_calculation.ipynb (전 코드셀)')
W('\n-- A2. Codex covariance.py (전문) --')
try:
    for i, ln in enumerate(open(CODEX, 'r', errors='replace')):
        W('   %4d| %s' % (i + 1, ln.rstrip()[:220]))
except Exception as e:
    W('   [열기 실패] %s' % str(e)[:80])
dump_nb(CHI_NB, 'A3. chi_square 체크포인트 — GDE/combined/likelihood 관련 셀',
        marker_any=['GDEs_combined', 'flux_values', 'GDE_covariance',
                    'sorted_data', 'flux_average'],
        cap_cell=6000, cap_cells=25)

# ============================== B. l-ROI 인벤토리 ==========================
W('\n' + '=' * 78)
W('== B. l-ROI .dat 인벤토리 (GC_analysis 트리) ==')
_l_rx = re.compile(r'^GCE_model_([A-Za-z]+).*?_l(-?\d+)\.dat$')
lsets = {}
center_cand = {}
_c_rx = re.compile(r'^GCE_model_([IVXLCDM]+)(_front(?:_back)?)_16yr\.dat$')
for root, dirs, files in os.walk(GCA, followlinks=False):
    for fn in files:
        m = _l_rx.match(fn)
        if m:
            bn = fn
            model = m.group(1)
            lval = int(m.group(2))
            ftag = ('front_back' if 'front_back' in bn
                    else ('front' if 'front' in bn else ''))
            ytag = ('16yr' if '16yr' in bn else
                    ('15yr' if '15yr' in bn else 'noyr'))
            key = (model, ftag, ytag)
            lsets.setdefault(key, {}).setdefault(lval, []).append(
                os.path.join(root, fn))
            continue
        m = _c_rx.match(fn)
        if m:
            key = m.group(2).lstrip('_')
            center_cand.setdefault(key, []).append(os.path.join(root, fn))
for key in sorted(lsets):
    lv = sorted(lsets[key])
    dup = sum(1 for l in lsets[key] if len(lsets[key][l]) > 1)
    W('  model=%-4s front=%-10s yr=%-5s  N_l=%d  dup경로=%d' % (
        key[0], key[1] or '-', key[2], len(lv), dup))
    W('     l = %s' % lv)
    W('     예: %s' % lsets[key][lv[0]][0])
W('\n  [center 후보: GCE_model_{M}_front(_back)_16yr.dat, _l·_cholis 없음]')
for k in sorted(center_cand):
    W('    %-10s : %d개' % (k, len(center_cand[k])))
    if center_cand[k]:
        W('       예: %s' % center_cand[k][0])


# ============================== 공통 유틸 =================================
def compare(Ct, Cc):
    st = np.sqrt(np.clip(np.diag(Ct).real, 0, None))
    sc = np.sqrt(np.clip(np.diag(Cc).real, 0, None))
    ok = (st > 0) & (sc > 0)
    if ok.sum() < 2:
        return None
    ratio = st[ok] / sc[ok]
    Rt = Ct.real / np.outer(np.where(st > 0, st, 1), np.where(st > 0, st, 1))
    Rc = Cc.real / np.outer(np.where(sc > 0, sc, 1), np.where(sc > 0, sc, 1))
    off = ~np.eye(len(Ct), dtype=bool)
    sub = np.outer(ok, ok) & off
    dR = np.abs(Rt - Rc)[sub]
    dC = np.abs(Ct.real - Cc.real).max()
    fro = (np.linalg.norm(Ct.real - Cc.real)
           / max(np.linalg.norm(Ct.real), 1e-300))
    return dict(r_med=float(np.median(ratio)), r_min=float(ratio.min()),
                r_max=float(ratio.max()), dR_max=float(dR.max()),
                fro=float(fro), dC_max=float(dC))


def flag(met):
    uni = (met['r_max'] / max(met['r_min'], 1e-300) - 1.0) < 1e-6
    if met['dR_max'] < 1e-8 and uni and abs(met['r_med'] - 1) < 1e-6:
        return 'EXACT'
    if met['fro'] < 1e-6:
        return 'EXACT(fro)'
    if met['dR_max'] < 1e-6 and uni:
        return 'R-EXACT, sigma x %.6g' % met['r_med']
    if met['dR_max'] < 0.02:
        return 'R-approx(<0.02)'
    return 'MISMATCH'


def emp_1overN(F):
    """F: (nbins, N). Eq.15 estimator: E[xy]-E[x]E[y] (1/N)."""
    mu = F.mean(axis=1)
    prod = (F[:, None, :] * F[None, :, :]).mean(axis=2)
    return prod - np.outer(mu, mu)


def topk_eigh(C, k):
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1][:k]
    return (V[:, idx] * w[idx]) @ V[:, idx].T


def topk_eig_asis(C, k):
    w, V = np.linalg.eig(C)
    w = np.real(w)
    V = np.real(V)
    idx = np.argsort(w)[::-1][:k]
    return (V[:, idx] * w[idx]) @ V[:, idx].T


def load_lset(key, lfilter=None):
    grp = lsets[key]
    lv = sorted(grp)
    if lfilter is not None:
        lv = [l for l in lv if lfilter(l)]
    E0, cols, used = None, [], []
    for l in lv:
        p = sorted(grp[l])[0]
        try:
            d = np.loadtxt(p)
        except Exception:
            continue
        if d.ndim != 2 or d.shape[1] < 2:
            continue
        if E0 is None:
            E0 = d[:, 0]
        if d.shape[0] != len(E0):
            continue
        cols.append(d[:, 1])
        used.append(l)
    if not cols:
        return None, None, []
    return E0, np.array(cols).T, used


def tload(path):
    try:
        C = np.load(path)
    except Exception:
        return None
    return C


# ============================== C. approx/full 대조 ========================
W('\n' + '=' * 78)
W('== C. approx/full 재구성 대조 ==')
console = []
jobs = []
for (model, ftag, ytag) in sorted(lsets):
    if model not in ('I', 'X'):
        continue
    if ytag == '16yr' and ftag in ('front', 'front_back'):
        sfx = 'front_back' if ftag == 'front_back' else 'front'
        jobs.append(((model, ftag, ytag), [
            ('approx_covariance_17x17_%s_model_%s_16yr.npy' % (sfx, model), 17),
            ('approx_covariance_14x14_%s_model_%s_16yr.npy' % (sfx, model), 14),
            ('approx_covariance_17x17_%s_16yr.npy' % sfx, 17),
        ]))
    if ytag == 'noyr' and ftag == '' and model == 'I':
        jobs.append(((model, ftag, ytag), [
            ('approx_covariance_14x14_12yr.npy', 14),
            ('approx_covariance_17x17_12yr.npy', 17),
            ('full_covariance_matrix_14x14_12yr.npy', 14),
        ]))

for key, tlist in jobs:
    W('\n--- l-set %s ---' % str(key))
    subsets = [('all', None), ('|l|>=20', lambda l: abs(l) >= 20),
               ('|l|>=20,l!=0', lambda l: abs(l) >= 20 and l != 0)]
    built = {}
    for sname, sf in subsets:
        E0, F, used = load_lset(key, sf)
        if F is None:
            continue
        built[sname] = (E0, F, used)
        W('  subset %-14s N_roi=%d  bins=%d  l=%s' % (
            sname, F.shape[1], F.shape[0], used))
    for tname, nb in tlist:
        tp = os.path.join(CDIR, tname)
        Ct = tload(tp)
        if Ct is None:
            W('  [타깃 없음] %s' % tname)
            continue
        cplx = np.iscomplexobj(Ct)
        W('  ### %s  shape=%s dtype=%s%s' % (
            tname, str(Ct.shape), Ct.dtype, ' [COMPLEX]' if cplx else ''))
        best = None
        for sname in built:
            E0, F, used = built[sname]
            Fb = F[:nb, :] if F.shape[0] >= nb else None
            if Fb is None:
                continue
            emp = emp_1overN(Fb)
            cands = {'full(1/N)': emp}
            for k in (2, 3, 4, 5, 6):
                cands['eigh top%d' % k] = topk_eigh(emp, k)
            for k in (3, 4):
                cands['eig-asis top%d' % k] = topk_eig_asis(emp, k)
            for cname, Cc in cands.items():
                if Cc.shape != Ct.shape:
                    continue
                met = compare(np.real(Ct), Cc)
                if met is None:
                    continue
                lab = '%s | %s' % (sname, cname)
                W('    [%-32s] dR_max=%.3g sr(med)=%.5g fro=%.3g' % (
                    lab, met['dR_max'], met['r_med'], met['fro']))
                score = (met['fro'], met['dR_max'])
                if best is None or score < best[0]:
                    best = (score, lab, met)
        if best is not None:
            _, lab, met = best
            fl = flag(met)
            W('    >>> best = [%s]  %s' % (lab, fl))
            console.append('%-52s -> [%s] fro=%.3g %s' % (
                tname, lab, met['fro'], fl))
        else:
            console.append('%-52s -> 후보 없음' % tname)

# ============================== D. GDE 대조 ================================
W('\n' + '=' * 78)
W('== D. GDE(80모델 산포) 재구성 대조 ==')
for ftag in ('front', 'front_back'):
    fl_ = center_cand.get(ftag, [])
    W('\n  [%s] center .dat 발견 %d개' % (ftag, len(fl_)))
    if len(fl_) < 60:
        W('   → 60 미만: GDE 재구성 보류 (A3 덤프에서 flux_values 소스 확인)')
        continue
    E0, cols, names = None, [], []
    for p in sorted(fl_):
        try:
            d = np.loadtxt(p)
        except Exception:
            continue
        if d.ndim != 2 or d.shape[1] < 2:
            continue
        if E0 is None:
            E0 = d[:, 0]
        if d.shape[0] != len(E0):
            continue
        cols.append(d[:, 1])
        names.append(os.path.basename(p))
    F = np.array(cols).T
    W('   적재 N=%d, bins=%d' % (F.shape[1], F.shape[0]))
    for tname in ('GDE_covariance_matrix_%s_16yr.npy' % ftag,
                  'GDE_covariance_matrix_%s_17x17_16yr.npy' % ftag,
                  'GDE_covariance_matrix_%s_14x14_16yr.npy' % ftag):
        Ct = tload(os.path.join(CDIR, tname))
        if Ct is None:
            continue
        nb = Ct.shape[0]
        for cname, Cc in [('np.cov ddof1', np.cov(F[:nb], ddof=1)),
                          ('np.cov ddof0', np.cov(F[:nb], ddof=0)),
                          ('cov17[:%d]sl ddof1' % nb,
                           np.cov(F, ddof=1)[:nb, :nb])]:
            met = compare(np.real(Ct), Cc)
            if met is None:
                continue
            W('   %s [%-18s] dR_max=%.3g sr(med)=%.5g fro=%.3g  %s' % (
                tname, cname, met['dR_max'], met['r_med'], met['fro'],
                flag(met)))

# ============================== E. 우리 fb raw 미리보기 =====================
W('\n' + '=' * 78)
W('== E. 우리 raw(22-ROI)의 고유스펙트럼·top-3 절단 미리보기 (후보 D 예비) ==')
for lab, p in [('fb17 raw', OUR_FB_RAW), ('front 운영 raw계보(F5 플래그)',
                                          OUR_FR_DIR)]:
    if not os.path.exists(p):
        W('  [%s] 없음: %s' % (lab, p))
        continue
    z = np.load(p)
    C = z['cov_matrix']
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    wsum = w.sum()
    W('\n  [%s] %s  shape=%s' % (lab, p, str(C.shape)))
    W('    w_i/Σw (상위 6) = %s' % ' '.join('%.4f' % (x / wsum) for x in w[:6]))
    W('    누적 w(1..3) = %.4f  (Cholis 기준 ≈0.99 대비)' % (w[:3].sum() / wsum))
    C3 = (V[:, :3] * w[:3]) @ V[:, :3].T
    s_r = np.sqrt(np.diag(C))
    s_3 = np.sqrt(np.clip(np.diag(C3), 0, None))
    W('    sigma raw    = %s' % ' '.join('%.2e' % v for v in s_r))
    W('    sigma top3   = %s' % ' '.join('%.2e' % v for v in s_3))
    W('    top3/raw     = %s' % ' '.join('%.3f' % v
                                         for v in (s_3 / s_r)))
    off = ~np.eye(len(C), dtype=bool)
    R3 = C3 / np.outer(np.where(s_3 > 0, s_3, 1), np.where(s_3 > 0, s_3, 1))
    W('    top3 R off-diag min/mean/max = %.4f/%.4f/%.4f' % (
        R3[off].min(), R3[off].mean(), R3[off].max()))

W('\n' + '=' * 78)
W('주의: C·D의 EXACT/MISMATCH 는 스캔한 후보군(estimator·정렬·부분집합·k) 한정')
W('  지표. MISMATCH 시 A절 덤프(빌더 실제 코드)로 규약을 교정해 재시도한다.')
W('  E절은 §3-3 후보 (D) 예비 수치일 뿐 — cov 구성 결정은 해밝 판정 사항.')
_rep.close()

print('[recon2] C절 대조 결과:')
for ln in console:
    print('   ', ln)
print('[recon2] 리포트: %s' % REP_PATH)
print('[recon2] D절(GDE)·E절(우리 raw 미리보기)은 리포트에서 확인.')
