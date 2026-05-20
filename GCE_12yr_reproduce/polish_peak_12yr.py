#!/usr/bin/env python3
"""
polish_peak_12yr.py — chain-max vs deterministic-peak diagnostic (12yr, 채택 config 12a)

목적: 17yr polish 와 동일한 estimator 진단을 12yr 채택 config
  (`_hisbub_noConstr` = his-bubble + Poisson-only) 에서 수행.
  per-bin lnL(chain argmax) 대비 결정론적 peak 의 Δ와 모델 간 차등을 측정.
  likelihood 는 12yr runner(run_one_model.py, 07-03 rank-toggle 패치본)와
  동일 구조: Poisson-only (USE_CONSTRAINT=0), bubble 은 BUBBLE_OVERRIDE_DIR,
  component 경로는 _tpl 규약 (NAT38 분기 포함).

환경 knob (runner 와 동일 의미; 기본값 = 채택 12a):
  RANK_SUFFIX          기본 '_hisbub_noConstr'
  USE_CONSTRAINT       기본 '0' (Poisson-only). '1' -> chi2_bub+chi2_iso 포함
  BUBBLE_OVERRIDE_DIR  hisbub 계열이면 필수 (refit 스크립트에서 값 복원):
                         grep -n "BUBBLE_OVERRIDE_DIR" mirrorfix_step2_refit.sh
  WORK_DIR_12          기본 './GC_analysis_DR2'
  MASK_VARIANT         기본 'DR2'
  NAT38                기본 '0' ('1' -> _38bin..._grp14 component)

안전장치: 각 모델 bin 0 에서 chain-argmax 지점 재계산 lnL 이 npz max_likelihood 와
  |Δ|>0.5 면 config knob 불일치로 판단하고 해당 모델 중단(+힌트). 1e-3 초과는 경고.

사용:
  cd ~/GCE-Chi-square-fitting/GCE_12yr_reproduce/
  conda activate fermi
  export BUBBLE_OVERRIDE_DIR=<refit 스크립트의 값>
  python polish_peak_12yr.py I X XV XLVIII XLIX LIII

Author: haebarg (2026), generated alongside Claude conversation.
"""

import os
import sys
import time
import csv
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.special import gammaln
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# ============================================================
# CONFIG — 12yr runner (rank-toggle 패치본) 와 동일 의미
# ============================================================
front       = '_front'
WORK_DIR    = os.environ.get('WORK_DIR_12', './GC_analysis_DR2').strip()
RESULTS_DIR = './results_12yr'
OUT_DIR     = './DIAG_polish_peak'

RANK_SUFFIX         = os.environ.get('RANK_SUFFIX', '_hisbub_noConstr').strip()
USE_CONSTRAINT      = (os.environ.get('USE_CONSTRAINT', '0').strip() != '0')
BUBBLE_OVERRIDE_DIR = os.environ.get('BUBBLE_OVERRIDE_DIR', '').strip()
NAT38               = (os.environ.get('NAT38', '0').strip() == '1')
MASK_VARIANT        = os.environ.get('MASK_VARIANT', 'DR2').strip()
PSC_MASK_FILE       = f'GC_mask_60x60_definitions_{MASK_VARIANT}.npy'

NPZ_PATTERN = RESULTS_DIR + '/GCE_model_{m}_front_12yr_cholis' + RANK_SUFFIX + '_fit.npz'
LHV_PATTERN = RESULTS_DIR + '/GCE_model_{m}_front_12yr_cholis' + RANK_SUFFIX + '_likelihood_value'

if ('hisbub' in RANK_SUFFIX) and (not BUBBLE_OVERRIDE_DIR):
    sys.exit('[FATAL] RANK_SUFFIX에 hisbub 포함인데 BUBBLE_OVERRIDE_DIR 미설정.\n'
             '        refit 스크립트에서 값 복원 후 export 하세요:\n'
             '          grep -n "BUBBLE_OVERRIDE_DIR\\|USE_CONSTRAINT\\|NAT38" mirrorfix_step2_refit.sh')

_MODEL = {'m': None}   # _tpl 이 참조

def _tpl(name, convol):
    """12yr runner 의 component 경로 규약 (bubble override + NAT38 분기)."""
    cv = '' if convol == 'yes' else '_no_convol'
    base = (f'GC_{name}_model{_MODEL["m"]}' if name in ('pion', 'bremss', 'ics')
            else f'GC_{name}_model')
    d = BUBBLE_OVERRIDE_DIR if (name == 'fermi_bubble' and BUBBLE_OVERRIDE_DIR) else WORK_DIR
    if NAT38:
        return f'{d}/{base}_38bin_12yr{front}_clean{cv}_grp14.fits'
    return f'{d}/{base}_12yr{front}_clean{cv}.fits'


# ============================================================
# Shared load — Poisson-only 면 ccube+mask 만 필요
# ============================================================

def roi_solid_angle(dl, db, b):
    return np.radians(dl) * np.radians(db) * np.cos(np.radians(b))


def log_factorial(O):
    return gammaln(np.asarray(O, dtype=float) + 1.0)


def load_shared():
    ccube_path = f'{WORK_DIR}/GC_ccube_12yr{front}_clean.fits'
    need = [ccube_path,
            f'{WORK_DIR}/Model/GC_disk_mask_60x60_definitions.npy',
            f'{WORK_DIR}/Model/{PSC_MASK_FILE}']
    if USE_CONSTRAINT:
        need += [f'{WORK_DIR}/GC_expcube_center_12yr{front}_clean.fits',
                 f'{WORK_DIR}/Model/bubble_constraints.txt',
                 f'{WORK_DIR}/Model/iso_constraints_full_err.txt']
    for p in need:
        if not os.path.exists(p):
            sys.exit(f'[FATAL] missing input: {p}')

    E_bounds = fits.open(ccube_path)[1].data
    nE = len(E_bounds)
    E = np.zeros(nE); delta_E = np.zeros(nE)
    for i in range(nE):
        E[i]       = np.sqrt(E_bounds[i][2] * E_bounds[i][1] * 1e-6) * 1e-3
        delta_E[i] = (E_bounds[i][2] - E_bounds[i][1]) * 1e-6

    disk_mask = np.load(f'{WORK_DIR}/Model/GC_disk_mask_60x60_definitions.npy')[100:500, 100:500]
    psc_mask  = np.load(f'{WORK_DIR}/Model/{PSC_MASK_FILE}')[:, 100:500, 100:500]

    S = dict(E=E, delta_E=delta_E, disk_mask=disk_mask, psc_mask=psc_mask, ster=None)

    if USE_CONSTRAINT:
        raw_map = fits.open(ccube_path)
        w = WCS(raw_map[0].header).dropaxis(2)
        width, height = np.shape(raw_map[0].data[0])
        ster = np.zeros([width, height])
        for i in range(height):
            for j in range(width):
                l, b = w.wcs_pix2world(j, i, 0)
                ster[i, j] = roi_solid_angle(0.1, 0.1, b)
        S['ster'] = ster
        bc = np.loadtxt(f'{WORK_DIR}/Model/bubble_constraints.txt')
        S['bubble_flux_data']        = interp1d(bc[:, 0], bc[:, 1], fill_value='extrapolate', kind='quadratic')(E)
        S['bubble_lower_error_data'] = interp1d(bc[:, 0], bc[:, 2], fill_value='extrapolate', kind='quadratic')(E)
        S['bubble_upper_error_data'] = interp1d(bc[:, 0], bc[:, 3], fill_value='extrapolate', kind='quadratic')(E)
        ic = np.loadtxt(f'{WORK_DIR}/Model/iso_constraints_full_err.txt')
        S['isotropic_flux_data']        = (E ** 2) * interp1d(ic[:, 0], ic[:, 1], fill_value='extrapolate', kind='quadratic')(E)
        S['isotropic_lower_error_data'] = (E ** 2) * interp1d(ic[:, 0], ic[:, 2], fill_value='extrapolate', kind='quadratic')(E)
        S['isotropic_upper_error_data'] = (E ** 2) * interp1d(ic[:, 0], ic[:, 3], fill_value='extrapolate', kind='quadratic')(E)
    return S


# ============================================================
# Likelihood — 12yr runner 자구 구조 (USE_CONSTRAINT 분기 = 패치와 동일)
# ============================================================

class Likelihood:
    def __init__(self, model, energy_bin, S):
        _MODEL['m'] = model
        self.model, self.energy_bin, self.S = model, energy_bin, S
        self.data = fits.open(f'{WORK_DIR}/GC_ccube_12yr{front}_clean.fits')[0].data[energy_bin, 100:500, 100:500]
        self.pion_bremss = (fits.open(_tpl('pion', 'yes'))[0].data[energy_bin, 100:500, 100:500]
                          + fits.open(_tpl('bremss', 'yes'))[0].data[energy_bin, 100:500, 100:500])
        self.ics    = fits.open(_tpl('ics', 'yes'))[0].data[energy_bin, 100:500, 100:500]
        self.GCE    = fits.open(_tpl('GCE', 'yes'))[0].data[energy_bin, 100:500, 100:500]
        self.bubble = fits.open(_tpl('fermi_bubble', 'yes'))[0].data[energy_bin, 100:500, 100:500]
        self.iso    = fits.open(_tpl('isotropic', 'yes'))[0].data[energy_bin, 100:500, 100:500]

        self.E, self.delta_E = S['E'], S['delta_E']
        if USE_CONSTRAINT:
            self.iso_no_convol    = fits.open(_tpl('isotropic', 'no'))[0].data[energy_bin, 100:500, 100:500]
            self.bubble_no_convol = fits.open(_tpl('fermi_bubble', 'no'))[0].data[energy_bin, 100:500, 100:500]
            self.exp_cube = (fits.open(f'{WORK_DIR}/GC_expcube_center_12yr{front}_clean.fits')[0].data[energy_bin]
                             * S['ster'])[100:500, 100:500]

        _psc = S['psc_mask'][energy_bin]
        self.full_mask = _psc * S['disk_mask']
        _obs = self.data[self.full_mask == 1].astype(float)
        self.observed_log_factorial_masked = log_factorial(_obs)

    def likelihood_constrained(self, parameter_set):
        S = self.S
        pion_bremss_param, ics_param, GCE_param, bubble_param, isotropic_param = parameter_set
        expected_pixel = (pion_bremss_param * self.pion_bremss
                          + ics_param       * self.ics
                          + GCE_param       * self.GCE
                          + isotropic_param * self.iso
                          + bubble_param    * self.bubble)
        observed_pixel = self.data[self.full_mask == 1]
        expected_pixel = expected_pixel[self.full_mask == 1]
        if (expected_pixel < 0).any():
            return np.inf
        observed_log_expected = observed_pixel * np.log(expected_pixel)
        lhd = 2 * (expected_pixel - observed_log_expected + self.observed_log_factorial_masked)
        if not USE_CONSTRAINT:
            return np.sum(lhd)                      # 12a: Poisson-only

        isotropic = (np.sum(self.full_mask * (self.iso_no_convol) / self.exp_cube)
                     * isotropic_param / np.sum(self.full_mask))
        isotropic_sed = (self.E[self.energy_bin] ** 2) * isotropic / (self.delta_E[self.energy_bin])
        bubble = (np.sum(self.full_mask * (self.bubble_no_convol) / self.exp_cube)
                  * bubble_param / np.sum(self.full_mask))
        bubble_sed = (self.E[self.energy_bin] ** 2) * bubble / (self.delta_E[self.energy_bin])

        larger_error = max([S['bubble_upper_error_data'][self.energy_bin],
                            S['bubble_lower_error_data'][self.energy_bin]])
        if S['bubble_flux_data'][self.energy_bin] < bubble_sed:
            chi2_bubble = ((bubble_sed - S['bubble_flux_data'][self.energy_bin])
                           / S['bubble_upper_error_data'][self.energy_bin]) ** 2
        elif S['bubble_flux_data'][self.energy_bin] > bubble_sed:
            chi2_bubble = ((bubble_sed - S['bubble_flux_data'][self.energy_bin])
                           / S['bubble_lower_error_data'][self.energy_bin]) ** 2
        else:
            chi2_bubble = ((bubble_sed - S['bubble_flux_data'][self.energy_bin]) / larger_error) ** 2

        isotropic_larger_error = max([S['isotropic_lower_error_data'][self.energy_bin],
                                      S['isotropic_upper_error_data'][self.energy_bin]])
        if S['isotropic_flux_data'][self.energy_bin] < isotropic_sed:
            chi2_isotropic = ((S['isotropic_flux_data'][self.energy_bin] - isotropic_sed)
                              / S['isotropic_upper_error_data'][self.energy_bin]) ** 2
        elif S['isotropic_flux_data'][self.energy_bin] > isotropic_sed:
            chi2_isotropic = ((S['isotropic_flux_data'][self.energy_bin] - isotropic_sed)
                              / S['isotropic_lower_error_data'][self.energy_bin]) ** 2
        else:
            chi2_isotropic = ((S['isotropic_flux_data'][self.energy_bin] - isotropic_sed)
                              / isotropic_larger_error) ** 2
        return np.sum(lhd) + chi2_bubble + chi2_isotropic


# ============================================================
# 고속 평가기 + polish (17yr v2 pos-domain 과 동일 기계)
# ============================================================

def build_fast_eval(lh):
    S = lh.S; j = lh.energy_bin
    m = (lh.full_mask == 1)
    obs    = lh.data[m].astype(float)
    logfac = lh.observed_log_factorial_masked
    A, B = lh.pion_bremss[m].astype(float), lh.ics[m].astype(float)
    G, Bu, Is = lh.GCE[m].astype(float), lh.bubble[m].astype(float), lh.iso[m].astype(float)
    BIG = 1e12

    if USE_CONSTRAINT:
        nm = np.sum(lh.full_mask)
        iso_unit = (S['E'][j] ** 2) * (np.sum(lh.full_mask * lh.iso_no_convol / lh.exp_cube) / nm) / S['delta_E'][j]
        bub_unit = (S['E'][j] ** 2) * (np.sum(lh.full_mask * lh.bubble_no_convol / lh.exp_cube) / nm) / S['delta_E'][j]
        bf, blo, bhi = (S['bubble_flux_data'][j], S['bubble_lower_error_data'][j], S['bubble_upper_error_data'][j])
        itf, ilo, ihi = (S['isotropic_flux_data'][j], S['isotropic_lower_error_data'][j], S['isotropic_upper_error_data'][j])
        b_larger, i_larger = max(bhi, blo), max(ilo, ihi)

    def f(p):
        p = np.asarray(p, dtype=float)
        if np.any(p < 0):
            return BIG * (1.0 + float(np.sum(-p[p < 0])))
        mu = p[0] * A + p[1] * B + p[2] * G + p[4] * Is + p[3] * Bu
        if np.any(mu < 0):
            return BIG * (1.0 + float(-mu.min()))
        with np.errstate(divide='ignore', invalid='ignore'):
            t = obs * np.log(mu)
        s = 2.0 * np.sum(mu - t + logfac)
        if not np.isfinite(s):
            return BIG
        if not USE_CONSTRAINT:
            return s
        bub_sed, iso_sed = bub_unit * p[3], iso_unit * p[4]
        if   bf < bub_sed: c2b = ((bub_sed - bf) / bhi) ** 2
        elif bf > bub_sed: c2b = ((bub_sed - bf) / blo) ** 2
        else:              c2b = ((bub_sed - bf) / b_larger) ** 2
        if   itf < iso_sed: c2i = ((itf - iso_sed) / ihi) ** 2
        elif itf > iso_sed: c2i = ((itf - iso_sed) / ilo) ** 2
        else:               c2i = ((itf - iso_sed) / i_larger) ** 2
        return s + c2b + c2i
    return f


def polish_bin(lh, starts, ref_point, restarts=0, seed=0):
    f_fast = build_fast_eval(lh)
    ref_val = lh.likelihood_constrained(ref_point)
    fid = abs(f_fast(ref_point) - ref_val) / max(1.0, abs(ref_val))
    f_use = lh.likelihood_constrained if fid > 1e-6 else f_fast
    if fid > 1e-6:
        print(f'    [WARN] fast-eval fidelity {fid:.2e} (bin {lh.energy_bin}) — verbatim으로만 진행')

    cands = [np.asarray(s, float) for s in starts]
    rng = np.random.default_rng(seed + lh.energy_bin)
    for _ in range(restarts):
        cands.append(np.clip(cands[0] * rng.lognormal(0.0, 0.3, 5), 0.0, None))

    best_p, best_f = np.asarray(ref_point, float), ref_val
    for p0 in cands:
        r1 = minimize(f_use, p0, method='Nelder-Mead',
                      options=dict(maxiter=20000, maxfev=20000, xatol=1e-7, fatol=1e-7, adaptive=True))
        r2 = minimize(f_use, r1.x, method='Powell',
                      options=dict(maxiter=20000, xtol=1e-9, ftol=1e-9))
        for r in (r1, r2):
            fx = f_use(r.x)
            if np.isfinite(fx) and fx < best_f and np.all(np.asarray(r.x) >= 0):
                best_f, best_p = fx, np.asarray(r.x, float)

    verb_best = lh.likelihood_constrained(best_p)
    if (not np.isfinite(verb_best)) or verb_best > ref_val:
        best_p, verb_best = np.asarray(ref_point, float), ref_val
    return best_p, verb_best, ref_val, fid


def run_model(model, S, restarts):
    npz_path = NPZ_PATTERN.format(m=model)
    if not os.path.exists(npz_path):
        print(f'[SKIP] {model}: npz 없음 ({npz_path})'); return None
    z = np.load(npz_path)
    P_chain = z['fitted_params']
    if 'fitted_params_median' in z.files:
        P_second = z['fitted_params_median']
    elif 'fitted_params_lower' in z.files and 'fitted_params_upper' in z.files:
        P_second = 0.5 * (z['fitted_params_lower'] + z['fitted_params_upper'])
    else:
        P_second = P_chain
    key = 'max_likelihood' if 'max_likelihood' in z.files else 'max_lhd'
    lnL_npz = z[key]
    nE = lnL_npz.shape[0]

    lhv_path = LHV_PATTERN.format(m=model)
    if os.path.exists(lhv_path):
        d = float(np.max(np.abs(np.loadtxt(lhv_path) - lnL_npz)))
        if d > 1e-6:
            print(f'  [WARN] {model}: likelihood_value ↔ npz 불일치 max|Δ|={d:.3e}')

    lnL_chain_re = np.zeros(nE); lnL_pol = np.zeros(nE)
    dpar_max = np.zeros(nE); P_pol = np.zeros_like(P_chain)

    print(f'\n===== Model {model} (12yr{RANK_SUFFIX}) =====')
    print(f'{"bin":>3} {"lnL_chain(npz)":>16} {"lnL_recomp":>14} {"lnL_polish":>14} '
          f'{"Δpolish":>10} {"|Δc|max":>9} {"sec":>6}')
    for j in range(nE):
        t0 = time.time()
        lh = Likelihood(model, j, S)
        p_c, p_m = P_chain[:, j].copy(), P_second[:, j].copy()
        best_p, f_best, f_ref, fid = polish_bin(lh, [p_c, p_m], p_c, restarts=restarts)
        lnL_chain_re[j], lnL_pol[j] = -0.5 * f_ref, -0.5 * f_best
        P_pol[:, j] = best_p
        dpar_max[j] = float(np.max(np.abs(best_p - p_c)))
        gap = lnL_chain_re[j] - lnL_npz[j]
        if abs(gap) > 0.5 and j == 0:
            print(f'  [FATAL@bin0] 재계산 lnL ↔ npz 차이 {gap:+.3f} — config knob 불일치 의심.\n'
                  f'    확인: BUBBLE_OVERRIDE_DIR / USE_CONSTRAINT / NAT38 / MASK_VARIANT / WORK_DIR_12\n'
                  f'    (grep -n "BUBBLE_OVERRIDE_DIR\\|USE_CONSTRAINT\\|NAT38" mirrorfix_step2_refit.sh)')
            return None
        if abs(gap) > 1e-3:
            print(f'  [WARN] bin {j}: 재계산 lnL ↔ npz 차이 {gap:+.4f}')
        print(f'{j:>3} {lnL_npz[j]:>16.3f} {lnL_chain_re[j]:>14.3f} {lnL_pol[j]:>14.3f} '
              f'{lnL_pol[j]-lnL_chain_re[j]:>10.3f} {dpar_max[j]:>9.4f} {time.time()-t0:>6.1f}')

    sum_chain, sum_pol = float(np.sum(lnL_chain_re)), float(np.sum(lnL_pol))
    print(f'--- Σ lnL: chain {sum_chain:.2f} → polish {sum_pol:.2f}   ΔΣ = {sum_pol-sum_chain:+.2f}')

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(f'{OUT_DIR}/polish12_{model}.npz',
             E=S['E'], lnL_npz=lnL_npz, lnL_chain_recomp=lnL_chain_re, lnL_polish=lnL_pol,
             params_chain=P_chain, params_polish=P_pol, dpar_max=dpar_max,
             sum_chain=sum_chain, sum_polish=sum_pol)
    return dict(model=model, sum_chain=sum_chain, sum_polish=sum_pol,
                dsum=sum_pol - sum_chain,
                max_bin_gain=float(np.max(lnL_pol - lnL_chain_re)),
                argmax_bin=int(np.argmax(lnL_pol - lnL_chain_re)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='*', default=['I', 'X', 'XV', 'XLVIII', 'XLIX', 'LIII'])
    ap.add_argument('--restarts', type=int, default=0)
    args = ap.parse_args()
    models = args.models if args.models else ['I', 'X', 'XV', 'XLVIII', 'XLIX', 'LIII']

    print(f'[config] WORK_DIR={WORK_DIR}  suffix={RANK_SUFFIX!r}  USE_CONSTRAINT={int(USE_CONSTRAINT)}')
    print(f'[config] BUBBLE_OVERRIDE_DIR={BUBBLE_OVERRIDE_DIR!r}  NAT38={int(NAT38)}  mask={PSC_MASK_FILE}')
    print(f'[config] models={models}  restarts={args.restarts}')
    t0 = time.time()
    S = load_shared()
    print(f'[shared] loaded in {time.time()-t0:.1f}s  (nE={len(S["E"])})')

    rows = []
    for m in models:
        r = run_model(m, S, args.restarts)
        if r:
            rows.append(r)
    if not rows:
        return

    print('\n===== SUMMARY (12yr — 모델 간 차등분이 랭킹에 들어가는 양) =====')
    print(f'{"model":>8} {"Σ chain":>16} {"Σ polish":>16} {"ΔΣ":>10} {"max bin Δ":>10} {"@bin":>5}')
    for r in rows:
        print(f'{r["model"]:>8} {r["sum_chain"]:>16.2f} {r["sum_polish"]:>16.2f} '
              f'{r["dsum"]:>10.3f} {r["max_bin_gain"]:>10.3f} {r["argmax_bin"]:>5}')
    ds = [r['dsum'] for r in rows]
    print(f'\nΔΣ 모델 간 산포: max−min = {max(ds)-min(ds):.3f}  '
          f'(참고: 12yr 인접 순위 gap median ≈ 44, 4-way 핸드오프 3094/70.0)')

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = f'{OUT_DIR}/polish12_summary.csv'
    new = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as fh:
        wcsv = csv.writer(fh)
        if new:
            wcsv.writerow(['model', 'suffix', 'sum_chain', 'sum_polish', 'dsum',
                           'max_bin_gain', 'argmax_bin'])
        for r in rows:
            wcsv.writerow([r['model'], RANK_SUFFIX, f'{r["sum_chain"]:.4f}', f'{r["sum_polish"]:.4f}',
                           f'{r["dsum"]:.4f}', f'{r["max_bin_gain"]:.4f}', r['argmax_bin']])
    print(f'[saved] {csv_path} + polish12_{{M}}.npz in {OUT_DIR}/')


if __name__ == '__main__':
    main()
