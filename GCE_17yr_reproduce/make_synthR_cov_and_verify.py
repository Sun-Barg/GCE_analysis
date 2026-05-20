#!/usr/bin/env python3
# make_synthR_cov_and_verify.py
# Sanghwan-R + our-sigma (SYNTH) systematic covariance: drop-in npz + bb-bar TS verification.
# Run from terminal (conda env: fermi). No MCMC -> seconds.
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator

# ---------- CONFIG (서버에서 4개 경로만 확인) ----------
WORK         = os.path.expanduser('~/GCE-Chi-square-fitting/GCE_17yr_reproduce')
OUR_COV_NPZ  = f'{WORK}/results_cov_17yr_PRE_widefix_20260603/GCE_systematic_covariance_matrix_17yr.npz'
SANGHWAN_COV = os.path.expanduser('~/GCE-Chi-square-fitting/Cov/approx_covariance_14x14_front_model_X_16yr.npy')
GCE_DAT      = f'{WORK}/results_17yr/GCE_model_X_front_17yr_cholis.dat'   # col0=E, col1=flux, col2=stat
PPPC4        = os.path.expanduser('~/GCE-Chi-square-fitting/PPPC4/particle_data/AtProduction_gammas.dat')
OUT_DIR      = f'{WORK}/results_cov_17yr_synthR'
J_FACTOR = 3.5251837158376415e+21      # per-sr convention (figure와 동일 -> Cholis Fig18 재현값)
SR       = 0.4288213187542626
m_arr = np.logspace(np.log10(10), np.log10(200), 60)
s_arr = np.logspace(-27, -25, 60)
LN10  = np.log(10.0)

# ---------- 1. SYNTH cov (our sigma + Sanghwan R) ----------
d        = np.load(OUR_COV_NPZ)
cov_our  = d['cov_matrix']
sig_our  = d['sigma_sys'] if 'sigma_sys' in d.files else np.sqrt(np.diag(cov_our))
cov_s    = np.load(SANGHWAN_COV)
sig_s    = np.sqrt(np.diag(cov_s))
R_s      = cov_s / np.outer(sig_s, sig_s)            # Sanghwan correlation structure
C_synth  = np.outer(sig_our, sig_our) * R_s          # our magnitude, his correlation
C_synth  = 0.5 * (C_synth + C_synth.T)               # enforce symmetry

os.makedirs(OUT_DIR, exist_ok=True)
out = {k: d[k] for k in d.files}                     # preserve full schema
out['cov_matrix']     = C_synth
out['inv_cov_matrix'] = np.linalg.pinv(C_synth)
out['sigma_sys']      = sig_our                      # unchanged (ours)
out['provenance']     = np.array(
    'SYNTH: diag(sigma_sys_17yr) @ corr(approx_covariance_14x14_front_model_X_16yr) @ diag(sigma_sys_17yr)')
np.savez(f'{OUT_DIR}/GCE_systematic_covariance_matrix_17yr.npz', **out)

# ---------- 2. PPPC4 bb-bar DM flux ----------
def load_pppc4_bbbar(path):
    raw = np.genfromtxt(path, skip_header=1)
    mDM, log10x, dN = raw[:, 0], raw[:, 1], raw[:, 13]   # 0-based col 13 = 'b' (bb-bar, EW=Yes)
    mg, xg = np.unique(mDM), np.unique(log10x)
    if len(dN) == len(mg) * len(xg):                     # standard rectangular grid
        return RegularGridInterpolator((mg, xg), dN.reshape(len(mg), len(xg)),
                                       bounds_error=False, fill_value=0.0)
    return LinearNDInterpolator(np.column_stack([mDM, log10x]), dN, fill_value=0.0)

def dm_flux_bbbar(m, emeans, interp):                    # E^2 dN/dE for sigma_v = 1 (linear in sigma_v)
    log10x    = np.log10(emeans / m)
    dNdlog10x = interp(np.column_stack([np.full_like(emeans, m), log10x]))
    dNdE      = np.where(emeans < m, dNdlog10x / (emeans * LN10), 0.0)
    return emeans**2 * dNdE * (1.0 / m**2) * J_FACTOR / SR

# ---------- 3. bb-bar TS ----------
def bb_TS(cov_sys, GCE_data, stat_err, emeans, interp):
    inv       = np.linalg.inv(np.diag(stat_err**2) + cov_sys)
    chi2_null = float(GCE_data @ inv @ GCE_data)
    grid, best = np.empty((len(s_arr), len(m_arr))), (np.nan, np.nan, np.inf)
    for jm, m in enumerate(m_arr):
        f1 = dm_flux_bbbar(m, emeans, interp)
        for js, s in enumerate(s_arr):
            delta = f1 * s - GCE_data
            c2    = float(delta @ inv @ delta)
            grid[js, jm] = c2
            if c2 < best[2]:
                best = (m, s, c2)
    return dict(TS=chi2_null - best[2], m=best[0], sv=best[1],
                min_chi2=best[2], chi2_null=chi2_null, grid=grid)

g = np.loadtxt(GCE_DAT)
emeans, GCE_data, stat_err = g[:, 0], g[:, 1], g[:, 2]
interp = load_pppc4_bbbar(PPPC4)

covs = {'OUR raw (PRE_widefix)': cov_our,
        'Sanghwan-approx 16yr':  cov_s,
        'SYNTH (Sanghwan-R)':    C_synth}
res     = {n: bb_TS(C, GCE_data, stat_err, emeans, interp) for n, C in covs.items()}
anchors = {'OUR raw (PRE_widefix)': 2.44, 'Sanghwan-approx 16yr': 57.33}

print(f"\n{'cov':24s} {'chi2_null':>9s} {'min_chi2':>9s} {'m[GeV]':>7s} {'sigma_v':>11s} {'TS':>7s}")
for n, r in res.items():
    tag = f"  (anchor {anchors[n]})" if n in anchors else "  <-- DELIVERABLE"
    print(f"{n:24s} {r['chi2_null']:9.2f} {r['min_chi2']:9.2f} {r['m']:7.1f} {r['sv']:11.3e} {r['TS']:7.2f}{tag}")

# ---------- 4. plot ----------
M, S   = np.meshgrid(m_arr, s_arr)
colors = {'OUR raw (PRE_widefix)': 'crimson',
          'Sanghwan-approx 16yr':  'steelblue',
          'SYNTH (Sanghwan-R)':    'darkorange'}
fig, ax = plt.subplots(figsize=(8, 6))
for n, r in res.items():
    ax.contour(M, S, r['grid'], levels=[r['min_chi2'] + 2.30, r['min_chi2'] + 6.18],
               colors=colors[n], linewidths=2, linestyles=['-', '--'])
    ax.plot(r['m'], r['sv'], 'o', color=colors[n], label=f"{n}: m={r['m']:.0f}, TS={r['TS']:.1f}")
ax.axhline(2.2e-26, ls=':', color='gray', label='thermal relic')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(10, 200); ax.set_ylim(1e-27, 1e-24)
ax.set_xlabel(r'$m_\chi$ [GeV]'); ax.set_ylabel(r'$\langle\sigma v\rangle$ [cm$^3$ s$^{-1}$]')
ax.set_title(r'$b\bar b$ — SYNTH (Sanghwan-R + our-$\sigma$) vs anchors (Model X, 17yr)')
ax.legend(fontsize=9); ax.grid(which='major', ls='--', lw=0.4)
fig.savefig(f'{OUT_DIR}/bb_synthR_TS_verification.png', dpi=140, bbox_inches='tight')
print(f"\nsaved: {OUT_DIR}/GCE_systematic_covariance_matrix_17yr.npz")
print(f"saved: {OUT_DIR}/bb_synthR_TS_verification.png")
