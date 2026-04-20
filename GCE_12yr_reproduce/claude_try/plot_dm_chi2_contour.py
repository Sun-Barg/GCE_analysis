"""
DM chi-square contour plot for Pipeline vs Zenodo at fiducial model.

Mirror of v8.8 Section 13 but adapted to Claude-v0 pipeline:
  - Reads the single-model pkl (XLIX by default) for fiducial flux
  - Builds systematic covariance from the available completed models
    (more models -> better coverage; early envelope may have few)
  - Reads Zenodo GCE_Statistical_errors.dat and cov_mat_21Dec02.npy
  - Computes 2D chi-square over (m_chi, sigma_v) for bb-bar channel
  - Overlays Zenodo reference contours on same grid

Inputs
------
  env MODEL         (default XLIX): which model to use as fiducial
  env FIT_SUFFIX    (default globalCgce_calibrated): pkl suffix
  env CHANNEL       (default bb): DM annihilation channel
                     bb=13, tau+tau-=11, e+e-=4, mu+mu-=7 (PPPC4 column)

Outputs
-------
  plot_dm_chi2_contour_{MODEL}_{SUFFIX}.png
"""
import os, sys, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator, interp1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR, TEMPLATES_DIR, ZENODO_FIG_DIR

MODEL = os.environ.get('MODEL', 'XLIX')
SUFFIX = os.environ.get('FIT_SUFFIX', 'globalCgce_calibrated')
CHANNEL = os.environ.get('CHANNEL', 'bb')

CHANNEL_COLS = {
    'ee': 4, 'mumu': 7, 'tautau': 11, 'bb': 13, 'tt': 14, 'WW': 17, 'ZZ': 19,
}
CHANNEL_LABELS = {
    'ee': r'$e^+ e^-$', 'mumu': r'$\mu^+ \mu^-$', 'tautau': r'$\tau^+ \tau^-$',
    'bb': r'$b \bar{b}$', 'tt': r'$t \bar{t}$',
    'WW': r'$W^+ W^-$', 'ZZ': r'$ZZ$',
}

CHANNEL_COL = CHANNEL_COLS.get(CHANNEL, 13)
CHANNEL_LABEL = CHANNEL_LABELS.get(CHANNEL, CHANNEL)

COV_DIR = os.path.join(TEMPLATES_DIR, 'Covariance_Matrix_Information')
ZENODO_FLUX_FILE = os.path.join(ZENODO_FIG_DIR,
                                 f'GCE_Model{MODEL}_flux_Inner40x40_masked_disk.dat')
STAT_ERR_FILE = os.path.join(COV_DIR, 'GCE_Statistical_errors.dat')
FULL_COV_FILE = os.path.join(COV_DIR, 'cov_mat_21Dec02.npy')

J_FACTOR = 3.5251837158376415e+21
SR_ROI_40x40 = (40.0 * np.pi / 180.0) ** 2

PKL_FILE = f'{OUT_DIR}/GCE_model_{MODEL}_haebarg_v_claude_{SUFFIX}.pkl'

PPPC4_SEARCH_PATHS = [
    os.path.join(WORK_DIR, 'Prompt_spectra/'),
    '/home/haebarg/Prompt_spectra/',
    '/home/haebarg/ipynb/',
    os.path.join(TEMPLATES_DIR, 'Prompt_spectra/'),
    './Prompt_spectra/',
]


def find_pppc4_file(filename):
    for p in PPPC4_SEARCH_PATHS:
        cand = os.path.join(p, filename)
        if os.path.exists(cand):
            return cand
    return None


def load_pppc4_spectrum(mass_gev, channel_col, which='gammas', EW='Yes'):
    filename = (f"AtProduction_{which}.dat" if EW == 'Yes'
                 else f"AtProduction_NoEW_{which}.dat")
    path = find_pppc4_file(filename)
    if path is None:
        raise FileNotFoundError(f"PPPC4 file not found: {filename}  "
                                 f"searched: {PPPC4_SEARCH_PATHS}")
    data = np.loadtxt(path, skiprows=1)
    m_grid = np.unique(data[:, 0])
    logx_grid = np.unique(data[:, 1])
    z = data[:, channel_col].reshape(len(m_grid), len(logx_grid))
    interp = RegularGridInterpolator((m_grid, logx_grid), z,
                                       bounds_error=False, fill_value=None)
    pts = np.array([[mass_gev, lx] for lx in logx_grid])
    dNdlogx = interp(pts)
    E_axis = mass_gev * (10 ** logx_grid)
    with np.errstate(divide='ignore', invalid='ignore'):
        dNdE = dNdlogx / (E_axis * np.log(10))
    return E_axis, np.nan_to_num(dNdE, nan=0.0, posinf=0.0, neginf=0.0)


def chi2_dm(dm_mass, sigma_v, data_flux, inv_cov, emeans_gev):
    warnings.simplefilter("ignore", category=RuntimeWarning)
    E_axis, dNdE_vals = load_pppc4_spectrum(dm_mass, CHANNEL_COL, 'gammas', 'Yes')
    dNdE_interp = interp1d(E_axis, dNdE_vals, fill_value=0, bounds_error=False,
                            kind='linear')(emeans_gev)
    model_flux = (emeans_gev ** 2) * dNdE_interp * \
                 (sigma_v / dm_mass ** 2) * J_FACTOR / SR_ROI_40x40
    delta = model_flux - data_flux
    return float(delta @ inv_cov @ delta)


def collect_completed_fluxes(suffix):
    import glob
    pattern = f'{OUT_DIR}/GCE_model_*_haebarg_v_claude_{suffix}.pkl'
    fluxes = []
    for fp in sorted(glob.glob(pattern)):
        try:
            with open(fp, 'rb') as f:
                d = pickle.load(f)
            if 'flux_best' in d:
                fluxes.append(d['flux_best'])
        except Exception:
            pass
    return np.array(fluxes) if fluxes else None


def main():
    os.chdir(WORK_DIR)
    print("=" * 80)
    print(f"DM chi-square contour : Model {MODEL}  channel {CHANNEL}")
    print(f"  pkl       : {PKL_FILE}")
    print(f"  zenodo ref: {ZENODO_FLUX_FILE}")
    print("=" * 80)

    for f in [PKL_FILE, ZENODO_FLUX_FILE, STAT_ERR_FILE, FULL_COV_FILE]:
        if not os.path.exists(f):
            print(f"ERROR: missing {f}")
            sys.exit(1)

    if find_pppc4_file("AtProduction_gammas.dat") is None:
        print("ERROR: PPPC4 file AtProduction_gammas.dat not found in any of:")
        for p in PPPC4_SEARCH_PATHS:
            print(f"  {p}")
        print("\nHint: download from http://www.marcocirelli.net/PPPC4DMID.html")
        sys.exit(1)

    with open(PKL_FILE, 'rb') as f:
        d = pickle.load(f)
    E = np.asarray(d['E'])
    pl_fid_flux = np.asarray(d['flux_best'])
    pl_fid_err = np.asarray(d.get('flux_std', pl_fid_flux * 0.1))
    n_bins = len(E)

    zen = np.loadtxt(ZENODO_FLUX_FILE)
    zen_E = zen[:n_bins, 0]
    zen_flux = zen[:n_bins, 1]

    stat_err_author = np.loadtxt(STAT_ERR_FILE)[:n_bins, 1]

    full_cov_raw = np.load(FULL_COV_FILE)
    Ua, Sa, Vha = np.linalg.svd(full_cov_raw)
    cov_sys_author = sum(Sa[i] * np.outer(Ua.T[i], Vha[i]) for i in range(3))
    cov_total_author = np.diag(stat_err_author ** 2) + cov_sys_author
    inv_cov_author = np.linalg.inv(cov_total_author)

    flux_matrix = collect_completed_fluxes(SUFFIX)
    if flux_matrix is None or flux_matrix.shape[0] < 3:
        print(f"\nWARNING: only {0 if flux_matrix is None else flux_matrix.shape[0]} "
              f"completed models found with suffix '{SUFFIX}'.")
        print("Using Zenodo author covariance directly for pipeline too (fallback).")
        cov_sys_pipeline = cov_sys_author.copy()
    else:
        print(f"  building systematic cov from {flux_matrix.shape[0]} completed models")
        cov_sys_raw = np.cov(flux_matrix.T, bias=False)
        stds = np.sqrt(np.diag(cov_sys_raw))
        stds_safe = np.where(stds > 0, stds, 1e-30)
        corr = cov_sys_raw / np.outer(stds_safe, stds_safe)
        Up, Sp, Vhp = np.linalg.svd(corr)
        corr_trunc = sum(Sp[i] * np.outer(Up[:, i], Vhp[i, :]) for i in range(3))
        residuals = np.maximum(1.0 - np.diag(corr_trunc), 0.0)
        corr_trunc += np.diag(residuals)
        cov_sys_pipeline = corr_trunc * np.outer(stds, stds)

    stat_err_pipeline = pl_fid_err.copy()
    bad = np.isnan(stat_err_pipeline) | (stat_err_pipeline <= 0)
    if bad.any():
        stat_err_pipeline[bad] = stat_err_author[bad]
    cov_total_pipeline = np.diag(stat_err_pipeline ** 2) + cov_sys_pipeline
    inv_cov_pipeline = np.linalg.inv(cov_total_pipeline)

    N_GRID = 40
    mass_range = np.logspace(np.log10(10), np.log10(200), N_GRID)
    sigmav_range = np.logspace(-27, -25, N_GRID)
    DM_MASS, SIGMAV = np.meshgrid(mass_range, sigmav_range)

    print(f"\n  scanning {N_GRID}x{N_GRID} grid for pipeline ...")
    chi2_pl = np.zeros_like(DM_MASS)
    for i in range(N_GRID):
        for j in range(N_GRID):
            chi2_pl[i, j] = chi2_dm(DM_MASS[i, j], SIGMAV[i, j],
                                      pl_fid_flux, inv_cov_pipeline, E)
        if (i+1) % 10 == 0:
            print(f"    pipeline: {i+1}/{N_GRID} rows done")

    print(f"\n  scanning {N_GRID}x{N_GRID} grid for Zenodo ...")
    chi2_zen = np.zeros_like(DM_MASS)
    for i in range(N_GRID):
        for j in range(N_GRID):
            chi2_zen[i, j] = chi2_dm(DM_MASS[i, j], SIGMAV[i, j],
                                       zen_flux, inv_cov_author, zen_E)
        if (i+1) % 10 == 0:
            print(f"    zenodo: {i+1}/{N_GRID} rows done")

    ipl = np.unravel_index(np.argmin(chi2_pl), chi2_pl.shape)
    izn = np.unravel_index(np.argmin(chi2_zen), chi2_zen.shape)
    bf = dict(pl_m=DM_MASS[ipl], pl_s=SIGMAV[ipl], pl_chi2=chi2_pl[ipl],
               zn_m=DM_MASS[izn], zn_s=SIGMAV[izn], zn_chi2=chi2_zen[izn])
    dof = n_bins - 2

    print(f"\n--- Pipeline v0 ({MODEL} fiducial, suffix={SUFFIX}) ---")
    print(f"  best-fit m_chi   = {bf['pl_m']:.1f} GeV")
    print(f"  best-fit sigma_v = {bf['pl_s']:.3e} cm^3/s")
    print(f"  chi2_min / dof   = {bf['pl_chi2']:.2f} / {dof} "
          f"= {bf['pl_chi2']/dof:.3f}")
    print(f"\n--- Zenodo (Cholis+2022 Model {MODEL}) ---")
    print(f"  best-fit m_chi   = {bf['zn_m']:.1f} GeV")
    print(f"  best-fit sigma_v = {bf['zn_s']:.3e} cm^3/s")
    print(f"  chi2_min / dof   = {bf['zn_chi2']:.2f} / {dof} "
          f"= {bf['zn_chi2']/dof:.3f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    levels_pl = [bf['pl_chi2'] + 2.30, bf['pl_chi2'] + 6.18]
    ax.contour(DM_MASS, SIGMAV, chi2_pl, levels=levels_pl,
                colors='crimson', linestyles=['--', ':'], linewidths=[2.5, 2.0])
    ax.plot(bf['pl_m'], bf['pl_s'], marker='*', color='crimson',
             markersize=18, markeredgecolor='black', zorder=10)

    levels_zn = [bf['zn_chi2'] + 2.30, bf['zn_chi2'] + 6.18]
    ax.contour(DM_MASS, SIGMAV, chi2_zen, levels=levels_zn,
                colors='royalblue', linestyles=['-', '-.'],
                linewidths=[3.0, 2.2], alpha=0.85)
    ax.plot(bf['zn_m'], bf['zn_s'], marker='o', color='royalblue',
             markersize=11, markeredgecolor='black', zorder=10)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(10, 200)
    ax.set_ylim(5e-27, 1e-25)
    ax.set_xlabel(r'$m_\chi$ [GeV]', fontsize=14)
    ax.set_ylabel(r'$\langle \sigma v \rangle$ [cm$^3$ s$^{-1}$]', fontsize=14)
    ax.set_title(f'DM constraints (Model {MODEL}, channel {CHANNEL_LABEL})',
                  fontsize=13, pad=12)
    ax.grid(True, which='major', linestyle='--', alpha=0.4)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)

    lines = [
        Line2D([0], [0], color='crimson', lw=2.5, linestyle='--'),
        Line2D([0], [0], color='crimson', lw=2.0, linestyle=':'),
        Line2D([0], [0], color='royalblue', lw=3.0, linestyle='-'),
        Line2D([0], [0], color='royalblue', lw=2.2, linestyle='-.'),
    ]
    labels = [
        f'Pipeline Model {MODEL} - 1sigma',
        'Pipeline - 2sigma',
        f'Zenodo Model {MODEL} - 1sigma',
        'Zenodo - 2sigma',
    ]
    ax.legend(lines, labels, loc='lower right', fontsize=10, framealpha=0.92)

    plt.tight_layout()
    out_png = f'{OUT_DIR}/plot_dm_chi2_contour_{MODEL}_{CHANNEL}_{SUFFIX}.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")

    out_npz = f'{OUT_DIR}/plot_dm_chi2_contour_{MODEL}_{CHANNEL}_{SUFFIX}.npz'
    np.savez(out_npz, DM_MASS=DM_MASS, SIGMAV=SIGMAV,
              chi2_pipeline=chi2_pl, chi2_zenodo=chi2_zen, bf=bf,
              E=E, pl_fid_flux=pl_fid_flux, zen_flux=zen_flux,
              cov_sys_pipeline=cov_sys_pipeline, cov_sys_author=cov_sys_author)
    print(f"Saved: {out_npz}")


if __name__ == '__main__':
    main()
