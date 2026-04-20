import os, sys, time, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import interp1d
from scipy.special import gammaln
import emcee
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    WORK_DIR, ANA_DIR, OUT_DIR,
    CCUBE, EXPCUBE_CENTER,
    BUBBLE_CONSTRAINTS, ISO_CONSTRAINTS,
    NWALKERS, NSTEPS, NBURN, NDIM, MIN_REL_ERROR_FLOOR,
    MODELS, ROI_SLICE,
)
from prepare_masks import FULL_MASK_PATH, prepare_masks


def solid_angle_per_pixel(width, height, wcs, dl=0.1, db=0.1):
    from config import SR_PER_PIXEL_NPY
    if os.path.exists(SR_PER_PIXEL_NPY):
        cached = np.load(SR_PER_PIXEL_NPY)
        if cached.shape == (height, width):
            return cached.astype(np.float64)
    dl_rad = np.deg2rad(dl); db_rad = np.deg2rad(db)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    _, b = wcs.wcs_pix2world(xx.ravel(), yy.ravel(), 0)
    b = b.reshape(height, width)
    sap = dl_rad * db_rad * np.cos(np.deg2rad(b))
    try:
        np.save(SR_PER_PIXEL_NPY, sap)
    except Exception:
        pass
    return sap


def load_energy_axis(ccube_path):
    e_bounds = fits.open(ccube_path)[1].data
    n = len(e_bounds)
    E = np.zeros(n); dE = np.zeros(n)
    for i in range(n):
        E[i] = np.sqrt(e_bounds[i][2] * e_bounds[i][1] * 1e-6) * 1e-3
        dE[i] = (e_bounds[i][2] - e_bounds[i][1]) * 1e-6
    return E, dE


def load_external_constraints(E_arr):
    bub = np.loadtxt(BUBBLE_CONSTRAINTS)
    iso = np.loadtxt(ISO_CONSTRAINTS)
    bub_f = interp1d(bub[:, 0], bub[:, 1], kind='quadratic', fill_value='extrapolate')
    bub_lo = interp1d(bub[:, 0], bub[:, 2], kind='quadratic', fill_value='extrapolate')
    bub_hi = interp1d(bub[:, 0], bub[:, 3], kind='quadratic', fill_value='extrapolate')
    iso_f = interp1d(iso[:, 0], iso[:, 1], kind='quadratic', fill_value='extrapolate')
    iso_lo = interp1d(iso[:, 0], iso[:, 2], kind='quadratic', fill_value='extrapolate')
    iso_hi = interp1d(iso[:, 0], iso[:, 3], kind='quadratic', fill_value='extrapolate')

    bub_data = np.maximum(bub_f(E_arr), 1e-30)
    bub_err_lo = np.maximum(bub_lo(E_arr), MIN_REL_ERROR_FLOOR * bub_data)
    bub_err_hi = np.maximum(bub_hi(E_arr), MIN_REL_ERROR_FLOOR * bub_data)

    iso_data = (E_arr ** 2) * iso_f(E_arr)
    iso_err_lo = (E_arr ** 2) * iso_lo(E_arr)
    iso_err_hi = (E_arr ** 2) * iso_hi(E_arr)
    iso_data = np.maximum(iso_data, 1e-30)
    iso_err_lo = np.maximum(iso_err_lo, MIN_REL_ERROR_FLOOR * iso_data)
    iso_err_hi = np.maximum(iso_err_hi, MIN_REL_ERROR_FLOOR * iso_data)

    return bub_data, bub_err_lo, bub_err_hi, iso_data, iso_err_lo, iso_err_hi


def load_component_maps(model, ebin):
    yroi, xroi = ROI_SLICE
    def _open(path):
        return fits.open(path)[0].data[ebin, yroi, xroi].astype(np.float64)
    base = './GC_analysis_sanghwan'
    pion_c = _open(f'{base}/GC_pion_model{model}_12yr_front_clean.fits')
    brem_c = _open(f'{base}/GC_bremss_model{model}_12yr_front_clean.fits')
    ics_c = _open(f'{base}/GC_ics_model{model}_12yr_front_clean.fits')
    gce_c = _open(f'{base}/GC_GCE_model_12yr_front_clean.fits')
    bub_c = _open(f'{base}/GC_fermi_bubble_model_12yr_front_clean.fits')
    iso_c = _open(f'{base}/GC_isotropic_model_12yr_front_clean.fits')
    iso_nc = _open(f'{base}/GC_isotropic_model_12yr_front_clean_no_convol.fits')
    bub_nc = _open(f'{base}/GC_fermi_bubble_model_12yr_front_clean_no_convol.fits')
    return pion_c + brem_c, ics_c, gce_c, bub_c, iso_c, iso_nc, bub_nc


class BinLikelihood:
    def __init__(self, model, ebin, ccube, exp_cube_pix_sr, full_mask,
                 E, dE, bub_d, bub_lo, bub_hi, iso_d, iso_lo, iso_hi):
        self.ebin = ebin
        yroi, xroi = ROI_SLICE
        self.data = ccube[ebin, yroi, xroi].astype(np.float64)
        self.exp = exp_cube_pix_sr[ebin]
        self.mask = full_mask[ebin]
        self.mask_bool = self.mask > 0.5
        self.mask_sum = float(self.mask.sum())
        pb, ic, gc, bb, iso, iso_nc, bub_nc = load_component_maps(model, ebin)
        self.pb = pb; self.ic = ic; self.gc = gc; self.bb = bb; self.iso = iso
        self.E = E[ebin]; self.dE = dE[ebin]
        self.bub_d = bub_d[ebin]; self.bub_lo = bub_lo[ebin]; self.bub_hi = bub_hi[ebin]
        self.iso_d = iso_d[ebin]; self.iso_lo = iso_lo[ebin]; self.iso_hi = iso_hi[ebin]

        with np.errstate(divide='ignore', invalid='ignore'):
            iso_per_pix = np.where(self.exp > 0, iso_nc / self.exp, 0.0)
            bub_per_pix = np.where(self.exp > 0, bub_nc / self.exp, 0.0)
        self.iso_avg = float((self.mask * iso_per_pix).sum() / self.mask_sum)
        self.bub_avg = float((self.mask * bub_per_pix).sum() / self.mask_sum)

        obs_m = self.data[self.mask_bool]
        self.obs_m = obs_m
        self.log_fact_obs_sum = float(gammaln(obs_m + 1.0).sum())

        self.pb_m = self.pb[self.mask_bool]
        self.ic_m = self.ic[self.mask_bool]
        self.gc_m = self.gc[self.mask_bool]
        self.bb_m = self.bb[self.mask_bool]
        self.iso_m = self.iso[self.mask_bool]

    def neg2_log_like(self, params):
        c_gas, c_ics, c_gce, c_bub, c_iso = params
        if c_gas < 0 or c_ics < 0 or c_gce < 0 or c_bub < 0 or c_iso < 0:
            return np.inf
        expected = (c_gas * self.pb_m + c_ics * self.ic_m + c_gce * self.gc_m
                    + c_bub * self.bb_m + c_iso * self.iso_m)
        if (expected <= 0).any():
            return np.inf
        poisson = 2.0 * (expected.sum() - (self.obs_m * np.log(expected)).sum() + self.log_fact_obs_sum)

        bub_sed = (self.E ** 2) * c_bub * self.bub_avg / self.dE
        iso_sed = (self.E ** 2) * c_iso * self.iso_avg / self.dE

        if bub_sed >= self.bub_d:
            chi2_bub = ((bub_sed - self.bub_d) / self.bub_hi) ** 2
        else:
            chi2_bub = ((self.bub_d - bub_sed) / self.bub_lo) ** 2
        if iso_sed >= self.iso_d:
            chi2_iso = ((iso_sed - self.iso_d) / self.iso_hi) ** 2
        else:
            chi2_iso = ((self.iso_d - iso_sed) / self.iso_lo) ** 2

        return poisson + chi2_bub + chi2_iso


_BIN_LIKE = None
def _log_prob(params):
    if (np.array(params) < 0).any():
        return -np.inf
    val = _BIN_LIKE.neg2_log_like(params)
    if not np.isfinite(val):
        return -np.inf
    return -0.5 * val


def fit_one_model(model):
    global _BIN_LIKE
    print(f"\n==== STAGE 2 :: MODEL {model} ====", flush=True)
    t_model = time.time()

    if not os.path.exists(FULL_MASK_PATH):
        prepare_masks()
    full_mask = np.load(FULL_MASK_PATH).astype(np.float32)

    ccube_data = fits.open(CCUBE)[0].data.astype(np.float64)
    E, dE = load_energy_axis(CCUBE)
    n_ebin = len(E)

    raw = fits.open(CCUBE)
    wcs = WCS(raw[0].header).dropaxis(2)
    yroi, xroi = ROI_SLICE
    sap_full = solid_angle_per_pixel(ccube_data.shape[2], ccube_data.shape[1], wcs)
    sap = sap_full[yroi, xroi]
    expcube_full = fits.open(EXPCUBE_CENTER)[0].data.astype(np.float64)
    exp_cube_pix_sr = expcube_full[:, yroi, xroi] * sap[None, :, :]

    bub_d, bub_lo, bub_hi, iso_d, iso_lo, iso_hi = load_external_constraints(E)

    flux_best = np.zeros(n_ebin)
    flux_lo = np.zeros(n_ebin)
    flux_hi = np.zeros(n_ebin)
    flux_std = np.zeros(n_ebin)
    coef_best = np.zeros((n_ebin, NDIM))
    coef_med = np.zeros((n_ebin, NDIM))
    coef_std = np.zeros((n_ebin, NDIM))
    max_logL = np.zeros(n_ebin)
    gce_avg_per_bin = np.zeros(n_ebin)

    for eb in range(n_ebin):
        t_bin = time.time()
        bl = BinLikelihood(model, eb, ccube_data, exp_cube_pix_sr, full_mask,
                           E, dE, bub_d, bub_lo, bub_hi, iso_d, iso_lo, iso_hi)
        _BIN_LIKE = bl

        with np.errstate(divide='ignore', invalid='ignore'):
            gce_per_pix = np.where(bl.exp > 0, bl.gc / bl.exp, 0.0)
        gce_avg = float((bl.mask * gce_per_pix).sum() / bl.mask_sum)
        gce_avg_per_bin[eb] = gce_avg

        rng = np.random.default_rng(seed=12345 + eb)
        p0 = np.column_stack([
            rng.uniform(0.3, 1.7, NWALKERS),
            rng.uniform(0.3, 1.7, NWALKERS),
            rng.uniform(0.3, 3.0, NWALKERS),
            rng.uniform(0.5, 6.0, NWALKERS),
            rng.uniform(0.5, 6.0, NWALKERS),
        ])

        sampler = emcee.EnsembleSampler(NWALKERS, NDIM, _log_prob)
        chunk = 100
        pos = p0
        for cs in range(0, NSTEPS, chunk):
            n = min(chunk, NSTEPS - cs)
            t0 = time.time()
            state = sampler.run_mcmc(pos, n, progress=False)
            pos = state.coords if hasattr(state, 'coords') else state[0]
            cum = cs + n
            print(f"    [{model} bin {eb:2d}] step {cum:>4}/{NSTEPS}  "
                  f"{n/(time.time()-t0):.2f} it/s", flush=True)

        log_prob_flat = sampler.get_log_prob(discard=NBURN, flat=True)
        chain_flat = sampler.get_chain(discard=NBURN, flat=True)
        idx_max = int(np.argmax(log_prob_flat))
        best = chain_flat[idx_max]
        med = np.median(chain_flat, axis=0)
        std = np.std(chain_flat, axis=0, ddof=1)
        lo16 = np.percentile(chain_flat, 16, axis=0)
        hi84 = np.percentile(chain_flat, 84, axis=0)

        coef_best[eb] = best
        coef_med[eb] = med
        coef_std[eb] = std
        max_logL[eb] = float(log_prob_flat[idx_max])

        e2_dE = (E[eb] ** 2) / dE[eb]
        flux_best[eb] = best[2] * gce_avg * e2_dE
        flux_std[eb] = std[2] * gce_avg * e2_dE
        flux_lo[eb] = lo16[2] * gce_avg * e2_dE
        flux_hi[eb] = hi84[2] * gce_avg * e2_dE

        print(f"    [{model} bin {eb:2d}] best={best.round(3)}  "
              f"flux={flux_best[eb]:.3e}  done in {(time.time()-t_bin)/60:.1f} min", flush=True)

    out_dat = os.path.join(OUT_DIR, f'GCE_model_{model}_haebarg_v_claude.dat')
    np.savetxt(out_dat,
               np.column_stack([E, flux_best, flux_std, flux_lo, flux_hi]),
               header='E[GeV]  E2dN/dE_best  std  lower_1sigma  upper_1sigma')
    out_logL = os.path.join(OUT_DIR, f'GCE_model_{model}_haebarg_v_claude_logL.txt')
    np.savetxt(out_logL, max_logL, header='per-bin max log-likelihood (= -0.5 * -2lnL)')
    out_pkl = os.path.join(OUT_DIR, f'GCE_model_{model}_haebarg_v_claude.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'model': model, 'E': E, 'dE': dE,
            'flux_best': flux_best, 'flux_std': flux_std,
            'flux_lo': flux_lo, 'flux_hi': flux_hi,
            'coef_best': coef_best, 'coef_med': coef_med, 'coef_std': coef_std,
            'max_logL': max_logL, 'gce_avg_per_bin': gce_avg_per_bin,
            'total_neg2_logL': float(-2.0 * max_logL.sum()),
        }, f)
    print(f"==== MODEL {model} done in {(time.time()-t_model)/60:.1f} min ====", flush=True)
    print(f"  total -2 log L = {-2.0*max_logL.sum():.2f}", flush=True)
    return out_pkl


if __name__ == '__main__':
    targets = sys.argv[1:] if len(sys.argv) > 1 else MODELS
    os.chdir(WORK_DIR)
    for m in targets:
        fit_one_model(m)
    print("\nALL STAGE 2 COMPLETE", flush=True)
