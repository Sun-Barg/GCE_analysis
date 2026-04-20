import os, sys, time, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
from astropy.io import fits
from astropy.wcs import WCS
import emcee
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    WORK_DIR, OUT_DIR,
    CCUBE, EXPCUBE_CENTER,
    NSTEPS, NBURN,
    MODELS, ROI_SLICE,
)
from prepare_masks import FULL_MASK_PATH, prepare_masks
from stage2_fit import (
    load_energy_axis, load_external_constraints, solid_angle_per_pixel,
    load_component_maps,
)

SUFFIX = os.environ.get('FIT_SUFFIX', 'globalCgce')
N_BINS = 14
NDIM_GLOBAL = 1 + 4 * N_BINS
NWALKERS_GLOBAL = 2 * NDIM_GLOBAL + 4


class GlobalLikelihood:
    def __init__(self, model, ccube, exp_cube_pix_sr, full_mask, E, dE,
                 bub_d, bub_lo, bub_hi, iso_d, iso_lo, iso_hi):
        self.n = N_BINS
        self.E = np.asarray(E, dtype=np.float64)
        self.dE = np.asarray(dE, dtype=np.float64)
        yroi, xroi = ROI_SLICE

        self.bub_d = np.asarray(bub_d); self.bub_lo = np.asarray(bub_lo); self.bub_hi = np.asarray(bub_hi)
        self.iso_d = np.asarray(iso_d); self.iso_lo = np.asarray(iso_lo); self.iso_hi = np.asarray(iso_hi)
        self.mask_sum = np.zeros(self.n)
        self.iso_avg = np.zeros(self.n)
        self.bub_avg = np.zeros(self.n)

        obs_list = []
        pb_list = []; ics_list = []; gc_list = []; bb_list = []; iso_list = []
        bin_idx_list = []
        log_fact_sum_total = 0.0

        for eb in range(self.n):
            d = ccube[eb, yroi, xroi].astype(np.float64)
            mb = full_mask[eb] > 0.5
            obs_m = d[mb]
            obs_list.append(obs_m)
            log_fact_sum_total += float(gammaln(obs_m + 1.0).sum())
            self.mask_sum[eb] = float(full_mask[eb].sum())

            pb, ic, gc, bb, iso, iso_nc, bub_nc = load_component_maps(model, eb)
            pb_list.append(pb[mb].astype(np.float64))
            ics_list.append(ic[mb].astype(np.float64))
            gc_list.append(gc[mb].astype(np.float64))
            bb_list.append(bb[mb].astype(np.float64))
            iso_list.append(iso[mb].astype(np.float64))

            bin_idx_list.append(np.full(int(mb.sum()), eb, dtype=np.int32))

            exp_eb = exp_cube_pix_sr[eb]
            with np.errstate(divide='ignore', invalid='ignore'):
                iso_pp = np.where(exp_eb > 0, iso_nc / exp_eb, 0.0)
                bub_pp = np.where(exp_eb > 0, bub_nc / exp_eb, 0.0)
            self.iso_avg[eb] = float((full_mask[eb] * iso_pp).sum() / self.mask_sum[eb])
            self.bub_avg[eb] = float((full_mask[eb] * bub_pp).sum() / self.mask_sum[eb])

        self.obs_all = np.concatenate(obs_list)
        self.pb_all = np.concatenate(pb_list)
        self.ics_all = np.concatenate(ics_list)
        self.gc_all = np.concatenate(gc_list)
        self.bb_all = np.concatenate(bb_list)
        self.iso_all = np.concatenate(iso_list)
        self.bin_idx = np.concatenate(bin_idx_list)
        self.log_fact_sum_total = log_fact_sum_total

        self.E2_over_dE = (self.E ** 2) / self.dE

    def neg2_logL(self, params):
        c_gce = params[0]
        c_gas = np.asarray(params[1:1+self.n])
        c_ics = np.asarray(params[1+self.n:1+2*self.n])
        c_bub = np.asarray(params[1+2*self.n:1+3*self.n])
        c_iso = np.asarray(params[1+3*self.n:1+4*self.n])
        if c_gce < 0 or (c_gas < 0).any() or (c_ics < 0).any() \
                or (c_bub < 0).any() or (c_iso < 0).any():
            return np.inf

        cg = c_gas[self.bin_idx]
        ci = c_ics[self.bin_idx]
        cb = c_bub[self.bin_idx]
        cs = c_iso[self.bin_idx]

        expected = (cg * self.pb_all
                    + ci * self.ics_all
                    + c_gce * self.gc_all
                    + cb * self.bb_all
                    + cs * self.iso_all)
        if (expected <= 0).any():
            return np.inf

        poisson_total = 2.0 * (expected.sum()
                                - (self.obs_all * np.log(expected)).sum()
                                + self.log_fact_sum_total)

        bub_sed = self.E2_over_dE * c_bub * self.bub_avg
        iso_sed = self.E2_over_dE * c_iso * self.iso_avg
        bub_err = np.where(bub_sed >= self.bub_d, self.bub_hi, self.bub_lo)
        iso_err = np.where(iso_sed >= self.iso_d, self.iso_hi, self.iso_lo)
        chi2_b = np.sum(((bub_sed - self.bub_d) / bub_err) ** 2)
        chi2_i = np.sum(((iso_sed - self.iso_d) / iso_err) ** 2)

        return poisson_total + chi2_b + chi2_i


_GL = None
def _log_prob(params):
    if (np.asarray(params) < 0).any():
        return -np.inf
    val = _GL.neg2_logL(params)
    if not np.isfinite(val):
        return -np.inf
    return -0.5 * val


def fit_one_model(model):
    global _GL
    print(f"\n==== Model {model} (global c_gce) START  pid={os.getpid()}", flush=True)
    print(f"  ndim={NDIM_GLOBAL}, nwalkers={NWALKERS_GLOBAL}, "
          f"nsteps={NSTEPS}, nburn={NBURN}", flush=True)
    t0 = time.time()
    os.chdir(WORK_DIR)

    if not os.path.exists(FULL_MASK_PATH):
        prepare_masks()
    full_mask = np.load(FULL_MASK_PATH).astype(np.float32)

    ccube_hdu = fits.open(CCUBE)
    ccube = ccube_hdu[0].data.astype(np.float64)
    E, dE = load_energy_axis(CCUBE)
    wcs = WCS(ccube_hdu[0].header).dropaxis(2)
    yroi, xroi = ROI_SLICE
    sap_full = solid_angle_per_pixel(ccube.shape[2], ccube.shape[1], wcs)
    expcube_full = fits.open(EXPCUBE_CENTER)[0].data.astype(np.float64)
    exp_cube_pix_sr = expcube_full[:, yroi, xroi] * sap_full[yroi, xroi][None, :, :]
    bub_d, bub_lo, bub_hi, iso_d, iso_lo, iso_hi = load_external_constraints(E)

    print(f"  [{model}] building GlobalLikelihood...", flush=True)
    _GL = GlobalLikelihood(model, ccube, exp_cube_pix_sr, full_mask, E, dE,
                            bub_d, bub_lo, bub_hi, iso_d, iso_lo, iso_hi)
    print(f"  [{model}] ready in {time.time()-t0:.1f} s", flush=True)

    ndim = 1 + 4 * N_BINS
    rng = np.random.default_rng(seed=42 + sum(ord(c) for c in model))

    centers = np.zeros(ndim)
    scales = np.zeros(ndim)
    centers[0] = 1.0;                    scales[0] = 0.3
    for eb in range(N_BINS):
        centers[1+eb]            = 1.0;  scales[1+eb]            = 0.3
        centers[1+N_BINS+eb]     = 1.0;  scales[1+N_BINS+eb]     = 0.3
        centers[1+2*N_BINS+eb]   = 2.0;  scales[1+2*N_BINS+eb]   = 1.0
        centers[1+3*N_BINS+eb]   = 2.0;  scales[1+3*N_BINS+eb]   = 1.0

    p0 = centers + scales * rng.standard_normal((NWALKERS_GLOBAL, ndim))
    p0 = np.clip(p0, 0.01, None)

    from emcee.moves import DEMove, StretchMove
    moves = [(DEMove(), 0.8), (StretchMove(), 0.2)]
    sampler = emcee.EnsembleSampler(NWALKERS_GLOBAL, ndim, _log_prob, moves=moves)

    print(f"  [{model}] timing single log_prob call...", flush=True)
    t_one = time.time()
    lp0 = _log_prob(p0[0])
    t_one = time.time() - t_one
    print(f"  [{model}] one log_prob = {t_one*1000:.2f} ms  (lp={lp0:.3f})", flush=True)

    print(f"  [{model}] verifying walker validity...", flush=True)
    valid = 0
    for w in p0:
        if np.isfinite(_log_prob(w)):
            valid += 1
    print(f"  [{model}] {valid}/{NWALKERS_GLOBAL} walkers have finite log_prob", flush=True)
    if valid < NWALKERS_GLOBAL * 0.5:
        print(f"  [{model}] WARNING: >50% walkers invalid. Sampling may fail.", flush=True)

    expected_per_step = t_one * NWALKERS_GLOBAL
    print(f"  [{model}] ~one step ≈ {expected_per_step:.2f} s  "
          f"(100 steps ≈ {expected_per_step*100/60:.1f} min, "
          f"{NSTEPS} steps ≈ {expected_per_step*NSTEPS/60:.1f} min)", flush=True)

    chunk = 20
    pos = p0
    t_mcmc = time.time()
    mcmc_failed = False
    last_successful_step = 0
    for cs in range(0, NSTEPS, chunk):
        n = min(chunk, NSTEPS - cs)
        t_chunk = time.time()
        try:
            state = sampler.run_mcmc(pos, n, progress=False,
                                      skip_initial_state_check=True)
            pos = state.coords if hasattr(state, 'coords') else state[0]
            last_successful_step = cs + n
        except Exception as e:
            print(f"    [{model}] step {cs}/{NSTEPS} FAILED: {type(e).__name__}: {e}",
                  flush=True)
            print(f"    [{model}] saving results from successful steps so far "
                  f"({last_successful_step})", flush=True)
            mcmc_failed = True
            break
        cum = cs + n
        rate = n / (time.time() - t_chunk)
        eta = (NSTEPS - cum) / rate / 60 if rate > 0 else 0
        print(f"    [{model}] step {cum:>4}/{NSTEPS}  "
              f"{rate:.2f} it/s  ETA={eta:.1f} min  "
              f"elapsed={(time.time()-t_mcmc)/60:.1f} min",
              flush=True)

    nburn_actual = min(NBURN, max(0, last_successful_step // 2))
    if mcmc_failed:
        print(f"  [{model}] MCMC failed early; using first half "
              f"({nburn_actual} steps) as burn-in discard", flush=True)
    print(f"  [{model}] discarding {nburn_actual} burn-in from "
          f"{last_successful_step} total steps", flush=True)

    chain_flat = sampler.get_chain(discard=nburn_actual, flat=True)
    logp_flat = sampler.get_log_prob(discard=nburn_actual, flat=True)
    finite = np.isfinite(logp_flat)
    if not finite.any():
        print(f"  [{model}] FATAL: no finite log_prob samples after burn-in", flush=True)
        return None
    chain_flat = chain_flat[finite]
    logp_flat = logp_flat[finite]
    idx_max = int(np.argmax(logp_flat))
    best = chain_flat[idx_max]
    std = np.std(chain_flat, axis=0, ddof=1)
    lo16 = np.percentile(chain_flat, 16, axis=0)
    hi84 = np.percentile(chain_flat, 84, axis=0)

    c_gce = float(best[0])
    c_gce_std = float(std[0])
    c_gce_lo = float(lo16[0])
    c_gce_hi = float(hi84[0])

    gce_avg = np.zeros(N_BINS)
    gc_cube = fits.open(
        f'{WORK_DIR}/GC_analysis_sanghwan/GC_GCE_model_12yr_front_clean.fits'
    )[0].data
    for eb in range(N_BINS):
        exp_eb = exp_cube_pix_sr[eb]
        gc_full = gc_cube[eb, yroi, xroi]
        with np.errstate(divide='ignore', invalid='ignore'):
            pp = np.where(exp_eb > 0, gc_full / exp_eb, 0.0)
        gce_avg[eb] = (full_mask[eb] * pp).sum() / full_mask[eb].sum()

    e2_dE = (E ** 2) / dE
    flux_best = c_gce * gce_avg * e2_dE
    flux_std = c_gce_std * gce_avg * e2_dE
    flux_lo = c_gce_lo * gce_avg * e2_dE
    flux_hi = c_gce_hi * gce_avg * e2_dE

    coef_per_bin = np.zeros((N_BINS, 5))
    coef_per_bin[:, 0] = best[1:1+N_BINS]
    coef_per_bin[:, 1] = best[1+N_BINS:1+2*N_BINS]
    coef_per_bin[:, 2] = c_gce
    coef_per_bin[:, 3] = best[1+2*N_BINS:1+3*N_BINS]
    coef_per_bin[:, 4] = best[1+3*N_BINS:1+4*N_BINS]

    max_logL = float(logp_flat[idx_max])

    out_dat = os.path.join(OUT_DIR, f'GCE_model_{model}_haebarg_v_claude_{SUFFIX}.dat')
    np.savetxt(out_dat,
               np.column_stack([E, flux_best, flux_std, flux_lo, flux_hi]),
               header=f'E[GeV] E2dN/dE std lo hi   global_c_gce = {c_gce:.4f}')
    out_pkl = os.path.join(OUT_DIR, f'GCE_model_{model}_haebarg_v_claude_{SUFFIX}.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'model': model, 'E': E, 'dE': dE,
            'c_gce_global': c_gce, 'c_gce_global_std': c_gce_std,
            'c_gce_global_lo': c_gce_lo, 'c_gce_global_hi': c_gce_hi,
            'flux_best': flux_best, 'flux_std': flux_std,
            'flux_lo': flux_lo, 'flux_hi': flux_hi,
            'coef_best': coef_per_bin,
            'max_logL_total': max_logL,
            'total_neg2_logL': float(-2.0 * max_logL),
            'gce_avg_per_bin': gce_avg,
        }, f)

    elapsed = (time.time() - t0) / 60
    print(f"==== Model {model} DONE in {elapsed:.1f} min ====", flush=True)
    print(f"  global c_gce = {c_gce:.4f} +{c_gce_hi-c_gce:.4f} -{c_gce-c_gce_lo:.4f}",
          flush=True)
    print(f"  total -2 log L = {-2.0*max_logL:.2f}", flush=True)
    return out_pkl


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: diagnostic_C_global_fit.py MODEL", file=sys.stderr)
        sys.exit(1)
    fit_one_model(sys.argv[1])
