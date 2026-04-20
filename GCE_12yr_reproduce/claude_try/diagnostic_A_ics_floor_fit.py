import os, sys, time, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
from astropy.io import fits
from astropy.wcs import WCS
import emcee
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    WORK_DIR, OUT_DIR,
    CCUBE, EXPCUBE_CENTER,
    NWALKERS, NSTEPS, NBURN, NDIM,
    MODELS, ROI_SLICE, STAGE2_WORKERS,
)
from prepare_masks import FULL_MASK_PATH, prepare_masks
from stage2_fit import (
    BinLikelihood, load_energy_axis, load_external_constraints, solid_angle_per_pixel,
)


C_ICS_FLOOR = float(os.environ.get('C_ICS_FLOOR', '0.1'))
SUFFIX = os.environ.get('FIT_SUFFIX', f'icsfloor{C_ICS_FLOOR:.2f}'.replace('.', 'p'))

_W = {}
_CURRENT_BL = None
_FLOOR = C_ICS_FLOOR


def _worker_init(work_dir, floor_val):
    global _FLOOR
    _FLOOR = floor_val
    os.chdir(work_dir)
    from astropy.io import fits as _fits
    from astropy.wcs import WCS as _WCS

    ccube_hdu = _fits.open(CCUBE)
    _W['ccube'] = ccube_hdu[0].data.astype(np.float64)
    E, dE = load_energy_axis(CCUBE)
    _W['E'] = E; _W['dE'] = dE

    wcs = _WCS(ccube_hdu[0].header).dropaxis(2)
    height, width = _W['ccube'].shape[1], _W['ccube'].shape[2]
    sap_full = solid_angle_per_pixel(width, height, wcs)
    yroi, xroi = ROI_SLICE

    expcube = _fits.open(EXPCUBE_CENTER)[0].data.astype(np.float64)
    _W['exp_cube'] = expcube[:, yroi, xroi] * sap_full[yroi, xroi][None, :, :]

    _W['mask'] = np.load(FULL_MASK_PATH).astype(np.float32)

    bub_d, bub_lo, bub_hi, iso_d, iso_lo, iso_hi = load_external_constraints(E)
    _W['bub_d'] = bub_d; _W['bub_lo'] = bub_lo; _W['bub_hi'] = bub_hi
    _W['iso_d'] = iso_d; _W['iso_lo'] = iso_lo; _W['iso_hi'] = iso_hi
    print(f"  [worker PID={os.getpid()}] init done; c_ics floor = {_FLOOR}", flush=True)


def _log_prob_floor(params):
    if (np.asarray(params) < 0).any():
        return -np.inf
    if params[1] < _FLOOR:
        return -np.inf
    val = _CURRENT_BL.neg2_log_like(params)
    if not np.isfinite(val):
        return -np.inf
    return -0.5 * val


def _fit_bin_task(task):
    global _CURRENT_BL
    model, ebin = task
    pid = os.getpid()
    t0 = time.time()

    bl = BinLikelihood(
        model, ebin,
        _W['ccube'], _W['exp_cube'], _W['mask'],
        _W['E'], _W['dE'],
        _W['bub_d'], _W['bub_lo'], _W['bub_hi'],
        _W['iso_d'], _W['iso_lo'], _W['iso_hi'],
    )
    _CURRENT_BL = bl

    seed = 12345 + ebin * 1000 + (sum(ord(c) for c in model) % 1000)
    rng = np.random.default_rng(seed=seed)
    p0 = np.column_stack([
        rng.uniform(0.3, 1.7, NWALKERS),
        rng.uniform(max(_FLOOR + 0.05, 0.3), 1.7, NWALKERS),
        rng.uniform(0.3, 3.0, NWALKERS),
        rng.uniform(0.5, 6.0, NWALKERS),
        rng.uniform(0.5, 6.0, NWALKERS),
    ])

    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, _log_prob_floor)
    sampler.run_mcmc(p0, NSTEPS, progress=False)

    log_prob_flat = sampler.get_log_prob(discard=NBURN, flat=True)
    chain_flat = sampler.get_chain(discard=NBURN, flat=True)
    idx_max = int(np.argmax(log_prob_flat))
    best = chain_flat[idx_max]
    std = np.std(chain_flat, axis=0, ddof=1)
    med = np.median(chain_flat, axis=0)
    lo16 = np.percentile(chain_flat, 16, axis=0)
    hi84 = np.percentile(chain_flat, 84, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        gce_per_pix = np.where(bl.exp > 0, bl.gc / bl.exp, 0.0)
    gce_avg = float((bl.mask * gce_per_pix).sum() / bl.mask_sum)

    e2_dE = (_W['E'][ebin] ** 2) / _W['dE'][ebin]
    flux_best = float(best[2]) * gce_avg * e2_dE
    flux_std = float(std[2]) * gce_avg * e2_dE
    flux_lo = float(lo16[2]) * gce_avg * e2_dE
    flux_hi = float(hi84[2]) * gce_avg * e2_dE

    elapsed = time.time() - t0
    print(f"  [{time.strftime('%H:%M:%S')}] [PID {pid}] {model} bin {ebin:2d} done in "
          f"{elapsed/60:.1f}min  c_ics={best[1]:.3f}  c_gce={best[2]:.3f}  flux={flux_best:.3e}",
          flush=True)

    return {
        'model': model, 'ebin': ebin,
        'E_val': float(_W['E'][ebin]), 'dE_val': float(_W['dE'][ebin]),
        'best': best, 'med': med, 'std': std, 'lo16': lo16, 'hi84': hi84,
        'max_logL': float(log_prob_flat[idx_max]),
        'gce_avg': gce_avg,
        'flux_best': flux_best, 'flux_std': flux_std,
        'flux_lo': flux_lo, 'flux_hi': flux_hi,
        'elapsed_sec': elapsed,
    }


def run_parallel(target_models=None, n_workers=None, n_ebins=14):
    if target_models is None:
        target_models = MODELS
    if n_workers is None:
        n_workers = STAGE2_WORKERS

    os.chdir(WORK_DIR)
    if not os.path.exists(FULL_MASK_PATH):
        prepare_masks()

    tasks = [(m, b) for m in target_models for b in range(n_ebins)]
    print(f"[{time.strftime('%H:%M:%S')}] EXPERIMENT A :: c_ics floor = {C_ICS_FLOOR} :: "
          f"suffix = {SUFFIX}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] {len(tasks)} tasks on {n_workers} workers "
          f"({len(target_models)} models × {n_ebins} bins)", flush=True)

    t_start = time.time()
    ctx = get_context('fork')
    with ctx.Pool(processes=n_workers, initializer=_worker_init,
                  initargs=(WORK_DIR, C_ICS_FLOOR)) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_fit_bin_task, tasks, chunksize=1), 1):
            results.append(r)
            print(f"  [{time.strftime('%H:%M:%S')}] progress {i}/{len(tasks)}", flush=True)

    total = (time.time() - t_start) / 60
    print(f"\n[{time.strftime('%H:%M:%S')}] all {len(tasks)} tasks done in {total:.1f} min", flush=True)

    for m in target_models:
        mr = sorted([r for r in results if r['model'] == m], key=lambda x: x['ebin'])
        if len(mr) != n_ebins:
            print(f"  [warn] model {m}: got {len(mr)}/{n_ebins} bins", flush=True)
            continue
        E = np.array([r['E_val'] for r in mr])
        dE = np.array([r['dE_val'] for r in mr])
        flux_best = np.array([r['flux_best'] for r in mr])
        flux_std = np.array([r['flux_std'] for r in mr])
        flux_lo = np.array([r['flux_lo'] for r in mr])
        flux_hi = np.array([r['flux_hi'] for r in mr])
        coef_best = np.array([r['best'] for r in mr])
        coef_med = np.array([r['med'] for r in mr])
        coef_std = np.array([r['std'] for r in mr])
        max_logL = np.array([r['max_logL'] for r in mr])
        gce_avg_per_bin = np.array([r['gce_avg'] for r in mr])

        out_dat = os.path.join(OUT_DIR, f'GCE_model_{m}_haebarg_v_claude_{SUFFIX}.dat')
        np.savetxt(out_dat,
                   np.column_stack([E, flux_best, flux_std, flux_lo, flux_hi]),
                   header=f'E[GeV]  E2dN/dE_best  std  lower_1sigma  upper_1sigma   (c_ics>={C_ICS_FLOOR})')
        out_pkl = os.path.join(OUT_DIR, f'GCE_model_{m}_haebarg_v_claude_{SUFFIX}.pkl')
        with open(out_pkl, 'wb') as f:
            pickle.dump({
                'model': m, 'E': E, 'dE': dE,
                'c_ics_floor': C_ICS_FLOOR,
                'flux_best': flux_best, 'flux_std': flux_std,
                'flux_lo': flux_lo, 'flux_hi': flux_hi,
                'coef_best': coef_best, 'coef_med': coef_med, 'coef_std': coef_std,
                'max_logL': max_logL, 'gce_avg_per_bin': gce_avg_per_bin,
                'total_neg2_logL': float(-2.0 * max_logL.sum()),
            }, f)
        print(f"  saved: {out_dat}", flush=True)

    return results


if __name__ == '__main__':
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    run_parallel(target_models=targets)
