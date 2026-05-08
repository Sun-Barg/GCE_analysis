#!/usr/bin/env python3
"""run_one_bin.py — Run MCMC for ONE (roi, bin) pair as a standalone subprocess.

Usage:
    python run_one_bin.py <ROI> <BIN>          # ROI in {-70..70 step 5, |roi|>=20}, BIN in 0..13
    python run_one_bin.py 25 0
    python run_one_bin.py -45 7

Output:
    GCE_cov_l<ROI>_bin<BIN>_front_17yr_cholis.npz   (single-bin chain results)

Strategy: each invocation = fresh Python process. Short-lived (~5-8 min).
If killed by VS Code or any other cause, just retry.

After all 14 bins for a ROI are done, run merge_bins_to_roi.py to assemble.

ASSUMES:
- All prep is done by run_one_roi.py first (gtexpcube2, gtbin, mask, XML, gtsrcmaps, gtmodel)
- This script ONLY does MCMC for the given (roi, bin)
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import emcee
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('roi', type=int, help='Galactic longitude (e.g., 25, -45)')
    p.add_argument('bin', type=int, help='Energy bin index (0-13)')
    p.add_argument('--force', action='store_true', help='re-run even if output exists')
    p.add_argument('--n-steps', type=int, default=500, help='MCMC steps')
    p.add_argument('--burn-in', type=int, default=100, help='Burn-in steps to discard')
    p.add_argument('--n-walkers', type=int, default=100, help='Number of walkers')
    args = p.parse_args()

    # Output filename
    out_npz = f'./GCE_cov_l{args.roi}_bin{args.bin:02d}_front_17yr_cholis.npz'
    if os.path.exists(out_npz) and not args.force:
        print(f'[skip] {out_npz} already exists', flush=True)
        return 0

    # Import run_one_roi as module (loads all globals: Likelihood, log_probability,
    # constraints, steradian_per_pixel, etc.)
    print(f'[load] importing run_one_roi.py module', flush=True)
    t_load = time.time()
    import importlib.util
    spec = importlib.util.spec_from_file_location('roi_module', 
                                                   os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                'run_one_roi.py'))
    roi_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(roi_module)
    print(f'[load] done in {time.time()-t_load:.1f}s', flush=True)

    # Verify the bin is valid
    if not (0 <= args.bin < 14):
        print(f'[FAIL] bin {args.bin} out of range [0, 14)', flush=True)
        return 1

    # MCMC
    print(f'[mcmc] starting roi={args.roi} bin={args.bin}', flush=True)
    t_mcmc = time.time()

    nwalkers = args.n_walkers
    ndim = 5
    initial_state = np.array([0.5, 1.5, 1.0, 100.0, 1.0]) + 0.01*np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, roi_module.log_probability,
        args=(args.bin, args.roi)
    )
    # CRITICAL: progress=False (tqdm interferes with VS Code terminal)
    sampler.run_mcmc(initial_state, args.n_steps, progress=False)
    flat = sampler.get_chain(discard=args.burn_in, thin=1, flat=True)
    log_prob = sampler.get_log_prob(discard=args.burn_in, thin=1, flat=True)

    max_idx = np.argmax(log_prob)
    max_pos = flat[max_idx]
    median_pos = np.median(flat, axis=0)
    std_pos = np.std(flat, axis=0)
    upper = np.percentile(flat, 84, axis=0)
    lower = np.percentile(flat, 16, axis=0)

    np.savez(
        out_npz,
        fitted_params=max_pos,
        fitted_params_std=std_pos,
        fitted_params_median=median_pos,
        fitted_params_upper=upper,
        fitted_params_lower=lower,
        max_likelihood=log_prob[max_idx],
        roi=args.roi,
        bin=args.bin,
    )

    dt = time.time() - t_mcmc
    print(f'[done] saved {out_npz}', flush=True)
    print(f'[done] roi={args.roi} bin={args.bin} in {dt/60:.1f}min ({args.n_steps/dt:.1f} it/s)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
