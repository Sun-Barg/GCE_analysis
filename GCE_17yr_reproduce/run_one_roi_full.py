#!/usr/bin/env python3
"""run_one_roi_full.py — Full pipeline for one cov ROI.

Does everything for one ROI:
  1. Prep (gtexpcube2, gtbin, mask, gtsrcmaps, gtmodel) via run_one_roi.py --prep-only
  2. MCMC for 14 bins via orchestrate_cov_bins.py (each bin = subprocess)
  3. Merge per-bin npz into per-ROI npz via merge_bins_to_roi.py

Usage:
    python run_one_roi_full.py 25
    python run_one_roi_full.py -70

Or in a loop for all 19 remaining ROIs:
    for roi in -70 -65 -60 -55 -50 -45 -40 -35 -30 -25 30 35 40 45 50 55 60 65 70; do
        python run_one_roi_full.py $roi
    done

Skips stages that are already complete (resume-friendly).
"""

import argparse
import os
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('roi', type=int)
    p.add_argument('--max-retries', type=int, default=10)
    args = p.parse_args()

    roi = args.roi
    final_npz = f'./GCE_cov_l{roi}_front_17yr_cholis_fit.npz'

    if os.path.exists(final_npz):
        print(f'[skip] roi={roi}: {final_npz} already exists')
        return 0

    print(f'================================================================')
    print(f' run_one_roi_full.py: roi={roi}  start={time.strftime("%H:%M:%S")}')
    print(f'================================================================')

    t_start = time.time()

    # Stage 1: Prep (with retry — mask_creation/gtsrcmaps may be killed by VS Code)
    prep_done_marker = f'./GC_analysis_FL16Y/GC_isotropic_model_17yr_front_clean_no_convol_l{roi}.fits'
    if os.path.exists(prep_done_marker):
        print(f'[skip] prep: {prep_done_marker} exists')
    else:
        print(f'\n[stage 1] PREP for roi={roi} (gtexpcube2 + gtbin + mask + XML + gtsrcmaps + gtmodel)')
        print(f'          ~1-2 hours expected')
        prep_success = False
        for prep_attempt in range(1, args.max_retries + 1):
            t0 = time.time()
            print(f'[run ] prep attempt {prep_attempt}/{args.max_retries}')
            result = subprocess.run(
                ['python', 'run_one_roi.py', str(roi), '--prep-only'],
            )
            dt = (time.time() - t0) / 60
            if result.returncode == 0 and os.path.exists(prep_done_marker):
                print(f'[ok ] prep done in {dt:.0f}min')
                prep_success = True
                break
            print(f'[fail] prep attempt {prep_attempt} rc={result.returncode} after {dt:.0f}min')
            if result.returncode == -9:
                print(f'        rc=-9 is SIGKILL (likely VS Code) - retrying. Most prep work cached, next attempt will resume.')
            time.sleep(5)
        if not prep_success:
            print(f'[FAIL] prep gave up after {args.max_retries} attempts for roi={roi}')
            return 1

    # Stage 2: MCMC for 14 bins (each as separate subprocess)
    all_bins_done = all(
        os.path.exists(f'./GCE_cov_l{roi}_bin{b:02d}_front_17yr_cholis.npz')
        for b in range(14)
    )
    if all_bins_done:
        print(f'[skip] all 14 per-bin npz files already exist')
    else:
        print(f'\n[stage 2] MCMC for 14 bins of roi={roi}')
        print(f'          ~7 min/bin × 14 = ~100min expected')
        t0 = time.time()
        result = subprocess.run(
            ['python', 'orchestrate_cov_bins.py', str(roi), 
             '--max-retries', str(args.max_retries)],
        )
        if result.returncode != 0:
            print(f'[WARN] some bins failed for roi={roi}, rc={result.returncode}')
            print(f'       check which bins are missing and re-run')
            return 1
        print(f'[ok ] MCMC done in {(time.time()-t0)/60:.0f}min')

    # Stage 3: Merge per-bin into per-ROI
    print(f'\n[stage 3] MERGE per-bin npz into per-ROI npz')
    result = subprocess.run(['python', 'merge_bins_to_roi.py', str(roi)])
    if result.returncode != 0:
        print(f'[FAIL] merge failed for roi={roi}')
        return 1

    total_min = (time.time() - t_start) / 60
    print(f'\n================================================================')
    print(f' [DONE] roi={roi} in {total_min:.0f}min')
    print(f' Output: {final_npz}')
    print(f'================================================================')
    return 0


if __name__ == '__main__':
    sys.exit(main())
