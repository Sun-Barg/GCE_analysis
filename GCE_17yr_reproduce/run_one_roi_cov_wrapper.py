#!/usr/bin/env python3
"""run_one_roi_cov_wrapper.py — cov pipeline phase-split wrapper.

Mirror of run_one_model_wrapper.py for the covariance pipeline.

launch_all_rois.py calls this as 'python run_one_roi_cov_wrapper.py ROI'.
It runs run_one_roi_cov.py as two sequential subprocesses:

  Phase 1 (prepare): XML build + gtsrcmaps x2 + gtmodel x12
                     -> fermitools state dies with the subprocess
  Phase 2 (mcmc)   : Likelihood + serial emcee x14 bin + save .dat
                     -> fresh process (no fermitools fork-unsafe state)

Root cause (main pipeline Jobs 3-8, 2026-05-14/15): running emcee in
the same Python process as gtsrcmaps/gtmodel triggers external SIGKILL.
cov has the same fermitools-then-MCMC structure + heavier RSS (~30-50GB
per ROI gtsrcmaps, ~1000+ FL16Y sources), so the same split is required.

interface:
  python run_one_roi_cov_wrapper.py ROI

exit codes:
  prepare fails -> that rc, mcmc not run
  mcmc fails    -> that rc
  both ok       -> 0
Env vars (e.g. DIAG_SAVE_CHAIN) propagate to both subprocesses.
"""
import os
import sys
import time
import subprocess


def _run_phase(runner, roi, phase):
    env = os.environ.copy()
    env['RUN_PHASE'] = phase
    print(f'\n---- phase: {phase} ----', flush=True)
    t = time.time()
    rc = subprocess.call(
        [sys.executable, '-u', runner, str(roi)],
        env=env,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )
    dt = (time.time() - t) / 60
    if rc != 0:
        print(f'---- phase {phase} FAIL  rc={rc}  elapsed={dt:.1f} min ----',
              flush=True)
    else:
        print(f'---- phase {phase} done  elapsed={dt:.1f} min ----', flush=True)
    return rc


def main():
    if len(sys.argv) < 2:
        print('Usage: python run_one_roi_cov_wrapper.py ROI', file=sys.stderr)
        sys.exit(2)
    roi = sys.argv[1].strip()

    runner = './run_one_roi_cov.py'
    if not os.path.exists(runner):
        runner = 'run_one_roi_cov.py'
    if not os.path.exists(runner):
        print(f'[FATAL] run_one_roi_cov.py not found in cwd ({os.getcwd()})',
              file=sys.stderr)
        sys.exit(2)

    print(f'==== cov wrapper start  roi={roi}  pid={os.getpid()}  '
          f'runner={runner} ====', flush=True)
    t0 = time.time()

    rc = _run_phase(runner, roi, 'prepare')
    if rc != 0:
        print(f'==== cov wrapper FAIL  total={(time.time()-t0)/60:.1f} min ====',
              flush=True)
        sys.exit(rc)

    rc = _run_phase(runner, roi, 'mcmc')
    if rc != 0:
        print(f'==== cov wrapper FAIL  total={(time.time()-t0)/60:.1f} min ====',
              flush=True)
        sys.exit(rc)

    print(f'==== cov wrapper done  roi={roi}  '
          f'total={(time.time()-t0)/60:.1f} min ====', flush=True)


if __name__ == '__main__':
    main()
