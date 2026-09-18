#!/usr/bin/env python3
"""run_one_model_wrapper_profiles.py — halo-profile (thesis 6.3) phase wrapper.

Identical phase-split logic to run_one_model_wrapper.py (Job 6 pattern:
fermitools state in a process + subsequent MCMC = SIGKILL, so prepare and
mcmc run as separate subprocesses), BUT it drives run_one_model_profiles.py
instead of run_one_model.py, so the GCE spatial template is the alt-profile
rho^2 map selected by GCE_PROFILE.

Why a separate wrapper (production run_one_model_wrapper.py untouched):
  - runner is run_one_model_profiles.py (NFW-production scripts unchanged).
  - GCE_PROFILE is REQUIRED here and must be 'Einasto2' or 'Burkert2'.
    This guards against accidentally writing NFW-named files through the
    profiles path: NFW2 (PROF_SUFFIX='') would collide with production
    outputs, so it is refused — re-run NFW via the production wrapper.
  - env = os.environ.copy() (as in the original) already propagates
    GCE_PROFILE to BOTH the prepare and mcmc subprocesses, so the
    profile-tagged paths are consistent across phases.

interface:
  GCE_PROFILE=Einasto2 python run_one_model_wrapper_profiles.py MODEL
  GCE_PROFILE=Burkert2 python run_one_model_wrapper_profiles.py MODEL

exit codes:
  prepare fail -> its rc, mcmc not run
  mcmc fail    -> its rc
  all ok       -> 0
Other env vars (e.g. DIAG_SAVE_CHAIN, MASK_VARIANT, PHASE2_CASE) are
passed through to both subprocesses unchanged.
"""
import os
import sys
import time
import subprocess

ALLOWED_PROFILES = ('Einasto2', 'Burkert2')


def _run_phase(runner, model, phase):
    env = os.environ.copy()
    env['RUN_PHASE'] = phase
    print(f'\n---- phase: {phase} ----', flush=True)
    t = time.time()
    rc = subprocess.call(
        [sys.executable, '-u', runner, model],
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
        print('Usage: GCE_PROFILE=Einasto2|Burkert2 '
              'python run_one_model_wrapper_profiles.py MODEL', file=sys.stderr)
        sys.exit(2)
    model = sys.argv[1].strip()

    profile = os.environ.get('GCE_PROFILE', '').strip()
    if profile not in ALLOWED_PROFILES:
        print(f'[FATAL] GCE_PROFILE must be one of {ALLOWED_PROFILES}, '
              f'got {profile!r}.', file=sys.stderr)
        print('        (To re-run NFW2, use the production '
              'run_one_model_wrapper.py instead — this profiles wrapper '
              'refuses NFW2 to avoid clobbering production outputs.)',
              file=sys.stderr)
        sys.exit(2)

    runner = './run_one_model_profiles.py'
    if not os.path.exists(runner):
        runner = 'run_one_model_profiles.py'
    if not os.path.exists(runner):
        print(f'[FATAL] run_one_model_profiles.py not found in cwd '
              f'({os.getcwd()})', file=sys.stderr)
        sys.exit(2)

    print(f'==== wrapper start  model={model}  profile={profile}  '
          f'pid={os.getpid()}  runner={runner} ====', flush=True)
    t0 = time.time()

    rc = _run_phase(runner, model, 'prepare')
    if rc != 0:
        print(f'==== wrapper FAIL  total={(time.time()-t0)/60:.1f} min ====',
              flush=True)
        sys.exit(rc)

    rc = _run_phase(runner, model, 'mcmc')
    if rc != 0:
        print(f'==== wrapper FAIL  total={(time.time()-t0)/60:.1f} min ====',
              flush=True)
        sys.exit(rc)

    print(f'==== wrapper done  model={model}  profile={profile}  '
          f'total={(time.time()-t0)/60:.1f} min ====', flush=True)


if __name__ == '__main__':
    main()
