#!/usr/bin/env python3
"""
launch_all_roi_prep.py — parallel launcher for the 22-ROI cov-prep stage.

Spawns up to N concurrent `prepare_one_roi_cov.py <roi>` subprocesses,
continuously refilling slots as workers finish. Drop-in companion to
launch_all_rois.py (cov MCMC launcher) with three substantive
differences:

  1. Memory budget is light (~1 GB/process vs 30-50 GB for cov gtsrcmaps),
     so default --workers is 4 (vs 2 for cov MCMC).
  2. Prep outputs stay in the working directory. There is no
     `results_*_dir` and no move_to_results — the outputs ARE the inputs
     to the next stage (run_one_roi_cov.py reads them in place).
  3. Prerequisite check at startup. prepare_one_roi_cov.py depends on 5
     outputs of prepare_common.py (Steps 3, 6, 7, 11, 12). If any are
     missing, this launcher aborts up front rather than spawning 22
     workers that all FATAL on the same missing file.

Completion of a ROI is decided by both rc == 0 AND the 3 expected
output files existing on disk:
    GC_analysis_FL16Y/GC_expcube_center_17yr_front_clean_l{roi}.fits
    GC_analysis_FL16Y/Model/GC_psc_model_FL16Y_l{roi}.xml
    GC_analysis_FL16Y/Model/GC_mask_60x60_definitions_FL16Y_l{roi}.npy

A worker exit code of 2 means FATAL (stale-file integrity failure,
missing prereq, etc.) — these are NOT retried (same condition would
recur). Exit codes other than 0/2 are treated as crashes and retried
up to --max-retries.

Usage:
    nohup python launch_all_roi_prep.py > launcher_prep.log 2>&1 &
    tail -f launcher_prep.log

    # subset:
    python launch_all_roi_prep.py --rois 25,-70 --workers 1

    # rebuild a specific step on a subset (passthrough to worker):
    python launch_all_roi_prep.py --rois 25 --worker-args "--force-step 3"

    # full rebuild of all 22 ROIs:
    python launch_all_roi_prep.py --worker-args "--force-all"

State:
    - .launcher_cov_prep.pid       single-instance lock
    - logs/cov_prep/roi_l{ROI}.log per-ROI log (append on retry)

Known limitation (mirrors launch_all_rois.py, intentionally NOT solved here):
    fermitools .par files in $PFILES are shared across all workers. With
    --workers >= 2, two gtexpcube2 invocations could race on
    gtexpcube2.par. In practice this is benign — each invocation passes
    all parameters explicitly via GtApp() — but if you see sporadic
    "could not lock .par" errors, reduce --workers to 1 or add per-worker
    PFILES isolation inside prepare_one_roi_cov.py.

Author: haebarg (2026)

Changes:
  [fb17-cov-v1] (2026-07-28) FB17=1 env -> front+back 17-bin cov variant
      (main pipeline과 동일 패턴): WORK_DIR/FRONT/evtype 전환, 결과는
      results_cov_fb17/ 분리(기존 front 22개 .dat와 카운트 충돌 방지).
      env 미설정 시 기존 fiducial 동작과 동일.
"""

import os
import sys
import time
import signal
import shlex
import argparse
import subprocess
from datetime import datetime

# ============================================================
# 22 control ROIs — same set as launch_all_rois.py (cov MCMC)
# ============================================================
ALL_ROIS = [r for r in range(-70, 75, 5) if r != 0 and abs(r) >= 20]
assert len(ALL_ROIS) == 22

FRONT          = '_front'
RUNNER_SCRIPT  = 'prepare_one_roi_cov.py'
LOCK_FILE      = '.launcher_cov_prep.pid'
LOG_DIR        = 'logs/cov_prep'

# WORK_DIR-relative paths to per-ROI prep outputs (KEEP IN SYNC with
# prepare_one_roi_cov.py — duplicated here for fast filesystem checks
# without importing the heavy fermitools-dependent worker module).
WORK_DIR       = './GC_analysis_FL16Y'

# FB17 variant — env switch (prepare_common.py 동일 패턴).
FB17 = bool(os.environ.get('FB17', '').strip())
if FB17:
    WORK_DIR = './GC_analysis_FL16Y_fb17'
    FRONT    = '_front_back'
    print(f'[config] FB17=1 -> WORK_DIR={WORK_DIR}, FRONT={FRONT!r}', flush=True)


def _per_roi_outputs(roi):
    return [
        f'{WORK_DIR}/GC_expcube_center_17yr{FRONT}_clean_l{roi}.fits',
        f'{WORK_DIR}/Model/GC_psc_model_FL16Y_l{roi}.xml',
        f'{WORK_DIR}/Model/GC_mask_60x60_definitions_FL16Y_l{roi}.npy',
    ]


# prepare_common.py outputs that prepare_one_roi_cov.py depends on
# (KEEP IN SYNC with prepare_one_roi_cov.check_prerequisites).
PREREQS = [
    (f'{WORK_DIR}/Allsky_ltcube_17yr{FRONT}_clean.fits',  'prepare_common.py Step 7  (gtltcube)'),
    (f'{WORK_DIR}/GC_ccube_17yr{FRONT}_clean.fits',       'prepare_common.py Step 6  (GC_ccube)'),
    (f'{WORK_DIR}/bin_definitions.fits',                  'prepare_common.py Step 3  (bin_definitions.fits)'),
    (f'{WORK_DIR}/Model/GC_psc_model_FL16Y.xml',          'prepare_common.py Step 11 (main psc XML)'),
    (f'{WORK_DIR}/Model/source_classification.npz',       'prepare_common.py Step 12 (source classification npz)'),
]


# ============================================================
# Helpers
# ============================================================

def _ts():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def is_complete(roi):
    """All three per-ROI prep outputs exist on disk (path level only).

    The integrity check (fits.verify, XML parse, mask shape/sum) lives
    inside prepare_one_roi_cov.py and runs on every worker invocation.
    A stale file therefore causes the worker to FATAL-abort (rc=2),
    which we detect via the rc check in the reap loop below — NOT via
    is_complete (which only knows the file exists).
    """
    return all(os.path.exists(f) for f in _per_roi_outputs(roi))


def launch_one(roi, log_path, env, worker_extra_args):
    log_file = open(log_path, 'a')
    log_file.write(f"\n========== {_ts()} START roi={roi} ==========\n")
    log_file.flush()
    cmd = [sys.executable, '-u', RUNNER_SCRIPT, str(roi)] + worker_extra_args
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return proc, log_file


def acquire_lock():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                old_pid = int(f.read().strip())
            try:
                os.kill(old_pid, 0)
                print(f"[FATAL] another cov-prep launcher (pid={old_pid}) is already running.")
                print(f"        if it's actually dead, remove {LOCK_FILE} and retry.")
                sys.exit(2)
            except ProcessLookupError:
                print(f"[warn] stale lockfile (pid {old_pid} gone), reclaiming.")
        except (ValueError, OSError):
            print(f"[warn] unreadable lockfile, reclaiming.")
    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))


def release_lock():
    try:
        if os.path.exists(LOCK_FILE):
            with open(LOCK_FILE) as f:
                pid = int(f.read().strip())
            if pid == os.getpid():
                os.remove(LOCK_FILE)
    except (ValueError, OSError):
        pass


def check_prereqs():
    """Verify all 5 prepare_common.py outputs exist before spawning any
    workers. Reports each missing file with its producing prep Step."""
    missing = [(p, src) for p, src in PREREQS if not os.path.exists(p)]
    if missing:
        print(f"[FATAL] missing prepare_common.py outputs — run that first:")
        for p, src in missing:
            print(f"    - {p}")
            print(f"        (produced by: {src})")
        sys.exit(2)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n', 1)[0])
    ap.add_argument('--workers',        type=int,   default=4,
                    help='max concurrent prep subprocesses (default: 4; '
                         'prep is light, ~1 GB/process; ~6 is the practical '
                         'ceiling)')
    ap.add_argument('--rois',           type=str,   default='',
                    help='comma-separated ROI subset, e.g. "25,-70" '
                         '(default: all 22)')
    ap.add_argument('--max-retries',    type=int,   default=3,
                    help='max relaunch attempts per ROI for non-FATAL crashes '
                         '(default: 3). rc=2 FATALs are never retried.')
    ap.add_argument('--poll-sec',       type=int,   default=15,
                    help='polling interval, seconds (default: 15; prep is '
                         'faster than cov MCMC)')
    ap.add_argument('--max-runtime-hr', type=float, default=6,
                    help='hard timeout, hours (default: 6; prep should take '
                         '~1-2 hr with --workers 4)')
    ap.add_argument('--worker-args',    type=str,   default='',
                    help='extra args passed verbatim to prepare_one_roi_cov.py '
                         '(e.g. "--force-step 3" or "--force-all"). Parsed '
                         'with shlex.')
    args = ap.parse_args()

    if not os.path.exists(RUNNER_SCRIPT):
        print(f"[FATAL] {RUNNER_SCRIPT} not found in cwd ({os.getcwd()}).")
        sys.exit(2)
    if args.workers > 6:
        print(f"[warn] --workers {args.workers} is unusually high for prep; "
              f"~6 is the practical ceiling.")
        print(f"       monitor `free -h` and watch for fermitools .par "
              f"contention.")

    check_prereqs()
    os.makedirs(LOG_DIR, exist_ok=True)
    acquire_lock()

    # ROI selection
    if args.rois.strip():
        try:
            rois = [int(r.strip()) for r in args.rois.split(',') if r.strip()]
        except ValueError:
            print(f"[FATAL] --rois must be integers, got: {args.rois!r}")
            release_lock(); sys.exit(2)
        bad = [r for r in rois if r not in ALL_ROIS]
        if bad:
            print(f"[FATAL] invalid ROI(s): {bad}")
            print(f"        valid: {ALL_ROIS}")
            release_lock(); sys.exit(2)
    else:
        rois = ALL_ROIS[:]

    # Worker extra args (shlex split, so quoted values work)
    try:
        worker_extra = shlex.split(args.worker_args) if args.worker_args.strip() else []
    except ValueError as e:
        print(f"[FATAL] could not parse --worker-args: {e}")
        release_lock(); sys.exit(2)

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    # State
    running          = {}      # roi -> (Popen, log_file_handle)
    attempts         = {r: 0 for r in rois}
    permanent_failed = set()
    t_start          = time.time()

    # Signal cleanup (mirrors launch_all_rois.py exactly)
    _interrupt_flag = {'stop': False}
    def handle_signal(signum, frame):
        if _interrupt_flag['stop']:
            return
        _interrupt_flag['stop'] = True
        print(f"\n[{_ts()}] signal {signum} received — terminating "
              f"{len(running)} children")
        for r, (proc, lf) in list(running.items()):
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        deadline = time.time() + 10
        for r, (proc, lf) in list(running.items()):
            remaining = max(0.1, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            try:
                lf.write(f"\n========== {_ts()} INTERRUPTED ==========\n")
                lf.close()
            except Exception:
                pass
        release_lock()
        sys.exit(130)

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    initially_complete = [r for r in rois if is_complete(r)]

    print(f"[{_ts()}] cov-prep launcher start  pid={os.getpid()}")
    print(f"  ROIs             : {len(rois)}")
    print(f"  workers          : {args.workers}")
    print(f"  worker args      : {worker_extra if worker_extra else '—'}")
    print(f"  log dir          : {LOG_DIR}")
    print(f"  max retries      : {args.max_retries}")
    print(f"  poll interval    : {args.poll_sec}s")
    print(f"  initially done   : {len(initially_complete)}/{len(rois)} "
          f"(path-existence only; worker re-verifies integrity)")

    # ============================================================
    # Main polling loop
    # ============================================================
    last_status_ts = 0.0
    try:
        while True:
            now = time.time()

            completed = [r for r in rois if is_complete(r)]
            not_done  = [r for r in rois
                         if r not in completed and r not in permanent_failed]

            if len(not_done) == 0:
                break
            if (now - t_start) / 3600 > args.max_runtime_hr:
                print(f"[{_ts()}] max runtime {args.max_runtime_hr}h "
                      f"exceeded — stopping")
                handle_signal(signal.SIGTERM, None)

            # Reap finished subprocesses
            for r in list(running.keys()):
                proc, log_file = running[r]
                rc = proc.poll()
                if rc is None:
                    continue
                log_file.write(f"\n========== {_ts()} END roi={r} rc={rc} ==========\n")
                log_file.close()
                del running[r]

                if rc == 0 and is_complete(r):
                    # Genuine success: clean exit AND all 3 outputs present.
                    print(f"[{_ts()}] [done ] roi={r:+4d}  rc=0")
                elif rc == 2:
                    # FATAL (stale-file integrity, missing prereq, etc.).
                    # Same condition would recur on retry — give up immediately.
                    permanent_failed.add(r)
                    print(f"[{_ts()}] [FATAL] roi={r:+4d}  rc=2 — worker "
                          f"aborted (likely stale-file integrity or missing "
                          f"prereq); see {LOG_DIR}/roi_l{r}.log")
                else:
                    # Crash (signal, OOM, segfault, etc.) — retry.
                    attempts[r] += 1
                    if attempts[r] >= args.max_retries:
                        permanent_failed.add(r)
                        print(f"[{_ts()}] [FAIL ] roi={r:+4d}  "
                              f"attempts={attempts[r]}  giving up "
                              f"(see {LOG_DIR}/roi_l{r}.log)")
                    else:
                        print(f"[{_ts()}] [retry] roi={r:+4d}  "
                              f"attempt {attempts[r]}/{args.max_retries}  "
                              f"rc={rc}")

            # Fill empty slots
            runnable = [r for r in not_done
                        if r not in running and r not in permanent_failed]
            while len(running) < args.workers and runnable:
                r = runnable.pop(0)
                log_path = os.path.join(LOG_DIR, f"roi_l{r}.log")
                try:
                    proc, log_file = launch_one(r, log_path, env, worker_extra)
                except Exception as e:
                    print(f"[{_ts()}] [error] failed to launch roi={r}: {e}")
                    attempts[r] += 1
                    if attempts[r] >= args.max_retries:
                        permanent_failed.add(r)
                    continue
                running[r] = (proc, log_file)
                print(f"[{_ts()}] [start] roi={r:+4d}  pid={proc.pid}  "
                      f"running={len(running)}/{args.workers}")

            # Periodic status summary (every 2 min — prep is faster than cov)
            if now - last_status_ts > 120:
                n_done = len(completed)
                n_run  = len(running)
                n_left = len(not_done) - n_run
                n_fail = len(permanent_failed)
                elapsed = (now - t_start) / 60
                print(f"[{_ts()}] [status] done={n_done:>3}  "
                      f"running={n_run:>2}  queued={n_left:>3}  "
                      f"failed={n_fail:>2}  elapsed={elapsed:.1f} min")
                sys.stdout.flush()
                last_status_ts = now

            time.sleep(args.poll_sec)

    finally:
        completed_final = [r for r in rois if is_complete(r)]
        elapsed_hr = (time.time() - t_start) / 3600
        print(f"\n=========== final ===========")
        print(f"  completed : {len(completed_final)}/{len(rois)}")
        print(f"  failed    : {len(permanent_failed)} "
              f"-> {sorted(permanent_failed)}")
        print(f"  elapsed   : {elapsed_hr:.2f} hr")
        release_lock()


if __name__ == '__main__':
    main()
