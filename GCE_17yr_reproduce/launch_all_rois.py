#!/usr/bin/env python3
"""
launch_all_rois.py — robust 22-ROI launcher for 17yr GCE covariance pipeline.

Identical robust pattern to launch_all_models.py:
- single-instance lockfile, SIGINT/SIGTERM cleanup
- skip-if-completed (auto-detect completed ROIs from filesystem)
- auto-move outputs to results_cov_17yr/ on completion
- retry-on-fail (max-retries, default 3)
- continuous slot refill — dead/finished workers replaced immediately

Memory constraint:
    cov gtsrcmaps loads ~1000+ FL16Y point sources per ROI and consumes
    ~30-50 GB RSS at peak. Default --workers 2 is the safe ceiling on
    neutrino (128 GB total, ~30 GB system reserve).
    DO NOT raise --workers above 3 without monitoring `free -h`.

Usage:
    nohup python launch_all_rois.py > launcher_cov.log 2>&1 &
    tail -f launcher_cov.log

    # subset:
    python launch_all_rois.py --rois -50,30,50 --workers 1

State:
    - .launcher_cov.pid          single-instance lock
    - results_cov_17yr/*.dat     completion marker
    - log_cov_l{ROI}.txt         per-ROI log (append on retry)

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
import argparse
import subprocess
import shutil
from datetime import datetime
from launcher_locks import (
    write_lock, remove_lock, cleanup_stale_locks,
    adopt_running_workers, adopted_alive,
)

# ============================================================
# 22 control ROIs (cov notebook: range(-70, 75, 5) skip 0 and |roi|<20)
# ============================================================
ALL_ROIS = [r for r in range(-70, 75, 5) if r != 0 and abs(r) >= 20]
assert len(ALL_ROIS) == 22

FRONT          = '_front'
RUNNER_SCRIPT  = 'run_one_roi_cov_wrapper.py'
LOCK_FILE      = '.launcher_cov.pid'
RESULT_EXTS    = ['.dat', '_fit.npz', '_likelihood_value']
DEFAULT_RESULTS_DIR = 'results_cov_17yr'

# FB17 variant — env switch: front_back 네이밍 + 결과 디렉토리 분리.
FB17 = bool(os.environ.get('FB17', '').strip())
if FB17:
    FRONT = '_front_back'
    DEFAULT_RESULTS_DIR = 'results_cov_fb17'
    print(f'[config] FB17=1 -> FRONT={FRONT!r}, '
          f'results_dir={DEFAULT_RESULTS_DIR}', flush=True)


def _ts():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _roi_files(roi):
    return [f'GCE_cov_l{roi}{FRONT}_17yr_cholis{ext}' for ext in RESULT_EXTS]


def is_complete(roi, results_dir, work_dir):
    dat = f'GCE_cov_l{roi}{FRONT}_17yr_cholis.dat'
    return os.path.exists(os.path.join(results_dir, dat)) \
        or os.path.exists(os.path.join(work_dir,    dat))


def move_to_results(roi, work_dir, results_dir):
    moved = []
    for fname in _roi_files(roi):
        src = os.path.join(work_dir,    fname)
        dst = os.path.join(results_dir, fname)
        if os.path.exists(src):
            if os.path.exists(dst):
                try:
                    os.remove(src)
                except OSError:
                    pass
            else:
                shutil.move(src, dst)
                moved.append(fname)
    return moved


def launch_one(roi, log_path, env):
    log_file = open(log_path, 'a')
    log_file.write(f"\n========== {_ts()} START roi={roi} ==========\n")
    log_file.flush()
    proc = subprocess.Popen(
        [sys.executable, '-u', RUNNER_SCRIPT, str(roi)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    write_lock(roi, proc.pid)
    return proc, log_file


def acquire_lock():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                old_pid = int(f.read().strip())
            try:
                os.kill(old_pid, 0)
                print(f"[FATAL] another cov launcher (pid={old_pid}) is already running.")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers',        type=int,   default=2,
                    help='max concurrent ROI subprocesses (default: 2, memory-limited)')
    ap.add_argument('--rois',           type=str,   default='',
                    help='comma-separated ROI subset, e.g. "-50,30,50" (default: all 22)')
    ap.add_argument('--results-dir',    type=str,   default=DEFAULT_RESULTS_DIR)
    ap.add_argument('--work-dir',       type=str,   default='.')
    ap.add_argument('--max-retries',    type=int,   default=3)
    ap.add_argument('--poll-sec',       type=int,   default=30)
    ap.add_argument('--max-runtime-hr', type=float, default=48)
    args = ap.parse_args()

    if not os.path.exists(RUNNER_SCRIPT):
        print(f"[FATAL] {RUNNER_SCRIPT} not found in cwd ({os.getcwd()}).")
        sys.exit(2)
    if args.workers > 3:
        print(f"[warn] --workers {args.workers} exceeds memory-safe ceiling of 3.")
        print(f"       monitor `free -h` carefully; cov gtsrcmaps peaks at 30-50 GB/process.")

    os.makedirs(args.results_dir, exist_ok=True)
    acquire_lock()

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

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    running          = {}
    adopted_pids     = {}      # roi -> external PID (orphan from prior launcher)
    attempts         = {r: 0 for r in rois}
    permanent_failed = set()
    t_start          = time.time()

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

    initially_complete = [r for r in rois
                          if is_complete(r, args.results_dir, args.work_dir)]
    for r in initially_complete:
        move_to_results(r, args.work_dir, args.results_dir)

    n_stale = cleanup_stale_locks(rois)
    if n_stale > 0:
        print(f'[{_ts()}] locks: removed {n_stale} stale ROI locks')
    adopted = adopt_running_workers(
        rois, adopted_pids, running,
        args.results_dir, args.work_dir, RUNNER_SCRIPT,
        is_complete, os.getpid(),
    )
    if adopted:
        print(f'[{_ts()}] adopted {len(adopted)} orphan workers: {adopted}')

    print(f"[{_ts()}] cov launcher start  pid={os.getpid()}")
    print(f"  ROIs             : {len(rois)}")
    print(f"  workers          : {args.workers}")
    print(f"  results dir      : {args.results_dir}")
    print(f"  max retries      : {args.max_retries}")
    print(f"  poll interval    : {args.poll_sec}s")
    print(f"  initially done   : {len(initially_complete)}/{len(rois)}")

    last_status_ts = 0.0
    try:
        while True:
            now = time.time()

            completed = [r for r in rois
                         if is_complete(r, args.results_dir, args.work_dir)]
            not_done  = [r for r in rois
                         if r not in completed and r not in permanent_failed]

            if len(not_done) == 0:
                break
            if (now - t_start) / 3600 > args.max_runtime_hr:
                print(f"[{_ts()}] max runtime {args.max_runtime_hr}h exceeded — stopping")
                handle_signal(signal.SIGTERM, None)

            # Reap finished
            for r in list(running.keys()):
                proc, log_file = running[r]
                rc = proc.poll()
                if rc is None:
                    continue
                log_file.write(f"\n========== {_ts()} END roi={r} rc={rc} ==========\n")
                log_file.close()
                del running[r]

                if is_complete(r, args.results_dir, args.work_dir):
                    moved = move_to_results(r, args.work_dir, args.results_dir)
                    remove_lock(r)
                    print(f"[{_ts()}] [done ] roi={r:+4d}  rc={rc}  moved={len(moved)} files")
                else:
                    attempts[r] += 1
                    remove_lock(r)
                    if attempts[r] >= args.max_retries:
                        permanent_failed.add(r)
                        print(f"[{_ts()}] [FAIL ] roi={r:+4d}  attempts={attempts[r]}  "
                              f"giving up (see log_cov_l{r}.txt)")
                    else:
                        print(f"[{_ts()}] [retry] roi={r:+4d}  attempt {attempts[r]}/"
                              f"{args.max_retries}  rc={rc}")

            # Reap adopted orphan workers (PID-based)
            for r in list(adopted_pids.keys()):
                pid = adopted_pids[r]
                if adopted_alive(pid):
                    continue
                del adopted_pids[r]
                if is_complete(r, args.results_dir, args.work_dir):
                    moved = move_to_results(r, args.work_dir, args.results_dir)
                    remove_lock(r)
                    print(f"[{_ts()}] [done ] roi={r:+4d} (adopted) moved={len(moved)} files")
                else:
                    attempts[r] = max(attempts[r], 1)
                    remove_lock(r)
                    if attempts[r] >= args.max_retries:
                        permanent_failed.add(r)
                        print(f"[{_ts()}] [FAIL ] roi={r:+4d} (adopted dead, no .dat)")
                    else:
                        print(f"[{_ts()}] [retry] roi={r:+4d} (adopted dead)")

            # Fill slots
            runnable = [r for r in not_done
                        if r not in running and r not in adopted_pids
                        and r not in permanent_failed]
            while len(running) < args.workers and runnable:
                r = runnable.pop(0)
                log_path = f"log_cov_l{r}.txt"
                try:
                    proc, log_file = launch_one(r, log_path, env)
                except Exception as e:
                    print(f"[{_ts()}] [error] failed to launch roi={r}: {e}")
                    attempts[r] += 1
                    if attempts[r] >= args.max_retries:
                        permanent_failed.add(r)
                    continue
                running[r] = (proc, log_file)
                print(f"[{_ts()}] [start] roi={r:+4d}  pid={proc.pid}  "
                      f"running={len(running)}/{args.workers}")

            if now - last_status_ts > 300:
                n_done = len(completed)
                n_run  = len(running)
                n_left = len(not_done) - n_run
                n_fail = len(permanent_failed)
                elapsed = (now - t_start) / 60
                print(f"[{_ts()}] [status] done={n_done:>3}  running={n_run:>2}  "
                      f"queued={n_left:>3}  failed={n_fail:>2}  "
                      f"elapsed={elapsed:.1f} min")
                sys.stdout.flush()
                last_status_ts = now

            time.sleep(args.poll_sec)

    finally:
        completed_final = [r for r in rois
                           if is_complete(r, args.results_dir, args.work_dir)]
        elapsed_hr = (time.time() - t_start) / 3600
        print(f"\n=========== final ===========")
        print(f"  completed : {len(completed_final)}/{len(rois)}")
        print(f"  failed    : {len(permanent_failed)} -> {sorted(permanent_failed)}")
        print(f"  elapsed   : {elapsed_hr:.2f} hr")
        for r in completed_final:
            move_to_results(r, args.work_dir, args.results_dir)
        release_lock()


if __name__ == '__main__':
    main()
