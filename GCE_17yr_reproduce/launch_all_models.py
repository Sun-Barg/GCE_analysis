#!/usr/bin/env python3
"""
launch_all_models.py — robust 80-model launcher for 17yr GCE main fit.

Spawns up to N concurrent `run_one_model.py M` subprocesses. Continuously
polls for completed / dead workers and refills the slot from the not-yet-
done list, so the parallel pool stays full even when individual workers
crash or hang-then-die. Auto-moves completed model outputs into
`results_17yr/`. Designed to be re-launchable: a second invocation
detects already-completed models from the filesystem and only runs the
missing ones.

[v3.1] startup auto-cleanup. The worker (run_one_model.py v3) FATALs
(rc=2) on any stale FITS/XML intermediate — which is the desired safety
behavior mid-run, but means a previous SIGKILL-interrupted launcher
session leaves partial files that would cause every retry to instantly
FATAL → permanent_failed. To prevent that, the launcher now scans
all per-model intermediates at startup, verifies each, and deletes
ONLY stale ones (healthy ones are preserved so workers can skip).
Auto-cleanup runs by default; disable with --no-cleanup.

Usage:
    nohup python launch_all_models.py > launcher.log 2>&1 &
    tail -f launcher.log

    # subset:
    python launch_all_models.py --models I,II,X --workers 4

    # different worker count:
    python launch_all_models.py --workers 8

    # skip startup cleanup (rare; for debugging):
    python launch_all_models.py --no-cleanup

State:
    - `.launcher.pid`        single-instance lock (deleted on exit)
    - `results_17yr/*.dat`   completion marker (per model)
    - `log_{M}.txt`          per-model log (append on retry, timestamp banner)

Resume semantics:
    - already in `results_17yr/`           -> skip
    - already in work dir (.dat present)   -> mv to results_17yr/, mark done
    - currently running (this launcher)    -> wait for poll
    - dead with no .dat                    -> retry until --max-retries
    - dead with rc=2 (worker FATAL)        -> permanent_failed (no retry;
                                              same condition would recur)
    - stale intermediates from prior SIGKILL -> auto-deleted at startup
                                                  (healthy ones preserved)

Cleanup:
    - SIGINT/SIGTERM: terminate all running children, then exit
    - Normal completion: print summary, remove lockfile

Author: haebarg (2026)
"""

import os
import sys
import time
import signal
import argparse
import subprocess
import shutil
from datetime import datetime

from cholis_masking import (
    verify_fits, verify_cube, verify_xml, verify_dat, verify_srcmap,
)
from launcher_locks import (
    write_lock, remove_lock, cleanup_stale_locks,
    adopt_running_workers, adopted_alive,
)

# ============================================================
# 80-model list (Cholis Roman numerals, NAMING_CONVENTION order)
# ============================================================
ALL_MODELS = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
    "XXI", "XXII", "XXIII", "XXIV", "XXV", "XXVI", "XXVII", "XXVIII", "XXIX", "XXX",
    "XXXI", "XXXII", "XXXIII", "XXXIV", "XXXV", "XXXVI", "XXXVII", "XXXVIII", "XXXIX", "XL",
    "XLI", "XLII", "XLIII", "XLIV", "XLV", "XLVI", "XLVII", "XLVIII", "XLIX", "L",
    "LI", "LII", "LIII", "LIV", "LV", "LVI", "LVII", "LVIII", "LIX", "LX",
    "LXI", "LXII", "LXIII", "LXIV", "LXV", "LXVI", "LXVII", "LXVIII", "LXIX", "LXX",
    "LXXI", "LXXII", "LXXIII", "LXXIV", "LXXV", "LXXVI", "LXXVII", "LXXVIII", "LXXIX", "LXXX",
]
assert len(ALL_MODELS) == 80

FRONT          = '_front'
RUNNER_SCRIPT  = 'run_one_model_wrapper.py'
LOCK_FILE      = '.launcher.pid'
RESULT_EXTS    = ['.dat', '_fit.npz', '_likelihood_value']


# ============================================================
# Helpers
# ============================================================

def _ts():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _model_files(model):
    """Return the three filename suffixes for a given model."""
    return [f'GCE_model_{model}{FRONT}_17yr_cholis{ext}' for ext in RESULT_EXTS]


def is_complete(model, results_dir, work_dir):
    """Completion = .dat exists in either results_dir or work_dir."""
    dat = f'GCE_model_{model}{FRONT}_17yr_cholis.dat'
    return os.path.exists(os.path.join(results_dir, dat)) \
        or os.path.exists(os.path.join(work_dir,    dat))


def move_to_results(model, work_dir, results_dir):
    """Move all output files of `model` from work_dir to results_dir.
    Returns the list of extensions actually moved."""
    moved = []
    for fname in _model_files(model):
        src = os.path.join(work_dir,    fname)
        dst = os.path.join(results_dir, fname)
        if os.path.exists(src):
            if os.path.exists(dst):
                # destination already has it (rare race) — keep destination,
                # remove source to avoid duplication.
                try:
                    os.remove(src)
                except OSError:
                    pass
            else:
                shutil.move(src, dst)
                moved.append(fname)
    return moved


def launch_one(model, log_path, env):
    """Start a subprocess for `model`; append-mode log with timestamp banner.

    start_new_session is intentionally False so SIGINT to launcher kills
    children. Children are otherwise standalone Python processes.
    """
    log_file = open(log_path, 'a')
    log_file.write(f"\n========== {_ts()} START model={model} ==========\n")
    log_file.flush()
    proc = subprocess.Popen(
        [sys.executable, '-u', RUNNER_SCRIPT, model],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    write_lock(model, proc.pid)
    return proc, log_file


def acquire_lock():
    """Single-instance lock. Aborts if another launcher is alive."""
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                old_pid = int(f.read().strip())
            try:
                os.kill(old_pid, 0)
                print(f"[FATAL] another launcher (pid={old_pid}) is already running.")
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


# ============================================================
# Startup auto-cleanup (v3.1)
#
# The worker run_one_model.py v3 FATALs (rc=2) on any stale FITS/XML
# intermediate. If a previous launcher session was SIGKILL'd mid-worker,
# partial intermediates remain. Without cleanup, every retry would
# instantly FATAL and the launcher would mark the model permanent_failed
# on first contact — exactly defeating the resume logic.
#
# This function scans every per-model intermediate, verifies each with
# the same verifiers run_one_model.py uses, and deletes ONLY the stale
# ones. Healthy intermediates are preserved so the worker can skip them
# on the next launch (resuming from the right step).
# ============================================================

# Per-model intermediates (14 files per model):
#   5 XMLs + 2 gtsrcmaps (convol yes/no) + 6 per-component gtmodel (3 components × 2 convol)
#   + final .dat (with 2 companions handled separately)
#
# Plus 6 model-INDEPENDENT shared template gtmodel files
# (GCE/iso/fermi_bubble × convol yes/no) — these are built once by
# whichever model runs first and reused.

def _verify_or_delete(path, verifier_fn, stats):
    """Helper: if path exists and verifier fails, delete the file.
    Three outcomes tracked in stats dict:
      - absent:   file not present (nothing to do)
      - kept:     verifier OK (preserved for worker skip)
      - deleted:  verifier failed (stale; removed)
    """
    if not os.path.exists(path):
        stats['absent'] += 1
        return
    ok, msg = verifier_fn()
    if ok:
        stats['kept'] += 1
        return
    print(f"  [cleanup] stale: {path}  ({msg})", flush=True)
    try:
        os.remove(path)
        stats['deleted'] += 1
    except OSError as e:
        print(f"  [warn   ] failed to delete {path}: {e}", flush=True)


def cleanup_stale_intermediates(models, work_dir, results_dir):
    """Scan all per-model intermediates and delete stale files.

    Returns:
        stats dict: {'deleted': N, 'kept': N, 'absent': N, 'skipped_complete': N}
    """
    stats = {'deleted': 0, 'kept': 0, 'absent': 0, 'skipped_complete': 0}

    model_dir = os.path.join(work_dir, 'GC_analysis_FL16Y', 'Model')
    fits_dir  = os.path.join(work_dir, 'GC_analysis_FL16Y')
    cwd_dir   = work_dir   # final .dat lives in cwd, not GC_analysis_FL16Y/

    for m in models:
        # Skip already-completed models — don't risk touching their files
        if is_complete(m, results_dir, work_dir):
            stats['skipped_complete'] += 1
            continue

        # --- Per-model XMLs (5 files) ---
        xml_specs = [
            (f'{model_dir}/GC_model{m}_test.xml',          100),  # PSC + 6 components
            (f'{model_dir}/GC_Extended_model{m}_test.xml', 1),    # 6 components only
            (f'{model_dir}/GC_pion_model{m}_test.xml',     1),    # 1 component
            (f'{model_dir}/GC_bremss_model{m}_test.xml',   1),
            (f'{model_dir}/GC_ics_model{m}_test.xml',      1),
        ]
        for path, min_n in xml_specs:
            _verify_or_delete(path,
                              lambda p=path, n=min_n: verify_xml(p, min_sources=n),
                              stats)

        # --- gtsrcmaps outputs (2 files, multi-HDU FITS) ---
        for convol_suffix in ['', '_no_convol']:
            path = f'{fits_dir}/GC_Extended_srcmap_17yr_front_clean_model_{m}{convol_suffix}.fits'
            _verify_or_delete(path,
                              lambda p=path: verify_srcmap(p),
                              stats)

        # --- Per-component gtmodel outputs (6 files, 3D cube 600x600x14) ---
        for comp in ['pion', 'bremss', 'ics']:
            for convol_suffix in ['', '_no_convol']:
                path = f'{fits_dir}/GC_{comp}_model{m}_17yr_front_clean{convol_suffix}.fits'
                _verify_or_delete(path,
                                  lambda p=path: verify_cube(p, expected_xy=(600, 600)),
                                  stats)

        # --- Final .dat + companions ---
        dat_path = os.path.join(cwd_dir, f'GCE_model_{m}_front_17yr_cholis.dat')
        if os.path.exists(dat_path):
            ok, msg = verify_dat(dat_path)
            if ok:
                stats['kept'] += 1
            else:
                # Stale .dat → also delete .npz + _likelihood_value companions
                # (they're written together at end of a successful run; if .dat
                # is corrupt, treat the trio as one logical unit).
                print(f"  [cleanup] stale: {dat_path}  ({msg})", flush=True)
                for ext in ['.dat', '_fit.npz', '_likelihood_value']:
                    p = os.path.join(cwd_dir, f'GCE_model_{m}_front_17yr_cholis{ext}')
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                            stats['deleted'] += 1
                            if ext != '.dat':
                                print(f"  [cleanup] removed companion: {p}",
                                      flush=True)
                        except OSError as e:
                            print(f"  [warn   ] failed to delete {p}: {e}",
                                  flush=True)
        else:
            stats['absent'] += 1

    # --- Model-INDEPENDENT shared template gtmodel (6 files) ---
    # GCE / isotropic / fermi_bubble × convol(yes, no).
    # These are built once by whichever model runs first. If ANY is stale,
    # delete all 6 for consistency — the first model to encounter their
    # absence will rebuild the whole set atomically.
    shared_paths = [
        f'{fits_dir}/GC_{src}_model_17yr_front_clean{conv}.fits'
        for src in ('GCE', 'isotropic', 'fermi_bubble')
        for conv in ('', '_no_convol')
    ]
    shared_present = [p for p in shared_paths if os.path.exists(p)]
    shared_stale = []
    for p in shared_present:
        ok, msg = verify_cube(p, expected_xy=(600, 600))
        if not ok:
            shared_stale.append((p, msg))

    if shared_stale:
        print(f"  [cleanup] shared template gtmodel: {len(shared_stale)}/"
              f"{len(shared_present)} stale → deleting all {len(shared_present)} "
              f"for consistency", flush=True)
        for p, msg in shared_stale:
            print(f"    stale: {p}  ({msg})", flush=True)
        for p in shared_present:
            try:
                os.remove(p)
                stats['deleted'] += 1
            except OSError as e:
                print(f"  [warn] failed to delete {p}: {e}", flush=True)
    else:
        stats['kept'] += len(shared_present)
        stats['absent'] += (len(shared_paths) - len(shared_present))

    return stats


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n', 1)[0])
    ap.add_argument('--workers',        type=int,   default=16,
                    help='max concurrent subprocesses (default: 16)')
    ap.add_argument('--models',         type=str,   default='',
                    help='comma-separated subset, e.g. "I,II,X" (default: all 80)')
    ap.add_argument('--results-dir',    type=str,   default='results_17yr')
    ap.add_argument('--work-dir',       type=str,   default='.')
    ap.add_argument('--max-retries',    type=int,   default=3,
                    help='max relaunch attempts per model (default: 3)')
    ap.add_argument('--poll-sec',       type=int,   default=30,
                    help='polling interval, seconds (default: 30)')
    ap.add_argument('--max-runtime-hr', type=float, default=24,
                    help='hard timeout, hours (default: 24)')
    ap.add_argument('--no-cleanup',    action='store_true',
                    help='skip startup auto-cleanup of stale intermediates '
                         '(v3.1; rare, for debugging — normally cleanup runs '
                         'by default and only deletes verifier-failing files)')
    args = ap.parse_args()

    # Verify runner exists
    if not os.path.exists(RUNNER_SCRIPT):
        print(f"[FATAL] {RUNNER_SCRIPT} not found in cwd ({os.getcwd()}).")
        sys.exit(2)

    os.makedirs(args.results_dir, exist_ok=True)
    acquire_lock()

    # Build model list
    if args.models.strip():
        models = [m.strip() for m in args.models.split(',') if m.strip()]
        unknown = [m for m in models if m not in ALL_MODELS]
        if unknown:
            print(f"[FATAL] unknown model name(s): {unknown}")
            print(f"        valid names: {ALL_MODELS}")
            release_lock(); sys.exit(2)
    else:
        models = ALL_MODELS[:]

    # Environment for children
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    # Make sure PHASE2_CASE is not inherited (case-test mode is
    # for diagnostics only; production = no case suffix).
    env.pop('PHASE2_CASE', None)

    # State
    running          = {}      # model -> (Popen, log_file_handle)
    adopted_pids     = {}      # model -> external PID (orphan from prior launcher)
    attempts         = {m: 0 for m in models}
    permanent_failed = set()
    t_start          = time.time()

    # Signal cleanup
    _interrupt_flag = {'stop': False}
    def handle_signal(signum, frame):
        if _interrupt_flag['stop']:
            return    # already handling
        _interrupt_flag['stop'] = True
        print(f"\n[{_ts()}] signal {signum} received — terminating "
              f"{len(running)} children")
        for m, (proc, lf) in list(running.items()):
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        # wait briefly then SIGKILL holdouts
        deadline = time.time() + 10
        for m, (proc, lf) in list(running.items()):
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

    # Initial completion scan + cleanup of any stranded work-dir outputs
    initially_complete = [m for m in models
                          if is_complete(m, args.results_dir, args.work_dir)]
    for m in initially_complete:
        move_to_results(m, args.work_dir, args.results_dir)

    # v3.1 startup auto-cleanup: scan per-model intermediates from any
    # previous SIGKILL-interrupted session and delete only the stale ones.
    # Without this step, workers re-launched against partial intermediates
    # would FATAL (rc=2) on first contact, and the rc=2 branch in the
    # reap loop would mark them permanent_failed — defeating resume.
    if not args.no_cleanup:
        print(f"[{_ts()}] cleanup: scanning per-model intermediates for "
              f"stale files...")
        t_cu = time.time()
        cu_stats = cleanup_stale_intermediates(models, args.work_dir,
                                                args.results_dir)
        cu_dt = time.time() - t_cu
        print(f"[{_ts()}] cleanup: done in {cu_dt:.1f}s  "
              f"deleted={cu_stats['deleted']}  kept={cu_stats['kept']}  "
              f"absent={cu_stats['absent']}  "
              f"skipped_complete={cu_stats['skipped_complete']}")
    else:
        print(f"[{_ts()}] cleanup: SKIPPED (--no-cleanup)")

    # Per-model PID locks: clean stale + adopt orphan workers
    # left over by a prior launcher death (watchdog scenario).
    n_stale = cleanup_stale_locks(models)
    if n_stale > 0:
        print(f'[{_ts()}] locks: removed {n_stale} stale model locks')
    adopted = adopt_running_workers(
        models, adopted_pids, running,
        args.results_dir, args.work_dir, RUNNER_SCRIPT,
        is_complete, os.getpid(),
    )
    if adopted:
        print(f'[{_ts()}] adopted {len(adopted)} orphan workers: '
              f'{adopted}')

    print(f"[{_ts()}] launcher start  pid={os.getpid()}")
    print(f"  models           : {len(models)}")
    print(f"  workers          : {args.workers}")
    print(f"  results dir      : {args.results_dir}")
    print(f"  max retries      : {args.max_retries}")
    print(f"  poll interval    : {args.poll_sec}s")
    print(f"  initially done   : {len(initially_complete)}/{len(models)}")

    # ============================================================
    # Main polling loop
    # ============================================================
    last_status_ts = 0.0
    try:
        while True:
            now = time.time()

            # (1) Recompute completion set
            completed = [m for m in models
                         if is_complete(m, args.results_dir, args.work_dir)]
            not_done  = [m for m in models
                         if m not in completed and m not in permanent_failed]

            # (2) Termination check
            if len(not_done) == 0:
                break
            if (now - t_start) / 3600 > args.max_runtime_hr:
                print(f"[{_ts()}] max runtime {args.max_runtime_hr}h exceeded — stopping")
                handle_signal(signal.SIGTERM, None)   # cleanup path

            # (3) Reap finished subprocesses
            for m in list(running.keys()):
                proc, log_file = running[m]
                rc = proc.poll()
                if rc is None:
                    continue
                # finished
                log_file.write(f"\n========== {_ts()} END model={m} rc={rc} ==========\n")
                log_file.close()
                del running[m]

                if rc == 0 and is_complete(m, args.results_dir, args.work_dir):
                    # Genuine success: clean exit AND final .dat present.
                    moved = move_to_results(m, args.work_dir, args.results_dir)
                    remove_lock(m)
                    print(f"[{_ts()}] [done ] {m:<8} rc=0  moved={len(moved)} files")
                elif rc == 2:
                    # FATAL from run_one_model.py v3 integrity check
                    # (stale FITS/XML intermediate, missing prerequisite,
                    # etc.). Same condition would recur on retry — give
                    # up immediately so the user can inspect.
                    permanent_failed.add(m)
                    remove_lock(m)
                    print(f"[{_ts()}] [FATAL] {m:<8} rc=2 — worker aborted "
                          f"on integrity check (likely stale FITS/XML "
                          f"intermediate); see log_{m}.txt")
                else:
                    # Crash (signal, OOM, segfault, etc.) — retry.
                    attempts[m] += 1
                    remove_lock(m)
                    if attempts[m] >= args.max_retries:
                        permanent_failed.add(m)
                        print(f"[{_ts()}] [FAIL ] {m:<8} attempts={attempts[m]}  "
                              f"giving up (see log_{m}.txt)")
                    else:
                        print(f"[{_ts()}] [retry] {m:<8} attempt {attempts[m]}/"
                              f"{args.max_retries}  rc={rc}")

            # (3b) Reap adopted orphan workers (PID-based, no Popen)
            for m in list(adopted_pids.keys()):
                pid = adopted_pids[m]
                if adopted_alive(pid):
                    continue
                # exited — check whether it produced a .dat
                del adopted_pids[m]
                if is_complete(m, args.results_dir, args.work_dir):
                    moved = move_to_results(m, args.work_dir, args.results_dir)
                    remove_lock(m)
                    print(f"[{_ts()}] [done ] {m:<8} (adopted) moved={len(moved)} files")
                else:
                    attempts[m] = max(attempts[m], 1)
                    remove_lock(m)
                    if attempts[m] >= args.max_retries:
                        permanent_failed.add(m)
                        print(f"[{_ts()}] [FAIL ] {m:<8} (adopted dead, no .dat, max retries)")
                    else:
                        print(f"[{_ts()}] [retry] {m:<8} (adopted dead, no .dat, "
                              f"attempt {attempts[m]}/{args.max_retries})")

            # (4) Fill empty slots
            runnable = [m for m in not_done
                        if m not in running and m not in adopted_pids
                        and m not in permanent_failed]
            while len(running) < args.workers and runnable:
                m = runnable.pop(0)
                log_path = f"log_{m}.txt"
                try:
                    proc, log_file = launch_one(m, log_path, env)
                except Exception as e:
                    print(f"[{_ts()}] [error] failed to launch {m}: {e}")
                    attempts[m] += 1
                    if attempts[m] >= args.max_retries:
                        permanent_failed.add(m)
                    continue
                running[m] = (proc, log_file)
                print(f"[{_ts()}] [start] {m:<8} pid={proc.pid}  "
                      f"running={len(running)}/{args.workers}")

            # (5) Periodic status summary (every 5 minutes)
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

            # (6) Sleep
            time.sleep(args.poll_sec)

    finally:
        # Final summary + lockfile cleanup (signal handler also calls release_lock)
        completed_final = [m for m in models
                           if is_complete(m, args.results_dir, args.work_dir)]
        elapsed_hr = (time.time() - t_start) / 3600
        print(f"\n=========== final ===========")
        print(f"  completed : {len(completed_final)}/{len(models)}")
        print(f"  failed    : {len(permanent_failed)} -> {sorted(permanent_failed)}")
        print(f"  elapsed   : {elapsed_hr:.2f} hr")
        # Move any stragglers
        for m in completed_final:
            move_to_results(m, args.work_dir, args.results_dir)
        release_lock()


if __name__ == '__main__':
    main()
