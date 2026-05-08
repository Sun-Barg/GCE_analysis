#!/usr/bin/env python3
"""
launch_80models_parallel.py — Launch Full 80-model pipeline in N-way parallel.

Runs the full Cholis+2022 model set by sharding 80 models across
N parallel `run_main_loop_subprocess.py` worker processes. Each worker
processes its own subset of models sequentially; workers run in parallel.

Usage:
    # Default: 5-way parallel, full 80 models
    python launch_80models_parallel.py

    # Custom: 3-way parallel (safer for HDD)
    python launch_80models_parallel.py --n-workers 3

    # Run only a subset (e.g. 40 extra models beyond the existing 14)
    python launch_80models_parallel.py --models-file remaining_models.txt

    # Dry run: show what would happen, don't launch
    python launch_80models_parallel.py --dry-run

    # Monitor existing run (no new launches)
    python launch_80models_parallel.py --monitor

How it works:
  1. Pre-flight check: ensures that shared data-prep files (CCUBE, LTCUBE,
     EXPCUBE center/edge, masks, base XMLs, bin_definitions) already exist
     with NDSKEYS headers. These must be run through the notebook ONCE
     before launching parallel workers.
  2. Shard the 80-model list into N roughly-equal groups.
  3. Spawn N subprocesses, each running run_main_loop_subprocess.py with
     its own shard (as a comma-separated model list).
  4. Save launch metadata (PID, shard, log path) to `./parallel_run_state.json`
  5. Each worker logs to `./logs_parallel/shard_{i}.log`.
  6. Script exits immediately after spawning (detached mode); workers keep
     running even if the launcher or its terminal dies.

Monitoring (in a different terminal):
    python launch_80models_parallel.py --monitor   # or use the helper below
    tail -f logs_parallel/shard_0.log              # watch one shard live

Stopping:
    python launch_80models_parallel.py --stop      # SIGTERM all workers

Resume after crash:
  Just re-run the same command. The needs_run() / skip_or_run() guards
  in run_main_loop_subprocess.py skip any model whose .dat is complete,
  so finished models cost ~1 second each to check and skip.

Config (edit constants below if needed):
  N_WORKERS_DEFAULT = 5
  RUNNER = 'run_main_loop_subprocess.py'
"""
import os
import sys
import json
import time
import shutil
import signal
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


# ============================================================================
# The canonical 80-model list from Cholis+2022
# (NAMING_CONVENTION_OF_DIFFUSE_EMISSION_MODELS.dat, col 1)
# ============================================================================
ALL_80_MODELS = [
    'I','II','III','IV','V','VI','VII','VIII','IX','X',
    'XI','XII','XIII','XIV','XV','XVI','XVII','XVIII','XIX','XX',
    'XXI','XXII','XXIII','XXIV','XXV','XXVI','XXVII','XXVIII','XXIX','XXX',
    'XXXI','XXXII','XXXIII','XXXIV','XXXV','XXXVI','XXXVII','XXXVIII','XXXIX','XL',
    'XLI','XLII','XLIII','XLIV','XLV','XLVI','XLVII','XLVIII','XLIX','L',
    'LI','LII','LIII','LIV','LV','LVI','LVII','LVIII','LIX','LX',
    'LXI','LXII','LXIII','LXIV','LXV','LXVI','LXVII','LXVIII','LXIX','LXX',
    'LXXI','LXXII','LXXIII','LXXIV','LXXV','LXXVI','LXXVII','LXXVIII','LXXIX','LXXX',
]

# Configurable defaults
N_WORKERS_DEFAULT = 5
RUNNER = 'run_main_loop_subprocess.py'
LOG_DIR = 'logs_parallel'
STATE_FILE = 'parallel_run_state.json'


# ============================================================================
# Pre-flight checks
# ============================================================================

# Files that must exist before ANY model-worker runs. These are the shared
# data-prep outputs produced by notebook Cells 13-18, 19-20, 29-32.
REQUIRED_SHARED_FILES = [
    'GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits',
    'GC_analysis_sanghwan/Allsky_ltcube_12yr_front_clean.fits',
    'GC_analysis_sanghwan/GC_expcube_center_12yr_front_clean.fits',
    'GC_analysis_sanghwan/Allsky_expcube_edge_12yr_front_clean.fits',
    'GC_analysis_sanghwan/Model/GC_model_DR2.xml',
    'GC_analysis_sanghwan/Model/GC_psc_model_DR2.xml',
    'GC_analysis_sanghwan/Model/empty_model.xml',
    'GC_analysis_sanghwan/Model/GC_mask_60x60_definitions_DR2.npy',
    'GC_analysis_sanghwan/Model/GC_disk_mask_60x60_definitions.npy',
    'GCE_template_NFW2.fits',
    'Fermi_Bubbles_template.fits',
    'isotropic_spectrum_ff.txt',
    'fermi_bubble_spectrum.txt',
]


def check_preflight(models_to_run: List[str]) -> bool:
    """Check that shared prep files exist and per-model MapCubes are present."""
    print("=" * 72)
    print("Pre-flight checks")
    print("=" * 72)

    # 1. Shared files from data-prep stage
    missing_shared = [p for p in REQUIRED_SHARED_FILES if not os.path.exists(p)]
    if missing_shared:
        print(f"\n  ✗ {len(missing_shared)} shared data-prep file(s) missing:")
        for p in missing_shared:
            print(f"      {p}")
        print(f"\n  ⚠ Run the notebook through Cell 32 ONCE before launching parallel.")
        print(f"    These shared inputs must exist or workers will race to create them.")
        return False
    print(f"  ✓ All {len(REQUIRED_SHARED_FILES)} shared data-prep files present")

    # 2. Per-model MapCubes (3 files per model: pion, bremss, ics)
    missing_by_model = {}
    for m in models_to_run:
        missing = []
        for comp in ['pion', 'bremss', 'ics']:
            p = f'MapCubes/{comp}_mapcube_model{m}.fits'
            if not os.path.exists(p):
                missing.append(p)
        if missing:
            missing_by_model[m] = missing
    if missing_by_model:
        print(f"\n  ✗ {len(missing_by_model)} model(s) are missing MapCubes:")
        for m, lst in list(missing_by_model.items())[:10]:
            print(f"      Model {m}: missing {len(lst)} files")
            for p in lst:
                print(f"          {p}")
        if len(missing_by_model) > 10:
            print(f"      ... and {len(missing_by_model) - 10} more models")
        print(f"\n  ⚠ Those models will fail. Either download the missing MapCubes,")
        print(f"    or remove them from the model list with --models-file.")
        return False
    print(f"  ✓ All {len(models_to_run)} models have pion/bremss/ics MapCubes")

    # 3. Sanity: runner exists
    if not os.path.exists(RUNNER):
        print(f"\n  ✗ Runner script missing: {RUNNER}")
        return False
    print(f"  ✓ Runner script present: {RUNNER}")

    # 4. Report which models already have completed .dat files (will be skipped)
    completed = [m for m in models_to_run if os.path.exists(f'GCE_model_{m}_12yr_cholis.dat')]
    to_compute = [m for m in models_to_run if m not in completed]
    print(f"\n  Status: {len(completed)}/{len(models_to_run)} already complete, "
          f"{len(to_compute)} remaining")
    if completed and len(completed) < 20:
        print(f"  Already done: {completed}")
    print()
    return True


# ============================================================================
# Shard / launch
# ============================================================================

def shard_models(models: List[str], n: int) -> List[List[str]]:
    """Split models into n roughly-balanced shards (round-robin)."""
    shards = [[] for _ in range(n)]
    # Round-robin assignment balances load even if some models take longer
    for i, m in enumerate(models):
        shards[i % n].append(m)
    return shards


def launch_shards(shards: List[List[str]], log_dir: Path, dry_run: bool = False,
                  force: bool = False) -> List[dict]:
    """Spawn one subprocess per shard. Returns list of launched jobs."""
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, shard in enumerate(shards):
        if not shard:
            continue
        log_path = log_dir / f'shard_{i:02d}.log'
        shard_arg = ",".join(shard)
        cmd = [sys.executable, RUNNER, shard_arg]
        if force:
            cmd.append('--force')
        if dry_run:
            print(f"  [DRY] shard {i}: {len(shard)} models ({shard[:3]}...{shard[-1:]})"
                  f" -> {log_path}")
            print(f"        cmd: {' '.join(cmd[:4])}...")
            jobs.append(dict(shard_index=i, models=shard, pid=None, log=str(log_path)))
            continue

        # Launch detached so launcher can exit
        log_file = open(log_path, 'ab')
        # Header so the log reader knows which shard this is
        log_file.write(f"\n\n=== shard {i:02d}  pid=? start={time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                       f"=== models ({len(shard)}): {shard} ===\n\n".encode())
        log_file.flush()
        p = subprocess.Popen(
            cmd,
            stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True,   # fully detach
        )
        log_file.close()
        print(f"  ✓ shard {i}: {len(shard)} models ({shard[0]}..{shard[-1]}) "
              f"PID={p.pid} -> {log_path}")
        jobs.append(dict(
            shard_index=i, models=shard, pid=p.pid, log=str(log_path),
            start_time=time.time(),
        ))
    return jobs


PATCH_MARKER = "# v3.8 parallel: accept comma-separated list"
RUNNER_VERSION_MARKER = 'RUNNER_VERSION = "v3.14"'


def supports_runner_comma_list() -> bool:
    """Check whether run_main_loop_subprocess.py accepts comma-separated list.

    v3.8.1 fix: previous check used `'split(",")' in src` which gives a
    false negative because the patched code uses single-quoted `split(',')`.
    We now check for the unique comment marker we inject, which is
    quote-style-independent.
    """
    if not os.path.exists(RUNNER):
        return False
    src = open(RUNNER).read()
    return PATCH_MARKER in src


def runner_has_npz_save() -> bool:
    """Check whether runner has the v3.10+ .npz save logic."""
    if not os.path.exists(RUNNER):
        return False
    src = open(RUNNER).read()
    return 'np.savez_compressed' in src and 'cholis_fit.npz' in src


def runner_version_ok() -> bool:
    """Check whether runner has the v3.14+ version marker."""
    if not os.path.exists(RUNNER):
        return False
    src = open(RUNNER).read()
    return RUNNER_VERSION_MARKER in src


def patch_runner_for_comma_list() -> None:
    """Add comma-separated-list support to run_main_loop_subprocess.py.

    Idempotent: if already patched (detected via PATCH_MARKER), returns
    immediately.

    v3.14 change: NEVER silently restores from .pre_parallel. If the runner
    appears to be an older version (missing v3.14 marker OR missing .npz save
    code), we ABORT with a clear error message. This is because a previous
    version of this launcher would restore from backup in some edge cases,
    and that backup could be missing v3.10+ features, silently producing
    runs without .npz output. We now refuse to patch a downrev runner.
    """
    # v3.14 gate: refuse to proceed if runner lacks essential features
    if not runner_version_ok():
        raise RuntimeError(
            f"\n  ✗ {RUNNER} is missing the v3.14 version marker.\n"
            f"  The runner in place may be an older version lacking .npz save\n"
            f"  logic or other features. Aborting to avoid data loss.\n\n"
            f"  To fix: copy the latest run_main_loop_subprocess.py from your\n"
            f"  patches/outputs directory, replacing the current file.\n"
            f"  You can verify with:\n"
            f"    grep 'RUNNER_VERSION = \"v3.14\"' {RUNNER}\n"
        )

    if not runner_has_npz_save():
        raise RuntimeError(
            f"\n  ✗ {RUNNER} is missing the .npz save logic (v3.10+ feature).\n"
            f"  V2 SED decomposition and V10 multi-model comparison require\n"
            f"  .npz output files. Aborting.\n\n"
            f"  To fix: replace {RUNNER} with the latest version.\n"
        )

    if supports_runner_comma_list():
        return  # already patched

    with open(RUNNER) as f:
        src = f.read()

    old = "    targets = MODEL_LIST_DEFAULT if arg.lower() == 'all' else [arg]"

    if old not in src:
        # v3.14: do NOT auto-restore from backup. User must handle manually.
        backup = RUNNER + '.pre_parallel'
        raise RuntimeError(
            f"\n  ✗ Could not locate target-assignment line in {RUNNER}:\n"
            f"    {old}\n"
            f"  The runner may be in an unexpected state.\n"
            f"  Backup exists: {os.path.exists(backup)}\n"
            f"  Manual intervention required. Do NOT auto-restore from .pre_parallel\n"
            f"  as that may overwrite newer features (v3.10 .npz save, etc).\n"
        )

    new = (f"    {PATCH_MARKER} (e.g. 'X,I,XV,XLIX')\n"
           "    if arg.lower() == 'all':\n"
           "        targets = MODEL_LIST_DEFAULT\n"
           "    elif ',' in arg:\n"
           "        targets = [x.strip() for x in arg.split(',') if x.strip()]\n"
           "    else:\n"
           "        targets = [arg]")
    src_new = src.replace(old, new)

    # Backup original (only on first patch)
    backup = RUNNER + '.pre_parallel'
    if not os.path.exists(backup):
        shutil.copy(RUNNER, backup)
        print(f"  Backed up original runner -> {backup}")

    with open(RUNNER, 'w') as f:
        f.write(src_new)
    print(f"  ✓ Patched {RUNNER} to accept comma-separated model lists")


# ============================================================================
# Monitor / status
# ============================================================================

def load_state() -> Optional[dict]:
    if not os.path.exists(STATE_FILE):
        return None
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def status_report(state: dict, models_to_run: List[str]) -> None:
    """Print overall + per-shard status."""
    print("=" * 78)
    print(f"Parallel run status  ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 78)
    jobs = state.get('jobs', [])
    started = state.get('start_time', time.time())
    elapsed_min = (time.time() - started) / 60
    print(f"  Started:      {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started))}")
    print(f"  Elapsed:      {elapsed_min:.1f} min")
    print(f"  Total models: {len(models_to_run)}")
    print()

    # Per-shard status
    print(f"  {'Shard':<6} {'PID':<8} {'Alive':<6} {'N':<4} {'Done':<5} "
          f"{'Running':<10} {'Last line':<50}")
    print('  ' + '-' * 80)
    total_done = 0
    for job in jobs:
        sid = job['shard_index']
        pid = job.get('pid')
        alive = pid_alive(pid) if pid else False
        shard_models = job.get('models', [])
        done_in_shard = [m for m in shard_models
                         if os.path.exists(f'GCE_model_{m}_12yr_cholis.dat')]
        n_done = len(done_in_shard)
        total_done += n_done

        # Read last line of log
        log_path = job.get('log', '')
        last_line = ''
        current = ''
        if os.path.exists(log_path):
            try:
                with open(log_path, 'rb') as f:
                    f.seek(0, 2)
                    end = f.tell()
                    # read last 2000 bytes
                    f.seek(max(0, end - 2000))
                    tail = f.read().decode('utf-8', errors='replace').strip().split('\n')
                last_line = tail[-1][:80] if tail else ''
                # find current model (last ==== MODEL X ==== marker)
                for ln in reversed(tail):
                    if 'MODEL' in ln and '====' in ln:
                        # extract: ==== MODEL X ====
                        parts = ln.split()
                        if len(parts) >= 3 and parts[-2] in ALL_80_MODELS:
                            current = parts[-2]
                            break
            except Exception:
                pass

        status = '✓' if alive else '✗'
        print(f"  {sid:<6} {pid or '-':<8} {status:<6} {len(shard_models):<4} "
              f"{n_done:<5} {current or '-':<10} {last_line[:47]}")
    print()
    print(f"  Overall: {total_done}/{len(models_to_run)} models complete "
          f"({100*total_done/max(len(models_to_run),1):.1f}%)")
    print()

    # ETA
    if elapsed_min > 5 and total_done > 0:
        rate = total_done / elapsed_min  # models per min
        remaining = len(models_to_run) - total_done
        eta_min = remaining / rate if rate > 0 else float('inf')
        print(f"  Throughput: {rate*60:.2f} models/hr,  "
              f"est. remaining: {eta_min/60:.1f} hr "
              f"({eta_min/60/24:.1f} days)")


def stop_all(state: dict) -> None:
    print("Sending SIGTERM to all workers...")
    for job in state.get('jobs', []):
        pid = job.get('pid')
        if pid and pid_alive(pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                print(f"  shard {job['shard_index']}: SIGTERM -> PID {pid}")
            except Exception as e:
                print(f"  shard {job['shard_index']}: failed to kill PID {pid}: {e}")
    # Wait a bit then report
    time.sleep(3)
    for job in state.get('jobs', []):
        pid = job.get('pid')
        if pid:
            alive = pid_alive(pid)
            print(f"  shard {job['shard_index']}: PID {pid} alive={alive}")


# ============================================================================
# Watchdog
# ============================================================================

def restart_dead_shards(state: dict, log_dir: Path, force: bool = False) -> int:
    """For each dead shard with unfinished models, relaunch as a fresh subprocess.

    Returns number of shards relaunched.
    """
    relaunched = 0
    for job in state.get('jobs', []):
        pid = job.get('pid')
        if pid and pid_alive(pid):
            continue  # still alive, leave it
        # find unfinished models in this shard
        unfinished = [m for m in job.get('models', [])
                      if not os.path.exists(f'GCE_model_{m}_12yr_cholis.dat')]
        if not unfinished:
            continue  # this shard is done
        sid = job['shard_index']
        log_path = log_dir / f'shard_{sid:02d}.log'
        shard_arg = ",".join(unfinished)
        cmd = [sys.executable, RUNNER, shard_arg]
        if force:
            cmd.append('--force')
        # Append a restart-marker to the existing log
        with open(log_path, 'ab') as lf:
            lf.write(f"\n\n=== shard {sid:02d} RESTART by watchdog "
                     f"at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                     f"=== unfinished models ({len(unfinished)}): {unfinished} ===\n"
                     f"=== restart-attempt count: {job.get('restart_count', 0) + 1} ===\n\n"
                     .encode())
        log_file = open(log_path, 'ab')
        p = subprocess.Popen(
            cmd,
            stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_file.close()
        # update job entry IN PLACE
        job['pid'] = p.pid
        job['models'] = unfinished
        job['restart_count'] = job.get('restart_count', 0) + 1
        job['last_restart'] = time.time()
        print(f"  ↻ shard {sid}: {len(unfinished)} models remaining, "
              f"new PID={p.pid} (restart #{job['restart_count']})")
        relaunched += 1
    return relaunched


def watchdog_loop(state_path: str, log_dir: Path, check_interval: int = 300,
                  max_restarts: int = 5, force: bool = False) -> int:
    """Poll worker PIDs and relaunch any that have died with unfinished work.

    Runs until either:
      - All models are complete (all .dat exist)
      - Some shard has been restarted more than max_restarts times (likely
        a real bug, not a transient OOM)
      - User Ctrl-C
    """
    print(f"=== Watchdog started ===")
    print(f"  check interval: {check_interval} sec")
    print(f"  max restarts per shard: {max_restarts}")
    print(f"  Press Ctrl-C to stop watchdog (workers keep running)")
    print()
    iter_count = 0
    try:
        while True:
            state = load_state()
            if state is None:
                print(f"  state file disappeared, exiting watchdog")
                return 1

            models_total = state.get('models_to_run', [])
            done = [m for m in models_total if os.path.exists(f'GCE_model_{m}_12yr_cholis.dat')]
            still_running = sum(1 for j in state['jobs']
                                if j.get('pid') and pid_alive(j['pid']))
            iter_count += 1
            print(f"[watchdog #{iter_count} {time.strftime('%H:%M:%S')}] "
                  f"alive workers: {still_running}, "
                  f"complete: {len(done)}/{len(models_total)}")

            # All done?
            if len(done) >= len(models_total):
                print(f"\n  ✓ All {len(models_total)} models complete. Watchdog exiting.")
                return 0

            # Check for runaway restart counts BEFORE relaunching
            stuck = [j for j in state['jobs']
                     if j.get('restart_count', 0) >= max_restarts]
            if stuck:
                print(f"\n  ✗ {len(stuck)} shard(s) restarted >= {max_restarts} times. "
                      f"Likely a persistent bug, not a transient crash.")
                for j in stuck:
                    print(f"    shard {j['shard_index']}: {j.get('restart_count')} restarts. "
                          f"Check {log_dir}/shard_{j['shard_index']:02d}.log")
                return 2

            # Restart any dead shards with unfinished work
            n = restart_dead_shards(state, log_dir, force=force)
            if n > 0:
                print(f"  ↻ relaunched {n} shard(s)")
                save_state(state)

            time.sleep(check_interval)
    except KeyboardInterrupt:
        print(f"\n  Watchdog stopped by user. Workers continue running in background.")
        return 0


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-workers', type=int, default=N_WORKERS_DEFAULT,
                    help=f'Number of parallel workers (default: {N_WORKERS_DEFAULT})')
    ap.add_argument('--models-file', default=None,
                    help='File with one model name per line (default: all 80 Cholis models)')
    ap.add_argument('--models', default=None,
                    help='Comma-separated explicit model list (overrides --models-file)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Show what would happen, do not launch')
    ap.add_argument('--monitor', action='store_true',
                    help='Report status of existing run, do not launch new workers')
    ap.add_argument('--watchdog', action='store_true',
                    help='Poll PIDs and auto-relaunch dead shards with unfinished models')
    ap.add_argument('--watchdog-interval', type=int, default=300,
                    help='Watchdog poll interval in seconds (default: 300)')
    ap.add_argument('--watchdog-max-restarts', type=int, default=5,
                    help='Max restarts per shard before giving up (default: 5)')
    ap.add_argument('--stop', action='store_true',
                    help='Send SIGTERM to existing workers, then exit')
    ap.add_argument('--force', action='store_true',
                    help='Pass --force to workers (recompute even if .dat exists)')
    ap.add_argument('--log-dir', default=LOG_DIR,
                    help=f'Directory for per-shard logs (default: {LOG_DIR})')
    args = ap.parse_args()

    # Stop existing run
    if args.stop:
        state = load_state()
        if state is None:
            print(f"No state file found ({STATE_FILE}); nothing to stop.")
            return 0
        stop_all(state)
        return 0

    # Monitor existing run
    if args.monitor:
        state = load_state()
        if state is None:
            print(f"No state file found ({STATE_FILE}).")
            print(f"Either no run has been launched, or launch it first without --monitor.")
            return 1
        # We need the full model list to compute progress
        ml = state.get('models_to_run') or ALL_80_MODELS
        status_report(state, ml)
        return 0

    # Watchdog mode (also doesn't launch new workers from scratch — it
    # polls and relaunches dead shards from an existing state)
    if args.watchdog:
        state = load_state()
        if state is None:
            print(f"No state file found ({STATE_FILE}). Launch workers first.")
            return 1
        log_dir = Path(args.log_dir)
        return watchdog_loop(STATE_FILE, log_dir,
                             check_interval=args.watchdog_interval,
                             max_restarts=args.watchdog_max_restarts,
                             force=args.force)

    # --- Normal launch path ---

    # Resolve model list
    if args.models:
        models = [m.strip() for m in args.models.split(',') if m.strip()]
    elif args.models_file:
        with open(args.models_file) as f:
            models = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    else:
        models = list(ALL_80_MODELS)
    print(f"Models to process: {len(models)}")

    # Pre-flight
    if not args.dry_run:
        if not check_preflight(models):
            print("\n✗ Pre-flight failed. Fix the issues above, then retry.")
            return 1
    else:
        print("  (skipping pre-flight in dry-run mode)")

    # Patch runner if needed
    if not args.dry_run:
        patch_runner_for_comma_list()

    # Check for previous run state
    prev = load_state()
    if prev is not None and not args.dry_run:
        print(f"\n⚠ Found existing state file {STATE_FILE}")
        alive_workers = [j for j in prev.get('jobs', []) if j.get('pid') and pid_alive(j['pid'])]
        if alive_workers:
            print(f"  {len(alive_workers)} worker(s) still alive:")
            for j in alive_workers:
                print(f"    shard {j['shard_index']}: PID {j['pid']}")
            print(f"\n  To monitor:  python {sys.argv[0]} --monitor")
            print(f"  To stop:     python {sys.argv[0]} --stop")
            print(f"  To relaunch: stop first, then re-run this command")
            return 1
        else:
            print(f"  No workers alive; archiving old state and starting fresh.")
            shutil.move(STATE_FILE, STATE_FILE + f'.{int(time.time())}.bak')

    # Shard
    shards = shard_models(models, args.n_workers)
    print(f"\nSharding {len(models)} models into {args.n_workers} workers:")
    for i, s in enumerate(shards):
        if s:
            print(f"  shard {i}: {len(s):>3} models  [{s[0]}..{s[-1]}]")

    if args.dry_run:
        print(f"\n=== DRY RUN ===")
        jobs = launch_shards(shards, Path(args.log_dir), dry_run=True, force=args.force)
        print(f"\nNo subprocesses were actually spawned.")
        print(f"Remove --dry-run to launch for real.")
        return 0

    # Launch
    print(f"\n=== Launching {args.n_workers} parallel workers ===\n")
    log_dir = Path(args.log_dir)
    jobs = launch_shards(shards, log_dir, dry_run=False, force=args.force)

    # Save state
    state = {
        'start_time': time.time(),
        'n_workers': args.n_workers,
        'models_to_run': models,
        'jobs': jobs,
        'runner': RUNNER,
    }
    save_state(state)
    print(f"\n  State saved to {STATE_FILE}")

    # Final instructions
    print(f"\n{'='*72}")
    print(f"All {len([j for j in jobs if j.get('pid')])} workers launched in background.")
    print(f"{'='*72}")
    print(f"\n  Monitor:  python {sys.argv[0]} --monitor")
    print(f"  Watchdog (auto-restart dead shards):")
    print(f"            python {sys.argv[0]} --watchdog")
    print(f"            (run in tmux/screen so it survives terminal close)")
    print(f"  Watch log of shard 0:  tail -f {log_dir}/shard_00.log")
    print(f"  Stop all:              python {sys.argv[0]} --stop")
    print(f"\n  Expected wall time: ~{len(models)*80/args.n_workers/60:.1f} hr "
          f"(@ 80 min/model / {args.n_workers} workers)")
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
