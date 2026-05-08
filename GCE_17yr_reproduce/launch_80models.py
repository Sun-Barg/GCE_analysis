#!/usr/bin/env python3
"""launch_80models.py — Run multiple GDE models in parallel.

Each worker = subprocess running run_one_model.py for one Roman-numeral model.
Each does XML build + gtsrcmaps + gtmodel + emcee for that model.

Usage:
    python launch_80models.py                          # 5 workers (default)
    python launch_80models.py --workers 12             # 12 parallel models
    python launch_80models.py --models X XLIX I        # only these models

Each model takes ~70 min (post-bubble-fix Model X verified).
With 12 workers, 80 models ~= 8 hours wall time.

Notes:
- Uses per-worker PFILES to avoid fermitools .par race condition
- run_one_model.py writes outputs to CWD; this launcher AUTO-MOVES them to
  ./results_17yr/ on each successful worker exit (no manual `mv` needed)
- run_one_model.py has its own [skip] guard if out_dat exists in CWD
- Resume-friendly: pre-skips done models in CWD or results_17yr/
- Designed to safely coexist with launch_cov_full.py (different file names + different PFILES dirs)
- Memory: main pipeline gtsrcmaps is light (~1-2GB per worker, vs cov's 30-50GB).
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


# ---- 80 Roman-numeral model list (I .. LXXX) ----------------------
def to_roman(n):
    """Convert integer 1..80 to Roman numeral. Cholis's 80 GDE models."""
    table = [(50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'),
             (5, 'V'),  (4,  'IV'), (1,  'I')]
    out = ''
    for v, sym in table:
        while n >= v:
            out += sym
            n -= v
    return out


ALL_MODELS = [to_roman(i) for i in range(1, 81)]
# = ['I','II','III','IV','V','VI','VII','VIII','IX','X','XI', ... 'LXXIX','LXXX']


# ---- output paths (run_one_model.py L1098, L1138) -----------------
FRONT = '_front'   # match run_one_model.py CONFIG (front='_front', evtype=1)
RESULTS_DIR = './results_17yr'   # final destination for completed model outputs


def out_path_cwd(model):
    return f'./GCE_model_{model}{FRONT}_17yr_cholis.dat'


def out_path_results(model):
    return f'{RESULTS_DIR}/GCE_model_{model}{FRONT}_17yr_cholis.dat'


def is_done(model):
    """Done = .dat exists in CWD OR results_17yr/."""
    return os.path.exists(out_path_cwd(model)) or os.path.exists(out_path_results(model))


def move_results_to_dir(model, dest_dir):
    """Move the 3 output files for `model` from CWD into dest_dir.
    Returns (moved_count, missing_list). Idempotent: skips files already at dest.
    """
    Path(dest_dir).mkdir(exist_ok=True)
    suffixes = ['.dat', '_fit.npz', '_likelihood_value']
    moved = 0
    missing = []
    for suf in suffixes:
        src = f'./GCE_model_{model}{FRONT}_17yr_cholis{suf}'
        dst = f'{dest_dir}/GCE_model_{model}{FRONT}_17yr_cholis{suf}'
        if os.path.exists(src):
            try:
                # os.replace overwrites cross-platform-safely
                os.replace(src, dst)
                moved += 1
            except OSError as e:
                # cross-device move can fail; fall back to copy+remove
                import shutil
                shutil.move(src, dst)
                moved += 1
        elif not os.path.exists(dst):
            missing.append(suf)
    return moved, missing


def main():
    ap = argparse.ArgumentParser(description='Parallel launcher for 80-model GCE analysis')
    ap.add_argument('--workers', type=int, default=5,
                    help='parallel models (default 5; main pipeline is memory-light)')
    ap.add_argument('--models', nargs='*', default=None,
                    help='specific models in Roman numerals (default: all 80)')
    ap.add_argument('--logs', default='./logs_80models')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--force', action='store_true',
                    help='pass --force to run_one_model.py (re-run even if done)')
    args = ap.parse_args()

    # Build request list
    if args.models:
        # Validate user-provided list
        bad = [m for m in args.models if m not in ALL_MODELS]
        if bad:
            print(f'[error] unknown model(s): {bad}', file=sys.stderr)
            print(f'[error] valid Roman numerals: {", ".join(ALL_MODELS)}', file=sys.stderr)
            return 2
        requested = list(args.models)
    else:
        requested = list(ALL_MODELS)

    if args.force:
        skipped, pending = [], list(requested)
    else:
        skipped = [m for m in requested if is_done(m)]
        pending = [m for m in requested if not is_done(m)]

    if skipped:
        print(f'[skip] {len(skipped)} models already done: {skipped}')
    if not pending:
        print('[done] nothing to do')
        return 0

    print(f'[plan] {len(pending)} models with {args.workers} workers')
    print(f'[plan] models to run: {pending}')

    Path(args.logs).mkdir(exist_ok=True)
    Path('./pfiles_80models').mkdir(exist_ok=True)

    queue = list(pending)
    running = {}    # model -> (proc, log_handle, start_time)
    completed = []
    failed = []

    state_file = Path('80models_state.json')
    started_at = time.strftime('%Y-%m-%d %H:%M:%S')

    def write_state():
        state_file.write_text(json.dumps({
            'started':   started_at,
            'workers':   args.workers,
            'pending':   queue,
            'running':   list(running.keys()),
            'completed': completed,
            'failed':    failed,
            'skipped':   skipped,
            'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
        }, indent=2))

    def launch(model):
        log_path = Path(args.logs) / f'model_{model}.log'
        log_h = open(log_path, 'w', buffering=1)
        cmd = [args.python, 'run_one_model.py', model]
        if args.force:
            cmd.append('--force')
        log_h.write(f'# launched: {" ".join(shlex.quote(c) for c in cmd)}\n')
        log_h.write(f'# at: {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        log_h.flush()

        # Per-worker PFILES (fermitools .par isolation)
        env = os.environ.copy()
        worker_pfiles = Path('./pfiles_80models') / f'model_{model}'
        worker_pfiles.mkdir(parents=True, exist_ok=True)
        sys_pfiles = env.get('PFILES', os.path.expanduser('~/pfiles'))
        if ';' in sys_pfiles:
            env['PFILES'] = f'{worker_pfiles};' + sys_pfiles.split(';', 1)[1]
        else:
            env['PFILES'] = f'{worker_pfiles};{sys_pfiles}'

        proc = subprocess.Popen(
            cmd,
            stdout=log_h,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        running[model] = (proc, log_h, time.time())
        print(f'  [launch] model={model}  pid={proc.pid}  log={log_path}')

    def reap():
        finished = []
        for model, (proc, log_h, t0) in running.items():
            rc = proc.poll()
            if rc is not None:
                finished.append((model, rc, t0))
        for model, rc, t0 in finished:
            proc, log_h, _ = running.pop(model)
            log_h.close()
            dt_min = (time.time() - t0) / 60
            done_total = len(completed) + len(failed)
            total = len(pending)
            if rc == 0 and is_done(model):
                # Auto-move outputs from CWD to results_17yr/ on success
                moved, missing = move_results_to_dir(model, RESULTS_DIR)
                move_note = ''
                if missing:
                    move_note = f'  [warn] missing: {missing}'
                completed.append(model)
                print(f'  [ok  ] model={model:<7}  {dt_min:.0f}min  '
                      f'({done_total+1}/{total} done, '
                      f'{len(running)} running, {len(queue)} queued)'
                      f'  moved={moved}{move_note}')
            else:
                failed.append((model, rc))
                print(f'  [FAIL rc={rc}] model={model:<7}  {dt_min:.0f}min  see log')

    def graceful_shutdown(sig=None, frame=None):
        print('\n[shutdown] terminating workers...')
        for model, (proc, log_h, _) in running.items():
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                log_h.close()
            except Exception:
                pass
        write_state()
        sys.exit(130)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    write_state()
    t_start = time.time()

    while queue or running:
        while queue and len(running) < args.workers:
            launch(queue.pop(0))
            write_state()

        time.sleep(30.0)   # each model is ~70min, no need to poll faster
        reap()
        write_state()

    dt_h = (time.time() - t_start) / 3600
    print(f'\n[finished] {len(completed)} ok, {len(failed)} failed, '
          f'wall time {dt_h:.1f} h')
    if completed:
        print(f'[done] outputs auto-moved to {RESULTS_DIR}/')
    if failed:
        for m, rc in failed:
            print(f'  failed: model={m} rc={rc} - see {args.logs}/model_{m}.log')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
