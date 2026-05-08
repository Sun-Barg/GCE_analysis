#!/usr/bin/env python3
"""launch_cov_full.py — Run multiple ROI full pipelines in parallel.

Each worker = subprocess running run_one_roi_full.py for one ROI.
Each does prep + MCMC + merge for that ROI.

Usage:
    python launch_cov_full.py                  # 3 workers (default, safe)
    python launch_cov_full.py --workers 5      # 5 parallel ROIs
    python launch_cov_full.py --rois 25 -25 30 # only these ROIs

Each ROI takes ~5.5 hours. With 3 workers, 19 ROIs = ~35 hours.

Notes:
- Uses per-worker PFILES to avoid fermitools .par race condition
- Reads/writes only its own ROI files (no contention except disk I/O)
- gtsrcmaps for different ROIs uses different ccube/expcube files (no conflict)
- Resume-friendly: skips ROIs whose final .npz exists
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

ALL_ROIS = [
    -70, -65, -60, -55, -50, -45, -40, -35, -30, -25,
     25,  30,  35,  40,  45,  50,  55,  60,  65,  70,
]


def out_path(roi):
    return f'./GCE_cov_l{roi}_front_17yr_cholis_fit.npz'


def is_done(roi):
    return os.path.exists(out_path(roi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3,
                    help='parallel ROIs (default 3; each worker uses ~1-2GB RAM)')
    ap.add_argument('--rois', type=int, nargs='*', default=None,
                    help='specific ROIs (default: all 20)')
    ap.add_argument('--logs', default='./logs_cov_full')
    ap.add_argument('--python', default=sys.executable)
    args = ap.parse_args()

    requested = list(args.rois) if args.rois else list(ALL_ROIS)
    skipped = [r for r in requested if is_done(r)]
    pending = [r for r in requested if not is_done(r)]

    if skipped:
        print(f'[skip] {len(skipped)} ROIs already done: {skipped}')
    if not pending:
        print('[done] nothing to do')
        return 0

    print(f'[plan] {len(pending)} ROIs with {args.workers} workers')
    print(f'[plan] ROIs to run: {pending}')

    Path(args.logs).mkdir(exist_ok=True)
    Path('./pfiles_cov_full').mkdir(exist_ok=True)

    queue = list(pending)
    running = {}    # roi -> (proc, log_handle, start_time)
    completed = []
    failed = []

    state_file = Path('cov_full_state.json')

    def write_state():
        state_file.write_text(json.dumps({
            'started':   time.strftime('%Y-%m-%d %H:%M:%S'),
            'workers':   args.workers,
            'pending':   queue,
            'running':   list(running.keys()),
            'completed': completed,
            'failed':    failed,
            'skipped':   skipped,
        }, indent=2))

    def launch(roi):
        log_path = Path(args.logs) / f'roi_{roi:+d}.log'
        log_h = open(log_path, 'w', buffering=1)
        cmd = [args.python, 'run_one_roi_full.py', str(roi)]
        log_h.write(f'# launched: {" ".join(shlex.quote(c) for c in cmd)}\n')
        log_h.write(f'# at: {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        log_h.flush()

        # Per-worker PFILES (fermitools .par isolation)
        env = os.environ.copy()
        worker_pfiles = Path('./pfiles_cov_full') / f'roi_{roi:+d}'
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
        running[roi] = (proc, log_h, time.time())
        print(f'  [launch] roi={roi:+d}  pid={proc.pid}  log={log_path}')

    def reap():
        finished = []
        for roi, (proc, log_h, t0) in running.items():
            rc = proc.poll()
            if rc is not None:
                finished.append((roi, rc, t0))
        for roi, rc, t0 in finished:
            proc, log_h, _ = running.pop(roi)
            log_h.close()
            dt_min = (time.time() - t0) / 60
            done_total = len(completed) + len(failed)
            total = len(pending)
            if rc == 0 and is_done(roi):
                completed.append(roi)
                print(f'  [ok  ] roi={roi:+d}  {dt_min:.0f}min  '
                      f'({done_total+1}/{total} done, {len(running)} running, {len(queue)} queued)')
            else:
                failed.append((roi, rc))
                print(f'  [FAIL rc={rc}] roi={roi:+d}  {dt_min:.0f}min  see log')

    def graceful_shutdown(sig=None, frame=None):
        print('\n[shutdown] terminating workers...')
        for roi, (proc, log_h, _) in running.items():
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

        time.sleep(30.0)   # check every 30s (each ROI is hours-long, no need to poll fast)
        reap()
        write_state()

    dt_h = (time.time() - t_start) / 3600
    print(f'\n[finished] {len(completed)} ok, {len(failed)} failed, '
          f'wall time {dt_h:.1f} h')
    if failed:
        for r, rc in failed:
            print(f'  failed: roi={r:+d} rc={rc} - see {args.logs}/roi_{r:+d}.log')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
