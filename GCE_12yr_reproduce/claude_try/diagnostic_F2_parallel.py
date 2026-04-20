"""
Parallel launcher for Diagnostic F2 with bounded concurrency.

Memory budget (125 GB total, Indico ~37 GB resident):
  - Each F2 subprocess peaks at ~10-12 GB (base fit + smeared cube overhead).
  - 2 concurrent: Indico 37 + 2x12 = 61 GB total, ~64 GB free -> safe.
  - 3 concurrent: 37 + 36 = 73 GB, ~15 GB free -> risky (may OOM).
  - 5 concurrent: 37 + 60 = 97 GB, ~unknown -> previously OOM-killed.

Default concurrency = 2. Override with --concurrency N or env F2_CONCURRENCY.

Usage:
    python diagnostic_F2_parallel.py                    # 2 concurrent, X + XLIX
    python diagnostic_F2_parallel.py X XV XLIX          # custom models
    F2_CONCURRENCY=3 python diagnostic_F2_parallel.py   # raise concurrency

Runtime: single model ~180 min. With concurrency=2, 2 models wall-clock ~180 min.
"""
import os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR

DEFAULT_MODELS = ['X', 'XLIX']
DEFAULT_CONCURRENCY = 2
SUFFIX = os.environ.get('FIT_SUFFIX', 'edispSmeared')
CONCURRENCY = int(os.environ.get('F2_CONCURRENCY', DEFAULT_CONCURRENCY))
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def launch_one(model):
    logf = os.path.join(OUT_DIR, f'log_diagnostic_F2_fit_{model}.txt')
    env = os.environ.copy()
    env['FIT_SUFFIX'] = SUFFIX
    cmd = ['python', '-u',
            os.path.join(PIPELINE_DIR, 'diagnostic_F2_edisp_smeared_fit.py'),
            model]
    fh = open(logf, 'w')
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                          env=env, cwd=WORK_DIR)
    return {'m': model, 'p': p, 'fh': fh, 'log': logf, 't': time.time()}


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MODELS
    os.chdir(WORK_DIR)
    print(f"[{time.strftime('%H:%M:%S')}] F2 parallel  models={targets}  "
          f"concurrency={CONCURRENCY}  suffix={SUFFIX}", flush=True)

    queue = []
    skipped = []
    for m in targets:
        out_pkl = os.path.join(OUT_DIR,
                                f'GCE_model_{m}_haebarg_v_claude_{SUFFIX}.pkl')
        if os.path.exists(out_pkl):
            skipped.append(m)
            continue
        queue.append(m)

    if skipped:
        print(f"  [skip] pkl already exists: {skipped}", flush=True)
    if not queue:
        print("  nothing to run.", flush=True)
        return

    print(f"  queued: {queue}", flush=True)
    print(f"  running up to {CONCURRENCY} at a time", flush=True)

    running = []
    pending = list(queue)
    done = []
    t_start = time.time()

    while pending or running:
        while pending and len(running) < CONCURRENCY:
            m = pending.pop(0)
            info = launch_one(m)
            running.append(info)
            print(f"[{time.strftime('%H:%M:%S')}] launched {m}  "
                  f"PID={info['p'].pid}  log={info['log']}", flush=True)

        time.sleep(60)

        still = []
        for info in running:
            rc = info['p'].poll()
            if rc is None:
                still.append(info)
                print(f"  [{time.strftime('%H:%M:%S')}] {info['m']} "
                      f"running ({(time.time()-info['t'])/60:.1f} min)", flush=True)
            else:
                info['fh'].close()
                info['rc'] = rc
                info['elapsed_min'] = (time.time() - info['t']) / 60
                st = 'OK' if rc == 0 else f'FAIL rc={rc}'
                print(f"  [{time.strftime('%H:%M:%S')}] {info['m']} {st} "
                      f"in {info['elapsed_min']:.1f} min", flush=True)
                done.append(info)
        running = still

    total = (time.time() - t_start) / 60
    n_fail = sum(1 for d in done if d.get('rc') != 0)
    print(f"\n[{time.strftime('%H:%M:%S')}] F2 parallel done in {total:.1f} min "
          f"({len(done)} ran, {n_fail} failed, {len(skipped)} skipped)", flush=True)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
