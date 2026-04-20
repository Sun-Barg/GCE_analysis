"""
Envelope Stage 1 - launch global-c_gce fits across 80 models.

Features
--------
- Reads runnable models from envelope_available_models.txt (Stage 0 output).
- Configurable concurrency (env ENVELOPE_CONCURRENCY, default 4).
- Configurable NSTEPS (env FIT_NSTEPS, default 500 for fast scan).
- Skips models whose pkl already exists.
- Checkpoint file: envelope_progress.json records which models completed
  so re-running continues where it left off.
- Per-model timeout (env ENVELOPE_TIMEOUT_MIN, default 240) kills runaway fits.
- Memory-observant polling: every 60s prints running jobs + free RAM.

Usage
-----
    # fast scan (default): NSTEPS=500, NBURN=150, concurrency=4
    python envelope_stage1_fit.py

    # full MCMC (slow): NSTEPS=1500, NBURN=500, concurrency=2
    FIT_NSTEPS=1500 FIT_NBURN=500 ENVELOPE_CONCURRENCY=2 \
        FIT_SUFFIX=globalCgce_envelope_full \
        python envelope_stage1_fit.py

    # custom model list (ignore stage 0 file)
    python envelope_stage1_fit.py X XV XLIX LIII XLVIII

Expected wall-clock
-------------------
- Fast scan: ~50 min per model -> ~20 hr total at 4 concurrent (80 models).
- Full MCMC: ~180 min per model -> ~120 hr total at 2 concurrent.

If killed mid-run, re-run the same command - completed models skip.
"""
import os, sys, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
AVAILABLE_MODELS_FILE = os.path.join(OUT_DIR, 'envelope_available_models.txt')
PROGRESS_FILE = os.path.join(OUT_DIR, 'envelope_progress.json')

SUFFIX = os.environ.get('FIT_SUFFIX', 'globalCgce_envelope')
MASK_CHOICE = os.environ.get('MASK_CHOICE', 'calibrated')
CONCURRENCY = int(os.environ.get('ENVELOPE_CONCURRENCY', '4'))
FIT_NSTEPS = os.environ.get('FIT_NSTEPS', '500')
FIT_NBURN = os.environ.get('FIT_NBURN', '150')
TIMEOUT_MIN = int(os.environ.get('ENVELOPE_TIMEOUT_MIN', '240'))
POLL_INTERVAL = int(os.environ.get('ENVELOPE_POLL_SEC', '60'))


def load_model_list(argv_models):
    if argv_models:
        return list(argv_models)
    if not os.path.exists(AVAILABLE_MODELS_FILE):
        print(f"ERROR: {AVAILABLE_MODELS_FILE} not found.")
        print("Run envelope_stage0_check_models.py first, or pass models on CLI.")
        sys.exit(1)
    models = []
    with open(AVAILABLE_MODELS_FILE) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith('#'):
                models.append(s)
    return models


def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {'done': {}, 'failed': {}}
    with open(PROGRESS_FILE) as f:
        return json.load(f)


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def pkl_path(model):
    return os.path.join(OUT_DIR,
                        f'GCE_model_{model}_haebarg_v_claude_{SUFFIX}.pkl')


def launch(model):
    logf = os.path.join(OUT_DIR, f'log_envelope_fit_{model}.txt')
    env = os.environ.copy()
    env['FIT_SUFFIX'] = SUFFIX
    env['FIT_NSTEPS'] = FIT_NSTEPS
    env['FIT_NBURN'] = FIT_NBURN
    env['MASK_CHOICE'] = MASK_CHOICE
    cmd = ['python', '-u',
            os.path.join(PIPELINE_DIR, 'envelope_fit_one.py'), model]
    fh = open(logf, 'w')
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                          env=env, cwd=WORK_DIR)
    return {'m': model, 'p': p, 'fh': fh, 'log': logf,
            't': time.time(), 'pid': p.pid}


def free_ram_gb():
    try:
        with open('/proc/meminfo') as f:
            kb = {}
            for line in f:
                k, v = line.split(':')
                kb[k.strip()] = int(v.strip().split()[0])
        return kb.get('MemAvailable', 0) / 1024 / 1024
    except Exception:
        return float('nan')


def main():
    os.chdir(WORK_DIR)
    argv_models = sys.argv[1:]
    models_all = load_model_list(argv_models)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Envelope Stage 1")
    print(f"  models total           : {len(models_all)}")
    print(f"  SUFFIX                 : {SUFFIX}")
    print(f"  MASK_CHOICE            : {MASK_CHOICE}")
    print(f"  NSTEPS / NBURN         : {FIT_NSTEPS} / {FIT_NBURN}")
    print(f"  CONCURRENCY            : {CONCURRENCY}")
    print(f"  TIMEOUT_MIN / POLL_SEC : {TIMEOUT_MIN} / {POLL_INTERVAL}")
    print(f"  progress file          : {PROGRESS_FILE}")
    print(f"  free RAM at start      : {free_ram_gb():.1f} GB")
    print()

    progress = load_progress()

    pending = []
    skipped_done = []
    for m in models_all:
        if os.path.exists(pkl_path(m)):
            skipped_done.append(m)
            progress.setdefault('done', {})[m] = progress.get('done', {}).get(m, 'pre-existing')
            continue
        pending.append(m)
    save_progress(progress)

    if skipped_done:
        print(f"  already done ({len(skipped_done)}): "
              f"{' '.join(skipped_done[:10])}"
              + ("..." if len(skipped_done) > 10 else ""))
    print(f"  to run   : {len(pending)} models")
    if not pending:
        print("  nothing to do.")
        return

    running = []
    finished_this_session = []
    failed_this_session = []
    t_start = time.time()

    while pending or running:
        while pending and len(running) < CONCURRENCY:
            m = pending.pop(0)
            info = launch(m)
            running.append(info)
            print(f"[{time.strftime('%H:%M:%S')}] launched {m}  PID={info['pid']}  "
                  f"running={[r['m'] for r in running]}  pending={len(pending)}",
                  flush=True)

        time.sleep(POLL_INTERVAL)

        still = []
        for info in running:
            rc = info['p'].poll()
            elapsed_min = (time.time() - info['t']) / 60

            if rc is None:
                if elapsed_min > TIMEOUT_MIN:
                    print(f"[{time.strftime('%H:%M:%S')}] {info['m']} TIMEOUT "
                          f"after {elapsed_min:.1f} min, killing", flush=True)
                    try:
                        info['p'].kill()
                    except Exception as e:
                        print(f"  kill failed: {e}")
                    info['fh'].close()
                    failed_this_session.append(info['m'])
                    progress.setdefault('failed', {})[info['m']] = 'timeout'
                    save_progress(progress)
                else:
                    still.append(info)
            else:
                info['fh'].close()
                out_exists = os.path.exists(pkl_path(info['m']))
                if rc == 0 and out_exists:
                    finished_this_session.append(info['m'])
                    progress.setdefault('done', {})[info['m']] = f'{elapsed_min:.1f}min'
                    print(f"[{time.strftime('%H:%M:%S')}] {info['m']} OK in "
                          f"{elapsed_min:.1f} min", flush=True)
                else:
                    failed_this_session.append(info['m'])
                    progress.setdefault('failed', {})[info['m']] = (
                        f'rc={rc} no_pkl' if not out_exists else f'rc={rc}')
                    print(f"[{time.strftime('%H:%M:%S')}] {info['m']} FAIL "
                          f"rc={rc} after {elapsed_min:.1f} min", flush=True)
                save_progress(progress)
        running = still

        if running or pending:
            free_gb = free_ram_gb()
            elapsed_total = (time.time() - t_start) / 60
            running_str = ' '.join([f"{r['m']}({(time.time()-r['t'])/60:.0f}m)"
                                     for r in running])
            print(f"  [{time.strftime('%H:%M:%S')}] total={elapsed_total:.0f}m  "
                  f"running=[{running_str}]  pending={len(pending)}  "
                  f"RAM_free={free_gb:.1f}GB", flush=True)

    total_min = (time.time() - t_start) / 60
    print()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Envelope Stage 1 complete")
    print(f"  wall-clock          : {total_min:.1f} min")
    print(f"  finished this run   : {len(finished_this_session)}")
    print(f"  failed this run     : {len(failed_this_session)}")
    print(f"  total done (all)    : {len(progress.get('done', {}))}")
    print(f"  total failed (all)  : {len(progress.get('failed', {}))}")
    sys.exit(0 if not failed_this_session else 1)


if __name__ == '__main__':
    main()
