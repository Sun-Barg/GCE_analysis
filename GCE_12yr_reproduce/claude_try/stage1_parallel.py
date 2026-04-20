import os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR, MODELS, STAGE1_PARALLEL_MODELS

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def _all_stage1_outputs_exist(model):
    base = os.path.join(WORK_DIR, 'GC_analysis_sanghwan')
    required = []
    for conv in ('', '_no_convol'):
        for comp in ('pion', 'bremss', 'ics'):
            required.append(f'{base}/GC_{comp}_model{model}_12yr_front_clean{conv}.fits')
        for comp in ('GCE', 'fermi_bubble', 'isotropic'):
            required.append(f'{base}/GC_{comp}_model_12yr_front_clean{conv}.fits')
    return all(os.path.exists(p) for p in required)


def launch_all(target_models=None):
    if target_models is None:
        target_models = MODELS
    procs = []
    for m in target_models:
        if _all_stage1_outputs_exist(m):
            print(f"[{time.strftime('%H:%M:%S')}] [skip stage1] model {m} — all outputs present", flush=True)
            continue
        logf = os.path.join(OUT_DIR, f'log_stage1_{m}.txt')
        cmd = ['python', '-u', os.path.join(PIPELINE_DIR, 'stage1_maps.py'), m]
        fh = open(logf, 'w')
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=WORK_DIR)
        procs.append({'model': m, 'proc': p, 'log': logf, 'fh': fh, 'start': time.time()})
        print(f"[{time.strftime('%H:%M:%S')}] launched {m}: PID {p.pid}  log={logf}", flush=True)

    if not procs:
        print(f"[{time.strftime('%H:%M:%S')}] Stage 1 — nothing to do.", flush=True)
        return 0

    n_fail = 0
    while procs:
        time.sleep(30)
        still = []
        for info in procs:
            rc = info['proc'].poll()
            if rc is None:
                elapsed = (time.time() - info['start']) / 60
                print(f"  [{time.strftime('%H:%M:%S')}] {info['model']} still running ({elapsed:.1f} min)", flush=True)
                still.append(info)
            else:
                info['fh'].close()
                elapsed = (time.time() - info['start']) / 60
                status = 'OK' if rc == 0 else f'FAIL (rc={rc})'
                print(f"  [{time.strftime('%H:%M:%S')}] {info['model']} {status} in {elapsed:.1f} min", flush=True)
                if rc != 0:
                    n_fail += 1
        procs = still
    return n_fail


if __name__ == '__main__':
    target = sys.argv[1:] if len(sys.argv) > 1 else None
    t0 = time.time()
    n_fail = launch_all(target)
    total = (time.time() - t0) / 60
    print(f"\n[{time.strftime('%H:%M:%S')}] Stage 1 parallel done in {total:.1f} min  ({n_fail} failed)", flush=True)
    sys.exit(0 if n_fail == 0 else 1)
