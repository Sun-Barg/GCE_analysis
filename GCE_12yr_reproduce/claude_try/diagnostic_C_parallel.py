import os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR, MODELS

SUFFIX = os.environ.get('FIT_SUFFIX', 'globalCgce')
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def launch():
    os.chdir(WORK_DIR)
    procs = []
    for m in MODELS:
        out_pkl = os.path.join(OUT_DIR, f'GCE_model_{m}_haebarg_v_claude_{SUFFIX}.pkl')
        if os.path.exists(out_pkl):
            print(f"[skip {m}] exists", flush=True)
            continue
        logf = os.path.join(OUT_DIR, f'log_diagnostic_C_fit_{m}.txt')
        env = os.environ.copy()
        env['FIT_SUFFIX'] = SUFFIX
        cmd = ['python', '-u',
                os.path.join(PIPELINE_DIR, 'diagnostic_C_global_fit.py'), m]
        fh = open(logf, 'w')
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              env=env, cwd=WORK_DIR)
        procs.append({'m': m, 'p': p, 'fh': fh, 'log': logf, 't': time.time()})
        print(f"[{time.strftime('%H:%M:%S')}] launched {m}: PID {p.pid}  log={logf}",
              flush=True)

    if not procs:
        print(f"[{time.strftime('%H:%M:%S')}] nothing to launch.", flush=True)
        return 0

    n_fail = 0
    while procs:
        time.sleep(60)
        still = []
        for info in procs:
            rc = info['p'].poll()
            if rc is None:
                still.append(info)
                print(f"  [{time.strftime('%H:%M:%S')}] {info['m']} running "
                      f"({(time.time()-info['t'])/60:.1f} min)", flush=True)
            else:
                info['fh'].close()
                st = 'OK' if rc == 0 else f'FAIL rc={rc}'
                print(f"  [{time.strftime('%H:%M:%S')}] {info['m']} {st} in "
                      f"{(time.time()-info['t'])/60:.1f} min", flush=True)
                if rc != 0:
                    n_fail += 1
        procs = still
    return n_fail


if __name__ == '__main__':
    t0 = time.time()
    n_fail = launch()
    total = (time.time() - t0) / 60
    print(f"\n[{time.strftime('%H:%M:%S')}] Experiment C parallel done in {total:.1f} min "
          f"({n_fail} failed)", flush=True)
    sys.exit(0 if n_fail == 0 else 1)
