"""
Sequential launcher for Diagnostic F2.

Runs one EDISP-smeared fit at a time to keep within memory.
Given Indico (~35GB fixed) and each subprocess peaks ~8GB, we run
one-at-a-time to stay safely below ~125GB total.

Each model takes ~180 min. Total for 5 models sequential ~15 hrs,
so we default to X + XLIX (top pair: Group A and Group B reference)
which is ~6 hrs.

Usage:
    python diagnostic_F2_sequential.py                    # runs X, XLIX
    python diagnostic_F2_sequential.py X XV XLVIII        # custom list
"""
import os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR

DEFAULT_MODELS = ['X', 'XLIX']
SUFFIX = os.environ.get('FIT_SUFFIX', 'edispSmeared')
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_one(model):
    logf = os.path.join(OUT_DIR, f'log_diagnostic_F2_fit_{model}.txt')
    out_pkl = os.path.join(OUT_DIR,
                            f'GCE_model_{model}_haebarg_v_claude_{SUFFIX}.pkl')
    if os.path.exists(out_pkl):
        print(f"[skip {model}] pkl exists: {out_pkl}", flush=True)
        return 0

    env = os.environ.copy()
    env['FIT_SUFFIX'] = SUFFIX
    cmd = ['python', '-u',
            os.path.join(PIPELINE_DIR, 'diagnostic_F2_edisp_smeared_fit.py'),
            model]

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] starting {model}  "
          f"log={logf}", flush=True)
    with open(logf, 'w') as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              env=env, cwd=WORK_DIR)
    elapsed = (time.time() - t0) / 60
    print(f"[{time.strftime('%H:%M:%S')}] {model} rc={rc} in {elapsed:.1f} min",
          flush=True)
    return rc


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MODELS
    os.chdir(WORK_DIR)
    print(f"[{time.strftime('%H:%M:%S')}] F2 sequential  models={targets}  "
          f"suffix={SUFFIX}", flush=True)

    t_start = time.time()
    n_fail = 0
    for m in targets:
        rc = run_one(m)
        if rc != 0:
            n_fail += 1
            print(f"  model {m} FAILED rc={rc}", flush=True)

    total = (time.time() - t_start) / 60
    print(f"[{time.strftime('%H:%M:%S')}] F2 done in {total:.1f} min "
          f"({n_fail} fails)", flush=True)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
