"""
Envelope fit wrapper - runs global c_gce fit for one model with
env-controlled NSTEPS / NBURN for faster 80-model scan.

Usage:
    FIT_NSTEPS=500 FIT_NBURN=150 python envelope_fit_one.py MODEL
"""
import os, sys

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

FIT_NSTEPS = int(os.environ.get('FIT_NSTEPS', '1500'))
FIT_NBURN = int(os.environ.get('FIT_NBURN', '500'))

import config
config.NSTEPS = FIT_NSTEPS
config.NBURN = FIT_NBURN

import diagnostic_C_global_fit as dcgf
dcgf.NSTEPS = FIT_NSTEPS
dcgf.NBURN = FIT_NBURN


def main():
    if len(sys.argv) < 2:
        print("usage: envelope_fit_one.py MODEL", file=sys.stderr)
        sys.exit(1)
    model = sys.argv[1]
    print(f"[envelope_fit_one] model={model} "
          f"NSTEPS={dcgf.NSTEPS} NBURN={dcgf.NBURN} "
          f"SUFFIX={os.environ.get('FIT_SUFFIX', 'globalCgce')} "
          f"MASK_CHOICE={os.environ.get('MASK_CHOICE', 'ours')}",
          flush=True)
    dcgf.fit_one_model(model)


if __name__ == '__main__':
    main()
