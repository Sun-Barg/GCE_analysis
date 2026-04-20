"""
Envelope Stage 0 - Check which models have Sanghwan srcmap files available.

Reads the naming convention from project base and probes each model's
Pi0 / Bremss / ICS file existence. Writes a list of runnable models.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR

ANA = f'{WORK_DIR}/GC_analysis_sanghwan'
NAMING_FILE = '/home/haebarg/GCE-Chi-square-fitting/GCE_TEMPLATES_FILES_v3/NAMING_CONVENTION_OF_DIFFUSE_EMISSION_MODELS.dat'

if not os.path.exists(NAMING_FILE):
    NAMING_FILE_LOCAL = f'{WORK_DIR}/NAMING_CONVENTION_OF_DIFFUSE_EMISSION_MODELS.dat'
    if os.path.exists(NAMING_FILE_LOCAL):
        NAMING_FILE = NAMING_FILE_LOCAL

AVAILABLE_MODELS_FILE = os.path.join(OUT_DIR, 'envelope_available_models.txt')


def load_all_model_names():
    names = []
    if not os.path.exists(NAMING_FILE):
        print(f"ERROR: naming file not found at {NAMING_FILE}")
        return names
    with open(NAMING_FILE, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) >= 1:
                names.append(parts[0])
    return names


def model_is_runnable(model_name):
    needed = [
        f'GC_pion_model{model_name}_12yr_front_clean.fits',
        f'GC_bremss_model{model_name}_12yr_front_clean.fits',
        f'GC_ics_model{model_name}_12yr_front_clean.fits',
    ]
    for fn in needed:
        if not os.path.exists(f'{ANA}/{fn}'):
            return False, fn
    return True, None


def main():
    os.chdir(WORK_DIR)
    all_models = load_all_model_names()
    print(f"Loaded {len(all_models)} model names from naming convention")

    runnable = []
    missing = []
    for m in all_models:
        ok, miss = model_is_runnable(m)
        if ok:
            runnable.append(m)
        else:
            missing.append((m, miss))

    print(f"  Runnable: {len(runnable)}")
    print(f"  Missing srcmap: {len(missing)}")
    if missing and len(missing) <= 15:
        print("  Missing models (example):")
        for m, miss in missing[:15]:
            print(f"    {m:<8} missing {miss}")
    elif missing:
        print(f"  Missing models (first 15 of {len(missing)}):")
        for m, miss in missing[:15]:
            print(f"    {m:<8} missing {miss}")

    with open(AVAILABLE_MODELS_FILE, 'w') as f:
        f.write("# Runnable models for envelope analysis\n")
        f.write(f"# Total: {len(runnable)} of {len(all_models)}\n")
        for m in runnable:
            f.write(f"{m}\n")
    print(f"\nSaved: {AVAILABLE_MODELS_FILE}")
    print(f"Runnable models ({len(runnable)}): {' '.join(runnable[:20])}"
          + ("..." if len(runnable) > 20 else ""))


if __name__ == '__main__':
    main()
