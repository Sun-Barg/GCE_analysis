#!/usr/bin/env python3
"""
prepare_one_roi_cov.py — per-ROI data preparation for the 17yr GCE
covariance pipeline.

Reproduces the cov notebook's per-ROI portions only (cells 10, 14). The
notebook also contains shared-once steps that are NOT this script's
responsibility:

  - cell 8  CMAP build (GC_cmap_17yr_front_clean.fits)
            only used as a WCS reference by cell 9 (wimp_map);
            the production worker (run_one_roi_cov.py) reads the main
            GC_ccube directly, so this CMAP is unused in production.
  - cell 9  wimp_map calculation
            handled by make_wimp_map_per_roi.py.
  - cell 11 Allsky_expcube_edge
            already produced by prepare_common.py Step 9.

This script's three per-ROI steps:
  1. gtexpcube2 (CENTER, xref=roi)
        -> GC_expcube_center_17yr_front_clean_l{roi}.fits           (cov cell 10)
  2. SourceList + iso/gal prune
        -> Model/GC_model_FL16Y_l{roi}.xml                          (cov cell 14)
        -> Model/GC_psc_model_FL16Y_l{roi}.xml                      (cov cell 14)
  3. Per-ROI PSC mask
        -> Model/GC_mask_60x60_definitions_FL16Y_l{roi}.npy         (cov cell 14)

Run from: ~/GCE-Chi-square-fitting/GCE_17yr_reproduce/   (working directory)
Usage:
    python3 prepare_one_roi_cov.py <roi>                # roi e.g. 25, -70
    python3 prepare_one_roi_cov.py <roi> --force-step 2
    python3 prepare_one_roi_cov.py <roi> --force-step 1,2,3
    python3 prepare_one_roi_cov.py <roi> --force-all

Skip policy (same as prepare_common.py):
    output present + integrity OK  -> skip
    output present + integrity BAD -> sys.exit(2) with explicit msg
                                       (NO silent stale-file reuse)
    output absent                  -> build, then integrity-check
    --force-step N                 -> delete output, then build + check

Prerequisites (produced by prepare_common.py; checked at startup):
  - {WORK_DIR}/Allsky_ltcube_17yr_front_clean.fits   (prep Step 7)
  - {WORK_DIR}/GC_ccube_17yr_front_clean.fits        (prep Step 6)
  - {WORK_DIR}/bin_definitions.fits                  (prep Step 3)
  - {WORK_DIR}/Model/GC_psc_model_FL16Y.xml          (prep Step 11)
  - {WORK_DIR}/Model/source_classification.npz       (prep Step 12)

================================================================
LEGACY BEHAVIOR PRESERVED (cov cell 14 L19-L20 quirk)
================================================================
The per-ROI mask in Step 3 is built using the MAIN source classification
(sig/not_sig from source_classification.npz, derived from main
GC_psc_model_FL16Y.xml), NOT from the per-ROI XML produced in Step 2 here.

This mirrors the cov notebook exactly: at cell 14 L19, the line
    tree = ET.parse(f'.../GC_psc_model_FL16Y_l{roi}.xml')
is commented out, and L20 falls back to
    tree = ET.parse('.../GC_psc_model_FL16Y.xml')

Consequence: the per-ROI mask content ends up bit-identical to the main
psc mask (same sources, same WCS from main GC_ccube). The per-ROI
filename is preserved for compatibility with run_one_roi_cov.py.

The per-ROI XML produced in Step 2 IS still used downstream by
run_one_roi_cov.py for gtsrcmaps/gtmodel, so Step 2 must remain.

If this legacy behavior is ever revisited (i.e. switch the mask to
genuine per-ROI sources), the change point is documented inline in
build_psc_mask_roi() below. NB: such a change would alter downstream cov
MCMC results.
================================================================

Author: haebarg (2026)

Changes:
  [fb17-cov-v1] (2026-07-28) FB17=1 env -> front+back 17-bin cov variant
      (main pipeline과 동일 패턴): WORK_DIR/FRONT/evtype 전환, 결과는
      results_cov_fb17/ 분리(기존 front 22개 .dat와 카운트 충돌 방지).
      env 미설정 시 기존 fiducial 동작과 동일.
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
from astropy.io import fits

from GtApp import GtApp
from LATSourceModel import SourceList

from cholis_masking import (
    masking, galactic_to_equatorial,
    verify_cube, verify_xml, verify_mask_npy,
    EXPECTED_NEBINS as _CM_NEBINS,
)


# ============================================================
# CONFIG  (mirrored verbatim from main notebook cell 3 v6 — the single
#          source of truth for paths and run parameters. MUST be kept
#          byte-for-byte consistent with prepare_common.py CONFIG.
#          Do NOT diverge from cell 3 without updating both scripts.)
# ============================================================
WORK_DIR = './GC_analysis_FL16Y'

# Event type / IRF
front          = '_front'              # filename suffix
evtype_number  = 1                     # 1=FRONT only

# FB17 variant (front+back, 17 bins) — env switch (prepare_common.py 동일).
FB17 = bool(os.environ.get('FB17', '').strip())
if FB17:
    WORK_DIR      = './GC_analysis_FL16Y_fb17'
    front         = '_front_back'
    evtype_number = 3
    print(f'[config] FB17=1 -> WORK_DIR={WORK_DIR}, front={front!r}, '
          f'evtype={evtype_number}', flush=True)
evclass_number = 256                   # 256 = P8R3 CLEAN
IRFS           = 'P8R3_CLEAN_V3'

# Catalog (FL16Y) + extended templates
DR_NUMBER      = 4                     # fermitools only supports DR=1..4; FL16Y uses DR=4
CATALOG_FILE   = '../GCE_17yr_data/gll_psc_v40.fit'
EXTENDED_DIR   = '../GCE_17yr_data/LAT_extended_sources_16years/Templates/'

# Diffuse + isotropic (CLEAN, matching IRFS)
GALACTIC_FILE  = '../gll_iem_v07.fits'
ISOTROPIC_FILE = '../iso_P8R3_CLEAN_V3_v1.txt'

# Inputs from prepare_common.py
MODEL_DIR             = f'{WORK_DIR}/Model'
ALLSKY_LTCUBE         = f'{WORK_DIR}/Allsky_ltcube_17yr{front}_clean.fits'
GC_CCUBE              = f'{WORK_DIR}/GC_ccube_17yr{front}_clean.fits'
BIN_DEF_FILE          = f'{WORK_DIR}/bin_definitions.fits'
GC_PSC_MODEL_XML      = f'{MODEL_DIR}/GC_psc_model_FL16Y.xml'   # main; used for legacy-equivalent mask
SOURCE_CLASSIFICATION = f'{MODEL_DIR}/source_classification.npz'

# Image dimensions (must match prepare_common.py CCUBE_NXPIX/NYPIX)
CCUBE_NXPIX, CCUBE_NYPIX = 600, 600

# ROI validation (legacy condition: i != 0 and abs(i) < 20 excluded;
# launchers typically use the 20-ROI set ±25..±70 step 5)
ROI_MIN_ABS = 20
ROI_MAX_ABS = 70
ROI_STEP    = 5


# ============================================================
# Step framework  (mirrors prepare_common.py)
# ============================================================

def _ts():
    return datetime.now().strftime('%H:%M:%S')


def _abort(msg):
    print(f'\n[FATAL] {msg}', flush=True)
    print(f'[FATAL] prepare_one_roi_cov.py aborts; resolve and re-run.',
          flush=True)
    sys.exit(2)


class Step:
    """One pipeline step.

    Wires together: identity, output paths, the integrity verifier, the
    build function. `run()` handles skip-if-valid, force, stale-file abort,
    and post-build verification.
    """
    def __init__(self, n, name, outputs, verifier, builder):
        self.n        = n
        self.name     = name
        self.outputs  = outputs        # list of paths (the step's deliverables)
        self.verifier = verifier       # () -> (ok, msg)
        self.builder  = builder        # callable, side-effects only

    def run(self, force_set, force_all):
        forced = force_all or (self.n in force_set)
        print(f'\n[{_ts()}] === Step {self.n}: {self.name} ===', flush=True)

        if forced:
            # Delete any existing outputs first, so the post-build verify is meaningful.
            for p in self.outputs:
                if os.path.exists(p):
                    print(f'  [force] removing existing {p}', flush=True)
                    try:
                        os.remove(p)
                    except OSError as e:
                        _abort(f'cannot remove {p}: {e}')
        else:
            outs_exist = [os.path.exists(p) for p in self.outputs]
            if all(outs_exist):
                ok, msg = self.verifier()
                if ok:
                    print(f'  [skip] verified OK ({msg})', flush=True)
                    return
                else:
                    _abort(f'Step {self.n} outputs exist but FAILED integrity '
                           f'check: {msg}\n        outputs: {self.outputs}\n'
                           f'        Either delete the stale file(s) manually, '
                           f'or run with --force-step {self.n}.')
            elif any(outs_exist):
                stragglers = [p for p, e in zip(self.outputs, outs_exist) if e]
                _abort(f'Step {self.n} has partial outputs (mixed presence). '
                       f'Existing: {stragglers}. Remove manually or use '
                       f'--force-step {self.n} to rebuild.')

        # Build
        t0 = time.time()
        try:
            self.builder()
        except Exception as e:
            _abort(f'Step {self.n} builder raised {type(e).__name__}: {e}')
        dt = time.time() - t0

        # Post-build verification
        ok, msg = self.verifier()
        if not ok:
            _abort(f'Step {self.n} built outputs FAILED integrity check '
                   f'immediately after build: {msg}\n        outputs: '
                   f'{self.outputs}')
        print(f'  [done] {dt:.1f}s — {msg}', flush=True)


# ============================================================
# Builders (parameterized by ROI through closures)
# ============================================================

def make_builders(roi, gc_expcube_roi, gc_model_xml_roi,
                  gc_psc_model_xml_roi, psc_mask_npy_roi):
    """Factory: returns three zero-arg builder callables tied to this ROI's
    output paths. Used by Step instances built in build_steps(roi).
    """

    # ----- Step 1: gtexpcube2 (CENTER, per-ROI xref) -----
    def build_gtexpcube_roi():
        """Cov cell 10, single iteration with xref=roi."""
        g = GtApp('gtexpcube2', 'Likelihood')
        g['infile']   = ALLSKY_LTCUBE
        g['cmap']     = 'none'
        g['outfile']  = gc_expcube_roi
        g['evtype']   = evtype_number
        g['coordsys'] = 'GAL'
        g['xref']     = roi
        g['yref']     = 0
        g['nxpix']    = CCUBE_NXPIX
        g['nypix']    = CCUBE_NYPIX
        g['proj']     = 'CAR'
        g['binsz']    = 0.1
        g['bincalc']  = 'CENTER'
        g['irfs']     = IRFS
        g['ebinalg']  = 'FILE'
        g['ebinfile'] = BIN_DEF_FILE
        g.run()

    # ----- Step 2: per-ROI SourceList XML + iso/gal prune -----
    def build_source_list_roi():
        """Cov cell 14 L1-15: SourceList centered at galactic (roi, 0), then
        prune iso/galactic from the resulting XML.

        Writes (atomically, via .tmp + rename):
          - {gc_model_xml_roi}      (raw SourceList output)
          - {gc_psc_model_xml_roi}  (iso/gal pruned)
        """
        ra, dec = galactic_to_equatorial(roi, 0)
        sl = SourceList(
            DR=DR_NUMBER,
            catalog_file=CATALOG_FILE,
            ROI=[ra, dec, 35],
            output_name=f'GC_model_FL16Y_l{roi}.xml',
            write_directory=f'{MODEL_DIR}/',
        )
        sl.make_model(
            extended_catalog_names=True,
            norms_free_only=True,
            galactic_index_free=True,
            extra_radius=5,
            free_radius=28.28,
            max_free_radius=28.28,
            variable_free=True,
            sigma_to_free=49,
            galactic_name='gll_iem',
            galactic_file=GALACTIC_FILE,
            isotropic_file=ISOTROPIC_FILE,
            isotropic_name='isotropic',
            extended_directory=EXTENDED_DIR,
        )
        # NB: SourceList.make_model() writes directly to gc_model_xml_roi
        # (no .tmp). If gtools is killed mid-write the file would be partial.
        # The post-build verifier (verify_xml) catches that.

        # Prune iso/gal -> per-ROI PSC XML (cov cell 14 L5-15)
        tree = ET.parse(gc_model_xml_roi)
        root = tree.getroot()
        for src in list(root.findall('.//source')):
            name = src.get('name', '')
            if 'isotropic' in name or 'gll_iem' in name:
                root.remove(src)
        tmp = gc_psc_model_xml_roi + '.tmp'
        tree.write(tmp, encoding='utf-8', xml_declaration=True)
        os.rename(tmp, gc_psc_model_xml_roi)

    # ----- Step 3: per-ROI PSC mask (using MAIN sources — legacy quirk) -----
    def build_psc_mask_roi():
        """Cov cell 14 L17-89 (mask portion), mirroring the legacy behavior
        of using the MAIN source classification (from prep Step 12) and the
        MAIN GC_ccube as the WCS reference frame.

        ================== GOTCHA / FIX POINT ==================
        cov cell 14 L19 has `tree = ET.parse(.../GC_psc_model_FL16Y_l{roi}.xml)`
        COMMENTED OUT in favor of L20 `tree = ET.parse(.../GC_psc_model_FL16Y.xml)`
        (main). This means the per-ROI mask is built from MAIN sources, not
        per-ROI sources. Result: the per-ROI mask content is bit-identical
        to the main psc mask (already produced by prepare_common.py Step 13).

        To switch to genuine per-ROI sources (i.e. "fix" the legacy quirk):
          1. Replace the np.load(SOURCE_CLASSIFICATION) block below with a
             fresh ET.parse(gc_psc_model_xml_roi) + Signif_Avg classification
             loop (cf. prepare_common.py build_source_classification).
          2. Coordinate the change with run_one_roi_cov.py and any existing
             cov matrix results — downstream cov MCMC outputs will change.
        ========================================================
        """
        d = np.load(SOURCE_CLASSIFICATION, allow_pickle=True)
        sig     = list(d['sig_ra_dec_values'])
        not_sig = list(d['not_sig_ra_dec_values'])

        # Per-bin energy (geometric mean of edges) in GeV, from main GC_ccube
        e_bounds = fits.open(GC_CCUBE)[1].data
        E = np.array([1e-3 * np.sqrt(e_bounds[i][2] * e_bounds[i][1] * 1e-6)
                      for i in range(len(e_bounds))])
        print(f'  E ({len(E)} bins): {E[0]:.3f} - {E[-1]:.3f} GeV', flush=True)

        raw_shape = fits.open(GC_CCUBE)[0].data.shape   # (14, 600, 600)
        mask_big   = np.zeros(raw_shape, dtype=np.float32)
        mask_small = np.zeros(raw_shape, dtype=np.float32)
        for i in range(len(E)):
            mask_big[i]   = masking(1, sig,     E[i], GC_CCUBE, mask_scale=1.0)
            mask_small[i] = masking(0, not_sig, E[i], GC_CCUBE, mask_scale=1.0)
            print(f'    bin {i:>2}: E={E[i]:.3f} GeV  '
                  f'unmasked_big={int(mask_big[i].sum())}  '
                  f'unmasked_small={int(mask_small[i].sum())}', flush=True)
        full_mask = (mask_big * mask_small).astype(np.float32)

        tmp = psc_mask_npy_roi + '.tmp.npy'
        np.save(tmp, full_mask)
        os.rename(tmp, psc_mask_npy_roi)

    return build_gtexpcube_roi, build_source_list_roi, build_psc_mask_roi


# ============================================================
# Step registry
# ============================================================

def build_steps(roi):
    ny, nx = CCUBE_NYPIX, CCUBE_NXPIX

    gc_expcube_roi       = f'{WORK_DIR}/GC_expcube_center_17yr{front}_clean_l{roi}.fits'
    gc_model_xml_roi     = f'{MODEL_DIR}/GC_model_FL16Y_l{roi}.xml'
    gc_psc_model_xml_roi = f'{MODEL_DIR}/GC_psc_model_FL16Y_l{roi}.xml'
    psc_mask_npy_roi     = f'{MODEL_DIR}/GC_mask_60x60_definitions_FL16Y_l{roi}.npy'

    b_exp, b_src, b_mask = make_builders(
        roi, gc_expcube_roi, gc_model_xml_roi,
        gc_psc_model_xml_roi, psc_mask_npy_roi,
    )

    def verify_per_roi_xmls():
        for p in (gc_model_xml_roi, gc_psc_model_xml_roi):
            ok, msg = verify_xml(p, min_sources=100)
            if not ok:
                return False, f'{p}: {msg}'
        return True, '2 per-ROI XMLs OK'

    return [
        Step(1, f'gtexpcube2 (CENTER, xref={roi}) -> per-ROI expcube',
             [gc_expcube_roi],
             lambda: verify_cube(gc_expcube_roi, expected_xy=(nx, ny)),
             b_exp),
        Step(2, f'SourceList + iso/gal prune (l={roi})',
             [gc_model_xml_roi, gc_psc_model_xml_roi],
             verify_per_roi_xmls,
             b_src),
        Step(3, f'Per-ROI PSC mask 14x600x600 '
                f'(MAIN sources, legacy-equivalent)',
             [psc_mask_npy_roi],
             lambda: verify_mask_npy(psc_mask_npy_roi, (_CM_NEBINS, ny, nx)),
             b_mask),
    ]


# ============================================================
# Main
# ============================================================

def parse_force_set(arg):
    if not arg:
        return set()
    out = set()
    for tok in arg.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.add(int(tok))
        except ValueError:
            _abort(f'--force-step value must be int(s), got {tok!r}')
    return out


def validate_roi(roi):
    """Sanity-check the requested ROI. Mirrors the cov notebook condition
    `if i == 0 or abs(i) < 20: continue`, extended with grid + range checks
    so a stray CLI arg can't silently produce nonsense outputs."""
    if roi == 0:
        _abort('roi=0 is excluded (this is the main GC analysis, '
               'not a control ROI).')
    if abs(roi) < ROI_MIN_ABS:
        _abort(f'|roi|={abs(roi)} is too close to GC '
               f'(legacy cutoff: abs(roi) < {ROI_MIN_ABS} excluded).')
    if abs(roi) > ROI_MAX_ABS:
        _abort(f'|roi|={abs(roi)} exceeds the cov ROI grid '
               f'(max abs = {ROI_MAX_ABS} deg from GC).')
    if roi % ROI_STEP != 0:
        _abort(f'roi={roi} is not on the {ROI_STEP}-degree grid.')


def check_prerequisites():
    """All Step inputs are produced by prepare_common.py. Check up-front so
    that a missing/incomplete common-prep doesn't cause a cryptic GtApp
    failure mid-Step. Each missing file is reported with its source step."""
    prereqs = [
        (ALLSKY_LTCUBE,         'prepare_common.py Step 7  (gtltcube)'),
        (GC_CCUBE,              'prepare_common.py Step 6  (gtbin -> GC_ccube)'),
        (BIN_DEF_FILE,          'prepare_common.py Step 3  (bin_definitions.fits)'),
        (GC_PSC_MODEL_XML,      'prepare_common.py Step 11 (main psc XML)'),
        (SOURCE_CLASSIFICATION, 'prepare_common.py Step 12 (source classification npz)'),
    ]
    missing = [(p, src) for p, src in prereqs if not os.path.exists(p)]
    if missing:
        lines = ['Missing prerequisites — run prepare_common.py first:']
        for p, src in missing:
            lines.append(f'    - {p}')
            lines.append(f'        (produced by: {src})')
        _abort('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(
        description='Per-ROI data prep for the 17yr GCE cov pipeline. '
                    'Produces gtexpcube_center, per-ROI XMLs, and per-ROI '
                    'psc mask for one cov control ROI.',
    )
    ap.add_argument('roi', type=int,
                    help='Control ROI longitude in degrees. Must be nonzero, '
                         'on a 5-deg grid, with abs(roi) in [20, 70]. '
                         'Typical set: ±25, ±30, ±35, ±40, ±45, ±50, ±55, '
                         '±60, ±65, ±70 (20 ROIs).')
    ap.add_argument('--force-step', type=str, default='',
                    help='Comma-separated step numbers to force rebuild '
                         '(e.g. "2" or "1,2,3").')
    ap.add_argument('--force-all', action='store_true',
                    help='Rebuild every step (USE WITH CARE — Step 3 mask '
                         'recompute is slow).')
    args = ap.parse_args()

    validate_roi(args.roi)
    force_set = parse_force_set(args.force_step)
    force_all = args.force_all

    # cwd check
    if not os.path.isdir(WORK_DIR):
        _abort(f'{WORK_DIR} not found; run from the working directory '
               f'(~/GCE-Chi-square-fitting/GCE_17yr_reproduce/).')
    os.makedirs(MODEL_DIR, exist_ok=True)

    check_prerequisites()

    print(f'[{_ts()}] prepare_one_roi_cov.py start  pid={os.getpid()}  '
          f'roi={args.roi:+d}')
    print(f'  cwd        : {os.getcwd()}')
    print(f'  force_step : {sorted(force_set) if force_set else "—"}')
    print(f'  force_all  : {force_all}')

    t0 = time.time()
    steps = build_steps(args.roi)
    for s in steps:
        s.run(force_set, force_all)
    dt = time.time() - t0

    print(f'\n[{_ts()}] prepare_one_roi_cov.py done  '
          f'roi={args.roi:+d}  elapsed={dt/60:.1f} min')
    print(f'  3/3 steps completed and integrity-verified.')


if __name__ == '__main__':
    main()
