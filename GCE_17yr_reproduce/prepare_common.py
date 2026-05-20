#!/usr/bin/env python3
"""
prepare_common.py — common data preparation for the 17yr GCE pipeline.

Reproduces main notebook cells 5–25 (data preparation only; the per-model
loop in cell 30 is owned by run_one_model.py). Also generates the three
model-independent template XMLs (GCE/isotropic/fermi_bubble) that previously
were a side-effect of run_one_model.py Step 5 — moving them here makes the
prerequisite chain explicit and removes the implicit "first model worker
builds shared XMLs" race.

Run from: ~/GCE-Chi-square-fitting/GCE_17yr_reproduce/   (working directory)
Usage:
    python3 prepare_common.py                       # normal: skip valid, build missing
    python3 prepare_common.py --force-step 7        # rebuild only step 7
    python3 prepare_common.py --force-step 7,8,9    # rebuild several
    python3 prepare_common.py --force-all           # rebuild everything

Skip policy (per step):
    output present + integrity OK  -> skip
    output present + integrity BAD -> sys.exit(2) with explicit msg
                                       (NO silent stale-file reuse)
    output absent                  -> build, then integrity-check
    --force-step N                 -> delete output, then build + check

16 steps:
     1. SC merge (weekly FT2 -> single merged FT2)
     2. Photon listfile
     3. bin_definitions.fits  (Cholis 14 bins)
     4. gtselect              -> Allsky_select
     5. gtmktime              -> Allsky_gti
     6. gtbin (CCUBE)         -> GC_ccube (main GC ROI)
     7. gtltcube              -> Allsky_ltcube
     8. gtexpcube2 center     -> GC_expcube_center
     9. gtexpcube2 edge       -> Allsky_expcube_edge (nebins+1 = 15, normal)
    10. SourceList XML        -> Model/GC_model_FL16Y.xml
    11. Prune iso/galactic    -> Model/GC_psc_model_FL16Y.xml
    12. Classify sources      -> Model/source_classification.npz (sig + not_sig)
    13. Main psc mask         -> Model/GC_mask_60x60_definitions_FL16Y.npy
    14. Disk mask             -> Model/GC_disk_mask_60x60_definitions.npy
    15. empty_model.xml       -> Model/empty_model.xml
    16. Template XMLs         -> Model/GC_{GCE,isotropic,fermi_bubble}_model.xml
    17. (FB17 only) Constraint tables -> Model/{bubble,iso} (staged + verified)

Author: haebarg (2026)

Changes:
  [fb17-v1] (2026-07-27) FB17=1 env -> front+back 17-bin variant:
      WORK_DIR=./GC_analysis_FL16Y_fb17, front='_front_back', evtype=3,
      bin edges = Cholis 14 + 3 high-E bins (upper edges 114.935 /
      252.809 / 556.077 GeV = Sanghwan 16yr front_back scheme-A values;
      bins 16/17 identical to his, bin 15 lower edge stays at our
      fiducial 51.9312 vs his 52.253). Step 7 reuses the fiducial FRONT
      ltcube after an exact GTI-equality check (ltcube is evtype/energy
      independent; FB17_LTCUBE_REBUILD=1 forces a fresh build). New
      Step 17 (FB17 only) stages + verifies the RAW external-constraint
      tables in WORK_DIR/Model (coverage >= top-bin center; quadratic
      EXTRApolation is not acceptable). With FB17 unset the behavior is
      byte-identical to the fiducial 14-bin FRONT pipeline.
"""

import argparse
import glob
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack

from GtApp import GtApp
from LATSourceModel import SourceList

from cholis_masking import (
    masking, equatorial_to_galactic,
    verify_fits, verify_cube, verify_event_file, verify_sc_merged,
    verify_ltcube, verify_xml, verify_mask_npy, verify_bin_def,
)


# ============================================================
# CONFIG  (mirrored verbatim from main notebook cell 3 v6 —
#          the single source of truth for paths and run parameters.
#          Do NOT diverge from cell 3 without updating both places.)
# ============================================================
WORK_DIR = './GC_analysis_FL16Y'

# Event type / IRF
front          = '_front'              # filename suffix
evtype_number  = 1                     # 1=FRONT only
evclass_number = 256                   # 256 = P8R3 CLEAN
IRFS           = 'P8R3_CLEAN_V3'

# Mask scaling: 1.0 = Cholis Table III strict
MASK_SCALE = 1.0

# Energy bins: Cholis 14 bins, 0.274698 - 51.9312 GeV (15 explicit Zenodo edges)
ENERGY_BIN_MAX_GEV = 51.9

# ------------------------------------------------------------
# FB17 variant (front+back, 17 bins) — env switch, PHASE2_CASE-style.
#   FB17 unset/'' -> fiducial (byte-identical behavior)
#   FB17=1        -> front+back 17-bin workdir/run
# ------------------------------------------------------------
FB17 = bool(os.environ.get('FB17', '').strip())
if FB17:
    WORK_DIR      = './GC_analysis_FL16Y_fb17'
    front         = '_front_back'
    evtype_number = 3
    print(f'[config] FB17=1 -> WORK_DIR={WORK_DIR}, front={front!r}, '
          f'evtype={evtype_number} (evclass/IRFS unchanged: 256/P8R3_CLEAN_V3)',
          flush=True)

# Catalog (FL16Y) + extended templates
DR_NUMBER       = 4
CATALOG_FILE    = '../GCE_17yr_data/gll_psc_v40.fit'
EXTENDED_DIR    = '../GCE_17yr_data/LAT_extended_sources_16years/Templates/'

# Diffuse + isotropic
GALACTIC_FILE   = '../gll_iem_v07.fits'
ISOTROPIC_FILE  = '../iso_P8R3_CLEAN_V3_v1.txt'

# Photon weeklies + listfile
PHOTON_LIST_FILE   = 'photon_data_17yr.txt'
PHOTON_INPUT       = PHOTON_LIST_FILE
PHOTON_WEEKLY_GLOB = '../GCE_allsky_data/photon_files/lat_photon_weekly_w*.fits'

# Spacecraft FT2 (gtmktime does NOT accept @listfile reliably; merge required)
SC_MERGED_FILE = '../GCE_allsky_data/lat_spacecraft_merged_17yr.fits'
SC_FILE        = SC_MERGED_FILE
SC_WEEKLY_GLOB = '../GCE_allsky_data/sc_files/lat_spacecraft_weekly_w*.fits'
SC_LIST_FILE   = 'sc_files_17yr.txt'

# Time window: 'INDEF' uses the full spacecraft time range
TMIN = 'INDEF'
TMAX = 'INDEF'

# Templates / spectra (haebarg actual file names)
WIMP_MAP_PATH         = './GCE_template_NFW2.fits'
BUBBLE_TEMPLATE       = './Fermi_Bubbles_template.fits'
ISO_SPECTRUM_FILE     = './isotropic_spectrum_ff.txt'
BUBBLE_SPECTRUM_FILE  = './fermi_bubble_spectrum.txt'

# Per-stage output paths (derived from WORK_DIR + front)
ALLSKY_SELECT   = f'{WORK_DIR}/Allsky_select_17yr{front}_clean.fits'
ALLSKY_GTI      = f'{WORK_DIR}/Allsky_gti_17yr{front}_clean.fits'
GC_CCUBE        = f'{WORK_DIR}/GC_ccube_17yr{front}_clean.fits'
ALLSKY_LTCUBE   = f'{WORK_DIR}/Allsky_ltcube_17yr{front}_clean.fits'
GC_EXPCUBE_CTR  = f'{WORK_DIR}/GC_expcube_center_17yr{front}_clean.fits'
ALLSKY_EXP_EDGE = f'{WORK_DIR}/Allsky_expcube_edge_17yr{front}_clean.fits'
BIN_DEF_FILE    = f'{WORK_DIR}/bin_definitions.fits'

MODEL_DIR              = f'{WORK_DIR}/Model'
GC_MODEL_XML           = f'{MODEL_DIR}/GC_model_FL16Y.xml'
GC_PSC_MODEL_XML       = f'{MODEL_DIR}/GC_psc_model_FL16Y.xml'
SOURCE_CLASSIFICATION  = f'{MODEL_DIR}/source_classification.npz'
PSC_MASK_NPY           = f'{MODEL_DIR}/GC_mask_60x60_definitions_FL16Y.npy'
DISK_MASK_NPY          = f'{MODEL_DIR}/GC_disk_mask_60x60_definitions.npy'
EMPTY_MODEL_XML        = f'{MODEL_DIR}/empty_model.xml'
GCE_MODEL_XML          = f'{MODEL_DIR}/GC_GCE_model.xml'
ISO_MODEL_XML          = f'{MODEL_DIR}/GC_isotropic_model.xml'
BUBBLE_MODEL_XML       = f'{MODEL_DIR}/GC_fermi_bubble_model.xml'

# Cholis 14-bin energy grid (GeV; 15 edges) — Zenodo exact values
CHOLIS_BIN_EDGES_GEV = np.array([
    0.274698, 0.357, 0.464, 0.603, 0.784, 1.02, 1.32, 1.72,
    2.24, 2.91, 3.78, 4.91, 10.8, 23.7, 51.9312,
])
assert len(CHOLIS_BIN_EDGES_GEV) == 15   # 14 bins => 15 edges

# FB17: 3 extra high-E bins. Upper edges = Sanghwan 16yr front_back
# scheme-A values verbatim (bins 16/17 identical to his; bin 15 lower
# edge stays at our fiducial 51.9312 vs his 52.253 — 0.6% caveat for
# the debug-baseline comparison). ~equal log-width extension to 556 GeV.
if FB17:
    CHOLIS_BIN_EDGES_GEV = np.append(
        CHOLIS_BIN_EDGES_GEV, [114.935, 252.809, 556.077])

NEBINS = len(CHOLIS_BIN_EDGES_GEV) - 1
assert NEBINS == (17 if FB17 else 14)
# Cross-check against cholis_masking's env-driven verifier contract: a
# mismatch means FB17/GCE_NEBINS was not exported to this process.
from cholis_masking import EXPECTED_NEBINS as _CM_NEBINS
assert _CM_NEBINS == NEBINS, (
    f'cholis_masking EXPECTED_NEBINS={_CM_NEBINS} != NEBINS={NEBINS}; '
    f'export FB17=1 (or GCE_NEBINS) in the SAME environment for every '
    f'pipeline process.')

CCUBE_NXPIX, CCUBE_NYPIX = 600, 600
EDGE_NXPIX,  EDGE_NYPIX  = 3600, 1800


# ============================================================
# Step framework
# ============================================================

def _ts():
    return datetime.now().strftime('%H:%M:%S')


def _abort(msg):
    print(f'\n[FATAL] {msg}', flush=True)
    print(f'[FATAL] prepare_common.py aborts; resolve the issue and re-run.',
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
        print(f'\n[{_ts()}] === Step {self.n:>2}: {self.name} ===', flush=True)

        if forced:
            # Delete any existing outputs first, so verify after build is meaningful.
            for p in self.outputs:
                if os.path.exists(p):
                    print(f'  [force] removing existing {p}', flush=True)
                    try:
                        os.remove(p)
                    except OSError as e:
                        _abort(f'cannot remove {p}: {e}')
        else:
            # Skip if all outputs exist and pass integrity.
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
# Builders
# ============================================================

# ----- Step 1: SC merge -----
def build_sc_merge():
    """Merge weekly LAT spacecraft FT2 files into single FT2, preserving
    time-system keywords. Atomic write via .tmp + rename."""
    files = sorted(glob.glob(SC_WEEKLY_GLOB))
    if not files:
        raise FileNotFoundError(f'no weekly SC files: {SC_WEEKLY_GLOB}')
    print(f'  merging {len(files)} weekly FT2 files...', flush=True)

    sc_tables, gti_starts, gti_stops = [], [], []
    primary_hdr = sc_data_hdr = gti_hdr = None

    for i, fn in enumerate(files):
        with fits.open(fn, memmap=False) as hdul:
            names = [h.name for h in hdul]
            if primary_hdr is None:
                primary_hdr = hdul[0].header.copy()
            if sc_data_hdr is None and 'SC_DATA' in names:
                sc_data_hdr = hdul['SC_DATA'].header.copy()
            if gti_hdr is None and 'GTI' in names:
                gti_hdr = hdul['GTI'].header.copy()
            if 'SC_DATA' in names:
                sc_tables.append(Table(hdul['SC_DATA'].data))
            if 'GTI' in names and len(hdul['GTI'].data) > 0:
                gti_starts.extend(list(hdul['GTI'].data['START']))
                gti_stops.extend (list(hdul['GTI'].data['STOP']))
        if (i + 1) % 100 == 0:
            print(f'    ... {i+1}/{len(files)}', flush=True)

    merged_sc = vstack(sc_tables)
    merged_sc.sort('START')
    new_tstart = float(merged_sc['START'][0])
    new_tstop  = float(merged_sc['STOP'][-1])

    if gti_starts:
        order = np.argsort(gti_starts)
        gti_starts = list(np.array(gti_starts)[order])
        gti_stops  = list(np.array(gti_stops )[order])

    primary_hdu = fits.PrimaryHDU(header=primary_hdr)
    primary_hdu.header['TSTART'] = new_tstart
    primary_hdu.header['TSTOP']  = new_tstop

    sc_hdu = fits.BinTableHDU(merged_sc, header=sc_data_hdr, name='SC_DATA')
    sc_hdu.header['TSTART'] = new_tstart
    sc_hdu.header['TSTOP']  = new_tstop

    gti_cols = [
        fits.Column(name='START', format='D', array=np.array(gti_starts)),
        fits.Column(name='STOP',  format='D', array=np.array(gti_stops)),
    ]
    gti_hdu = fits.BinTableHDU.from_columns(gti_cols, name='GTI')
    if gti_hdr is not None:
        for key in ('TIMESYS','MJDREFI','MJDREFF','TIMEUNIT','TIMEREF','TASSIGN'):
            if key in gti_hdr:
                gti_hdu.header[key] = gti_hdr[key]
    gti_hdu.header['TSTART'] = new_tstart
    gti_hdu.header['TSTOP']  = new_tstop

    tmp = SC_MERGED_FILE + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    fits.HDUList([primary_hdu, sc_hdu, gti_hdu]).writeto(tmp, overwrite=False)
    os.rename(tmp, SC_MERGED_FILE)


# ----- Step 2: photon listfile -----
def build_photon_listfile():
    paths = sorted(glob.glob(PHOTON_WEEKLY_GLOB))
    if not paths:
        raise FileNotFoundError(f'no weekly photon files: {PHOTON_WEEKLY_GLOB}')
    paths = [os.path.abspath(p) for p in paths]
    tmp = PHOTON_LIST_FILE + '.tmp'
    with open(tmp, 'w') as f:
        f.write('\n'.join(paths) + '\n')
    os.rename(tmp, PHOTON_LIST_FILE)
    print(f'  wrote {PHOTON_LIST_FILE} with {len(paths)} entries', flush=True)


def verify_photon_listfile():
    if not os.path.exists(PHOTON_LIST_FILE):
        return False, f'missing: {PHOTON_LIST_FILE}'
    with open(PHOTON_LIST_FILE) as f:
        entries = [ln.strip() for ln in f if ln.strip()]
    if not entries:
        return False, 'empty listfile'
    missing = [p for p in entries if not os.path.exists(p)]
    if missing:
        return False, f'{len(missing)} listed files do not exist (e.g. {missing[0]})'
    weeks = sorted({int(m.group(1)) for p in entries
                    for m in [re.search(r'_w(\d+)_', os.path.basename(p))] if m})
    return True, f'{len(entries)} entries, weeks w{weeks[0]:03d}..w{weeks[-1]:03d}'


# ----- Step 3: bin_definitions.fits -----
def build_bin_definitions():
    """Energy bins (NEBINS = 14 fiducial / 17 FB17); keV, fermitools convention."""
    e_kev = CHOLIS_BIN_EDGES_GEV * 1.0e6   # GeV -> keV
    cols = [
        fits.Column(name='CHANNEL', format='I', array=np.arange(1, NEBINS + 1)),
        fits.Column(name='E_MIN',   format='E', array=e_kev[:-1]),
        fits.Column(name='E_MAX',   format='E', array=e_kev[1:]),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name='ENERGYBINS')
    tmp = BIN_DEF_FILE + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(tmp)
    os.rename(tmp, BIN_DEF_FILE)


# ----- Step 4: gtselect -----
def build_gtselect():
    f = GtApp('gtselect', 'dataSubselector')
    f['evclass'] = evclass_number
    f['evtype']  = evtype_number
    f['ra']      = 'INDEF'
    f['dec']     = 'INDEF'
    f['rad']     = 'INDEF'
    f['emin']    = 100
    f['emax']    = 1_000_000
    f['zmax']    = 100
    f['tmin']    = TMIN
    f['tmax']    = TMAX
    f['infile']  = PHOTON_LIST_FILE
    f['outfile'] = ALLSKY_SELECT
    f.run()


# ----- Step 5: gtmktime -----
def build_gtmktime():
    m = GtApp('gtmktime', 'dataSubselector')
    m['scfile']  = SC_MERGED_FILE
    m['filter']  = 'DATA_QUAL==1 && LAT_CONFIG==1 && ABS(ROCK_ANGLE) < 52'
    m['roicut']  = 'no'
    m['evfile']  = ALLSKY_SELECT
    m['outfile'] = ALLSKY_GTI
    m.run()


# ----- Step 6: gtbin (main CCUBE) -----
def build_gtbin_main():
    b = GtApp('gtbin', 'evtbin')
    b['algorithm'] = 'CCUBE'
    b['evfile']    = ALLSKY_GTI
    b['outfile']   = GC_CCUBE
    b['nxpix']     = CCUBE_NXPIX
    b['nypix']     = CCUBE_NYPIX
    b['binsz']     = 0.1
    b['coordsys']  = 'GAL'
    b['xref']      = 0
    b['yref']      = 0
    b['axisrot']   = 0
    b['proj']      = 'CAR'
    b['ebinalg']   = 'FILE'
    b['ebinfile']  = BIN_DEF_FILE
    b.run()


# ----- Step 7: gtltcube -----
# FB17 reuse: the livetime cube depends only on (SC file, GTIs, zmax,
# dcostheta, binsz) — NOT on evtype or energy selection. gtselect's
# evtype/emax change the event content but not the GTIs (same weeklies,
# same gtmktime filter, same tmin/tmax), so the fiducial FRONT ltcube is
# bit-valid for front+back IF the GTI tables match exactly. We verify
# that equality and copy; any mismatch (or FB17_LTCUBE_REBUILD=1) falls
# back to a fresh gtltcube build.
FRONT_LTCUBE_REUSE_SRC = './GC_analysis_FL16Y/Allsky_ltcube_17yr_front_clean.fits'
FRONT_GTI_REUSE_REF    = './GC_analysis_FL16Y/Allsky_gti_17yr_front_clean.fits'


def _gti_tables_equal(path_a, path_b):
    """Exact equality of GTI START/STOP arrays between two event files."""
    try:
        with fits.open(path_a, memmap=True) as ha, \
             fits.open(path_b, memmap=True) as hb:
            ga, gb = ha['GTI'].data, hb['GTI'].data
            if len(ga) != len(gb):
                return False, f'GTI row count {len(ga)} != {len(gb)}'
            if not (np.array_equal(ga['START'], gb['START'])
                    and np.array_equal(ga['STOP'], gb['STOP'])):
                return False, 'GTI START/STOP arrays differ'
            return True, f'GTI identical ({len(ga)} rows)'
    except Exception as e:
        return False, f'GTI compare failed: {type(e).__name__}: {e}'


def build_gtltcube():
    if FB17 and not os.environ.get('FB17_LTCUBE_REBUILD', '').strip():
        ok_src, msg_src = (verify_ltcube(FRONT_LTCUBE_REUSE_SRC)
                           if os.path.exists(FRONT_LTCUBE_REUSE_SRC)
                           else (False, f'missing: {FRONT_LTCUBE_REUSE_SRC}'))
        if ok_src and os.path.exists(FRONT_GTI_REUSE_REF):
            ok_gti, msg_gti = _gti_tables_equal(ALLSKY_GTI, FRONT_GTI_REUSE_REF)
            if ok_gti:
                tmp = ALLSKY_LTCUBE + '.tmp'
                if os.path.exists(tmp):
                    os.remove(tmp)
                shutil.copy2(FRONT_LTCUBE_REUSE_SRC, tmp)
                os.rename(tmp, ALLSKY_LTCUBE)
                print(f'  [reuse] fiducial FRONT ltcube copied '
                      f'({msg_gti}; src: {msg_src})', flush=True)
                return
            print(f'  [reuse-SKIP] {msg_gti} -> building fresh ltcube '
                  f'(unexpected for identical selections; report this)',
                  flush=True)
        else:
            print(f'  [reuse-SKIP] fiducial ltcube/GTI unavailable '
                  f'({msg_src}) -> building fresh ltcube', flush=True)
    lt = GtApp('gtltcube', 'Likelihood')
    lt['evfile']    = ALLSKY_GTI
    lt['scfile']    = SC_MERGED_FILE
    lt['outfile']   = ALLSKY_LTCUBE
    lt['dcostheta'] = 0.025
    lt['binsz']     = 1
    lt['zmax']      = 100
    lt.run()


# ----- Step 8: gtexpcube2 center -----
def build_gtexpcube_center():
    g = GtApp('gtexpcube2', 'Likelihood')
    g['infile']   = ALLSKY_LTCUBE
    g['cmap']     = 'none'
    g['outfile']  = GC_EXPCUBE_CTR
    g['evtype']   = evtype_number
    g['coordsys'] = 'GAL'
    g['xref']     = 0
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


# ----- Step 9: gtexpcube2 edge -----
def build_gtexpcube_edge():
    g = GtApp('gtexpcube2', 'Likelihood')
    g['infile']   = ALLSKY_LTCUBE
    g['cmap']     = 'none'
    g['outfile']  = ALLSKY_EXP_EDGE
    g['evtype']   = evtype_number
    g['coordsys'] = 'GAL'
    g['xref']     = 0
    g['yref']     = 0
    g['nxpix']    = EDGE_NXPIX
    g['nypix']    = EDGE_NYPIX
    g['proj']     = 'CAR'
    g['binsz']    = 0.1
    g['bincalc']  = 'EDGE'        # -> nebins+1 layers (normal)
    g['irfs']     = IRFS
    g['ebinalg']  = 'FILE'
    g['ebinfile'] = BIN_DEF_FILE
    g.run()


# ----- Step 10: SourceList XML (main GC) -----
def build_source_list_main():
    sl = SourceList(
        DR=DR_NUMBER,
        catalog_file=CATALOG_FILE,
        ROI=[266, -29, 35],
        output_name='GC_model_FL16Y.xml',
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


# ----- Step 11: Prune iso/galactic from XML -----
def build_psc_model_xml():
    tree = ET.parse(GC_MODEL_XML)
    root = tree.getroot()
    for src in list(root.findall('.//source')):
        name = src.get('name', '')
        if 'isotropic' in name or 'gll_iem' in name:
            root.remove(src)
    tmp = GC_PSC_MODEL_XML + '.tmp'
    tree.write(tmp, encoding='utf-8', xml_declaration=True)
    os.rename(tmp, GC_PSC_MODEL_XML)


# ----- Step 12: Source classification (sig / not_sig) -----
def build_source_classification():
    """Parse GC_psc_model_FL16Y.xml + FL16Y catalog -> two lists of
    [name, ra, dec] keyed by Signif_Avg > 49 (sig) vs <= 49 (not_sig).
    Reproduces main cells 18-19. Cached to npz for traceability."""
    tree = ET.parse(GC_PSC_MODEL_XML)
    root = tree.getroot()

    ra_dec_values, spatial_ra_dec_values = [], []
    for src in root.findall('.//source'):
        name = src.attrib.get('name', '')
        stype = src.attrib.get('type', '')
        if stype == 'PointSource':
            ra_p  = src.find(".//spatialModel/parameter[@name='RA']")
            dec_p = src.find(".//spatialModel/parameter[@name='DEC']")
            if ra_p is not None and dec_p is not None:
                ra  = float(ra_p.attrib['value'])
                dec = float(dec_p.attrib['value'])
                ra_dec_values.append([name, ra, dec])
        elif stype == 'DiffuseSource':
            sm = src.find('.//spatialModel')
            if sm is None:
                continue
            if sm.attrib.get('type') == 'SpatialMap':
                fp = sm.attrib.get('file')
                if fp and os.path.exists(fp):
                    try:
                        ra  = fits.open(fp)[0].header['CRVAL1']
                        dec = fits.open(fp)[0].header['CRVAL2']
                        spatial_ra_dec_values.append([name, ra, dec])
                    except (KeyError, OSError):
                        pass
            else:
                ra_p  = src.find(".//spatialModel/parameter[@name='RA']")
                dec_p = src.find(".//spatialModel/parameter[@name='DEC']")
                if ra_p is not None and dec_p is not None:
                    ra  = float(ra_p.attrib['value'])
                    dec = float(dec_p.attrib['value'])
                    spatial_ra_dec_values.append([name, ra, dec])

    # FL16Y catalog Signif_Avg > 49 -> sig list
    not_sig = ra_dec_values + spatial_ra_dec_values
    sig = []
    cat = fits.open(CATALOG_FILE)[1].data
    src_names_to_signif = {row['Source_Name'].strip(): float(row['Signif_Avg'])
                           for row in cat}
    for entry in list(not_sig):
        s = src_names_to_signif.get(entry[0].strip())
        if s is not None and s > 49:
            sig.append(entry)
            not_sig.remove(entry)

    print(f'  classified: sig={len(sig)} (Signif>49), not_sig={len(not_sig)}',
          flush=True)

    # np.savez auto-appends '.npz' if the path doesn't already end in it,
    # so pre-include the extension in the tmp name to avoid a rename mismatch.
    tmp = SOURCE_CLASSIFICATION + '.tmp.npz'
    np.savez(
        tmp,
        sig_ra_dec_values=np.array(sig, dtype=object),
        not_sig_ra_dec_values=np.array(not_sig, dtype=object),
        n_sig=len(sig), n_not_sig=len(not_sig),
    )
    os.rename(tmp, SOURCE_CLASSIFICATION)


def verify_source_classification():
    if not os.path.exists(SOURCE_CLASSIFICATION):
        return False, f'missing: {SOURCE_CLASSIFICATION}'
    try:
        d = np.load(SOURCE_CLASSIFICATION, allow_pickle=True)
        n_sig = int(d['n_sig'])
        n_not_sig = int(d['n_not_sig'])
    except Exception as e:
        return False, f'load failed: {e}'
    if n_sig == 0 and n_not_sig == 0:
        return False, 'empty classification'
    return True, f'sig={n_sig}, not_sig={n_not_sig}'


# ----- Step 13: Main psc mask -----
def build_psc_mask():
    d = np.load(SOURCE_CLASSIFICATION, allow_pickle=True)
    sig = list(d['sig_ra_dec_values'])
    not_sig = list(d['not_sig_ra_dec_values'])

    # Energy per bin (geometric mean of edges) in GeV
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

    tmp = PSC_MASK_NPY + '.tmp.npy'
    np.save(tmp, full_mask)
    os.rename(tmp, PSC_MASK_NPY)


# ----- Step 14: Disk mask -----
def build_disk_mask():
    from astropy.wcs import WCS
    src = fits.open(GC_CCUBE)
    w = WCS(src[0].header).dropaxis(2)
    nx = np.shape(src[0].data[0])[1]
    ny = np.shape(src[0].data[0])[0]
    b_list = np.zeros(ny)
    for i in range(ny):
        if np.abs(w.wcs_pix2world(0, i, 0)[1]) <= 2:
            b_list[i] = i
    b_max = int(np.max(b_list))
    b_min = int(np.min(b_list[b_list != 0])) if (b_list != 0).any() else 0
    disk = np.ones(np.shape(src[0].data[0]), dtype=np.float32)
    disk[b_min:b_max + 1, :] = 0.0

    tmp = DISK_MASK_NPY + '.tmp.npy'
    np.save(tmp, disk)
    os.rename(tmp, DISK_MASK_NPY)


# ----- Step 15: empty_model.xml -----
def build_empty_model_xml():
    tree = ET.parse(GC_MODEL_XML)
    root = tree.getroot()
    for src in list(root.findall('.//source')):
        if 'quark' not in src.get('name', ''):
            root.remove(src)
    tmp = EMPTY_MODEL_XML + '.tmp'
    tree.write(tmp, encoding='utf-8', xml_declaration=True)
    os.rename(tmp, EMPTY_MODEL_XML)


# ----- Step 16: Template XMLs (GCE / iso / fermi_bubble) -----
# Verbatim from run_one_model.py Step 5 (L282-326), moved here to make the
# prerequisite chain explicit and remove the implicit "first worker builds
# shared XMLs" race. Workers' skip-if-exists logic remains unchanged.
_TEMPLATE_XMLS = {
    GCE_MODEL_XML: f"""
  <source name="GCE" type="DiffuseSource">
    <spectrum type="BrokenPowerLaw">
    <parameter free="0" max="1000.0" min="0.001" name="Prefactor" scale="1e-11" value="7*3"/>
    <parameter free="0" max="-1.0" min="-5." name="Index1" scale="1.0" value="-1.42"/>
    <parameter free="0" max="3000.0" min="30.0" name="BreakValue" scale="1.0" value="2006"/>
    <parameter free="0" max="-1.0" min="-5." name="Index2" scale="1.0" value="-2.63"/>
    </spectrum>
    <spatialModel file="{WIMP_MAP_PATH}" type="SpatialMap" map_based_integral="true">
    </spatialModel>
  </source>
""",
    ISO_MODEL_XML: f"""
  <source name="isotropic" type="DiffuseSource">
    <spectrum apply_edisp="true" file="{ISO_SPECTRUM_FILE}" type="FileFunction">
      <parameter free="1" max="10" min="1" name="Normalization" scale="1" value="1" />
    </spectrum>
    <spatialModel type="ConstantValue">
      <parameter free="0" max="10" min="0" name="Value" scale="1" value="1" />
    </spatialModel>
  </source>
""",
    BUBBLE_MODEL_XML: f"""
  <source name="Fermi_bubble" type="DiffuseSource">
    <spectrum apply_edisp="true" file="{BUBBLE_SPECTRUM_FILE}" type="FileFunction">
      <parameter free="1" max="1e+10" min="0" name="Normalization" scale="1" value="1" />
    </spectrum>
    <spatialModel file="{BUBBLE_TEMPLATE}" type="SpatialMap" map_based_integral="true">
    </spatialModel>
  </source>
""",
}


def build_template_xmls():
    """Build GCE / isotropic / fermi_bubble XMLs from empty_model.xml."""
    for out_path, src_xml in _TEMPLATE_XMLS.items():
        new_sources_root = ET.fromstring(f'<sources>{src_xml}</sources>')
        tree = ET.parse(EMPTY_MODEL_XML)
        root = tree.getroot()
        for new_src in new_sources_root:
            root.append(new_src)
        tmp = out_path + '.tmp'
        tree.write(tmp, encoding='utf-8', xml_declaration=True)
        os.rename(tmp, out_path)
        print(f'  wrote {out_path}', flush=True)


def verify_template_xmls():
    for out_path in _TEMPLATE_XMLS:
        ok, msg = verify_xml(out_path, min_sources=1)
        if not ok:
            return False, f'{out_path}: {msg}'
    return True, f'3 template XMLs OK'


# ----- Step 17 (FB17 only): external-constraint tables staged in Model/ -----
# run_one_model.py Step 7 loads {WORK_DIR}/Model/{bubble_constraints.txt,
# iso_constraints_full_err.txt} — RAW source-paper tables (bubble:
# arXiv:1407.7905 Tab.2, to ~487 GeV; iso: arXiv:1410.3696 Tab.3, to
# ~690 GeV) — and evaluates them at the bin centers via quadratic
# interp1d. For FB17 the tables must therefore COVER the top-bin center
# (~375 GeV); quadratic EXTRApolation of a truncated/evaluated table is
# not acceptable. Fiducial mode does not register this step (its Model/
# files are the originals and must never be force-deleted).
FIDUCIAL_MODEL_DIR = './GC_analysis_FL16Y/Model'
CONSTRAINT_FILES = ['bubble_constraints.txt', 'iso_constraints_full_err.txt']


def _top_bin_center_gev():
    return float(np.sqrt(CHOLIS_BIN_EDGES_GEV[-2] * CHOLIS_BIN_EDGES_GEV[-1]))


def build_constraint_tables():
    for name in CONSTRAINT_FILES:
        dst = f'{MODEL_DIR}/{name}'
        if os.path.exists(dst):
            continue
        src_p = f'{FIDUCIAL_MODEL_DIR}/{name}'
        if not os.path.exists(src_p):
            raise FileNotFoundError(
                f'{dst} absent and fiducial source {src_p} absent too')
        tmp = dst + '.tmp'
        shutil.copy2(src_p, tmp)
        os.rename(tmp, dst)
        print(f'  [copy] {src_p} -> {dst}', flush=True)


def verify_constraint_tables():
    need = _top_bin_center_gev()
    for name in CONSTRAINT_FILES:
        p = f'{MODEL_DIR}/{name}'
        if not os.path.exists(p):
            return False, f'missing: {p}'
        try:
            arr = np.loadtxt(p)
        except Exception as e:
            return False, f'{name}: loadtxt failed: {e}'
        if arr.ndim != 2 or arr.shape[1] != 4:
            return False, f'{name}: shape={arr.shape}, expected (N, 4)'
        if not np.all(np.isfinite(arr)):
            return False, f'{name}: non-finite entries'
        emax = float(np.max(arr[:, 0]))
        if emax < need:
            return False, (f'{name}: max E={emax:.1f} GeV < top-bin center '
                           f'{need:.1f} GeV — RAW source-paper table required '
                           f'(a 14-row evaluated file cannot be used for FB17)')
    return True, f'2 tables OK, coverage >= {need:.1f} GeV (top-bin center)'


# ============================================================
# Step registry
# ============================================================

def build_steps():
    ny, nx = CCUBE_NYPIX, CCUBE_NXPIX
    steps = [
        Step(1,  'SC merge (weeklies -> single FT2)',
             [SC_MERGED_FILE],
             lambda: verify_sc_merged(SC_MERGED_FILE),
             build_sc_merge),
        Step(2,  'Photon listfile',
             [PHOTON_LIST_FILE],
             verify_photon_listfile,
             build_photon_listfile),
        Step(3,  f'bin_definitions.fits ({NEBINS} bins)',
             [BIN_DEF_FILE],
             lambda: verify_bin_def(BIN_DEF_FILE),
             build_bin_definitions),
        Step(4,  'gtselect -> Allsky_select',
             [ALLSKY_SELECT],
             lambda: verify_event_file(ALLSKY_SELECT, min_events=10_000_000),
             build_gtselect),
        Step(5,  'gtmktime -> Allsky_gti',
             [ALLSKY_GTI],
             lambda: verify_event_file(ALLSKY_GTI, min_events=1_000_000),
             build_gtmktime),
        Step(6,  'gtbin -> GC_ccube (main GC ROI)',
             [GC_CCUBE],
             lambda: verify_cube(GC_CCUBE, expected_xy=(nx, ny)),
             build_gtbin_main),
        Step(7,  'gtltcube -> Allsky_ltcube',
             [ALLSKY_LTCUBE],
             lambda: verify_ltcube(ALLSKY_LTCUBE),
             build_gtltcube),
        Step(8,  'gtexpcube2 (center) -> GC_expcube_center',
             [GC_EXPCUBE_CTR],
             lambda: verify_cube(GC_EXPCUBE_CTR, expected_xy=(nx, ny)),
             build_gtexpcube_center),
        Step(9,  f'gtexpcube2 (edge) -> Allsky_expcube_edge (nebins+1={NEBINS + 1} normal)',
             [ALLSKY_EXP_EDGE],
             lambda: verify_cube(ALLSKY_EXP_EDGE,
                                 expected_xy=(EDGE_NXPIX, EDGE_NYPIX),
                                 allow_nebins_plus_one=True),
             build_gtexpcube_edge),
        Step(10, 'SourceList XML (main GC)',
             [GC_MODEL_XML],
             lambda: verify_xml(GC_MODEL_XML, min_sources=100),
             build_source_list_main),
        Step(11, 'Prune iso/galactic -> GC_psc_model_FL16Y.xml',
             [GC_PSC_MODEL_XML],
             lambda: verify_xml(GC_PSC_MODEL_XML, min_sources=100),
             build_psc_model_xml),
        Step(12, 'Classify sources (sig / not_sig)',
             [SOURCE_CLASSIFICATION],
             verify_source_classification,
             build_source_classification),
        Step(13, f'Main psc mask npy ({NEBINS}x600x600)',
             [PSC_MASK_NPY],
             lambda: verify_mask_npy(PSC_MASK_NPY, (NEBINS, ny, nx)),
             build_psc_mask),
        Step(14, 'Disk mask npy (600x600)',
             [DISK_MASK_NPY],
             lambda: verify_mask_npy(DISK_MASK_NPY, (ny, nx)),
             build_disk_mask),
        Step(15, 'empty_model.xml',
             [EMPTY_MODEL_XML],
             lambda: verify_xml(EMPTY_MODEL_XML, min_sources=0),
             build_empty_model_xml),
        Step(16, 'Template XMLs (GCE / isotropic / fermi_bubble)',
             [GCE_MODEL_XML, ISO_MODEL_XML, BUBBLE_MODEL_XML],
             verify_template_xmls,
             build_template_xmls),
    ]
    if FB17:
        steps.append(Step(
            17, 'External-constraint tables in Model/ (FB17)',
            [f'{MODEL_DIR}/{n}' for n in CONSTRAINT_FILES],
            verify_constraint_tables,
            build_constraint_tables))
    return steps


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force-step', type=str, default='',
                    help='comma-separated step numbers to force rebuild '
                         '(e.g. "7" or "7,8,9")')
    ap.add_argument('--force-all', action='store_true',
                    help='rebuild every step (USE WITH CARE — gtltcube is slow)')
    args = ap.parse_args()

    force_set = parse_force_set(args.force_step)
    force_all = args.force_all

    # cwd check: the fiducial workdir anchors "we are in
    # GCE_17yr_reproduce/" regardless of mode; the FB17 workdir is created.
    if not os.path.isdir('./GC_analysis_FL16Y'):
        _abort(f'./GC_analysis_FL16Y not found; run from the working '
               f'directory (~/GCE-Chi-square-fitting/GCE_17yr_reproduce/).')
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f'[{_ts()}] prepare_common.py start  pid={os.getpid()}')
    print(f'  cwd        : {os.getcwd()}')
    print(f'  force_step : {sorted(force_set) if force_set else "—"}')
    print(f'  force_all  : {force_all}')
    print(f'  FB17       : {FB17}  (WORK_DIR={WORK_DIR}, front={front!r}, '
          f'evtype={evtype_number}, NEBINS={NEBINS})')

    t0 = time.time()
    steps = build_steps()
    for s in steps:
        s.run(force_set, force_all)
    dt = time.time() - t0

    print(f'\n[{_ts()}] prepare_common.py done  elapsed={dt/60:.1f} min')
    print(f'  {len(steps)}/{len(steps)} steps completed and integrity-verified.')


if __name__ == '__main__':
    main()
