#!/usr/bin/env python3
"""
run_one_model.py — 16yr GCE main fit for a single GDE model.

Per-model subprocess wrapper for the main fit pipeline. Reproduces the logic
of GC_analysis-60x60-models_16yr_v13.ipynb cell 30, restricted to one model.

Changes vs cell 30:
  1. CONFIG (cell 3) mirrored at top of file
  2. 80-model loop removed; model taken from sys.argv[1]
  3. Bug A + Bug B + Bug C patches applied (chi2_iso asymmetric error
     direction; isotropic_flux_data[i] -> [self.energy_bin]; Fermi_bubble
     XML Normalization scale="1" value="1" in all three XML blocks)
  4. emcee progress=False (VS Code Remote-SSH SIGKILL mitigation)
  5. multiprocessing Pool(processes=4) — for 16-way parallel launch on
     64-core neutrino (4 * 16 = 64). Adjust if running fewer parallel
     subprocesses.
  6. matplotlib backend 'Agg' — safe in non-X11 subprocess; plt.show() is
     no-op so the cell-30 plot calls survive without modification.
  7. .npz save added at the end (fitted_params, *_std, *_median,
     *_upper, *_lower, max_likelihood, E, delta_E, GCE) so visualization
     notebook cell 5/9 can read chain percentiles.
  8. [v2] pandas + chainconsumer imports removed (Agg backend has no
     corner-plot output; chainconsumer's backend probe at fork suspected
     as BrokenPipeError trigger).
  9. [v2] Likelihood instance cached in module-global `_LH` and built
     in run_mcmc_for_bin BEFORE Pool fork. Workers inherit via
     copy-on-write; no fits.open per walker eval. Eliminates 800
     simultaneous fits.open at fork.
 10. [v2] all per-bin prints use flush=True for live tail -f.
 11. [v3] Integrity-checked skip pattern (mirrors prepare_common.py
     Step.run() policy): every FITS / XML intermediate and the final
     .dat are verifier-checked on skip. Stale FITS/XML -> sys.exit(2)
     (FATAL); stale final .dat -> auto-delete + rerun (since the
     rerun re-derives a fresh .dat from clean intermediates).
     Atomic .tmp + rename writes for all XML outputs. Post-build
     verification on every gtsrcmaps/gtmodel/XML write — a tool that
     returns success but produces a corrupt file is caught immediately
     rather than poisoning downstream cascading. Companion to the
     Phase 2 prep separation (prepare_common.py, prepare_one_roi_cov.py).

Usage:
    python run_one_model.py <ROMAN>
    e.g.  python run_one_model.py X

Skips if final .dat exists.

PREREQUISITE: the notebook cells 0-29 must have been executed already so
that XML/mask/CCUBE/LTCUBE/expcube/empty_model.xml/bin_definitions.fits
are all in place.

Author: haebarg (2026), generated alongside Claude conversation.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.special import gammaln
from scipy.interpolate import interp1d
from multiprocessing import Pool
import xml.etree.ElementTree as ET
import emcee
from GtApp import GtApp

from cholis_masking import (
    verify_fits, verify_cube, verify_xml, verify_dat, verify_srcmap,
)


# ============================================================
# CONFIG — mirror of notebook cell 3
# Edit here AND in cell 3 together to keep notebook and subprocess in sync.
# ============================================================

# Event type / IRF
front          = '_front'              # filename suffix; FRONT-only
evtype_number  = 1                     # 1 = FRONT only
evclass_number = 256                   # 256 = CLEAN
IRFS           = 'P8R3_CLEAN_V3'

# Mask scaling
MASK_SCALE = 1.0                       # Cholis Table III strict

# Catalog
DR_NUMBER    = 4
CATALOG_FILE = '/home/sanghwan/FermiLAT/Sanghwan/GC_analysis/gll_psc_v35.fit'  # vestigial here (unused); kept consistent

# Diffuse + isotropic
GALACTIC_FILE  = '../GCE_17yr_data/gll_iem_v07.fits'
ISOTROPIC_FILE = '../GCE_17yr_data/iso_P8R3_CLEAN_V3_v1.txt'

# Spacecraft
SC_MERGED_FILE = '../GCE_allsky_data/lat_spacecraft_merged_17yr.fits'
SC_FILE        = SC_MERGED_FILE

# Templates / spectra
WIMP_MAP_PATH        = './GCE_template_NFW2.fits'
BUBBLE_TEMPLATE      = './Fermi_Bubbles_template.fits'
ISO_SPECTRUM_FILE    = '/home/sanghwan/FermiLAT/Sanghwan/GC_14yr/data/Model/isotropic_spectrum.txt'
BUBBLE_SPECTRUM_FILE = '/home/sanghwan/FermiLAT/Sanghwan/GC_14yr/data/Model/fermi_bubble_spectrum.txt'

# GDE MapCubes
MAPCUBE_DIR_PION   = './MapCubes'
MAPCUBE_DIR_BREMSS = './MapCubes'
MAPCUBE_DIR_ICS    = './MapCubes'
MAPCUBE_EXT        = '.fits'

# Working directory
WORK_DIR = './GC_analysis_DR4'

# Subprocess-specific
POOL_PROCESSES = 4                     # emcee worker count per subprocess


# ============================================================
# Integrity-check helpers (v3 — mirrors prepare_common.py Step.run policy)
# ============================================================

def _check_or_abort(path, verifier_fn, label):
    """Three-way integrity-checked skip-or-build decision.

    Mirrors the prepare_common.py Step.run() integrity policy: never
    silently reuse a stale file.

      - file exists + verifier OK   -> True  (caller skips build)
      - file exists + verifier FAIL -> sys.exit(2)  (stale-file FATAL)
      - file absent                 -> False (caller proceeds to build)

    Args:
        path:        the file to check
        verifier_fn: zero-arg callable returning (ok: bool, msg: str)
        label:       short identifier for log output (e.g. 'gtsrcmaps (yes)')

    Atomic-write convention for callers building XML / FITS outputs:
    write to '<path>.tmp' then os.rename to <path>, so that a SIGKILL
    mid-write leaves the .tmp partial but the real path untouched.
    """
    if not os.path.exists(path):
        return False
    ok, msg = verifier_fn()
    if ok:
        print(f'[skip] {label}: {msg}', flush=True)
        return True
    print(f'[FATAL] stale {label} at {path}: {msg}', flush=True)
    print(f'        delete this file manually (or run the bulk per-model',
          flush=True)
    print(f'        cleanup before re-launch) and re-run model {model}.',
          flush=True)
    sys.exit(2)


def _verify_built_or_abort(path, verifier_fn, label):
    """Post-build sanity check. If the just-built output fails integrity,
    FATAL immediately rather than letting downstream gtmodel/MCMC cascade
    on a corrupt input. This catches tools that return success but write
    a truncated/malformed file (e.g. SIGKILL during the final flush)."""
    ok, msg = verifier_fn()
    if not ok:
        print(f'[FATAL] {label} produced output that FAILED integrity '
              f'check immediately after build: {msg}', flush=True)
        print(f'        path: {path}', flush=True)
        sys.exit(2)
    print(f'[done ] {label}: {msg}', flush=True)


# ============================================================
# Argument
# ============================================================
if len(sys.argv) != 2:
    print("usage: python run_one_model.py <ROMAN>")
    sys.exit(1)
model = sys.argv[1].strip()

out_dat = f'./GCE_model_{model}{front}_16yr_cholis.dat'
if os.path.exists(out_dat):
    ok, msg = verify_dat(out_dat)
    if ok:
        print(f'[skip] model {model}: final .dat already exists and is OK ({msg})')
        sys.exit(0)
    # Stale .dat from a partial previous run — auto-delete and proceed.
    # The full rerun re-derives the .dat from clean intermediates; if
    # those intermediates are themselves stale, the downstream
    # _check_or_abort calls will FATAL on them.
    print(f'[warn] stale final .dat at {out_dat}: {msg}')
    print(f'       deleting and re-running model {model} from scratch.')
    os.remove(out_dat)
    for _ext in ['_fit.npz', '_likelihood_value']:
        _companion = f'./GCE_model_{model}{front}_16yr_cholis{_ext}'
        if os.path.exists(_companion):
            os.remove(_companion)
            print(f'       removed companion {_companion}')

print(f'[start] model={model}  front={front}  pool={POOL_PROCESSES}')
t_start = time.time()


# ============================================================
# Step 1 — Build first 5-source XML (psc + 5 components)
#         file: GC_model{M}_test.xml
# ============================================================
new_sources = f"""
<source name="bremss" type="DiffuseSource">
    <spectrum type="ConstantValue">
      <parameter error="0.02899312444" free="1" max="100" min="1" name="Value" scale="1" value="1" />
    </spectrum>
    <spatialModel file="{MAPCUBE_DIR_BREMSS}/bremss_mapcube_model{model}{MAPCUBE_EXT}" type="MapCubeFunction" map_based_integral="true">
      <parameter free="0" max="1000" min="0" name="Normalization" scale="1" value="1" />
    </spatialModel>
  </source>
  <source name="ics" type="DiffuseSource">
    <spectrum type="ConstantValue">
      <parameter error="0.04073673429" free="1" max="100" min="1" name="Value" scale="1" value="1" />
    </spectrum>
    <spatialModel file="{MAPCUBE_DIR_ICS}/ics_mapcube_model{model}{MAPCUBE_EXT}" type="MapCubeFunction" map_based_integral="true">
      <parameter free="0" max="1000" min="0" name="Normalization" scale="1" value="1" />
    </spatialModel>
  </source>
  <source name="pion" type="DiffuseSource">
    <spectrum type="ConstantValue">
      <parameter error="0.02899312444" free="1" max="100" min="1" name="Value" scale="1" value="1" />
    </spectrum>
    <spatialModel file="{MAPCUBE_DIR_PION}/pion_mapcube_model{model}{MAPCUBE_EXT}" type="MapCubeFunction" map_based_integral="true">
      <parameter free="0" max="1000" min="0" name="Normalization" scale="1" value="1" />
    </spatialModel>
  </source>
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
  <source name="isotropic" type="DiffuseSource">
    <spectrum apply_edisp="true" file="{ISO_SPECTRUM_FILE}" type="FileFunction">
      <parameter free="1" max="10" min="1" name="Normalization" scale="1" value="1" />
    </spectrum>
    <spatialModel type="ConstantValue">
      <parameter free="0" max="10" min="0" name="Value" scale="1" value="1" />
    </spatialModel>
  </source>
  <source name="Fermi_bubble" type="DiffuseSource">
    <spectrum apply_edisp="true" file="{BUBBLE_SPECTRUM_FILE}" type="FileFunction">
      <parameter free="1" max="1e+10" min="0" name="Normalization" scale="1" value="1" />
    </spectrum>
    <spatialModel file="{BUBBLE_TEMPLATE}" type="SpatialMap" map_based_integral="true">
    </spatialModel>
  </source>
"""
# ^^^ Bug C patched here: Fermi_bubble Normalization scale="1" value="1"

new_sources_root = ET.fromstring(f"<sources>{new_sources}</sources>")
tree = ET.parse(f'{WORK_DIR}/Model/GC_psc_model_DR4.xml')
root = tree.getroot()
for new_source in new_sources_root:
    root.append(new_source)

_xml_path = f'{WORK_DIR}/Model/GC_model{model}_test.xml'
_label = f'GC_model{model}_test.xml (Step 1)'
if not _check_or_abort(_xml_path, lambda: verify_xml(_xml_path, min_sources=100), _label):
    _tmp = _xml_path + '.tmp'
    tree.write(_tmp, encoding='utf-8', xml_declaration=True)
    os.rename(_tmp, _xml_path)
    _verify_built_or_abort(_xml_path, lambda: verify_xml(_xml_path, min_sources=100), _label)


# ============================================================
# Step 2 — Build second 5-source XML (5 components only) — srcmap input
#         file: GC_Extended_model{M}_test.xml
# ============================================================
# Same XML text as Step 1 (note the lower indentation in the original
# notebook — kept identical here for reproducibility)
new_sources_root = ET.fromstring(f"<sources>{new_sources}</sources>")
tree = ET.parse(f'{WORK_DIR}/Model/empty_model.xml')
root = tree.getroot()
for new_source in new_sources_root:
    root.append(new_source)

_xml_path = f'{WORK_DIR}/Model/GC_Extended_model{model}_test.xml'
_label = f'GC_Extended_model{model}_test.xml (Step 2)'
if not _check_or_abort(_xml_path, lambda: verify_xml(_xml_path, min_sources=1), _label):
    _tmp = _xml_path + '.tmp'
    tree.write(_tmp, encoding='utf-8', xml_declaration=True)
    os.rename(_tmp, _xml_path)
    _verify_built_or_abort(_xml_path, lambda: verify_xml(_xml_path, min_sources=1), _label)


# ============================================================
# Step 3 — gtsrcmaps × 2 (convol=yes / convol=no)
# ============================================================
for convol_setting, convol_suffix in [('yes', ''), ('no', '_no_convol')]:
    _srcmap_out = f'{WORK_DIR}/GC_Extended_srcmap_16yr{front}_clean_model_{model}{convol_suffix}.fits'
    _label = f'gtsrcmaps (convol={convol_setting}, model {model})'
    if _check_or_abort(_srcmap_out, lambda: verify_srcmap(_srcmap_out), _label):
        continue
    print(f'[run ] {_label} -> {_srcmap_out}')
    srcMaps = GtApp('gtsrcmaps', 'Likelihood')
    srcMaps['scfile']  = SC_FILE
    srcMaps['expcube'] = f'{WORK_DIR}/Allsky_ltcube_16yr{front}_clean.fits'
    srcMaps['cmap']    = f'{WORK_DIR}/GC_ccube_16yr{front}_clean.fits'
    srcMaps['bexpmap'] = f'{WORK_DIR}/Allsky_expcube_edge_16yr{front}_clean.fits'
    srcMaps['srcmdl']  = f'{WORK_DIR}/Model/GC_Extended_model{model}_test.xml'
    srcMaps['outfile'] = _srcmap_out
    srcMaps['irfs']    = IRFS
    srcMaps['convol']  = convol_setting
    srcMaps['evtype']  = evtype_number
    srcMaps.run()
    _verify_built_or_abort(_srcmap_out, lambda: verify_srcmap(_srcmap_out), _label)


# ============================================================
# Step 4 — Per-component XML (pion, bremss, ics) + gtmodel × 6
# ============================================================
for component in ['bremss', 'ics', 'pion']:
    new_sources = f"""
    <source name="{component}" type="DiffuseSource">
        <spectrum type="ConstantValue">
          <parameter error="0.02899312444" free="1" max="100" min="1" name="Value" scale="1" value="1" />
        </spectrum>
        <spatialModel file="{MAPCUBE_DIR_PION}/{component}_mapcube_model{model}{MAPCUBE_EXT}" type="MapCubeFunction" map_based_integral="true">
          <parameter free="0" max="1000" min="0" name="Normalization" scale="1" value="1" />
        </spatialModel>
      </source>
    """
    new_sources_root = ET.fromstring(f"<sources>{new_sources}</sources>")
    tree = ET.parse(f'{WORK_DIR}/Model/empty_model.xml')
    root = tree.getroot()
    for new_source in new_sources_root:
        root.append(new_source)
    _xml_path = f'{WORK_DIR}/Model/GC_{component}_model{model}_test.xml'
    _label = f'GC_{component}_model{model}_test.xml (Step 4)'
    if not _check_or_abort(_xml_path, lambda p=_xml_path: verify_xml(p, min_sources=1), _label):
        _tmp = _xml_path + '.tmp'
        tree.write(_tmp, encoding='utf-8', xml_declaration=True)
        os.rename(_tmp, _xml_path)
        _verify_built_or_abort(_xml_path, lambda p=_xml_path: verify_xml(p, min_sources=1), _label)


for convol_setting, convol_suffix in [('yes', ''), ('no', '_no_convol')]:
    for component in ['pion', 'bremss', 'ics']:
        _gtm_out = f'{WORK_DIR}/GC_{component}_model{model}_16yr{front}_clean{convol_suffix}.fits'
        _label = f'gtmodel ({component}, convol={convol_setting}, model {model})'
        if _check_or_abort(_gtm_out,
                           lambda p=_gtm_out: verify_cube(p, expected_xy=(600, 600)),
                           _label):
            continue
        print(f'[run ] {_label} -> {_gtm_out}')
        gtmodel = GtApp('gtmodel', 'Likelihood')
        gtmodel['irfs']    = IRFS
        gtmodel['outtype'] = 'ccube'
        gtmodel['srcmdl']  = f'{WORK_DIR}/Model/GC_{component}_model{model}_test.xml'
        gtmodel['outfile'] = _gtm_out
        gtmodel['expcube'] = f'{WORK_DIR}/Allsky_ltcube_16yr{front}_clean.fits'
        gtmodel['bexpmap'] = f'{WORK_DIR}/Allsky_expcube_edge_16yr{front}_clean.fits'
        gtmodel['convol']  = convol_setting
        gtmodel['evtype']  = evtype_number
        gtmodel['srcmaps'] = f'{WORK_DIR}/GC_Extended_srcmap_16yr{front}_clean_model_{model}{convol_suffix}.fits'
        gtmodel.run()
        _verify_built_or_abort(_gtm_out,
                               lambda p=_gtm_out: verify_cube(p, expected_xy=(600, 600)),
                               _label)


# ============================================================
# Step 5 — Per-single-source XML (GCE / fermi_bubble / isotropic)
#         + gtmodel × 6  (these are model-independent template maps;
#         skip if already built by a previous model run)
# ============================================================
_src_specs = {
    'GCE': f"""
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
    'isotropic': f"""
  <source name="isotropic" type="DiffuseSource">
    <spectrum apply_edisp="true" file="{ISO_SPECTRUM_FILE}" type="FileFunction">
      <parameter free="1" max="10" min="1" name="Normalization" scale="1" value="1" />
    </spectrum>
    <spatialModel type="ConstantValue">
      <parameter free="0" max="10" min="0" name="Value" scale="1" value="1" />
    </spatialModel>
  </source>
""",
    'fermi_bubble': f"""
  <source name="Fermi_bubble" type="DiffuseSource">
    <spectrum apply_edisp="true" file="{BUBBLE_SPECTRUM_FILE}" type="FileFunction">
      <parameter free="1" max="1e+10" min="0" name="Normalization" scale="1" value="1" />
    </spectrum>
    <spatialModel file="{BUBBLE_TEMPLATE}" type="SpatialMap" map_based_integral="true">
    </spatialModel>
  </source>
""",
}
# ^^^ Bug C patched here too: fermi_bubble Normalization scale="1" value="1"

for _src_name, _src_xml in _src_specs.items():
    _src_model_xml = f'{WORK_DIR}/Model/GC_{_src_name}_model.xml'
    _label = f'GC_{_src_name}_model.xml (Step 5, model-indep template)'
    if _check_or_abort(_src_model_xml,
                       lambda p=_src_model_xml: verify_xml(p, min_sources=1),
                       _label):
        continue
    _new_sources_root = ET.fromstring(f"<sources>{_src_xml}</sources>")
    _tree = ET.parse(f'{WORK_DIR}/Model/empty_model.xml')
    _root = _tree.getroot()
    for _new_src in _new_sources_root:
        _root.append(_new_src)
    _tmp = _src_model_xml + '.tmp'
    _tree.write(_tmp, encoding='utf-8', xml_declaration=True)
    os.rename(_tmp, _src_model_xml)
    _verify_built_or_abort(_src_model_xml,
                           lambda p=_src_model_xml: verify_xml(p, min_sources=1),
                           _label)

for _convol_setting, _convol_suffix in [('yes', ''), ('no', '_no_convol')]:
    _src_srcmap = f'{WORK_DIR}/GC_Extended_srcmap_16yr{front}_clean_model_{model}{_convol_suffix}.fits'
    for _comp_name in ['GCE', 'fermi_bubble', 'isotropic']:
        _comp_out = f'{WORK_DIR}/GC_{_comp_name}_model_16yr{front}_clean{_convol_suffix}.fits'
        _label = f'gtmodel ({_comp_name}, convol={_convol_setting}, template)'
        if _check_or_abort(_comp_out,
                           lambda p=_comp_out: verify_cube(p, expected_xy=(600, 600)),
                           _label):
            continue
        print(f'[run ] {_label} -> {_comp_out}')
        _gtm = GtApp('gtmodel', 'Likelihood')
        _gtm['irfs']      = IRFS
        _gtm['outtype']   = 'ccube'
        _gtm['srcmdl']    = f'{WORK_DIR}/Model/GC_{_comp_name}_model.xml'
        _gtm['outfile']   = _comp_out
        _gtm['expcube']   = f'{WORK_DIR}/Allsky_ltcube_16yr{front}_clean.fits'
        _gtm['bexpmap']   = f'{WORK_DIR}/Allsky_expcube_edge_16yr{front}_clean.fits'
        _gtm['convol']    = _convol_setting
        _gtm['evtype']    = evtype_number
        _gtm['srcmaps']   = _src_srcmap
        _gtm.run()
        _verify_built_or_abort(_comp_out,
                               lambda p=_comp_out: verify_cube(p, expected_xy=(600, 600)),
                               _label)


# ============================================================
# Phase split — RUN_PHASE='prepare' 종료점.
# fermitools (GtApp)을 실행한 process에 남는 fork-unsafe state(GALPROP/fits mmap)
# 가 후속 MCMC 진입 시 SIGKILL trigger (Job 5, 2026-05-14 확인).
# wrapper(run_one_model_wrapper.py)가 RUN_PHASE를 환경변수로 set하여
# prepare와 mcmc를 별도 subprocess로 분리 — Job 6 패턴 영구 적용.
# RUN_PHASE 미설정 시('all') 기존 동작 유지 (호환성).
# ============================================================
if os.environ.get('RUN_PHASE', 'all') == 'prepare':
    print(f'[prepare done] model={model}  '
          f'phase=prepare  elapsed={(time.time()-t_start)/60:.1f} min',
          flush=True)
    sys.exit(0)


# ============================================================
# Step 6 — Data load (CCUBE, exp, mask, components, observed counts)
# ============================================================

def roi_solid_angle(delta_l_deg, delta_b_deg, b_deg):
    delta_l_rad = np.radians(delta_l_deg)
    delta_b_rad = np.radians(delta_b_deg)
    b_rad       = np.radians(b_deg)
    return delta_l_rad * delta_b_rad * np.cos(b_rad)

raw_map = fits.open(f'{WORK_DIR}/GC_ccube_16yr{front}_clean.fits')
w = WCS(raw_map[0].header).dropaxis(2)
width, height = np.shape(raw_map[0].data[0])

steradian_per_pixel = np.zeros([width, height])
for i in range(0, height, 1):
    for j in range(0, width, 1):
        l, b = w.wcs_pix2world(j, i, 0)
        steradian_per_pixel[i, j] = roi_solid_angle(0.1, 0.1, b)

disk_mask = np.load(f'{WORK_DIR}/Model/GC_disk_mask_60x60_definitions.npy')[100:500, 100:500]
psc_mask  = np.load(f'{WORK_DIR}/Model/GC_mask_60x60_definitions_DR4.npy')[:, 100:500, 100:500]

E_bounds = fits.open(f'{WORK_DIR}/GC_ccube_16yr{front}_clean.fits')[1].data
E = np.zeros(len(E_bounds))
for i in range(0, len(E_bounds), 1):
    E[i] = np.sqrt(E_bounds[i][2] * E_bounds[i][1] * 1e-6) * 1e-3
delta_E = np.zeros(len(E_bounds))
for i in range(0, len(E_bounds), 1):
    delta_E[i] = (E_bounds[i][2] - E_bounds[i][1]) * 1e-6

exp_cube = (fits.open(f'{WORK_DIR}/GC_expcube_center_16yr{front}_clean.fits')[0].data
            [:, 100:500, 100:500] * steradian_per_pixel[100:500, 100:500])

# Mask-averaged template & data flux per bin
def _mask_avg(path):
    out = np.zeros(len(E_bounds))
    d = fits.open(path)[0].data
    for i in range(len(E_bounds)):
        out[i] = np.sum(disk_mask * (d[i][100:500, 100:500] / exp_cube[i])) / np.sum(disk_mask)
    return out

pion   = _mask_avg(f'{WORK_DIR}/GC_pion_model{model}_16yr{front}_clean_no_convol.fits')
bremss = _mask_avg(f'{WORK_DIR}/GC_bremss_model{model}_16yr{front}_clean_no_convol.fits')
ics    = _mask_avg(f'{WORK_DIR}/GC_ics_model{model}_16yr{front}_clean_no_convol.fits')
GCE    = _mask_avg(f'{WORK_DIR}/GC_GCE_model_16yr{front}_clean_no_convol.fits')
bubble = _mask_avg(f'{WORK_DIR}/GC_fermi_bubble_model_16yr{front}_clean_no_convol.fits')
isotropic = _mask_avg(f'{WORK_DIR}/GC_isotropic_model_16yr{front}_clean_no_convol.fits')

counts_per_exp     = np.zeros(len(E_bounds))
counts_per_exp_err = np.zeros(len(E_bounds))
ccube_data = fits.open(f'{WORK_DIR}/GC_ccube_16yr{front}_clean.fits')[0].data
for i in range(len(E_bounds)):
    c = ccube_data[i][100:500, 100:500]
    counts_per_exp[i]     = np.sum(disk_mask * (c / exp_cube[i])) / np.sum(disk_mask)
    counts_per_exp_err[i] = np.sqrt(np.sum(((np.sqrt(disk_mask * c) / exp_cube[i]) ** 2))) / np.sum(disk_mask)


# ============================================================
# Step 7 — External constraints (bubble & isotropic)
# ============================================================

def log_factorial(O):
    return gammaln(np.asarray(O, dtype=float) + 1.0)

bubble_constraints = np.loadtxt(f'{WORK_DIR}/Model/bubble_constraints.txt')
bc_E   = bubble_constraints[:, 0]
bc_flx = bubble_constraints[:, 1]
bc_lo  = bubble_constraints[:, 2]
bc_hi  = bubble_constraints[:, 3]

bubble_fluxint      = interp1d(bc_E, bc_flx, fill_value='extrapolate', kind='quadratic')
bubble_lower_errint = interp1d(bc_E, bc_lo,  fill_value='extrapolate', kind='quadratic')
bubble_upper_errint = interp1d(bc_E, bc_hi,  fill_value='extrapolate', kind='quadratic')

bubble_flux_data        = bubble_fluxint(E)
bubble_lower_error_data = bubble_lower_errint(E)
bubble_upper_error_data = bubble_upper_errint(E)

iso_constraints = np.loadtxt(f'{WORK_DIR}/Model/iso_constraints_full_err.txt')
ic_E  = iso_constraints[:, 0]
ic_fl = iso_constraints[:, 1]
ic_lo = iso_constraints[:, 2]
ic_hi = iso_constraints[:, 3]

isotropic_fluxint      = interp1d(ic_E, ic_fl, fill_value='extrapolate', kind='quadratic')
isotropic_lower_errint = interp1d(ic_E, ic_lo, fill_value='extrapolate', kind='quadratic')
isotropic_upper_errint = interp1d(ic_E, ic_hi, fill_value='extrapolate', kind='quadratic')

isotropic_flux_data        = (E ** 2) * isotropic_fluxint(E)
isotropic_lower_error_data = (E ** 2) * isotropic_lower_errint(E)
isotropic_upper_error_data = (E ** 2) * isotropic_upper_errint(E)


# ============================================================
# Step 8 — Likelihood class (with Bug A + Bug B patches applied)
# ============================================================

class Likelihood:
    def __init__(self, model, energy_bin):
        self.model       = model
        self.energy_bin  = energy_bin
        self.data        = fits.open(f'{WORK_DIR}/GC_ccube_16yr{front}_clean.fits')[0].data[energy_bin, 100:500, 100:500]
        self.pion_bremss = (fits.open(f'{WORK_DIR}/GC_pion_model{model}_16yr{front}_clean.fits')[0].data[energy_bin, 100:500, 100:500]
                          + fits.open(f'{WORK_DIR}/GC_bremss_model{model}_16yr{front}_clean.fits')[0].data[energy_bin, 100:500, 100:500])
        self.ics    = fits.open(f'{WORK_DIR}/GC_ics_model{model}_16yr{front}_clean.fits')[0].data[energy_bin, 100:500, 100:500]
        self.GCE    = fits.open(f'{WORK_DIR}/GC_GCE_model_16yr{front}_clean.fits')[0].data[energy_bin, 100:500, 100:500]
        self.bubble = fits.open(f'{WORK_DIR}/GC_fermi_bubble_model_16yr{front}_clean.fits')[0].data[energy_bin, 100:500, 100:500]
        self.iso    = fits.open(f'{WORK_DIR}/GC_isotropic_model_16yr{front}_clean.fits')[0].data[energy_bin, 100:500, 100:500]
        self.iso_no_convol    = fits.open(f'{WORK_DIR}/GC_isotropic_model_16yr{front}_clean_no_convol.fits')[0].data[energy_bin, 100:500, 100:500]
        self.bubble_no_convol = fits.open(f'{WORK_DIR}/GC_fermi_bubble_model_16yr{front}_clean_no_convol.fits')[0].data[energy_bin, 100:500, 100:500]

        self.E       = E
        self.delta_E = delta_E
        self.exp_cube = (fits.open(f'{WORK_DIR}/GC_expcube_center_16yr{front}_clean.fits')[0].data[energy_bin]
                         * steradian_per_pixel)[100:500, 100:500]

        _psc_mask = psc_mask[energy_bin]
        self.disk_mask = disk_mask
        self.full_mask = _psc_mask * disk_mask
        _obs_masked = self.data[self.full_mask == 1].astype(float)
        self.observed_log_factorial_masked = log_factorial(_obs_masked)

    def likelihood_constrained(self, parameter_set):
        pion_bremss_param, ics_param, GCE_param, bubble_param, isotropic_param = parameter_set

        expected_pixel = (pion_bremss_param * self.pion_bremss
                          + ics_param         * self.ics
                          + GCE_param         * self.GCE
                          + isotropic_param   * self.iso
                          + bubble_param      * self.bubble)
        observed_pixel = self.data

        observed_pixel = observed_pixel[self.full_mask == 1]
        expected_pixel = expected_pixel[self.full_mask == 1]

        if (expected_pixel < 0).any():
            return np.inf

        observed_log_expected = observed_pixel * np.log(expected_pixel)
        lhd = 2 * (expected_pixel - observed_log_expected + self.observed_log_factorial_masked)

        # --- isotropic SED ---
        isotropic = (np.sum(self.full_mask * (self.iso_no_convol) / self.exp_cube)
                     * isotropic_param / np.sum(self.full_mask))
        isotropic_sed = (self.E[self.energy_bin] ** 2) * isotropic / (self.delta_E[self.energy_bin])

        # --- bubble SED ---
        bubble = (np.sum(self.full_mask * (self.bubble_no_convol) / self.exp_cube)
                  * bubble_param / np.sum(self.full_mask))
        bubble_sed = (self.E[self.energy_bin] ** 2) * bubble / (self.delta_E[self.energy_bin])

        # --- chi2_bubble (unchanged; original convention is correct) ---
        larger_error = max([bubble_upper_error_data[self.energy_bin],
                            bubble_lower_error_data[self.energy_bin]])
        if bubble_flux_data[self.energy_bin] < bubble_sed:
            chi2_bubble = ((bubble_sed - bubble_flux_data[self.energy_bin])
                           / bubble_upper_error_data[self.energy_bin]) ** 2
        elif bubble_flux_data[self.energy_bin] > bubble_sed:
            chi2_bubble = ((bubble_sed - bubble_flux_data[self.energy_bin])
                           / bubble_lower_error_data[self.energy_bin]) ** 2
        else:
            chi2_bubble = ((bubble_sed - bubble_flux_data[self.energy_bin])
                           / larger_error) ** 2

        # --- chi2_isotropic — Bug A + Bug B patched ---
        # data < sed  (model above data) -> upper error
        # data > sed  (model below data) -> lower error
        # also: isotropic_flux_data[i] -> [self.energy_bin]
        isotropic_larger_error = max([isotropic_lower_error_data[self.energy_bin],
                                      isotropic_upper_error_data[self.energy_bin]])
        if isotropic_flux_data[self.energy_bin] < isotropic_sed:
            chi2_isotropic = ((isotropic_flux_data[self.energy_bin] - isotropic_sed)
                              / isotropic_upper_error_data[self.energy_bin]) ** 2
        elif isotropic_flux_data[self.energy_bin] > isotropic_sed:
            chi2_isotropic = ((isotropic_flux_data[self.energy_bin] - isotropic_sed)
                              / isotropic_lower_error_data[self.energy_bin]) ** 2
        else:
            chi2_isotropic = ((isotropic_flux_data[self.energy_bin] - isotropic_sed)
                              / isotropic_larger_error) ** 2

        return np.sum(lhd) + chi2_bubble + chi2_isotropic


_LH = None  # populated per-bin by run_mcmc_for_bin BEFORE Pool fork


def log_likelihood(params, energy_bin):
    # _LH is set in run_mcmc_for_bin before Pool fork; worker copies it
    # via copy-on-write and avoids re-opening fits files per walker eval.
    return -(1.0 / 2.0) * _LH.likelihood_constrained(params)


def log_prior(params):
    limits = [(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)]
    for i, (lo, hi) in enumerate(limits):
        if not (lo <= params[i] <= hi):
            return -np.inf
    return 0.0


def log_probability(params, energy_bin):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, energy_bin)


# ============================================================
# Step 9 — MCMC per bin
# ============================================================

def run_mcmc_for_bin(energy_bin):
    global _LH
    _LH = Likelihood(model, energy_bin)  # build BEFORE Pool fork; workers
                                          # inherit via copy-on-write
    ndim, nwalkers, nsteps, burn_in_steps = 5, 100, 1000, 400
    t0 = time.time()
    print(f'[bin {energy_bin}] start', flush=True)
    initial_params = np.vstack([
        np.random.uniform(0, 3,  [nwalkers]),
        np.random.uniform(0, 3,  [nwalkers]),
        np.random.uniform(0, 3,  [nwalkers]),
        np.random.uniform(0, 10, [nwalkers]),
        np.random.uniform(0, 10, [nwalkers]),
    ]).T
    # [v3.2] emcee Pool removed. 12yr lesson #10 (REF_12yr_final_code_for_16yr_SUMMARY.md):
    # "Pool 시도하면 Fermi tools fork 이슈 가능". Confirmed in 16yr by Job 3.neutrino
    # (2026-05-14): single-worker PBS without launcher; gtsrcmaps + gtmodel
    # completed normally, then SIGKILL'd immediately after `[bin 0] start`
    # (the Pool fork point) with Pool workers showing BrokenPipe on IPC.
    # fermitools leaves fork-unsafe state (GALPROP/fits mmap) in the main
    # process; emcee Pool fork on top of that triggers parent kill.
    # Serial emcee removes the fork conflict; wall time ~70-80 min/model
    # (12yr baseline, gammaln-fast log_probability).
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(energy_bin,))
    pos, prob, state = sampler.run_mcmc(initial_params, nsteps, progress=False)

    # [DIAG] save full chain when DIAG_SAVE_CHAIN env is set (C1+C2 diagnostics)
    if os.environ.get('DIAG_SAVE_CHAIN'):
        _flat_chain = sampler.get_chain(discard=burn_in_steps, flat=True)
        _acc_frac   = sampler.acceptance_fraction.copy()
        try:
            _autocorr = sampler.get_autocorr_time(quiet=True, tol=0)
        except Exception:
            _autocorr = np.full(ndim, -1.0)
        np.savez(f'./DIAG_chain_{model}_bin{energy_bin:02d}.npz',
                 flat_chain=_flat_chain,
                 acceptance_fraction=_acc_frac,
                 autocorr_time=_autocorr,
                 energy_bin=energy_bin)
        print(f'  [DIAG] saved chain bin {energy_bin}: '
              f'shape={_flat_chain.shape}, acc={_acc_frac.mean():.3f}', flush=True)

    log_prob_samples  = sampler.get_log_prob(discard=burn_in_steps, flat=True)
    max_prob_index    = np.argmax(log_prob_samples)
    max_lhd           = log_prob_samples[max_prob_index]
    best_fit_params   = sampler.get_chain(discard=burn_in_steps, flat=True)[max_prob_index]
    flat_samples      = sampler.get_chain(discard=burn_in_steps, flat=True)

    lower_1sigma = np.percentile(flat_samples, 16, axis=0)
    upper_1sigma = np.percentile(flat_samples, 84, axis=0)
    median_1sigma = np.median(flat_samples, axis=0)
    std_1sigma    = np.std(flat_samples, axis=0, ddof=1)

    dt = time.time() - t0
    print(f'[bin {energy_bin}] done in {dt/60:.1f} min  best={best_fit_params}', flush=True)
    return (best_fit_params, median_1sigma, std_1sigma, max_lhd,
            upper_1sigma, lower_1sigma)


n = len(E)
fitted_params         = np.zeros(n * 5)
fitted_params_median  = np.zeros(n * 5)
fitted_params_std     = np.zeros(n * 5)
fitted_params_upper   = np.zeros(n * 5)
fitted_params_lower   = np.zeros(n * 5)
max_likelihood        = np.zeros(n)

for i in range(n):
    max_v, med_v, std_v, mlh, upp_v, low_v = run_mcmc_for_bin(i)
    for k in range(5):
        fitted_params[n*k + i]        = max_v[k]
        fitted_params_median[n*k + i] = med_v[k]
        fitted_params_std[n*k + i]    = std_v[k]
        fitted_params_upper[n*k + i]  = upp_v[k]
        fitted_params_lower[n*k + i]  = low_v[k]
    max_likelihood[i] = mlh


# ============================================================
# Step 10 — Save .dat / _likelihood_value / .npz
# ============================================================
GCE_arr = GCE  # alias for clarity
np.savetxt(
    out_dat,
    np.vstack([
        E,
        fitted_params[n*2:n*3]       * GCE_arr * (E**2) / delta_E,
        fitted_params_std[n*2:n*3]   * GCE_arr * (E**2) / delta_E,
        fitted_params_lower[n*2:n*3] * GCE_arr * (E**2) / delta_E,
        fitted_params_upper[n*2:n*3] * GCE_arr * (E**2) / delta_E,
    ]).T
)
np.savetxt(f'./GCE_model_{model}{front}_16yr_cholis_likelihood_value', max_likelihood)

# .npz — shape (5, n) for visualization-notebook compatibility
np.savez(
    f'./GCE_model_{model}{front}_16yr_cholis_fit.npz',
    fitted_params        = fitted_params.reshape(5, n),
    fitted_params_median = fitted_params_median.reshape(5, n),
    fitted_params_std    = fitted_params_std.reshape(5, n),
    fitted_params_upper  = fitted_params_upper.reshape(5, n),
    fitted_params_lower  = fitted_params_lower.reshape(5, n),
    max_likelihood       = max_likelihood,
    E                    = E,
    delta_E              = delta_E,
    GCE                  = GCE_arr,
    pion                 = pion,
    bremss               = bremss,
    ics                  = ics,
    bubble               = bubble,
    isotropic            = isotropic,
)

print(f'[done] model={model}  total={(time.time()-t_start)/60:.1f} min  '
      f'sum_logL={np.sum(max_likelihood):.1f}')
