#!/usr/bin/env python3
"""
run_one_roi_cov.py — 17yr GCE covariance pipeline for a single control ROI.

Per-ROI subprocess wrapper for the covariance pipeline. Reproduces the
logic of GCE_covariance_marix_calculation_17yr_v13.ipynb cells 16-27,
restricted to one ROI and one model (model='X', same as Sanghwan 16yr).

Changes vs cov notebook:
  1. CONFIG mirrored at top
  2. ROI loop removed; roi from sys.argv[1]
  3. Bug A + Bug B + Bug C patches applied (chi2_iso asymmetric error
     direction; isotropic_flux_data[i] -> [self.energy_bin]; Fermi_bubble
     XML Normalization scale="1" value="1")
  4. emcee progress=False (VS Code Remote-SSH SIGKILL mitigation)
  5. multiprocessing Pool(processes=4) for 2-way ROI parallel launch
     (2 ROI x 4 worker = 8 active processes; gtsrcmaps memory-bound
     so wider parallel hurts. Adjust POOL_PROCESSES if launching
     differently.)
  6. matplotlib backend 'Agg' (no plot code in this runner; safe regardless)
  7. .npz save added (memory pattern: GCE_cov_l{ROI}_front_17yr_cholis_fit.npz)
  8. Likelihood instance cached in module-global `_LH` and built in
     run_mcmc_for_bin BEFORE Pool fork (same pattern as main runner v2 —
     avoids fits.open per walker evaluation).

Usage:
    python run_one_roi_cov.py <ROI>
    e.g.  python run_one_roi_cov.py 50    (ROI center at l=+50, b=0)
          python run_one_roi_cov.py -25   (ROI center at l=-25, b=0)

Valid ROIs: [-70, -65, ..., -25, -20, 20, 25, ..., 65, 70] (22 total).

PREREQUISITES (run once before any ROI subprocess):
  - cov notebook cells 0-15 executed (per-ROI ccube/expcube/ltcube,
    per-ROI psc_mask, disk_mask, bin_definitions, isotropic & bubble
    constraint files, empty_model.xml, GC_psc_model_FL16Y_l{roi}.xml)
  - `python make_wimp_map_per_roi.py` finished (22 wimp_map_l*.fits)
  - main runner has produced GC_fermi_bubble_model.xml and
    GC_isotropic_model.xml (or these are generated separately)

Output (in working directory; launcher moves to results_cov_17yr/):
    GCE_cov_l{ROI}_front_17yr_cholis.dat              5 columns
    GCE_cov_l{ROI}_front_17yr_cholis_fit.npz          fitted_params/std/median/upper/lower, E, GCE, delta_E
    GCE_cov_l{ROI}_front_17yr_cholis_likelihood_value 14-bin max log-prob

Author: haebarg (2026)

Changes:
  [fb17-cov-v1] (2026-07-28) FB17=1 env -> front+back 17-bin cov variant
      (main pipeline과 동일 패턴): WORK_DIR/FRONT/evtype 전환, 결과는
      results_cov_fb17/ 분리(기존 front 22개 .dat와 카운트 충돌 방지).
      env 미설정 시 기존 fiducial 동작과 동일.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
from astropy.io import fits
from astropy.wcs import WCS
from scipy.special import gammaln
from scipy.interpolate import interp1d
from multiprocessing import Pool
import xml.etree.ElementTree as ET
import emcee
from GtApp import GtApp


# ============================================================
# CONFIG (mirror of cov notebook cell 3)
# ============================================================
front          = '_front'
evtype_number  = 1
IRFS           = 'P8R3_CLEAN_V3'
MAPCUBE_DIR_PION   = './MapCubes_wide'
MAPCUBE_DIR_BREMSS = './MapCubes_wide'
MAPCUBE_DIR_ICS    = './MapCubes_wide'
MAPCUBE_EXT        = '.fits'
ISO_SPECTRUM_FILE    = './isotropic_spectrum_ff.txt'
BUBBLE_SPECTRUM_FILE = './fermi_bubble_spectrum.txt'
BUBBLE_TEMPLATE      = './Fermi_Bubbles_template.fits'
SC_MERGED_FILE = '../GCE_allsky_data/lat_spacecraft_merged_17yr.fits'
SC_FILE        = SC_MERGED_FILE
WORK_DIR       = './GC_analysis_FL16Y'

# FB17 variant (front+back, 17 bins) — env switch (main runner 동일 패턴).
FB17 = bool(os.environ.get('FB17', '').strip())
if FB17:
    WORK_DIR      = './GC_analysis_FL16Y_fb17'
    front         = '_front_back'
    evtype_number = 3
    print(f'[config] FB17=1 -> WORK_DIR={WORK_DIR}, front={front!r}, '
          f'evtype={evtype_number}', flush=True)

MODEL = 'X'    # cov pipeline: single GDE model (Sanghwan 16yr convention)
POOL_PROCESSES = 4

VALID_ROIS = [r for r in range(-70, 75, 5) if r != 0 and abs(r) >= 20]


# ============================================================
# Argument
# ============================================================
if len(sys.argv) != 2:
    print("usage: python run_one_roi_cov.py <ROI>")
    sys.exit(1)
try:
    roi = int(sys.argv[1])
except ValueError:
    print(f"[FATAL] ROI must be integer, got: {sys.argv[1]!r}")
    sys.exit(1)
if roi not in VALID_ROIS:
    print(f"[FATAL] ROI={roi} not in valid list: {VALID_ROIS}")
    sys.exit(1)

out_dat = f'./GCE_cov_l{roi}{front}_17yr_cholis.dat'
if os.path.exists(out_dat):
    print(f'[skip] roi={roi}: final .dat already exists ({out_dat})')
    sys.exit(0)

model = MODEL
print(f'[start] roi={roi}  model={model}  pool={POOL_PROCESSES}', flush=True)
t_start = time.time()


# ============================================================
# Step 1 — Per-ROI 5-source XML (cov cell 16; Bug C patched)
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
        <parameter free="0" max="1000.0" min="0.001" name="Prefactor" scale="1e-04" value="7*3.6276211633314497"/>
        <parameter free="0" max="-1.0" min="-5." name="Index1" scale="1.0" value="-1.42"/>
        <parameter free="0" max="3000.0" min="30.0" name="BreakValue" scale="1.0" value="2006"/>
        <parameter free="0" max="-1.0" min="-5." name="Index2" scale="1.0" value="-2.63"/>
    </spectrum>
        <spatialModel file="{WORK_DIR}/Model/wimp_map_l{roi}.fits" type="SpatialMap" map_based_integral="true">
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

_ext_xml = f'{WORK_DIR}/Model/GC_Extended_model{model}_l{roi}.xml'
if not os.path.exists(_ext_xml):
    new_sources_root = ET.fromstring(f"<sources>{new_sources}</sources>")
    tree = ET.parse(f'{WORK_DIR}/Model/empty_model.xml')
    root = tree.getroot()
    for new_src in new_sources_root:
        root.append(new_src)
    tree.write(_ext_xml, encoding='utf-8', xml_declaration=True)
    print(f'[done] wrote {_ext_xml}', flush=True)


# Single-source GCE XML for this ROI (cov cell 16 second block)
gce_xml_l = f"""
      <source name="GCE" type="DiffuseSource">
        <spectrum type="BrokenPowerLaw">
        <parameter free="0" max="1000.0" min="0.001" name="Prefactor" scale="1e-12" value="3"/>
        <parameter free="0" max="-1.0" min="-5." name="Index1" scale="1.0" value="-1.42"/>
        <parameter free="0" max="3000.0" min="30.0" name="BreakValue" scale="1.0" value="2006"/>
        <parameter free="0" max="-1.0" min="-5." name="Index2" scale="1.0" value="-2.63"/>
    </spectrum>
        <spatialModel file="{WORK_DIR}/Model/wimp_map_l{roi}.fits" type="SpatialMap" map_based_integral="true">
        </spatialModel>
      </source>
    """
_gce_xml_path = f'{WORK_DIR}/Model/GC_GCE_model_l{roi}.xml'
if not os.path.exists(_gce_xml_path):
    new_sources_root = ET.fromstring(f"<sources>{gce_xml_l}</sources>")
    tree = ET.parse(f'{WORK_DIR}/Model/empty_model.xml')
    root = tree.getroot()
    for new_src in new_sources_root:
        root.append(new_src)
    tree.write(_gce_xml_path, encoding='utf-8', xml_declaration=True)
    print(f'[done] wrote {_gce_xml_path}', flush=True)


# ============================================================
# Step 2 — Per-component XML (cov cell 18, ROI-independent)
#         model='X' fixed; idempotent skip-if-exists
# ============================================================
for component in ['bremss', 'ics', 'pion']:
    _comp_xml = f'{WORK_DIR}/Model/GC_{component}_model{model}.xml'
    if os.path.exists(_comp_xml):
        continue
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
    for new_src in new_sources_root:
        root.append(new_src)
    tree.write(_comp_xml, encoding='utf-8', xml_declaration=True)
    print(f'[done] wrote {_comp_xml}', flush=True)

# Sanity check: bubble/isotropic XML must exist (built by main runner)
for _need in ['GC_fermi_bubble_model.xml', 'GC_isotropic_model.xml']:
    _p = f'{WORK_DIR}/Model/{_need}'
    if not os.path.exists(_p):
        print(f'[FATAL] required XML missing: {_p}')
        print(f'        (these are built by run_one_model.py; run main pipeline first or build manually)')
        sys.exit(2)


# ============================================================
# Step 3 — Per-ROI gtsrcmaps × 2 (cov cell 17)
# ============================================================
for convol_setting, convol_suffix in [('yes', ''), ('no', '_no_convol')]:
    _srcmap_out = f'{WORK_DIR}/GC_Extended_srcmap_17yr_model{model}{front}_clean{convol_suffix}_l{roi}.fits'
    if os.path.exists(_srcmap_out):
        print(f'[skip] gtsrcmaps ({convol_setting}, roi={roi}): {_srcmap_out}', flush=True)
        continue
    print(f'[run ] gtsrcmaps (convol={convol_setting}, roi={roi}) -> {_srcmap_out}', flush=True)
    srcMaps = GtApp('gtsrcmaps', 'Likelihood')
    srcMaps['scfile']  = SC_FILE
    srcMaps['expcube'] = f'{WORK_DIR}/Allsky_ltcube_17yr{front}_clean.fits'
    srcMaps['cmap']    = f'{WORK_DIR}/GC_ccube_17yr{front}_clean_l{roi}.fits'
    srcMaps['bexpmap'] = f'{WORK_DIR}/Allsky_expcube_edge_17yr{front}_clean.fits'
    srcMaps['srcmdl']  = _ext_xml
    srcMaps['outfile'] = _srcmap_out
    srcMaps['irfs']    = IRFS
    srcMaps['emapbnds']= 'yes'
    srcMaps['convol']  = convol_setting
    srcMaps['evtype']  = evtype_number
    srcMaps.run()


# ============================================================
# Step 4 — Per-ROI per-component gtmodel × 6 (cov cell 19)
# ============================================================
for convol_setting, convol_suffix in [('yes', ''), ('no', '_no_convol')]:
    for component in ['pion', 'bremss', 'ics']:
        _gtm_out = f'{WORK_DIR}/GC_{component}_model{model}_17yr{front}_clean{convol_suffix}_l{roi}.fits'
        if os.path.exists(_gtm_out):
            print(f'[skip] gtmodel ({component}, {convol_setting}, roi={roi}): {_gtm_out}', flush=True)
            continue
        print(f'[run ] gtmodel ({component}, convol={convol_setting}, roi={roi}) -> {_gtm_out}', flush=True)
        gtmodel = GtApp('gtmodel', 'Likelihood')
        gtmodel['irfs']    = IRFS
        gtmodel['outtype'] = 'ccube'
        gtmodel['srcmdl']  = f'{WORK_DIR}/Model/GC_{component}_model{model}.xml'
        gtmodel['outfile'] = _gtm_out
        gtmodel['expcube'] = f'{WORK_DIR}/Allsky_ltcube_17yr{front}_clean.fits'
        gtmodel['bexpmap'] = f'{WORK_DIR}/Allsky_expcube_edge_17yr{front}_clean.fits'
        gtmodel['convol']  = convol_setting
        gtmodel['evtype']  = evtype_number
        gtmodel['srcmaps'] = f'{WORK_DIR}/GC_Extended_srcmap_17yr_model{model}{front}_clean{convol_suffix}_l{roi}.fits'
        gtmodel.run()


# ============================================================
# Step 5 — Per-ROI GCE/bubble/isotropic gtmodel × 6 (cov cell 20)
# ============================================================
for convol_setting, convol_suffix in [('yes', ''), ('no', '_no_convol')]:
    _src_srcmap = f'{WORK_DIR}/GC_Extended_srcmap_17yr_model{model}{front}_clean{convol_suffix}_l{roi}.fits'
    for comp in ['GCE', 'fermi_bubble', 'isotropic']:
        _comp_out = f'{WORK_DIR}/GC_{comp}_model_17yr{front}_clean{convol_suffix}_l{roi}.fits'
        if os.path.exists(_comp_out):
            print(f'[skip] gtmodel ({comp}, {convol_setting}, roi={roi}): {_comp_out}', flush=True)
            continue
        print(f'[run ] gtmodel ({comp}, convol={convol_setting}, roi={roi}) -> {_comp_out}', flush=True)
        gtm = GtApp('gtmodel', 'Likelihood')
        gtm['irfs']     = IRFS
        gtm['outtype']  = 'ccube'
        if comp == 'GCE':
            gtm['srcmdl'] = f'{WORK_DIR}/Model/GC_{comp}_model_l{roi}.xml'
        else:
            gtm['srcmdl'] = f'{WORK_DIR}/Model/GC_{comp}_model.xml'
        gtm['outfile'] = _comp_out
        gtm['expcube'] = f'{WORK_DIR}/Allsky_ltcube_17yr{front}_clean.fits'
        gtm['bexpmap'] = f'{WORK_DIR}/Allsky_expcube_edge_17yr{front}_clean.fits'
        gtm['convol']  = convol_setting
        gtm['evtype']  = evtype_number
        gtm['srcmaps'] = _src_srcmap
        gtm.run()


# ============================================================
# Phase split (SIGKILL fix, REF_cov_pipeline_17yr_FINAL.md §1):
# prepare exits HERE so fermitools (gtsrcmaps/gtmodel) state dies with
# this subprocess; the wrapper then runs mcmc in a fresh process.
# In the mcmc phase Steps 1-5 above all hit their [skip] paths (outputs
# already exist) so no GtApp is invoked -> no fork-unsafe state.
# ============================================================
if os.environ.get('RUN_PHASE', '') == 'prepare':
    print(f'[prepare done] roi={roi}  Steps 1-5 complete; '
          f'mcmc runs in a fresh process', flush=True)
    sys.exit(0)


# ============================================================
# Step 6 — Data load (cov cell 22 globals + cell 27 per-ROI products)
# ============================================================

def roi_solid_angle(delta_l_deg, delta_b_deg, b_deg):
    delta_l_rad = np.radians(delta_l_deg)
    delta_b_rad = np.radians(delta_b_deg)
    b_rad       = np.radians(b_deg)
    return delta_l_rad * delta_b_rad * np.cos(b_rad)


# steradian_per_pixel is built from the GLOBAL ccube WCS (cov notebook
# convention; cov cell 22 references this as a module-global). cov cell
# 22's exp_cube uses [self.energy_bin] without [100:500, 100:500] slicing
# because steradian_per_pixel is built over the full 600x600 grid.
_raw = fits.open(f'{WORK_DIR}/GC_ccube_17yr{front}_clean.fits')
_w   = WCS(_raw[0].header).dropaxis(2)
_width, _height = np.shape(_raw[0].data[0])
steradian_per_pixel = np.zeros([_width, _height])
for _i in range(_height):
    for _j in range(_width):
        _l, _b = _w.wcs_pix2world(_j, _i, 0)
        steradian_per_pixel[_i, _j] = roi_solid_angle(0.1, 0.1, _b)

E_bounds = fits.open(f'{WORK_DIR}/GC_ccube_17yr{front}_clean.fits')[1].data
E = np.zeros(len(E_bounds))
for _i in range(len(E_bounds)):
    E[_i] = np.sqrt(E_bounds[_i][2] * E_bounds[_i][1] * 1e-6) * 1e-3
delta_E = np.zeros(len(E_bounds))
for _i in range(len(E_bounds)):
    delta_E[_i] = (E_bounds[_i][2] - E_bounds[_i][1]) * 1e-6

# Bin-count contract (main runner 동일): FB17 env가 이 프로세스와
# 준비된 workdir에서 일관되게 설정되었는지 검사.
_EXPECT_NEBINS = int(os.environ.get('GCE_NEBINS', '17' if FB17 else '14'))
assert len(E) == _EXPECT_NEBINS, (
    f'CCUBE has {len(E)} bins but expected {_EXPECT_NEBINS}; '
    f'export FB17=1 (or GCE_NEBINS) consistently.')

# Per-ROI exp_cube (used for component-flux extraction in Step 7)
exp_cube_per_roi = (fits.open(f'{WORK_DIR}/GC_expcube_center_17yr{front}_clean_l{roi}.fits')[0].data
                    [:, 100:500, 100:500] * steradian_per_pixel[100:500, 100:500])

disk_mask = np.load(f'{WORK_DIR}/Model/GC_disk_mask_60x60_definitions.npy')[100:500, 100:500]


def _mask_avg_per_roi(path):
    """cov cell 27: per-ROI mask-averaged flux per bin."""
    out = np.zeros(len(E_bounds))
    d = fits.open(path)[0].data
    for _i in range(len(E_bounds)):
        out[_i] = np.sum(disk_mask * (d[_i][100:500, 100:500] / exp_cube_per_roi[_i])) / np.sum(disk_mask)
    return out

pion   = _mask_avg_per_roi(f'{WORK_DIR}/GC_pion_model{model}_17yr{front}_clean_no_convol_l{roi}.fits')
bremss = _mask_avg_per_roi(f'{WORK_DIR}/GC_bremss_model{model}_17yr{front}_clean_no_convol_l{roi}.fits')
ics    = _mask_avg_per_roi(f'{WORK_DIR}/GC_ics_model{model}_17yr{front}_clean_no_convol_l{roi}.fits')
GCE    = _mask_avg_per_roi(f'{WORK_DIR}/GC_GCE_model_17yr{front}_clean_no_convol_l{roi}.fits')
bubble = _mask_avg_per_roi(f'{WORK_DIR}/GC_fermi_bubble_model_17yr{front}_clean_no_convol_l{roi}.fits')
isotropic = _mask_avg_per_roi(f'{WORK_DIR}/GC_isotropic_model_17yr{front}_clean_no_convol_l{roi}.fits')

# Observed counts (per-ROI, mask-averaged) — for diagnostics, written to .npz
counts_per_exp     = np.zeros(len(E_bounds))
counts_per_exp_err = np.zeros(len(E_bounds))
_ccube_data = fits.open(f'{WORK_DIR}/GC_ccube_17yr{front}_clean_l{roi}.fits')[0].data
for _i in range(len(E_bounds)):
    _c = _ccube_data[_i][100:500, 100:500]
    counts_per_exp[_i]     = np.sum(disk_mask * (_c / exp_cube_per_roi[_i])) / np.sum(disk_mask)
    counts_per_exp_err[_i] = np.sqrt(np.sum(((np.sqrt(disk_mask * _c) / exp_cube_per_roi[_i]) ** 2))) / np.sum(disk_mask)


# ============================================================
# Step 7 — External constraints (cov cell 22 globals)
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
# Step 8 — Likelihood class (cov cell 22; Bug A+B patches applied)
#         Note: /np.sum(600*600) normalization is intentional cov design
#         (mask-independent, fixed denominator); main runner differs.
# ============================================================

class Likelihood:
    def __init__(self, model, energy_bin, roi):
        self.roi        = roi
        self.model      = model
        self.energy_bin = energy_bin
        self.data        = fits.open(f'{WORK_DIR}/GC_ccube_17yr{front}_clean_l{self.roi}.fits')[0].data[energy_bin, 100:500, 100:500]
        self.pion_bremss = (fits.open(f'{WORK_DIR}/GC_pion_model{model}_17yr{front}_clean_l{self.roi}.fits')[0].data[energy_bin, 100:500, 100:500]
                          + fits.open(f'{WORK_DIR}/GC_bremss_model{model}_17yr{front}_clean_l{self.roi}.fits')[0].data[energy_bin, 100:500, 100:500])
        self.ics    = fits.open(f'{WORK_DIR}/GC_ics_model{model}_17yr{front}_clean_l{self.roi}.fits')[0].data[energy_bin, 100:500, 100:500]
        self.GCE    = fits.open(f'{WORK_DIR}/GC_GCE_model_17yr{front}_clean_l{self.roi}.fits')[0].data[energy_bin, 100:500, 100:500]
        self.bubble = fits.open(f'{WORK_DIR}/GC_fermi_bubble_model_17yr{front}_clean_l{self.roi}.fits')[0].data[energy_bin, 100:500, 100:500]
        self.iso    = fits.open(f'{WORK_DIR}/GC_isotropic_model_17yr{front}_clean_l{self.roi}.fits')[0].data[energy_bin, 100:500, 100:500]

        # for SED computation (no-convol templates)
        self.iso_no_convol    = fits.open(f'{WORK_DIR}/GC_isotropic_model_17yr{front}_clean_no_convol_l{self.roi}.fits')[0].data[energy_bin]
        self.bubble_no_convol = fits.open(f'{WORK_DIR}/GC_fermi_bubble_model_17yr{front}_clean_no_convol_l{self.roi}.fits')[0].data[energy_bin]
        # NOTE: cov uses full 600x600 exp_cube here, NOT sliced. matches cell 22.
        self.exp_cube_full = (fits.open(f'{WORK_DIR}/GC_expcube_center_17yr{front}_clean_l{self.roi}.fits')[0].data[energy_bin]
                              * steradian_per_pixel)

        self.E       = E
        self.delta_E = delta_E

        _psc_mask = np.load(f'{WORK_DIR}/Model/GC_mask_60x60_definitions_FL16Y_l{self.roi}.npy')[energy_bin, 100:500, 100:500]
        _disk     = disk_mask
        self.full_mask = _psc_mask * _disk
        _obs_masked = self.data[self.full_mask == 1].astype(float)
        self.observed_log_factorial_masked = log_factorial(_obs_masked)

    def likelihood_constrained(self, parameter_set):
        pion_bremss_param, ics_param, GCE_param, bubble_param, isotropic_param = parameter_set

        expected_pixel = (pion_bremss_param * self.pion_bremss
                          + ics_param        * self.ics
                          + GCE_param        * self.GCE
                          + isotropic_param  * self.iso
                          + bubble_param     * self.bubble)
        observed_pixel = self.data

        observed_pixel = observed_pixel[self.full_mask == 1]
        expected_pixel = expected_pixel[self.full_mask == 1]

        if (expected_pixel < 0).any():
            return np.inf

        observed_log_expected = observed_pixel * np.log(expected_pixel)
        # cov cell 22 intentionally drops the constant +log(O!) term — does
        # NOT affect fit (only changes likelihood absolute value)
        lhd = 2 * (expected_pixel - observed_log_expected)

        # --- isotropic SED with /np.sum(600*600) normalization ---
        isotropic_v = (np.sum(self.iso_no_convol / self.exp_cube_full)
                       * isotropic_param / np.sum(600 * 600))
        isotropic_sed = (self.E[self.energy_bin] ** 2) * isotropic_v / (self.delta_E[self.energy_bin])

        # --- bubble SED with /np.sum(600*600) normalization ---
        bubble_v = (np.sum(self.bubble_no_convol / self.exp_cube_full)
                    * bubble_param / np.sum(600 * 600))
        bubble_sed = (self.E[self.energy_bin] ** 2) * bubble_v / (self.delta_E[self.energy_bin])

        # --- chi2_bubble (unchanged convention) ---
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
        # data < sed (model > data) -> upper error
        # data > sed (model < data) -> lower error
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
    return -(1.0 / 2.0) * _LH.likelihood_constrained(params)


def log_prior(params):
    # cov cell 23 limits: c_gce ∈ (-inf, +inf) — control region allows negative
    limits = [(0, 10), (0, 10), (-np.inf, np.inf), (0, np.inf), (0, np.inf)]
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
    _LH = Likelihood(model, energy_bin, roi)

    ndim, nwalkers, nsteps, burn_in_steps = 5, 100, 1000, 400
    t0 = time.time()
    print(f'[bin {energy_bin}] start  roi={roi}', flush=True)
    # serial emcee — Pool removed (fermitools fork-unsafe -> SIGKILL; REF §1)
    # cov cell 23 initial parameter ranges
    initial_params = np.vstack([
        np.random.uniform(0, 1,  [nwalkers]),
        np.random.uniform(0, 1,  [nwalkers]),
        np.random.uniform(-5, 5, [nwalkers]),
        np.random.uniform(-5, 5, [nwalkers]),
        np.random.uniform(0, 5,  [nwalkers]),
    ]).T
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(energy_bin,))
    pos, prob, state = sampler.run_mcmc(initial_params, nsteps, progress=False)

    log_prob_samples = sampler.get_log_prob(discard=burn_in_steps, flat=True)
    max_prob_index   = np.argmax(log_prob_samples)
    max_lhd          = log_prob_samples[max_prob_index]
    best_fit_params  = sampler.get_chain(discard=burn_in_steps, flat=True)[max_prob_index]
    flat_samples     = sampler.get_chain(discard=burn_in_steps, flat=True)

    lower_1sigma  = np.percentile(flat_samples, 16, axis=0)
    upper_1sigma  = np.percentile(flat_samples, 84, axis=0)
    median_1sigma = np.median(flat_samples, axis=0)
    std_1sigma    = np.std(flat_samples, axis=0, ddof=1)

    dt = time.time() - t0
    print(f'[bin {energy_bin}] done in {dt/60:.1f} min  best={best_fit_params}', flush=True)
    return (best_fit_params, median_1sigma, std_1sigma, max_lhd,
            upper_1sigma, lower_1sigma)


n = len(E)
fitted_params        = np.zeros(n * 5)
fitted_params_median = np.zeros(n * 5)
fitted_params_std    = np.zeros(n * 5)
fitted_params_upper  = np.zeros(n * 5)
fitted_params_lower  = np.zeros(n * 5)
max_likelihood       = np.zeros(n)

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
# Step 10 — Save outputs
# ============================================================
GCE_arr = GCE  # alias
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
np.savetxt(f'./GCE_cov_l{roi}{front}_17yr_cholis_likelihood_value', max_likelihood)

np.savez(
    f'./GCE_cov_l{roi}{front}_17yr_cholis_fit.npz',
    roi                  = roi,
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
    counts_per_exp       = counts_per_exp,
    counts_per_exp_err   = counts_per_exp_err,
)

print(f'[done] roi={roi}  total={(time.time()-t_start)/60:.1f} min  '
      f'sum_logL={np.sum(max_likelihood):.1f}', flush=True)
