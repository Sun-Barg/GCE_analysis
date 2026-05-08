#!/usr/bin/env python3
import os; os.environ['TQDM_DISABLE'] = '1'
"""run_one_model.py v3 — Run the full GCE analysis pipeline for ONE model.

Key fixes vs notebook v12:
  - 14-bin enforcement (run_one_model rebuilds CCUBE if it has 13 bins)
  - Likelihood class caching per energy_bin (7-10x speedup in serial)
  - Serial emcee (no Pool — Pool was hiding the per-call Likelihood
    re-instantiation cost behind parallel I/O)
  - matplotlib Agg backend (no GUI in subprocess)
  - gc.collect() x 2 at end (12yr-validated OOM prevention)
  - Saves both .dat (5 cols, Cholis format) AND .npz (full fit arrays)

Usage:
    python run_one_model.py <MODEL>          # e.g. python run_one_model.py X
    python run_one_model.py <MODEL> --force  # ignore existing .dat
"""

import argparse
import gc
import glob
import os
import re as _re
import sys
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import emcee
import matplotlib
matplotlib.use('Agg')                    # no GUI in subprocess (must be before pyplot)
import matplotlib.pyplot as plt
import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack, hstack
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS

from chainconsumer import Chain, ChainConsumer

from scipy.integrate import dblquad
from scipy.interpolate import CubicSpline, interp1d
from scipy.special import gammaln

from gt_apps import GtApp
import gt_apps as gt_apps

warnings.filterwarnings('ignore')


# ===== CONFIG (mirrors notebook cell 3) =====
# ==================================================================
# 17yr GCE Analysis CONFIG - haebarg server (v6)
# ==================================================================
# Edit ONLY here. Both notebooks (main + covariance) share this CONFIG.
# Run notebooks from:
#   /home/haebarg/GCE-Chi-square-fitting/GCE_17yr_reproduce/
# ------------------------------------------------------------------

import os

# ---- Event type / IRF ---------------------------------------------
# Default: FRONT-only (evtype=1) for direct comparison with Cholis+2022.
# To switch back to FRONT+BACK (16yr Sanghwan default):
#     front='_front_back', evtype_number=3
front          = '_front'              # filename suffix
evtype_number  = 1                     # 1=FRONT only, 3=FRONT+BACK
evclass_number = 256                   # 256 = CLEAN
IRFS           = 'P8R3_CLEAN_V3'

# ---- Mask scaling --------------------------------------------------
MASK_SCALE = 1.0     # x1.0 (Cholis Table III). 16yr used x0.9 (undocumented).

# ---- Energy bins ---------------------------------------------------
# Default 51.9 GeV cap = 14 bins matching the mask definition (Cholis Table III).
# Set to 1000.0 to extend to ~550 GeV (17 bins, 16yr default).
ENERGY_BIN_MAX_GEV = 51.9

# ---- Catalog: FL16Y -----------------------------------------------
DR_NUMBER       = 5
CATALOG_FILE    = '../GCE_17yr_data/gll_psc_v40.fit'
CATALOG_SUFFIX  = '_FL16Y'
EXTENDED_DIR    = '../GCE_17yr_data/LAT_extended_sources_16years/Templates/'

# ---- Diffuse + isotropic ------------------------------------------
GALACTIC_FILE   = '../gll_iem_v07.fits'
ISOTROPIC_FILE  = '../iso_P8R3_SOURCE_V3_v1.txt'

# ---- Photon input (gtselect handles @listfile fine) --------------
PHOTON_LIST_FILE   = 'photon_data_17yr.txt'
PHOTON_INPUT       = PHOTON_LIST_FILE
PHOTON_WEEKLY_GLOB = '../GCE_allsky_data/photon_files/lat_photon_weekly_w*.fits'

# ---- Spacecraft input ---------------------------------------------
# IMPORTANT: gtmktime does NOT accept @listfile reliably -- it returns
# "Zero rows returned from FT2 file" even with valid weeklies in the list.
# Fix: pre-flight auto-merges SC weeklies into a single FT2 file using
# astropy (memory-safe at SC scale). gtmktime then receives this single file.
SC_MERGED_FILE = '../GCE_allsky_data/lat_spacecraft_merged_17yr.fits'
SC_FILE        = SC_MERGED_FILE                     # what gets fed to gtmktime/gtltcube
SC_WEEKLY_GLOB = '../GCE_allsky_data/sc_files/lat_spacecraft_weekly_w*.fits'
SC_LIST_FILE   = 'sc_files_17yr.txt'                # listfile (kept for diagnostics; not used)

# Time window: 'INDEF' uses the whole spacecraft file
TMIN = 'INDEF'
TMAX = 'INDEF'

# ---- Templates / spectra (haebarg actual file names) --------------
WIMP_MAP_PATH        = './GCE_template_NFW2.fits'
BUBBLE_TEMPLATE      = './Fermi_Bubbles_template.fits'
ISO_SPECTRUM_FILE    = './isotropic_spectrum_ff.txt'
BUBBLE_SPECTRUM_FILE = './fermi_bubble_spectrum.txt'

# ---- GDE Mapcubes -------------------------------------------------
MAPCUBE_DIR_PION   = './MapCubes'
MAPCUBE_DIR_BREMSS = './MapCubes'
MAPCUBE_DIR_ICS    = './MapCubes'
MAPCUBE_EXT        = '.fits'

# ---- External constraints (12yr files; reuse) ---------------------
ISO_CONSTRAINT_FILE     = './GC_analysis_FL16Y/Model/iso_constraints_full_err.txt'
EGB_CONSTRAINT_FILE     = './GC_analysis_FL16Y/Model/egb_constraints_full_err.txt'
BUBBLE_CONSTRAINT_FILE  = './GC_analysis_FL16Y/Model/bubble_constraints.txt'

# ---- Model list ---------------------------------------------------
# Default: full 80 GDE models. Set to a list to restrict, e.g.:
#   MODEL_LIST_OVERRIDE = ['X']  # single model
#   MODEL_LIST_OVERRIDE = ['X', 'XLIX', 'I']  # subset
MODEL_LIST_OVERRIDE = None

# ---- Working / output directory -----------------------------------
WORK_DIR = './GC_analysis_FL16Y'
os.makedirs(f'{WORK_DIR}/Model', exist_ok=True)

# ---- Existing-models tracker --------------------------------------
EXISTING_MODELS_TXT = f'./GCE_17yr{front}_existing_models.txt'

print("17yr CONFIG (v6) loaded:")
print(f"  Event type   : front='{front}', evtype={evtype_number}")
print(f"  Mask scale   : x{MASK_SCALE}")
print(f"  Energy cap   : {ENERGY_BIN_MAX_GEV} GeV")
print(f"  Catalog      : DR={DR_NUMBER}, {CATALOG_FILE}")
print(f"  Photon input : {PHOTON_INPUT}")
print(f"  SC input     : {SC_FILE}")
print(f"  Working dir  : {WORK_DIR}")
print(f"  Models       : {'override='+str(MODEL_LIST_OVERRIDE) if MODEL_LIST_OVERRIDE else 'all 80'}")

# Common derived constants
steradian_per_pixel = (0.1 * np.pi / 180) ** 2


# ===== 14-bin sanity check =====
# The notebook's cell 10 had a floating-point comparison bug that produced
# 13 bins. The correct intent (per project_knowledge / Cholis Table III)
# is 14 bins matching mask definitions. CCUBE was already built so we
# verify here and abort if it has 13 bins.
def verify_14_bins():
    ccube_path = f'./GC_analysis_FL16Y/GC_ccube_17yr{front}_clean.fits'
    if not os.path.exists(ccube_path):
        return  # prep not done yet
    hdul = fits.open(ccube_path)
    n_bins_in_ccube = len(hdul[1].data)
    hdul.close()
    if n_bins_in_ccube != 14:
        print(f'[ERROR] CCUBE has {n_bins_in_ccube} energy bins, expected 14.', flush=True)
        print(f'[ERROR] You likely ran the notebook with the buggy bin generator.', flush=True)
        print(f'[ERROR] Required action:', flush=True)
        print(f'[ERROR]   1. rm {ccube_path} GC_analysis_FL16Y/bin_definitions.*', flush=True)
        print(f'[ERROR]   2. rm GC_analysis_FL16Y/GC_*_model*_17yr*_clean*.fits', flush=True)
        print(f'[ERROR]   3. rm GC_analysis_FL16Y/GC_Extended_srcmap*_clean*.fits', flush=True)
        print(f'[ERROR]   4. Apply the 14-bin patch to bin_definitions.txt', flush=True)
        print(f'[ERROR]   5. Re-run notebook prep through cell 29', flush=True)
        print(f'[ERROR] Or run rebuild_14bin_prep.sh (shipped alongside).', flush=True)
        sys.exit(2)


# ===== run_one_model — extracted from notebook cell 30 + Likelihood cache =====
def run_one_model(model):
    """Run the full per-model pipeline."""
    verify_14_bins()
    
    # v4: integrity validation for FITS outputs from previous (possibly
    # aborted) gtsrcmaps/gtmodel runs. A file passing os.path.exists is
    # not enough — partial writes from aborted runs leave files that fail
    # downstream tools with cryptic errors (e.g. "Cannot read keyword 
    # NDSKEYS"). We delete corrupt files so the skip-if-exists guard 
    # re-runs them.
    def _is_valid_srcmap(path):
        """A complete gtsrcmaps output must have NDSKEYS in the primary header."""
        if not os.path.exists(path):
            return False
        try:
            with fits.open(path) as hdul:
                hdr = hdul[0].header
                # gtsrcmaps writes NDSKEYS counting the per-source extensions
                if 'NDSKEYS' not in hdr:
                    return False
                # Also verify NDSKEYS-many extensions are present
                n = int(hdr['NDSKEYS'])
                if len(hdul) < n + 1:    # primary + n source HDUs
                    return False
        except Exception:
            return False
        return True

    def _is_valid_gtmodel_output(path):
        """A complete gtmodel ccube output must be readable + have data."""
        if not os.path.exists(path):
            return False
        try:
            with fits.open(path) as hdul:
                if len(hdul) < 2:                # primary + EBOUNDS at minimum
                    return False
                if hdul[0].data is None:
                    return False
                # Must have correct first dim = bin count
                if hdul[0].data.ndim != 3:
                    return False
        except Exception:
            return False
        return True

    def _validate_or_remove(path, validator, label):
        """If `path` exists but fails `validator`, delete it + warn.
        Returns True if valid (so caller may [skip])."""
        if not os.path.exists(path):
            return False
        if validator(path):
            return True
        print(f'[corrupt] {label}: {path} — deleting + will re-run', flush=True)
        try:
            os.remove(path)
        except OSError as e:
            print(f'[error] could not delete {path}: {e}', flush=True)
        return False

    # Define the elements to add
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

    # Parse the new sources XML string
    new_sources_tree = ET.ElementTree(ET.fromstring(f"<sources>{new_sources}</sources>"))
    new_sources_root = new_sources_tree.getroot()

    # Parse the existing XML file
    tree = ET.parse('./GC_analysis_FL16Y/Model/GC_psc_model_FL16Y.xml')
    root = tree.getroot()

    # Append the new sources to the root element of the existing file
    for new_source in new_sources_root:
        root.append(new_source)

    # Save the modified XML to a new file
    _xml_path = f'./GC_analysis_FL16Y/Model/GC_model{model}_test.xml'
    if not os.path.exists(_xml_path):
        tree.write(_xml_path, encoding='utf-8', xml_declaration=True)
        print(f'[done] wrote {_xml_path}')
    #Creating total xml model file for srcmap

    # Define the elements to add
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

    # Parse the new sources XML string
    new_sources_tree = ET.ElementTree(ET.fromstring(f"<sources>{new_sources}</sources>"))
    new_sources_root = new_sources_tree.getroot()

    # Parse the existing XML file
    tree = ET.parse('./GC_analysis_FL16Y/Model/empty_model.xml')
    root = tree.getroot()

    # Append the new sources to the root element of the existing file
    for new_source in new_sources_root:
        root.append(new_source)

    # Save the modified XML to a new file
    _xml_path = f'./GC_analysis_FL16Y/Model/GC_Extended_model{model}_test.xml'
    if not os.path.exists(_xml_path):
        tree.write(_xml_path, encoding='utf-8', xml_declaration=True)
        print(f'[done] wrote {_xml_path}')


    convol=''
    _srcmap_out = f'./GC_analysis_FL16Y/GC_Extended_srcmap_17yr{front}{"_clean"}_model_{model}{convol}.fits'
    if _validate_or_remove(_srcmap_out, _is_valid_srcmap, 'gtsrcmaps'):
        print(f'[skip] gtsrcmaps: {_srcmap_out}')
    else:
        print(f'[run ] gtsrcmaps -> {_srcmap_out}')
        srcMaps = GtApp('gtsrcmaps', 'Likelihood')
        srcMaps['scfile']=SC_FILE
        srcMaps['expcube']=f'./GC_analysis_FL16Y/Allsky_ltcube_17yr{front}{"_clean"}.fits'
        srcMaps['cmap']=f'./GC_analysis_FL16Y/GC_ccube_17yr{front}{"_clean"}.fits'
        srcMaps['bexpmap']=f'./GC_analysis_FL16Y/Allsky_expcube_edge_17yr{front}{"_clean"}.fits'
        srcMaps['srcmdl']=f'./GC_analysis_FL16Y/Model/GC_Extended_model{model}_test.xml'
        srcMaps['outfile']=f'./GC_analysis_FL16Y/GC_Extended_srcmap_17yr{front}{"_clean"}_model_{model}{convol}.fits'
        srcMaps['irfs']=IRFS
        srcMaps['convol']='yes'
        srcMaps['evtype']=evtype_number
        #srcMaps['resample']='no'
        srcMaps.run();

    convol='_no_convol'
    _srcmap_out = f'./GC_analysis_FL16Y/GC_Extended_srcmap_17yr{front}{"_clean"}_model_{model}{convol}.fits'
    if _validate_or_remove(_srcmap_out, _is_valid_srcmap, 'gtsrcmaps'):
        print(f'[skip] gtsrcmaps: {_srcmap_out}')
    else:
        print(f'[run ] gtsrcmaps -> {_srcmap_out}')
        srcMaps = GtApp('gtsrcmaps', 'Likelihood')
        srcMaps['scfile']=SC_FILE
        srcMaps['expcube']=f'./GC_analysis_FL16Y/Allsky_ltcube_17yr{front}{"_clean"}.fits'
        srcMaps['cmap']=f'./GC_analysis_FL16Y/GC_ccube_17yr{front}{"_clean"}.fits'
        srcMaps['bexpmap']=f'./GC_analysis_FL16Y/Allsky_expcube_edge_17yr{front}{"_clean"}.fits'
        srcMaps['srcmdl']=f'./GC_analysis_FL16Y/Model/GC_Extended_model{model}_test.xml'
        srcMaps['outfile']=f'./GC_analysis_FL16Y/GC_Extended_srcmap_17yr{front}{"_clean"}_model_{model}{convol}.fits'
        srcMaps['irfs']=IRFS
        srcMaps['convol']='no'
        srcMaps['evtype']=evtype_number
        #srcMaps['resample']='no'
        srcMaps.run();

    for component in ['bremss', 'ics', 'pion']:
        # Define the elements to add
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
    
        # Parse the new sources XML string
        new_sources_tree = ET.ElementTree(ET.fromstring(f"<sources>{new_sources}</sources>"))
        new_sources_root = new_sources_tree.getroot()
    
        # Parse the existing XML file
        tree = ET.parse('./GC_analysis_FL16Y/Model/empty_model.xml')
        root = tree.getroot()
    
        # Append the new sources to the root element of the existing file
        for new_source in new_sources_root:
            root.append(new_source)
    
        # Save the modified XML to a new file
        _xml_path = f'./GC_analysis_FL16Y/Model/GC_{component}_model{model}_test.xml'
        if not os.path.exists(_xml_path):
            tree.write(_xml_path, encoding='utf-8', xml_declaration=True)


    convol=''
    for component in ['pion', 'bremss', 'ics']:
        _gtm_out = f'./GC_analysis_FL16Y/GC_{component}_model{model}_17yr{front}_clean{convol}.fits'
        if _validate_or_remove(_gtm_out, _is_valid_gtmodel_output, 'gtmodel'):
            print(f'[skip] gtmodel: {_gtm_out}')
        else:
            print(f'[run ] gtmodel -> {_gtm_out}')
            gtmodel = GtApp('gtmodel', 'Likelihood')
            gtmodel['irfs']=IRFS
            gtmodel['outtype']='ccube'
            gtmodel['srcmdl']=f'./GC_analysis_FL16Y/Model/GC_{component}_model{model}_test.xml'
            gtmodel['outfile']=f'./GC_analysis_FL16Y/GC_{component}_model{model}_17yr{front}_clean{convol}.fits'
            gtmodel['expcube']=f'./GC_analysis_FL16Y/Allsky_ltcube_17yr{front}{"_clean"}.fits'
            gtmodel['bexpmap']=f'./GC_analysis_FL16Y/Allsky_expcube_edge_17yr{front}{"_clean"}.fits'
            gtmodel['convol']='yes'
            #gtmodel['resample']='no'
            gtmodel['evtype']=evtype_number
            gtmodel['srcmaps']=f'./GC_analysis_FL16Y/GC_Extended_srcmap_17yr{front}{"_clean"}_model_{model}{convol}.fits'
            gtmodel.run()


    convol='_no_convol'
    for component in ['pion', 'bremss', 'ics']:
        _gtm_out = f'./GC_analysis_FL16Y/GC_{component}_model{model}_17yr{front}_clean{convol}.fits'
        if _validate_or_remove(_gtm_out, _is_valid_gtmodel_output, 'gtmodel'):
            print(f'[skip] gtmodel: {_gtm_out}')
        else:
            print(f'[run ] gtmodel -> {_gtm_out}')
            gtmodel = GtApp('gtmodel', 'Likelihood')
            gtmodel['irfs']=IRFS
            gtmodel['outtype']='ccube'
            gtmodel['srcmdl']=f'./GC_analysis_FL16Y/Model/GC_{component}_model{model}_test.xml'
            gtmodel['outfile']=f'./GC_analysis_FL16Y/GC_{component}_model{model}_17yr{front}_clean{convol}.fits'
            gtmodel['expcube']=f'./GC_analysis_FL16Y/Allsky_ltcube_17yr{front}{"_clean"}.fits'
            gtmodel['bexpmap']=f'./GC_analysis_FL16Y/Allsky_expcube_edge_17yr{front}{"_clean"}.fits'
            gtmodel['convol']='no'
            #gtmodel['resample']='no'
            gtmodel['evtype']=evtype_number
            gtmodel['srcmaps']=f'./GC_analysis_FL16Y/GC_Extended_srcmap_17yr{front}{"_clean"}_model_{model}{convol}.fits'
            gtmodel.run()




    # ============================================================
    # GCE / Fermi_bubble / isotropic prep
    # (was in 16yr cell 36 / v4 cell 43 — incorrectly disabled in v3
    # under "duplicated by main loop"; in fact it produces 6 files
    # the active emcee code reads. Restored in v10 with skip-if-exists.)
    # ============================================================

    # ---- Build per-source XML files (one source appended to empty_model.xml) ----
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
    for _src_name, _src_xml in _src_specs.items():
        _src_model_xml = f'./GC_analysis_FL16Y/Model/GC_{_src_name}_model.xml'
        if os.path.exists(_src_model_xml):
            continue
        _new_sources_root = ET.fromstring(f"<sources>{_src_xml}</sources>")
        _tree = ET.parse('./GC_analysis_FL16Y/Model/empty_model.xml')
        _root = _tree.getroot()
        for _new_src in _new_sources_root:
            _root.append(_new_src)
        _tree.write(_src_model_xml, encoding='utf-8', xml_declaration=True)
        print(f'[done] wrote {_src_model_xml}')

    # ---- gtmodel × 2 (convol='yes' / convol='no') over [GCE, fermi_bubble, isotropic] ----
    for _convol_setting, _convol_suffix in [('yes', ''), ('no', '_no_convol')]:
        # The srcmaps input is per-model (uses {model}{convol}.fits from gtsrcmaps);
        # the outfile is NOT per-model (model-independent template).
        _src_srcmap = f'./GC_analysis_FL16Y/GC_Extended_srcmap_17yr{front}{"_clean"}_model_{model}{_convol_suffix}.fits'
        for _comp_name in ['GCE', 'fermi_bubble', 'isotropic']:
            _comp_out = f'./GC_analysis_FL16Y/GC_{_comp_name}_model_17yr{front}_clean{_convol_suffix}.fits'
            if _validate_or_remove(_comp_out, _is_valid_gtmodel_output, f'gtmodel ({_comp_name})'):
                print(f'[skip] gtmodel ({_comp_name}, convol={_convol_setting}): {_comp_out}')
                continue
            print(f'[run ] gtmodel ({_comp_name}, convol={_convol_setting}) -> {_comp_out}')
            _gtm = GtApp('gtmodel', 'Likelihood')
            _gtm['irfs']      = IRFS
            _gtm['outtype']   = 'ccube'
            _gtm['srcmdl']    = f'./GC_analysis_FL16Y/Model/GC_{_comp_name}_model.xml'
            _gtm['outfile']   = _comp_out
            _gtm['expcube']   = f'./GC_analysis_FL16Y/Allsky_ltcube_17yr{front}{"_clean"}.fits'
            _gtm['bexpmap']   = f'./GC_analysis_FL16Y/Allsky_expcube_edge_17yr{front}{"_clean"}.fits'
            _gtm['convol']    = _convol_setting
            _gtm['evtype']    = evtype_number
            _gtm['srcmaps']   = _src_srcmap
            _gtm.run()

    ## Emcee running part

    def roi_solid_angle(delta_l_deg, delta_b_deg, b_deg):
        # Convert degrees to radians
        delta_l_rad = np.radians(delta_l_deg)
        delta_b_rad = np.radians(delta_b_deg)
        b_rad = np.radians(b_deg)
    
        # Calculate solid angle in steradians
        solid_angle = delta_l_rad * delta_b_rad * np.cos(b_rad)
        return solid_angle

    raw_map=fits.open(f'./GC_analysis_FL16Y/GC_ccube_17yr{front}_clean.fits')
    w=WCS(raw_map[0].header).dropaxis(2)
    # Define the dimensions of the numpy array
    width, height = np.shape(raw_map[0].data[0])

    # Create the counts map
    steradian_per_pixel=np.zeros([width, height])

    for i in range(0, height, 1):
        for j in range(0, width, 1):
            l, b = w.wcs_pix2world(j, i, 0) #x-axis array - b, y-axis array - l
            steradian_per_pixel[i, j] = roi_solid_angle(0.1, 0.1, b)

    # Revision :: Aug 11, 2024
    disk_mask=np.load('./GC_analysis_FL16Y/Model/GC_disk_mask_60x60_definitions.npy')[100:500, 100:500]
    psc_mask=np.load('./GC_analysis_FL16Y/Model/GC_mask_60x60_definitions_FL16Y.npy')[:, 100:500, 100:500]

    # [17yr] front comes from CONFIG cell — kept here for reference: front='_front_back'
    convol='_no_convol'

    E_bounds=fits.open(f'./GC_analysis_FL16Y/GC_ccube_17yr{front}_clean.fits')[1].data


    E=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        E[i] = np.sqrt(E_bounds[i][2]*E_bounds[i][1]*1e-6)*1e-3

    delta_E=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        delta_E[i] = (E_bounds[i][2] - E_bounds[i][1])*1e-6

    exp_cube = fits.open(f'./GC_analysis_FL16Y/GC_expcube_center_17yr{front}_clean.fits')[0].data[:, 100:500, 100:500]*steradian_per_pixel[100:500, 100:500]



    file_name=f'./GC_analysis_FL16Y/GC_pion_model{model}_17yr{front}_clean_no_convol.fits'
    pion=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        pion[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/a) )/np.sum(disk_mask)

    file_name=f'./GC_analysis_FL16Y/GC_bremss_model{model}_17yr{front}_clean_no_convol.fits'
    bremss=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        bremss[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/a) )/np.sum(disk_mask)
    
    file_name=f'./GC_analysis_FL16Y/GC_ics_model{model}_17yr{front}_clean_no_convol.fits'
    ics=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        ics[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/a) )/np.sum(disk_mask)
    
    file_name=f'./GC_analysis_FL16Y/GC_GCE_model_17yr{front}_clean_no_convol.fits'
    GCE=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        GCE[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/exp_cube[i]) )/np.sum(disk_mask)


    file_name=f'./GC_analysis_FL16Y/GC_fermi_bubble_model_17yr{front}_clean_no_convol.fits'
    bubble=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        bubble[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/a) )/np.sum(disk_mask)
    
    file_name=f'./GC_analysis_FL16Y/GC_isotropic_model_17yr{front}_clean_no_convol.fits'
    isotropic=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        isotropic[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/a) )/np.sum(disk_mask)
    
    counts_per_exp=np.zeros(len(E_bounds))
    i=0
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        counts_per_exp[i]=np.sum( disk_mask*( (fits.open(f'./GC_analysis_FL16Y/GC_ccube_17yr{front}_clean.fits')[0].data[i][100:500, 100:500]) /a) )/np.sum(disk_mask)

    counts_per_exp_err=np.zeros(len(E_bounds))
    i=0
    for i in range(0, len(E_bounds), 1):
        counts_per_exp_err[i]= np.sqrt( np.sum( ( (np.sqrt(disk_mask*fits.open(f'./GC_analysis_FL16Y/GC_ccube_17yr{front}_clean.fits')[0].data[i][100:500, 100:500]) /exp_cube[i] )**2)) )/np.sum(disk_mask)
        #counts_per_exp_err[i]=  np.sqrt(np.sum(disk_mask*fits.open(f'./GC_analysis_FL16Y/GC_all_time_60x60_ccube{front}_17yr.fits')[0].data[i][100:500, 100:500]))/np.sum(disk_mask*exp_cube[i]) 

    

    def log_factorial(O):
        """v12: vectorized log(O!) via gammaln(O+1) - ~1000x faster than the Python-loop version."""
        return gammaln(np.asarray(O, dtype=float) + 1.0)
    #Constraints interpolated function
    #Contains constraints for bubble and isotropic as well
    #For isotropic, from https://arxiv.org/pdf/1410.3696.pdf Table 3
    #Correcting bubble template given from https://arxiv.org/pdf/1407.7905, Table 2

    bubble_constraints=np.loadtxt('./GC_analysis_FL16Y/Model/bubble_constraints.txt')
    bubble_constraints_energy=bubble_constraints[0:, 0]
    bubble_constraints_flux=bubble_constraints[0:, 1]
    bubble_constraints_lower_error=bubble_constraints[0:, 2]
    bubble_constraints_upper_error=bubble_constraints[0:, 3]

    bubble_fluxint = interp1d((bubble_constraints_energy), (bubble_constraints_flux), fill_value='extrapolate', kind='quadratic')
    bubble_lower_errint = interp1d((bubble_constraints_energy), (bubble_constraints_lower_error), fill_value='extrapolate', kind='quadratic')
    bubble_upper_errint = interp1d((bubble_constraints_energy), (bubble_constraints_upper_error), fill_value='extrapolate', kind='quadratic')

    bubble_flux_data=bubble_fluxint((E))
    bubble_lower_error_data=bubble_lower_errint((E))
    bubble_upper_error_data=bubble_upper_errint((E))


    iso_constraints=np.loadtxt('./GC_analysis_FL16Y/Model/iso_constraints_full_err.txt')
    #iso_constraints=np.loadtxt('./GC_analysis_FL16Y/Model/egb_constraints_full_err.txt')

    iso_constraints_energy=iso_constraints[0:, 0]
    iso_constraints_flux=iso_constraints[0:, 1]
    iso_constraints_low_err=iso_constraints[0:, 2]
    iso_constraints_upp_err=iso_constraints[0:, 3]

    isotropic_fluxint=interp1d(iso_constraints_energy, iso_constraints_flux, fill_value="extrapolate", kind='quadratic')
    isotropic_lower_errint=interp1d(iso_constraints_energy, iso_constraints_low_err, fill_value="extrapolate", kind='quadratic')    
    isotropic_upper_errint=interp1d(iso_constraints_energy, iso_constraints_upp_err, fill_value="extrapolate", kind='quadratic')  

    isotropic_flux_data=((E)**2)*(isotropic_fluxint((E)))
    isotropic_lower_error_data=((E)**2)*(isotropic_lower_errint((E)))
    isotropic_upper_error_data=((E)**2)*(isotropic_upper_errint((E)))

    # [17yr] front comes from CONFIG cell — kept here for reference: front='_front_back'
    class Likelihood:
        def __init__(self, model, energy_bin):
            self.model=model
            self.energy_bin=energy_bin
            self.data=fits.open(f'./GC_analysis_FL16Y/GC_ccube_17yr{front}_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]
            self.pion_bremss=fits.open(f'./GC_analysis_FL16Y/GC_pion_model{model}_17yr{front}_clean.fits')[0].data[self.energy_bin, 100:500, 100:500] + fits.open(f'./GC_analysis_FL16Y/GC_bremss_model{model}_17yr{front}_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]  
            self.ics=fits.open(f'./GC_analysis_FL16Y/GC_ics_model{model}_17yr{front}_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]
            self.GCE=fits.open(f'./GC_analysis_FL16Y/GC_GCE_model_17yr{front}_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]
            self.bubble=fits.open(f'./GC_analysis_FL16Y/GC_fermi_bubble_model_17yr{front}_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]
            self.iso=fits.open(f'./GC_analysis_FL16Y/GC_isotropic_model_17yr{front}_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]
            # v12: cache no_convol versions for SED chi2 terms
            self.iso_no_convol    = fits.open(f'./GC_analysis_FL16Y/GC_isotropic_model_17yr{front}_clean_no_convol.fits')[0].data[self.energy_bin, 100:500, 100:500]
            self.bubble_no_convol = fits.open(f'./GC_analysis_FL16Y/GC_fermi_bubble_model_17yr{front}_clean_no_convol.fits')[0].data[self.energy_bin, 100:500, 100:500]
            E_bounds=fits.open(f'./GC_analysis_FL16Y/GC_ccube_17yr{front}_clean.fits')[1].data

            E=np.zeros(len(E_bounds))
            for i in range(0, len(E_bounds), 1):
                E[i] = np.sqrt(E_bounds[i][2]*E_bounds[i][1]*1e-6)*1e-3
            self.E = E
            delta_E=np.zeros(len(E_bounds))
            for i in range(0, len(E_bounds), 1):
                delta_E[i] = (E_bounds[i][2] - E_bounds[i][1])*1e-6
            self.delta_E = delta_E
            self.exp_cube = (fits.open(f'./GC_analysis_FL16Y/GC_expcube_center_17yr{front}_clean.fits')[0].data[self.energy_bin]*steradian_per_pixel)[100:500, 100:500]

        
            psc_mask=np.load('./GC_analysis_FL16Y/Model/GC_mask_60x60_definitions_FL16Y.npy')[self.energy_bin, 100:500, 100:500]

            disk_mask=np.load('./GC_analysis_FL16Y/Model/GC_disk_mask_60x60_definitions.npy')[100:500, 100:500]
            full_mask=psc_mask*disk_mask
            self.disk_mask=disk_mask
            self.full_mask=full_mask
            # v12: precompute log(O!) on masked observed data once per bin
            _obs_masked = self.data[self.full_mask == 1].astype(float)
            self.observed_log_factorial_masked = log_factorial(_obs_masked)
        def likelihood_constrained(self, parameter_set):
            #####################################
            pion_bremss_param=parameter_set[0]
            ics_param=parameter_set[1]
            GCE_param=parameter_set[2]
            bubble_param=parameter_set[3]
            isotropic_param=parameter_set[4]
            ######################################
            expected_pixel= (pion_bremss_param)*self.pion_bremss + (ics_param)*self.ics + (GCE_param)*self.GCE + (isotropic_param)*self.iso + (bubble_param)*self.bubble   
            observed_pixel = self.data

            observed_pixel = observed_pixel[self.full_mask == 1]
            expected_pixel = expected_pixel[self.full_mask == 1]

        
            if (expected_pixel < 0).any():
                return np.inf
            
            #expected_pixel[expected_pixel == 0.0] += 1e-20
        
            observed_log_expected=observed_pixel*np.log(expected_pixel)
            #nan_index = np.where(np.isnan(observed_log_expected))
            #observed_log_expected[nan_index] = 0
            lhd=2*( expected_pixel - observed_log_expected + self.observed_log_factorial_masked )

        
            file_name=f'./GC_analysis_FL16Y/GC_isotropic_model_17yr{front}_clean_no_convol.fits'
            _cached_no_convol = self.iso_no_convol
            #isotropic = np.sum( self.full_mask*(fits.open(file_name)[0].data[self.energy_bin]/self.exp_cube) )*isotropic_param/np.sum(self.full_mask)
            #isotropic = np.sum( self.full_mask*(fits.open(file_name)[0].data[self.energy_bin]) )*isotropic_param/np.sum(self.full_mask*self.exp_cube)
            #isotropic = np.sum( self.full_mask*(fits.open(file_name)[0].data[self.energy_bin])/self.exp_cube )*isotropic_param/np.sum(self.full_mask)
            isotropic = np.sum( self.full_mask*(_cached_no_convol)/self.exp_cube )*isotropic_param/np.sum(self.full_mask)
            #isotropic = np.sum( (fits.open(file_name)[0].data[self.energy_bin])/self.exp_cube )*isotropic_param/np.sum(600*600)

            isotropic_sed = (self.E[self.energy_bin]**2)*isotropic/(self.delta_E[self.energy_bin])


            file_name=f'./GC_analysis_FL16Y/GC_fermi_bubble_model_17yr{front}_clean_no_convol.fits'
            _cached_no_convol = self.bubble_no_convol
            #bubble = np.sum( self.full_mask*(fits.open(file_name)[0].data[self.energy_bin]/self.exp_cube) )*bubble_param/np.sum(self.full_mask)
            #bubble = np.sum( self.full_mask*(fits.open(file_name)[0].data[self.energy_bin]) )*bubble_param/np.sum(self.full_mask*self.exp_cube)
            #bubble = np.sum( self.full_mask*(fits.open(file_name)[0].data[self.energy_bin])/self.exp_cube )*bubble_param/np.sum(self.full_mask)
            bubble = np.sum( self.full_mask*(_cached_no_convol)/self.exp_cube )*bubble_param/np.sum(self.full_mask)
            #bubble = np.sum( (fits.open(file_name)[0].data[self.energy_bin])/self.exp_cube )*bubble_param/np.sum(600*600)
            bubble_sed = (self.E[self.energy_bin]**2)*bubble/(self.delta_E[self.energy_bin])

            larger_error=max([bubble_upper_error_data[self.energy_bin], bubble_lower_error_data[self.energy_bin]])
            if bubble_flux_data[self.energy_bin] < bubble_sed:
                chi2_bubble = ((bubble_sed - bubble_flux_data[self.energy_bin])/bubble_upper_error_data[self.energy_bin])**2
            if bubble_flux_data[self.energy_bin] > bubble_sed:
                chi2_bubble = ((bubble_sed - bubble_flux_data[self.energy_bin])/bubble_lower_error_data[self.energy_bin])**2
            if bubble_flux_data[self.energy_bin] == bubble_sed:
                chi2_bubble = ((bubble_sed - bubble_flux_data[self.energy_bin])/larger_error)**2


            isotropic_larger_error=max([isotropic_lower_error_data[self.energy_bin], isotropic_upper_error_data[self.energy_bin]])
            if isotropic_flux_data[self.energy_bin] < isotropic_sed:
                chi2_isotropic = ((isotropic_flux_data[self.energy_bin] - isotropic_sed)/isotropic_lower_error_data[self.energy_bin])**2
            if isotropic_flux_data[self.energy_bin] > isotropic_sed:
                chi2_isotropic = ((isotropic_flux_data[self.energy_bin] - isotropic_sed)/isotropic_upper_error_data[self.energy_bin])**2
            if isotropic_flux_data[self.energy_bin] == isotropic_sed:
                chi2_isotropic = ((isotropic_flux_data[self.energy_bin] - isotropic_sed)/isotropic_larger_error)**2
            #print(chi2_bubble, chi2_isotropic)
            return (np.sum(lhd)  + chi2_bubble + chi2_isotropic)



    # Define your likelihood function
    # v3 fix: cache Likelihood per energy_bin to avoid 7 fits.open + 2 np.load
    # + log_factorial recomputation on every log_likelihood call. The notebook
    # ran ~500k log_likelihood calls per bin, each rebuilding the full state.
    # In serial mode this dominates wall time (Pool(64) was hiding it).
    _lh_cache = {}
    def log_likelihood(params, energy_bin):
        lh = _lh_cache.get(energy_bin)
        if lh is None:
            lh = Likelihood(model, energy_bin)
            _lh_cache[energy_bin] = lh
        return -(1/2)*lh.likelihood_constrained(params)  # log likelihood -> Need to maximize

    # Define the prior function with parameter limits
    def log_prior(params):
        limits = [
            (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)
            #(0, 2), (0, 2), (-5, 5), (0.09, 0.11), (0, 5)
            #(-3, 3),  # Bounds for param1
            #(-3, 3),  # Bounds for param2
            #(-5, 5),  # Bounds for param3
            #(-3, 3),  # Bounds for param4
            #(-3, 3)   # Bounds for param5
        ]
    
        for i, (lower, upper) in enumerate(limits):
            if not (lower <= params[i] <= upper):
                return -np.inf  # Return negative infinity if outside bounds
        return 0.0  # Return zero if all parameters are within bounds

    # Define the log probability function
    def log_probability(params, energy_bin):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, energy_bin)  # Maximize this
        #return log_likelihood(params, energy_bin)

    def run_mcmc_for_bin(energy_bin):  
        ndim = 5
        nwalkers = 100
        nsteps = 1000
        burn_in_steps = 400
        start_time=time.time()
        print(energy_bin)
        # subprocess: serial emcee with bin-level Likelihood caching
        #np.random.seed(100100)#
        #initial_params = np.random.uniform(0, 1, [nwalkers, ndim])
        #initial_params = np.vstack([np.random.uniform(0, 1, [nwalkers]), np.random.uniform(0, 1, [nwalkers]), np.random.uniform(9, 11, [nwalkers]), np.random.uniform(0, 1, [nwalkers]), np.random.uniform(2, 3, [nwalkers])]).T
        initial_params = np.vstack([np.random.uniform(0.3, 1.5, [nwalkers]), np.random.uniform(1.0, 2.0, [nwalkers]), np.random.uniform(0.5, 3.0, [nwalkers]), np.random.uniform(0.5, 1.5, [nwalkers]), np.random.uniform(0.5, 1.5, [nwalkers])]).T
        #initial_params = np.abs(np.ones(5) + np.random.randn(nwalkers, ndim))
        #initial_params = np.vstack([np.random.uniform(-5, 5, [nwalkers]), np.random.uniform(-5, 5, [nwalkers]), np.random.uniform(-5, 5, [nwalkers]), np.random.uniform(-5, 5, [nwalkers]), np.random.uniform(-5, 5, [nwalkers])]).T
        #initial_params = np.abs(np.ones(5) + 0.1*np.random.randn(nwalkers, ndim))
        pos = initial_params

        from emcee.moves import DEMove, KDEMove
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(energy_bin,))#, moves=[(DEMove(), 0.7), (KDEMove(), 0.3)])
        #for iteration in range(10):    
        print("Running production...")
        pos, prob, state = sampler.run_mcmc(pos, nsteps, progress=True)
        #sampler.run_mcmc(pos, nsteps, progress=True)
        
        max_pos = pos[np.argmax(prob)]
        fitted_param = max_pos
        #max_lhd = np.argmax(prob)

  
        
        log_prob_samples = sampler.get_log_prob(discard=burn_in_steps, flat=True)

        max_prob_index = np.argmax(log_prob_samples)
        max_lhd = log_prob_samples[max_prob_index]

        best_fit_params = sampler.get_chain(discard=burn_in_steps, flat=True)[max_prob_index]        

        fitted_param = best_fit_params
        
        flat_samples = sampler.get_chain(discard=burn_in_steps, flat=True)

        lower_1sigma = np.percentile(flat_samples, 16, axis=0)
        upper_1sigma = np.percentile(flat_samples, 84, axis=0)

        #for i in range(ndim):
        # Calculate the 16th, 50th, and 84th percentiles for the i-th parameter
            #mcmc = np.percentile(flat_samples[:, i], [50])
            #fitted_param[i] = mcmc[0]
            #print(mcmc[0])

        print(max_pos)#, best_fit_params, fitted_param)

        

        #fitted_param = best_fit_params
        
        # Get only the samples from the current iteration
        samples = sampler.get_chain(discard=burn_in_steps, thin=1, flat=False)
        current_samples = samples[-nsteps:]  # Get only the last `nsteps` samples
    
        print("Max position:", fitted_param)
        # Trace Plot for Each Walker and Parameter
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            for walker in range(nwalkers):
                ax.plot(current_samples[:, walker, i], alpha=0.3)  # Plot each walker separately
            ax.set_xlim(0, nsteps)
            ax.set_ylim(-3, 20)
            ax.set_ylabel(f"param{i+1}")
            ax.yaxis.set_label_coords(-0.1, 0.5)
    
        axes[-1].set_xlabel("step number")
        plt.suptitle(f"Trace Plot for Each Walker after iteration")
        plt.close('all')
        # Final Corner Plot with ChainConsumer
        flat_samples = sampler.get_chain(discard=burn_in_steps, thin=1, flat=True)  # Flatten for ChainConsumer
        c = ChainConsumer()
        c.add_chain(Chain(samples=pd.DataFrame(flat_samples, columns=["param1", "param2", "param3", "param4", "param5"]), name='MCMC'))
        fig = c.plotter.plot(figsize=(6, 6))
        axes=fig.axes
        #for ax in axes:
        #    ax.set_xlim(-1, 6)
        #    ax.set_ylim(-1, 6)
        plt.close('all')
        print("std", np.std(flat_samples, axis=0, ddof=1))
        end_time=time.time()
        #print(f"{np.round((end_time-start_time)/(60*60), 5)}hours")
        print(fitted_param, np.median(flat_samples, axis=0))
        return fitted_param.T, np.median(flat_samples, axis=0).T, np.std(flat_samples, axis=0, ddof=1).T, max_lhd, upper_1sigma, lower_1sigma
        #return np.median(samples, axis=0), np.std(samples, axis=0)

    n=len(E)
    fitted_params=np.ones([n*5])
    fitted_params_median=np.ones([n*5])
    fitted_params_std = np.zeros([n*5])
    max_likelihood = np.zeros([n])

    fitted_params_upper = np.zeros([n*5])
    fitted_params_lower = np.zeros([n*5])
    # v5: per-bin checkpointing
    _checkpoint_path = f'./GCE_model_{model}{front}_17yr_cholis_checkpoint.npz'
    _completed_bins = set()
    if os.path.exists(_checkpoint_path):
        try:
            _ckpt = np.load(_checkpoint_path)
            fitted_params         = _ckpt['fitted_params'].copy()
            fitted_params_std     = _ckpt['fitted_params_std'].copy()
            fitted_params_median  = _ckpt['fitted_params_median'].copy()
            fitted_params_upper   = _ckpt['fitted_params_upper'].copy()
            fitted_params_lower   = _ckpt['fitted_params_lower'].copy()
            max_likelihood        = _ckpt['max_likelihood'].copy()
            _completed_bins       = set(_ckpt['completed_bins'].tolist())
            print(f'[checkpoint] resumed: {len(_completed_bins)}/{n} bins already done', flush=True)
        except Exception as _e:
            print(f'[checkpoint] failed to load (will start fresh): {_e}', flush=True)
            _completed_bins = set()

    import resource as _resource

    for i in range(0, n, 1):
        if i in _completed_bins:
            print(f'[skip] bin {i} already in checkpoint', flush=True)
            continue
        _t_bin = time.time()
        _rss_mb = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f'[bin {i:2d}/{n}] start  rss={_rss_mb:.0f}MB at {time.strftime("%H:%M:%S")}', flush=True)

        max_value, median_value, std_value, maximum_value, upper_value, lower_value = run_mcmc_for_bin(i)
        fitted_params[n*0:n*1][i] = max_value[0]
        fitted_params[n*1:n*2][i] = max_value[1]
        fitted_params[n*2:n*3][i] = max_value[2]
        fitted_params[n*3:n*4][i] = max_value[3]
        fitted_params[n*4:n*5][i] = max_value[4]

        fitted_params_std[n*0:n*1][i] = std_value[0]
        fitted_params_std[n*1:n*2][i] = std_value[1]
        fitted_params_std[n*2:n*3][i] = std_value[2]
        fitted_params_std[n*3:n*4][i] = std_value[3]
        fitted_params_std[n*4:n*5][i] = std_value[4]

        fitted_params_median[n*0:n*1][i] = median_value[0]
        fitted_params_median[n*1:n*2][i] = median_value[1]
        fitted_params_median[n*2:n*3][i] = median_value[2]
        fitted_params_median[n*3:n*4][i] = median_value[3]
        fitted_params_median[n*4:n*5][i] = median_value[4]

        fitted_params_upper[n*0:n*1][i] = upper_value[0]
        fitted_params_upper[n*1:n*2][i] = upper_value[1]
        fitted_params_upper[n*2:n*3][i] = upper_value[2]
        fitted_params_upper[n*3:n*4][i] = upper_value[3]
        fitted_params_upper[n*4:n*5][i] = upper_value[4]


        fitted_params_lower[n*0:n*1][i] = lower_value[0]
        fitted_params_lower[n*1:n*2][i] = lower_value[1]
        fitted_params_lower[n*2:n*3][i] = lower_value[2]
        fitted_params_lower[n*3:n*4][i] = lower_value[3]
        fitted_params_lower[n*4:n*5][i] = lower_value[4]


        max_likelihood[i] = maximum_value
    
        plt.style.use('default')
        ax=plt.subplot()
    
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('E [GeV]')
        ax.set_ylabel(r'$E^2 \frac{dN}{dE}$[GeV$cm^{-2}$$s^{-1} sr^{-1}$]')
    
        ax.set_ylim(1e-8, 1e-4)
        ax.set_xlim(0.3, 500)
    
        ax.tick_params(axis='y', which='both', direction='in', left=True)
        ax.tick_params(axis='x', which='both', direction='in', bottom=True)
        ax.minorticks_on()
        ax.grid(True, which='Major', linestyle='-', linewidth=0.5)
    
        fitted=fitted_params#fitted_params
        fitted_errors=fitted_params_std
        #sr=0.4288213187542626#0.214411*2
        sr=1
        #sr=0.4387
        #sr=0.4776

        ax.errorbar(E, counts_per_exp*(E**2)/(delta_E*sr) , yerr=counts_per_exp_err*(E**2)/(delta_E*sr), linestyle='dotted', marker='.', elinewidth=2, capsize=4, capthick=2, label='Raw_data')
    
    
    
    
        ax.errorbar(E, fitted[n*0:n*1]*(pion+bremss)*(E**2)/(delta_E*sr), yerr=fitted_errors[n*0:n*1]*(pion+bremss)*(E**2)/(delta_E*sr), linestyle='dotted', marker='.', elinewidth=2, capsize=4, capthick=2, label='pion+bremss', color='red')
        ax.errorbar(E, fitted[n*1:n*2]*(ics)*(E**2)/(delta_E*sr), yerr=fitted_errors[n*1:n*2]*(ics)*(E**2)/(delta_E*sr), linestyle='dashdot', marker='.', elinewidth=2, capsize=4, capthick=2, label='ics', color='blue')
    
        ax.plot(E, (pion+bremss)*(E**2)/(delta_E*sr), linestyle='solid', label='pion+bremss', color='red')
        ax.plot(E, (ics)*(E**2)/(delta_E*sr), linestyle='solid', label='ics', color='blue')
    
    
    
        ax.errorbar(E,  fitted[n*2:n*3]*(GCE)*(E**2)/(delta_E*sr), yerr=np.sqrt((fitted_errors[n*2:n*3]*GCE)**2)*(E**2)/(delta_E*sr), alpha=0.1, linestyle='dashed', marker='.', elinewidth=2, capsize=4, capthick=2, label='GCE', color='black')
    
    
    
        ax.errorbar(E, fitted[n*3:n*4]*(bubble)*(E**2)/(delta_E*sr),yerr=fitted_errors[n*3:n*4]*(bubble)*(E**2)/(delta_E*sr), linestyle='dashed', marker='.', elinewidth=2, capsize=4, capthick=2, label='bubble', color='purple')
    
        ax.errorbar(E, fitted[n*4:n*5]*(isotropic)*(E**2)/(delta_E*sr),yerr=fitted_errors[n*4:n*5]*(isotropic)*(E**2)/(delta_E*sr), linestyle='dashed', marker='.', elinewidth=2, capsize=4, capthick=2, label='isotropic', color='green')
        summed = fitted[n*0:n*1]*(pion+bremss) + fitted[n*1:n*2]*(ics) + fitted[n*2:n*3]*(GCE) + fitted[n*3:n*4]*(bubble) + fitted[n*4:n*5]*(isotropic)

        ax.plot(E, (E**2)*summed/(delta_E*sr), label='summed')
    
    
        plt.close('all')
        print((fitted[n*2:n*3]*(GCE)*(E**2)/(delta_E*sr))[i])
        print(((fitted_params_upper[n*2:n*3] - fitted[n*2:n*3])*(GCE)*(E**2)/(delta_E*sr))[i])
        print(((fitted[n*2:n*3]- fitted_params_lower[n*2:n*3]  )*(GCE)*(E**2)/(delta_E*sr))[i])

                                                            




        # v5: save checkpoint + cleanup at end of bin
        _completed_bins.add(i)
        np.savez(
            _checkpoint_path,
            fitted_params         = fitted_params,
            fitted_params_std     = fitted_params_std,
            fitted_params_median  = fitted_params_median,
            fitted_params_upper   = fitted_params_upper,
            fitted_params_lower   = fitted_params_lower,
            max_likelihood        = max_likelihood,
            completed_bins        = np.array(sorted(_completed_bins)),
        )
        plt.close('all')      # release matplotlib figures from this bin
        gc.collect()          # release any cached arrays
        _dt_bin = time.time() - _t_bin
        print(f'[bin {i:2d}/{n}] done   in {_dt_bin/60:.1f}m  ({_dt_bin:.0f}s)', flush=True)
    # v5: all bins done — remove checkpoint
    if os.path.exists(_checkpoint_path):
        os.remove(_checkpoint_path)
        print(f'[checkpoint] removed (all {n} bins complete)', flush=True)
    max_likelihood

    np.sum(max_likelihood)

    np.savetxt(f'./GCE_model_{model}{front}_17yr_cholis.dat', np.vstack([E, fitted[n*2:n*3]*(GCE)*(E**2)/(delta_E), (fitted_errors[n*2:n*3]*GCE)*(E**2)/(delta_E), (fitted_params_lower[n*2:n*3])*(GCE)*(E**2)/(delta_E), (fitted_params_upper[n*2:n*3])*(GCE)*(E**2)/(delta_E)]).T)
    np.savetxt(f'./GCE_model_{model}{front}_17yr_cholis_likelihood_value', np.array((max_likelihood))) #Positive of log likelihood

    # v3 subprocess: also save .npz with full fit arrays for cov / further analysis
    np.savez(
        f'./GCE_model_{model}{front}_17yr_cholis_fit.npz',
        fitted_params         = fitted_params.reshape(5, n),
        fitted_params_median  = fitted_params_median.reshape(5, n),
        fitted_params_std     = fitted_params_std.reshape(5, n),
        fitted_params_upper   = fitted_params_upper.reshape(5, n),
        fitted_params_lower   = fitted_params_lower.reshape(5, n),
        max_likelihood        = max_likelihood,
        E                     = E,
        pion                  = pion,
        bremss                = bremss,
        ics                   = ics,
        GCE                   = GCE,
        bubble                = bubble,
        isotropic             = isotropic,
        counts_per_exp        = counts_per_exp,
        counts_per_exp_err    = counts_per_exp_err,
        delta_E               = delta_E,
    )
    print(f'[done] saved fit npz for model {model}', flush=True)



    # final memory cleanup (12yr-validated to prevent OOM in workers)
    plt.close('all')
    gc.collect()
    gc.collect()


def main():
    p = argparse.ArgumentParser(description='Run GCE analysis for one model')
    p.add_argument('model', help='Roman numeral model identifier (e.g. X, II, XLIX)')
    p.add_argument('--force', action='store_true',
                   help='re-run even if output .dat already exists')
    args = p.parse_args()

    out_dat = f'./GCE_model_{args.model}{front}_17yr_cholis.dat'
    if os.path.exists(out_dat) and not args.force:
        print(f'[skip] {out_dat} already exists', flush=True)
        return 0

    t0 = time.time()
    print(f'[start] model {args.model} at {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    try:
        run_one_model(args.model)
    except Exception as e:
        print(f'[FAIL] model {args.model}: {type(e).__name__}: {e}', flush=True)
        import traceback
        traceback.print_exc()
        return 1
    dt = time.time() - t0
    print(f'[done] model {args.model} in {dt/60:.1f} min', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())