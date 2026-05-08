#!/usr/bin/env python3
"""
run_main_loop_subprocess.py — external runner for the GCE main loop.

VERSION: v3.14 (2026-04-24)
  v3.10 changes: saves .npz with fitted_params for V2 SED reconstruction
  v3.14 changes: adds version marker so launcher can detect silent reverts

RUNNER_VERSION = "v3.14"    # DO NOT REMOVE — launcher integrity check

Usage:
    python run_main_loop_subprocess.py X         # one model
    python run_main_loop_subprocess.py X,I       # comma-separated list
    python run_main_loop_subprocess.py all       # all 80 models

This script is invoked as a subprocess by Cell 36 of the v3 notebook.
Running it standalone (e.g. via `nohup`) is equivalent.

Features:
- Per-model guard: skips models whose final .dat is done.
- Per-step guard via skip_or_run(): skips gtsrcmaps/gtmodel calls whose
  outfile already exists.
- Includes the GCE / Fermi_bubble / isotropic gtmodel block (v3 E4 fix).
- Stays alive even if the parent Jupyter kernel dies.

Working directory: must be run from inside ./GCE_12yr_reproduce/
"""
import os, sys, time, warnings
import gc
warnings.filterwarnings("ignore")

# ---- imports (same as notebook Cell 1) ----
from GtApp import GtApp
import gt_apps as gt_apps
import matplotlib
matplotlib.use("Agg")  # no display in subprocess
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import xml.etree.ElementTree as ET
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.interpolate import interp1d, CubicSpline
from scipy.integrate import dblquad
import emcee
from chainconsumer import ChainConsumer
from multiprocessing import Pool

# ---- skip-helper (v3.4) ----

# ============================================================================
# Skip-if-output-exists helpers (haebarg port v3.4 — with integrity check)
# ----------------------------------------------------------------------------
# v3.4 change: the existence check is no longer enough. A mid-run gtsrcmaps
# crash leaves a partial .fits file on disk that fools v3.3's needs_run().
# v3.4 also validates the file is readable and structurally complete.
# ============================================================================
import os as _os

FORCE_RECOMPUTE = False  # flip to True to force re-running every guarded cell


def _check_file_integrity(path):
    """Return (is_healthy, reason). False means the file should be regenerated.

    Validates by extension:
      - .fits  : astropy can open, primary HDU exists
                 special case: srcmap files must have NDSKEYS header keyword
      - .npy   : numpy can load
      - .xml   : ElementTree can parse
      - other  : just check file exists and is non-empty
    """
    if not _os.path.exists(path):
        return False, "missing"
    if _os.path.getsize(path) == 0:
        return False, "zero-byte file"

    lower = path.lower()
    try:
        if lower.endswith(".fits"):
            from astropy.io import fits as _fits
            with _fits.open(path) as hdul:
                # Must have at least the primary HDU
                if len(hdul) == 0:
                    return False, "empty FITS (no HDUs)"
                # gtsrcmaps outputs are recognized by filename pattern
                # ('Extended_srcmap' or '_srcmap_'). Those need NDSKEYS.
                base = _os.path.basename(path)
                if "Extended_srcmap" in base or "_srcmap_" in base:
                    if "NDSKEYS" not in hdul[0].header:
                        return False, "srcmap missing NDSKEYS keyword (gtsrcmaps crashed mid-write)"
        elif lower.endswith(".npy"):
            import numpy as _np
            _np.load(path, allow_pickle=False, mmap_mode='r')
        elif lower.endswith(".xml"):
            import xml.etree.ElementTree as _ET
            _ET.parse(path)
        # other extensions: existence + non-zero is enough
    except Exception as e:
        return False, f"unreadable ({type(e).__name__}: {e})"

    return True, "OK"


def needs_run(*output_paths):
    """True if any listed output is missing, corrupt, or FORCE_RECOMPUTE.

    Unhealthy files are deleted automatically so the producing tool will
    write fresh output without needing clobber=yes argument tweaks.
    """
    if FORCE_RECOMPUTE:
        return True
    for p in output_paths:
        ok, reason = _check_file_integrity(p)
        if not ok:
            if reason != "missing":
                print(f"[v3.4 GUARD] removing unhealthy file: {p}  ({reason})", flush=True)
                try:
                    _os.remove(p)
                except OSError as _e:
                    print(f"[v3.4 GUARD] WARNING could not remove {p}: {_e}", flush=True)
            return True
    return False


def skip_or_run(app, label=""):
    """Run a Fermi GtApp only when its `outfile` is missing or unhealthy."""
    try:
        out = app['outfile']
    except Exception:
        app.run()
        return
    name = label or getattr(app, 'appName', '') or 'GtApp'
    if needs_run(out):
        print(f"[RUN ]  {name:<10} -> {out}", flush=True)
        app.run()
        # Verify the output is actually healthy before declaring DONE
        ok, reason = _check_file_integrity(out)
        if not ok:
            raise RuntimeError(f"{name} appeared to succeed but produced unhealthy output: {reason} ({out})")
        print(f"[DONE]  {name}", flush=True)
    else:
        print(f"[SKIP]  {name:<10} -> {out} (already exists)", flush=True)


def equatorial_to_galactic(ra, dec):
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame="icrs")
    return c.galactic.l.degree, c.galactic.b.degree


# =============================================================================
# Main loop body (transcribed from notebook in-cell version, with all v3 fixes)
# =============================================================================
MODEL_LIST_DEFAULT = ['X', 'XLIX', 'I', 'IV', 'V', 'VI', 'VII', 'IX',
                      'XV', 'XLI', 'XLVII', 'XLVIII', 'L', 'LII']


def run_one_model(model):
    print(f"\n{'='*60}", flush=True)
    print(f"==== MODEL {model} ====", flush=True)
    print(f"{'='*60}", flush=True)

    _final_dat = f'./GCE_model_{model}_12yr_cholis.dat'
    if not needs_run(_final_dat):
        print(f"[SKIP MODEL {model}] {_final_dat} already exists", flush=True)
        return True

    front = '_front'

    # ---- per-model XML generation ----
    new_sources = f"""
    <source name="bremss" type="DiffuseSource">
        <spectrum type="ConstantValue">
          <parameter error="0.02899312444" free="1" max="100" min="1" name="Value" scale="1" value="1" />
        </spectrum>
        <spatialModel file="./MapCubes/bremss_mapcube_model{model}.fits" type="MapCubeFunction" map_based_integral="true">
          <parameter free="0" max="1000" min="0" name="Normalization" scale="1" value="1" />
        </spatialModel>
      </source>
      <source name="ics" type="DiffuseSource">
        <spectrum type="ConstantValue">
          <parameter error="0.04073673429" free="1" max="100" min="1" name="Value" scale="1" value="1" />
        </spectrum>
        <spatialModel file="./MapCubes/ics_mapcube_model{model}.fits" type="MapCubeFunction" map_based_integral="true">
          <parameter free="0" max="1000" min="0" name="Normalization" scale="1" value="1" />
        </spatialModel>
      </source>
      <source name="pion" type="DiffuseSource">
        <spectrum type="ConstantValue">
          <parameter error="0.02899312444" free="1" max="100" min="1" name="Value" scale="1" value="1" />
        </spectrum>
        <spatialModel file="./MapCubes/pion_mapcube_model{model}.fits" type="MapCubeFunction" map_based_integral="true">
          <parameter free="0" max="1000" min="0" name="Normalization" scale="1" value="1" />
        </spatialModel>
      </source>
      <source name="GCE" type="DiffuseSource">
        <spectrum type="BrokenPowerLaw">
        <parameter free="0" max="1000.0" min="0.001" name="Prefactor" scale="1e-11" value="21"/>
        <parameter free="0" max="-1.0" min="-5." name="Index1" scale="1.0" value="-1.42"/>
        <parameter free="0" max="3000.0" min="30.0" name="BreakValue" scale="1.0" value="2006"/>
        <parameter free="0" max="-1.0" min="-5." name="Index2" scale="1.0" value="-2.63"/>
    </spectrum>
        <spatialModel file="./GCE_template_NFW2.fits" type="SpatialMap" map_based_integral="true">
        </spatialModel>
      </source>
      <source name="isotropic" type="DiffuseSource">
        <spectrum apply_edisp="true" file="./isotropic_spectrum_ff.txt" type="FileFunction">
          <parameter free="1" max="10" min="1" name="Normalization" scale="1" value="1" />
        </spectrum>
        <spatialModel type="ConstantValue">
          <parameter free="0" max="10" min="0" name="Value" scale="1" value="1" />
        </spatialModel>
      </source>
      <source name="Fermi_bubble" type="DiffuseSource">
        <spectrum apply_edisp="true" file="./fermi_bubble_spectrum.txt" type="FileFunction">
          <parameter free="1" max="1e+10" min="0" name="Normalization" scale="1e-3" value="5" />
        </spectrum>
        <spatialModel file="./Fermi_Bubbles_template.fits" type="SpatialMap" map_based_integral="true">
        </spatialModel>
      </source>
    """

    # full XML for psc-only srcmap
    new_sources_root = ET.fromstring(f"<sources>{new_sources}</sources>")
    tree = ET.parse('./GC_analysis_sanghwan/Model/GC_psc_model_DR2.xml')
    root = tree.getroot()
    for new_source in new_sources_root:
        root.append(new_source)
    tree.write(f'./GC_analysis_sanghwan/Model/GC_model{model}_test.xml',
               encoding='utf-8', xml_declaration=True)

    # full XML for Extended srcmap (built from empty_model.xml)
    new_sources_root = ET.fromstring(f"<sources>{new_sources}</sources>")
    tree = ET.parse('./GC_analysis_sanghwan/Model/empty_model.xml')
    root = tree.getroot()
    for new_source in new_sources_root:
        root.append(new_source)
    tree.write(f'./GC_analysis_sanghwan/Model/GC_Extended_model{model}_test.xml',
               encoding='utf-8', xml_declaration=True)

    # ---- gtsrcmaps × 2 (convol yes/no) ----
    for convol_val, convol_suffix in [('yes', ''), ('no', '_no_convol')]:
        srcMaps = GtApp('gtsrcmaps', 'Likelihood')
        srcMaps['scfile']  = '../GCE_allsky_data/lat_spacecraft_merged_12yr.fits'
        srcMaps['expcube'] = f'./GC_analysis_sanghwan/Allsky_ltcube_12yr{front}_clean.fits'
        srcMaps['cmap']    = f'./GC_analysis_sanghwan/GC_ccube_12yr{front}_clean.fits'
        srcMaps['bexpmap'] = f'./GC_analysis_sanghwan/Allsky_expcube_edge_12yr{front}_clean.fits'
        srcMaps['srcmdl']  = f'./GC_analysis_sanghwan/Model/GC_Extended_model{model}_test.xml'
        srcMaps['outfile'] = f'./GC_analysis_sanghwan/GC_Extended_srcmap_12yr{front}_clean_model_{model}{convol_suffix}.fits'
        srcMaps['irfs']    = 'P8R3_CLEAN_V3'
        srcMaps['convol']  = convol_val
        srcMaps['evtype']  = 1
        skip_or_run(srcMaps, label=f'gtsrcmaps convol={convol_val}')

    # ---- per-component XMLs for pion/bremss/ics ----
    for component in ['bremss', 'ics', 'pion']:
        comp_xml = f"""
        <source name="{component}" type="DiffuseSource">
            <spectrum type="ConstantValue">
              <parameter error="0.02899312444" free="1" max="100" min="1" name="Value" scale="1" value="1" />
            </spectrum>
            <spatialModel file="./MapCubes/{component}_mapcube_model{model}.fits" type="MapCubeFunction" map_based_integral="true">
              <parameter free="0" max="1000" min="0" name="Normalization" scale="1" value="1" />
            </spatialModel>
          </source>
        """
        new_sources_root = ET.fromstring(f"<sources>{comp_xml}</sources>")
        tree = ET.parse('./GC_analysis_sanghwan/Model/empty_model.xml')
        root = tree.getroot()
        for new_source in new_sources_root:
            root.append(new_source)
        tree.write(f'./GC_analysis_sanghwan/Model/GC_{component}_model{model}_test.xml',
                   encoding='utf-8', xml_declaration=True)

    # ---- gtmodel for pion/bremss/ics × convol yes/no ----
    for convol_val, convol_suffix in [('yes', ''), ('no', '_no_convol')]:
        for component in ['pion', 'bremss', 'ics']:
            gtmodel = GtApp('gtmodel', 'Likelihood')
            gtmodel['irfs']    = 'P8R3_CLEAN_V3'
            gtmodel['outtype'] = 'ccube'
            gtmodel['srcmdl']  = f'./GC_analysis_sanghwan/Model/GC_{component}_model{model}_test.xml'
            gtmodel['outfile'] = f'./GC_analysis_sanghwan/GC_{component}_model{model}_12yr{front}_clean{convol_suffix}.fits'
            gtmodel['expcube'] = f'./GC_analysis_sanghwan/Allsky_ltcube_12yr{front}_clean.fits'
            gtmodel['bexpmap'] = f'./GC_analysis_sanghwan/Allsky_expcube_edge_12yr{front}_clean.fits'
            gtmodel['convol']  = convol_val
            gtmodel['evtype']  = 1
            gtmodel['srcmaps'] = f'./GC_analysis_sanghwan/GC_Extended_srcmap_12yr{front}_clean_model_{model}{convol_suffix}.fits'
            skip_or_run(gtmodel, label=f'gtmodel {component} convol={convol_val}')

    # ---- v3 E4 fix: gtmodel for GCE / Fermi_bubble / isotropic ----
    extra_components = [
        ('GCE',          'GCE'),
        ('Fermi_bubble', 'fermi_bubble'),
        ('isotropic',    'isotropic'),
    ]
    extra_xml_frags = {
        'GCE': """
    <source name="GCE" type="DiffuseSource">
        <spectrum type="BrokenPowerLaw">
        <parameter free="0" max="1000.0" min="0.001" name="Prefactor" scale="1e-11" value="21"/>
        <parameter free="0" max="-1.0" min="-5." name="Index1" scale="1.0" value="-1.42"/>
        <parameter free="0" max="3000.0" min="30.0" name="BreakValue" scale="1.0" value="2006"/>
        <parameter free="0" max="-1.0" min="-5." name="Index2" scale="1.0" value="-2.63"/>
    </spectrum>
        <spatialModel file="./GCE_template_NFW2.fits" type="SpatialMap" map_based_integral="true">
        </spatialModel>
      </source>
""",
        'Fermi_bubble': """
    <source name="Fermi_bubble" type="DiffuseSource">
        <spectrum apply_edisp="true" file="./fermi_bubble_spectrum.txt" type="FileFunction">
          <parameter free="1" max="1e+10" min="0" name="Normalization" scale="1e-3" value="5" />
        </spectrum>
        <spatialModel file="./Fermi_Bubbles_template.fits" type="SpatialMap" map_based_integral="true">
        </spatialModel>
      </source>
""",
        'isotropic': """
    <source name="isotropic" type="DiffuseSource">
        <spectrum apply_edisp="true" file="./isotropic_spectrum_ff.txt" type="FileFunction">
          <parameter free="1" max="10" min="1" name="Normalization" scale="1" value="1" />
        </spectrum>
        <spatialModel type="ConstantValue">
          <parameter free="0" max="10" min="0" name="Value" scale="1" value="1" />
        </spatialModel>
      </source>
""",
    }
    for src_name, fname in extra_components:
        xml_path = f'./GC_analysis_sanghwan/Model/GC_{fname}_singlecomp.xml'
        if needs_run(xml_path):
            new_sources_root = ET.fromstring(f"<sources>{extra_xml_frags[src_name]}</sources>")
            tree = ET.parse('./GC_analysis_sanghwan/Model/empty_model.xml')
            root = tree.getroot()
            for s in new_sources_root:
                root.append(s)
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        for convol_val, convol_suffix in [('yes', ''), ('no', '_no_convol')]:
            gtm = GtApp('gtmodel', 'Likelihood')
            gtm['irfs']    = 'P8R3_CLEAN_V3'
            gtm['outtype'] = 'ccube'
            gtm['srcmdl']  = xml_path
            gtm['outfile'] = f'./GC_analysis_sanghwan/GC_{fname}_model_12yr{front}_clean{convol_suffix}.fits'
            gtm['expcube'] = f'./GC_analysis_sanghwan/Allsky_ltcube_12yr{front}_clean.fits'
            gtm['bexpmap'] = f'./GC_analysis_sanghwan/Allsky_expcube_edge_12yr{front}_clean.fits'
            gtm['convol']  = convol_val
            gtm['evtype']  = 1
            gtm['srcmaps'] = f'./GC_analysis_sanghwan/GC_Extended_srcmap_12yr{front}_clean_model_{model}{convol_suffix}.fits'
            skip_or_run(gtm, label=f'gtmodel {fname} convol={convol_val}')

    # ========================================================================
    # emcee section — verbatim from Sanghwan original (transcribed)
    # ========================================================================

    def roi_solid_angle(delta_l_deg, delta_b_deg, b_deg):
        return np.radians(delta_l_deg) * np.radians(delta_b_deg) * np.cos(np.radians(b_deg))

    raw_map = fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr{front}_clean.fits')
    w = WCS(raw_map[0].header).dropaxis(2)
    width, height = np.shape(raw_map[0].data[0])
    steradian_per_pixel = np.zeros([width, height])
    for i in range(height):
        for j in range(width):
            l, b = w.wcs_pix2world(j, i, 0)
            steradian_per_pixel[i, j] = roi_solid_angle(0.1, 0.1, b)

    disk_mask = np.load('./GC_analysis_sanghwan/Model/GC_disk_mask_60x60_definitions.npy')[100:500, 100:500]

    E_bounds = fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr{front}_clean.fits')[1].data
    E       = np.array([np.sqrt(b[2]*b[1]*1e-6)*1e-3 for b in E_bounds])
    delta_E = np.array([(b[2] - b[1])*1e-6 for b in E_bounds])

    exp_cube = (
        fits.open(f'./GC_analysis_sanghwan/GC_expcube_center_12yr{front}_clean.fits')[0].data[:, 100:500, 100:500]
        * steradian_per_pixel[100:500, 100:500]
    )

    def _flux_per_bin(filename):
        out = np.zeros(len(E_bounds))
        d = fits.open(filename)[0].data
        for i in range(len(E_bounds)):
            out[i] = np.sum(disk_mask * (d[i][100:500, 100:500] / exp_cube[i])) / np.sum(disk_mask)
        return out

    pion      = _flux_per_bin(f'./GC_analysis_sanghwan/GC_pion_model{model}_12yr{front}_clean_no_convol.fits')
    bremss    = _flux_per_bin(f'./GC_analysis_sanghwan/GC_bremss_model{model}_12yr{front}_clean_no_convol.fits')
    ics       = _flux_per_bin(f'./GC_analysis_sanghwan/GC_ics_model{model}_12yr{front}_clean_no_convol.fits')
    GCE       = _flux_per_bin(f'./GC_analysis_sanghwan/GC_GCE_model_12yr{front}_clean_no_convol.fits')
    bubble    = _flux_per_bin(f'./GC_analysis_sanghwan/GC_fermi_bubble_model_12yr{front}_clean_no_convol.fits')
    isotropic = _flux_per_bin(f'./GC_analysis_sanghwan/GC_isotropic_model_12yr{front}_clean_no_convol.fits')

    counts_per_exp = np.zeros(len(E_bounds))
    counts_per_exp_err = np.zeros(len(E_bounds))
    ccube_data = fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr{front}_clean.fits')[0].data
    for i in range(len(E_bounds)):
        counts_per_exp[i] = np.sum(disk_mask * (ccube_data[i][100:500, 100:500] / exp_cube[i])) / np.sum(disk_mask)
        counts_per_exp_err[i] = np.sqrt(
            np.sum((np.sqrt(disk_mask * ccube_data[i][100:500, 100:500]) / exp_cube[i])**2)
        ) / np.sum(disk_mask)

    # Constraint files for chi^2 priors
    bubble_constraints = np.loadtxt('./GC_analysis_sanghwan/Model/bubble_constraints.txt')
    bubble_fluxint     = interp1d(bubble_constraints[:,0], bubble_constraints[:,1], fill_value='extrapolate', kind='quadratic')
    bubble_lower_errint= interp1d(bubble_constraints[:,0], bubble_constraints[:,2], fill_value='extrapolate', kind='quadratic')
    bubble_upper_errint= interp1d(bubble_constraints[:,0], bubble_constraints[:,3], fill_value='extrapolate', kind='quadratic')
    bubble_flux_data        = bubble_fluxint(E)
    bubble_lower_error_data = bubble_lower_errint(E)
    bubble_upper_error_data = bubble_upper_errint(E)

    iso_constraints = np.loadtxt('./GC_analysis_sanghwan/Model/iso_constraints_full_err.txt')
    isotropic_fluxint     = interp1d(iso_constraints[:,0], iso_constraints[:,1], fill_value='extrapolate', kind='quadratic')
    isotropic_lower_errint= interp1d(iso_constraints[:,0], iso_constraints[:,2], fill_value='extrapolate', kind='quadratic')
    isotropic_upper_errint= interp1d(iso_constraints[:,0], iso_constraints[:,3], fill_value='extrapolate', kind='quadratic')
    isotropic_flux_data         = (E**2) * isotropic_fluxint(E)
    isotropic_lower_error_data  = (E**2) * isotropic_lower_errint(E)
    isotropic_upper_error_data  = (E**2) * isotropic_upper_errint(E)

    # ---- Likelihood class ----
    # OPTIMIZATION: use scipy.special.gammaln instead of Python-level log_factorial
    # (88x speedup, mathematically identical)
    from scipy.special import gammaln

    class Likelihood:
        def __init__(self, model_name, energy_bin):
            self.model = model_name
            self.energy_bin = energy_bin
            self.data        = fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr{front}_clean.fits')[0].data[energy_bin, 100:500, 100:500]
            self.pion_bremss = (fits.open(f'./GC_analysis_sanghwan/GC_pion_model{model_name}_12yr{front}_clean.fits')[0].data[energy_bin, 100:500, 100:500]
                              + fits.open(f'./GC_analysis_sanghwan/GC_bremss_model{model_name}_12yr{front}_clean.fits')[0].data[energy_bin, 100:500, 100:500])
            self.ics    = fits.open(f'./GC_analysis_sanghwan/GC_ics_model{model_name}_12yr{front}_clean.fits')[0].data[energy_bin, 100:500, 100:500]
            self.GCE    = fits.open(f'./GC_analysis_sanghwan/GC_GCE_model_12yr{front}_clean.fits')[0].data[energy_bin, 100:500, 100:500]
            self.bubble = fits.open(f'./GC_analysis_sanghwan/GC_fermi_bubble_model_12yr{front}_clean.fits')[0].data[energy_bin, 100:500, 100:500]
            self.iso    = fits.open(f'./GC_analysis_sanghwan/GC_isotropic_model_12yr{front}_clean.fits')[0].data[energy_bin, 100:500, 100:500]

            self.E       = E
            self.delta_E = delta_E
            self.exp_cube = (fits.open(f'./GC_analysis_sanghwan/GC_expcube_center_12yr{front}_clean.fits')[0].data[energy_bin]
                             * steradian_per_pixel)[100:500, 100:500]

            psc_mask  = np.load('./GC_analysis_sanghwan/Model/GC_mask_60x60_definitions_DR2.npy')[energy_bin, 100:500, 100:500]
            disk_mask = np.load('./GC_analysis_sanghwan/Model/GC_disk_mask_60x60_definitions.npy')[100:500, 100:500]
            self.full_mask = psc_mask * disk_mask

            # Pre-compute log_factorial(observed) since observed is fixed
            obs_flat = self.data[self.full_mask == 1].astype(int)
            self._log_fact_obs = gammaln(obs_flat + 1)

            # Pre-load no_convol fits (for isotropic/bubble integrand)
            self._iso_noconv    = fits.open(f'./GC_analysis_sanghwan/GC_isotropic_model_12yr{front}_clean_no_convol.fits')[0].data[energy_bin, 100:500, 100:500]
            self._bub_noconv    = fits.open(f'./GC_analysis_sanghwan/GC_fermi_bubble_model_12yr{front}_clean_no_convol.fits')[0].data[energy_bin, 100:500, 100:500]

        def likelihood_constrained(self, parameter_set):
            pb_p, ics_p, gce_p, bub_p, iso_p = parameter_set
            expected_pixel = pb_p*self.pion_bremss + ics_p*self.ics + gce_p*self.GCE + iso_p*self.iso + bub_p*self.bubble
            expected_pixel = expected_pixel[self.full_mask == 1]
            observed_pixel = self.data[self.full_mask == 1]

            if (expected_pixel < 0).any():
                return np.inf

            observed_log_expected = observed_pixel * np.log(expected_pixel)
            lhd = 2*(expected_pixel - observed_log_expected + self._log_fact_obs)

            # Bubble chi^2
            bubble = np.sum(self.full_mask * (self._bub_noconv / self.exp_cube)) * bub_p / np.sum(self.full_mask)
            bubble_sed = (self.E[self.energy_bin]**2) * bubble / self.delta_E[self.energy_bin]
            larger_error = max(bubble_upper_error_data[self.energy_bin], bubble_lower_error_data[self.energy_bin])
            if bubble_flux_data[self.energy_bin] < bubble_sed:
                chi2_bubble = ((bubble_sed - bubble_flux_data[self.energy_bin]) / bubble_upper_error_data[self.energy_bin])**2
            elif bubble_flux_data[self.energy_bin] > bubble_sed:
                chi2_bubble = ((bubble_sed - bubble_flux_data[self.energy_bin]) / bubble_lower_error_data[self.energy_bin])**2
            else:
                chi2_bubble = ((bubble_sed - bubble_flux_data[self.energy_bin]) / larger_error)**2

            # Isotropic chi^2
            isotropic = np.sum(self.full_mask * (self._iso_noconv / self.exp_cube)) * iso_p / np.sum(self.full_mask)
            isotropic_sed = (self.E[self.energy_bin]**2) * isotropic / self.delta_E[self.energy_bin]
            iso_larger = max(isotropic_lower_error_data[self.energy_bin], isotropic_upper_error_data[self.energy_bin])
            if isotropic_flux_data[self.energy_bin] < isotropic_sed:
                chi2_isotropic = ((isotropic_flux_data[self.energy_bin] - isotropic_sed) / isotropic_lower_error_data[self.energy_bin])**2
            elif isotropic_flux_data[self.energy_bin] > isotropic_sed:
                chi2_isotropic = ((isotropic_flux_data[self.energy_bin] - isotropic_sed) / isotropic_upper_error_data[self.energy_bin])**2
            else:
                chi2_isotropic = ((isotropic_flux_data[self.energy_bin] - isotropic_sed) / iso_larger)**2

            return np.sum(lhd) + chi2_bubble + chi2_isotropic

    # cache Likelihood instances per (model, bin) to avoid redundant fits.open
    _lh_cache = {}
    def get_likelihood(model_name, energy_bin):
        key = (model_name, energy_bin)
        if key not in _lh_cache:
            _lh_cache[key] = Likelihood(model_name, energy_bin)
        return _lh_cache[key]

    def log_likelihood(params, energy_bin):
        return -0.5 * get_likelihood(model, energy_bin).likelihood_constrained(params)

    def log_prior(params):
        for p in params:
            if p < 0:
                return -np.inf
        return 0.0

    def log_probability(params, energy_bin):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params, energy_bin)

    def run_mcmc_for_bin(energy_bin):
        ndim = 5; nwalkers = 100; nsteps = 1000; burn_in_steps = 400
        t0 = time.time()
        print(f"  [emcee bin {energy_bin}] starting", flush=True)
        initial_params = np.vstack([
            np.random.uniform(0, 3, [nwalkers]),
            np.random.uniform(0, 3, [nwalkers]),
            np.random.uniform(0, 3, [nwalkers]),
            np.random.uniform(0, 10, [nwalkers]),
            np.random.uniform(0, 10, [nwalkers]),
        ]).T
        # NOTE: serial sampling (no Pool) — gammaln optimization makes this fast enough,
        # and avoids fork-related Fermi tools issues if any.
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(energy_bin,))
        sampler.run_mcmc(initial_params, nsteps, progress=False)

        log_prob_samples = sampler.get_log_prob(discard=burn_in_steps, flat=True)
        max_idx = np.argmax(log_prob_samples)
        best_fit = sampler.get_chain(discard=burn_in_steps, flat=True)[max_idx]
        flat_samples = sampler.get_chain(discard=burn_in_steps, flat=True)
        max_lhd = log_prob_samples[max_idx]
        upper_1sigma = np.percentile(flat_samples, 84, axis=0)
        lower_1sigma = np.percentile(flat_samples, 16, axis=0)

        elapsed = time.time() - t0
        print(f"  [emcee bin {energy_bin}] done in {elapsed:.1f}s, max_lhd={max_lhd:.2f}", flush=True)
        return (best_fit.T,
                np.median(flat_samples, axis=0).T,
                np.std(flat_samples, axis=0, ddof=1).T,
                max_lhd, upper_1sigma, lower_1sigma)

    n = len(E)
    fitted_params       = np.ones(n*5)
    fitted_params_std   = np.zeros(n*5)
    fitted_params_upper = np.zeros(n*5)
    fitted_params_lower = np.zeros(n*5)
    max_likelihood      = np.zeros(n)

    for i in range(n):
        max_v, med_v, std_v, mxl, up_v, lo_v = run_mcmc_for_bin(i)
        for k in range(5):
            fitted_params[n*k:n*(k+1)][i]       = max_v[k]
            fitted_params_std[n*k:n*(k+1)][i]   = std_v[k]
            fitted_params_upper[n*k:n*(k+1)][i] = up_v[k]
            fitted_params_lower[n*k:n*(k+1)][i] = lo_v[k]
        max_likelihood[i] = mxl

    fitted = fitted_params
    fitted_errors = fitted_params_std
    np.savetxt(_final_dat, np.vstack([
        E,
        fitted[n*2:n*3] * GCE * (E**2) / delta_E,
        (fitted_errors[n*2:n*3] * GCE) * (E**2) / delta_E,
        fitted_params_lower[n*2:n*3] * GCE * (E**2) / delta_E,
        fitted_params_upper[n*2:n*3] * GCE * (E**2) / delta_E,
    ]).T)
    np.savetxt(f'./GCE_model_{model}_12yr_cholis_likelihood_value', max_likelihood)

    # v3.10: also save full fit result as .npz so V2 can reconstruct
    # per-component SEDs (pion+bremss, ICS, bubble, iso) with fitted
    # coefficients, matching Sanghwan's original Cell 42.
    # Layout: fitted_params is 5*n-length flat array, grouped as
    #   [0:n]       c_pion_bremss (per-bin)
    #   [n:2n]      c_ics
    #   [2n:3n]     c_gce
    #   [3n:4n]     c_bubble
    #   [4n:5n]     c_isotropic
    np.savez_compressed(
        f'./GCE_model_{model}_12yr_cholis_fit.npz',
        E=E, delta_E=delta_E,
        fitted_params=fitted_params,
        fitted_params_std=fitted_params_std,
        fitted_params_lower=fitted_params_lower,
        fitted_params_upper=fitted_params_upper,
        max_likelihood=max_likelihood,
        # template flux arrays (non-convol, for SED reconstruction)
        pion=pion, bremss=bremss, ics=ics,
        GCE=GCE, bubble=bubble, isotropic=isotropic,
    )

    print(f"[MODEL {model} done] -> {_final_dat}", flush=True)
    return True


if __name__ == "__main__":
    args = list(sys.argv[1:])
    if "--force" in args:
        FORCE_RECOMPUTE = True
        args.remove("--force")
    if not args:
        print("Usage: python run_main_loop_subprocess.py MODEL_NAME | all  [--force]", file=sys.stderr)
        sys.exit(1)
    arg = args[0]
    # v3.8 parallel: accept comma-separated list (e.g. 'X,I,XV,XLIX')
    if arg.lower() == 'all':
        targets = MODEL_LIST_DEFAULT
    elif ',' in arg:
        targets = [x.strip() for x in arg.split(',') if x.strip()]
    else:
        targets = [arg]

    overall_t0 = time.time()
    for model in targets:
        t0 = time.time()
        try:
            run_one_model(model)
        except Exception as e:
            import traceback
            print(f"\n[ERROR MODEL {model}] {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            print(f"[continuing with next model]\n", flush=True)
            continue
        print(f"[model={model}] elapsed {(time.time()-t0)/60:.1f} min", flush=True)

        # --- v3.9 memory cleanup between models ---
        # Without this, ~160 MB of numpy arrays per model + uncloseable fits
        # mmap regions accumulate, triggering OOM around the 3rd model when
        # running many workers in parallel.
        gc.collect()
        gc.collect()  # second pass for cyclic refs in astropy fits/emcee
        try:
            with open('/proc/self/status') as _fps:
                for _line in _fps:
                    if _line.startswith('VmRSS:'):
                        print(f"[mem after model={model}] {_line.strip()}", flush=True)
                        break
        except Exception:
            pass
        time.sleep(1)  # give kernel a moment to reclaim pages

    print(f"\n=== Total elapsed: {(time.time()-overall_t0)/60:.1f} min ===", flush=True)
