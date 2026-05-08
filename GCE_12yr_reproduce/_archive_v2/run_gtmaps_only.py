#!/usr/bin/env python3
"""
Standalone main loop for the Sanghwan GC analysis pipeline.
Executes all 80-model-grid analysis tasks (XML gen, gtsrcmaps, gtmodel,
likelihood fitting, plotting, .dat saving) in a fresh Python process,
to avoid Jupyter kernel silent-kill issues observed during astropy/emcee operations.

Usage:
    cd /home/haebarg/GCE-Chi-square-fitting/GCE_12yr_reproduce/
    python main_loop.py 2>&1 | tee main_loop.log

To run in background:
    nohup python main_loop.py > main_loop.log 2>&1 &
    tail -f main_loop.log
"""
import sys, os, pickle, warnings
warnings.filterwarnings("ignore")

# === Imports (mirrors notebook cell 1) ===
from GtApp import GtApp
import gt_apps as gt_apps
import matplotlib
matplotlib.use("Agg")  # No interactive backend in subprocess
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style, wcsaxes
plt.style.use(astropy_mpl_style)
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from astropy.table import Table, hstack
import xml.etree.ElementTree as ET
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.interpolate import interp1d, CubicSpline
from scipy.integrate import dblquad
import emcee
from chainconsumer import ChainConsumer
from multiprocessing import Pool
import time

# === Coordinate helpers (notebook cells 4, 5) ===
def galactic_to_equatorial(l, b):
    c = SkyCoord(l=l*u.degree, b=b*u.degree, frame="galactic")
    return c.icrs.ra.degree, c.icrs.dec.degree

def equatorial_to_galactic(ra, dec):
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame="icrs")
    return c.galactic.l.degree, c.galactic.b.degree

# === Load source classification from pickle (notebook cell 14) ===
print("Loading source classification...", flush=True)
with open("./GC_analysis_sanghwan/Model/source_classification.pkl", "rb") as f:
    _d = pickle.load(f)
ra_dec_values         = _d["ra_dec_values"]
spatial_ra_dec_values = _d["spatial_ra_dec_values"]
not_sig_ra_dec_values = _d["not_sig_ra_dec_values"]
sig_ra_dec_values     = _d["sig_ra_dec_values"]
print(f"  point={{len(ra_dec_values)}}, spatial={{len(spatial_ra_dec_values)}}, sig={{len(sig_ra_dec_values)}}, not_sig={{len(not_sig_ra_dec_values)}}", flush=True)

# === E_bounds, E, delta_E (notebook cell 18) ===
E_bounds = fits.open("./GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits")[1].data
E = np.array([np.sqrt(E_bounds[i][2]*E_bounds[i][1]*1e-6)*1e-3 for i in range(len(E_bounds))])
delta_E = np.array([(E_bounds[i][2] - E_bounds[i][1])*1e-6 for i in range(len(E_bounds))])
print(f"  energy bins: {{len(E_bounds)}}, range {{E[0]:.3f}}-{{E[-1]:.3f}} GeV", flush=True)

# === masking() function (notebook cell 20 / original cell 28) ===
def masking(significance, locations, energy, image_file):
    data = np.array([
        [0.275, 0.357, 1.125, 3.75], [0.357, 0.464, 0.975, 3.25],
        [0.464, 0.603, 0.788, 2.63], [0.603, 0.784, 0.600, 2.00],
        [0.784, 1.02,  0.450, 1.50], [1.02,  1.32,  0.375, 1.25],
        [1.32,  1.72,  0.300, 1.00], [1.72,  2.24,  0.225, 0.750],
        [2.24,  2.91,  0.188, 0.625], [2.91,  3.78,  0.162, 0.540],
        [3.78,  4.91,  0.125, 0.417], [4.91,  10.8,  0.100, 0.333],
        [10.8,  23.7,  0.060, 0.200], [23.7,  51.9,  0.053, 0.175],
    ])
    mask_energy = np.sqrt(data[:,0]*data[:,1])
    mask_small = data[:,2]*0.9
    mask_large = data[:,3]*0.9
    mask_small_int = interp1d(mask_energy, mask_small, fill_value="extrapolate")
    mask_large_int = interp1d(mask_energy, mask_large, fill_value="extrapolate")
    mask_small_size = float(mask_small_int(energy)); mask_small_size = max(mask_small_size, 0)
    mask_large_size = float(mask_large_int(energy)); mask_large_size = max(mask_large_size, 0)
    
    hdulist = fits.open(image_file)
    data_arr = hdulist[0].data[0]
    header = hdulist[0].header
    masked_data = np.ones(np.shape(data_arr))
    wcs = WCS(header).dropaxis(2)
    
    for points in locations:
        l, b = equatorial_to_galactic(float(points[1]), float(points[2]))
        gc = SkyCoord(l=l*u.degree, b=b*u.degree, frame="galactic")
        px, py = wcs.world_to_pixel(gc)
        x_c = np.round(float(px), 0); y_c = np.round(float(py), 0)
        ps_x = wcs.wcs.cdelt[0]; ps_y = wcs.wcs.cdelt[1]
        rad_deg = mask_large_size if significance == 1 else mask_small_size
        rad_pix = min(abs(rad_deg/ps_x), abs(rad_deg/ps_y))
        y, x = np.ogrid[:data_arr.shape[0], :data_arr.shape[1]]
        mask = (x - x_c)**2 + (y - y_c)**2 < rad_pix**2
        masked_data[mask] = 0
    return masked_data

# === model_list (notebook cell 27) ===
def generate_roman_numerals(n):
    roman_map = [(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),
                 (50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]
    def to_roman(num):
        r = ""
        for v, s in roman_map:
            while num >= v:
                r += s; num -= v
        return r
    return [to_roman(i) for i in range(1, n+1)]

model_list = ["X", "XLIX", "I", "IV", "V", "VI", "VII", "IX", "XV", "XLI", "XLVII", "XLVIII", "L", "LII"]
print(f"model_list: {{model_list}}", flush=True)

# === Main loop body (notebook cell 29) ===
#model='I'
for model in model_list:
    print(f'\n==== MODEL {model} START ====', flush=True)
    # Define the elements to add
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
        <parameter free="0" max="1000.0" min="0.001" name="Prefactor" scale="1e-11" value="7*3"/>
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
    
    # Parse the new sources XML string
    new_sources_tree = ET.ElementTree(ET.fromstring(f"<sources>{new_sources}</sources>"))
    new_sources_root = new_sources_tree.getroot()
    
    # Parse the existing XML file
    tree = ET.parse('./GC_analysis_sanghwan/Model/GC_psc_model_DR2.xml')
    root = tree.getroot()
    
    # Append the new sources to the root element of the existing file
    for new_source in new_sources_root:
        root.append(new_source)
    
    # Save the modified XML to a new file
    tree.write(f'./GC_analysis_sanghwan/Model/GC_model{model}_test.xml', encoding='utf-8', xml_declaration=True)
    #Creating total xml model file for srcmap
    
    # Define the elements to add
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
        <parameter free="0" max="1000.0" min="0.001" name="Prefactor" scale="1e-11" value="7*3"/>
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
    
    # Parse the new sources XML string
    new_sources_tree = ET.ElementTree(ET.fromstring(f"<sources>{new_sources}</sources>"))
    new_sources_root = new_sources_tree.getroot()
    
    # Parse the existing XML file
    tree = ET.parse('./GC_analysis_sanghwan/Model/empty_model.xml')
    root = tree.getroot()
    
    # Append the new sources to the root element of the existing file
    for new_source in new_sources_root:
        root.append(new_source)
    
    # Save the modified XML to a new file
    tree.write(f'./GC_analysis_sanghwan/Model/GC_Extended_model{model}_test.xml', encoding='utf-8', xml_declaration=True)
    
    
    front='_front'
    convol=''
    srcMaps = GtApp('gtsrcmaps', 'Likelihood')
    srcMaps['scfile']='../GCE_allsky_data/lat_spacecraft_merged_12yr.fits'
    srcMaps['expcube']=f'./GC_analysis_sanghwan/Allsky_ltcube_12yr{front}{"_clean"}.fits'
    srcMaps['cmap']=f'./GC_analysis_sanghwan/GC_ccube_12yr{front}{"_clean"}.fits'
    srcMaps['bexpmap']=f'./GC_analysis_sanghwan/Allsky_expcube_edge_12yr{front}{"_clean"}.fits'
    srcMaps['srcmdl']=f'./GC_analysis_sanghwan/Model/GC_Extended_model{model}_test.xml'
    srcMaps['outfile']=f'./GC_analysis_sanghwan/GC_Extended_srcmap_12yr{front}{"_clean"}_model_{model}{convol}.fits'
    srcMaps['irfs']='P8R3_CLEAN_V3'
    srcMaps['convol']='yes'
    srcMaps['evtype']=1
    #srcMaps['resample']='no'
    print(f'  [model={model}] running gtsrcmaps (convol=yes)...', flush=True)
    srcMaps.run();
    
    convol='_no_convol'
    srcMaps = GtApp('gtsrcmaps', 'Likelihood')
    srcMaps['scfile']='../GCE_allsky_data/lat_spacecraft_merged_12yr.fits'
    srcMaps['expcube']=f'./GC_analysis_sanghwan/Allsky_ltcube_12yr{front}{"_clean"}.fits'
    srcMaps['cmap']=f'./GC_analysis_sanghwan/GC_ccube_12yr{front}{"_clean"}.fits'
    srcMaps['bexpmap']=f'./GC_analysis_sanghwan/Allsky_expcube_edge_12yr{front}{"_clean"}.fits'
    srcMaps['srcmdl']=f'./GC_analysis_sanghwan/Model/GC_Extended_model{model}_test.xml'
    srcMaps['outfile']=f'./GC_analysis_sanghwan/GC_Extended_srcmap_12yr{front}{"_clean"}_model_{model}{convol}.fits'
    srcMaps['irfs']='P8R3_CLEAN_V3'
    srcMaps['convol']='no'
    srcMaps['evtype']=1
    #srcMaps['resample']='no'
    print(f'  [model={model}] running gtsrcmaps (convol=no)...', flush=True)
    srcMaps.run();
    
    print(f'  [model={model}] generating single-component XMLs for pion/bremss/ics...', flush=True)
    for component in ['bremss', 'ics', 'pion']:
        # Define the elements to add
        new_sources = f"""
        <source name="{component}" type="DiffuseSource">
            <spectrum type="ConstantValue">
              <parameter error="0.02899312444" free="1" max="100" min="1" name="Value" scale="1" value="1" />
            </spectrum>
            <spatialModel file="./MapCubes/{component}_mapcube_model{model}.fits" type="MapCubeFunction" map_based_integral="true">
              <parameter free="0" max="1000" min="0" name="Normalization" scale="1" value="1" />
            </spatialModel>
          </source>
        """
        
        # Parse the new sources XML string
        new_sources_tree = ET.ElementTree(ET.fromstring(f"<sources>{new_sources}</sources>"))
        new_sources_root = new_sources_tree.getroot()
        
        # Parse the existing XML file
        tree = ET.parse('./GC_analysis_sanghwan/Model/empty_model.xml')
        root = tree.getroot()
        
        # Append the new sources to the root element of the existing file
        for new_source in new_sources_root:
            root.append(new_source)
        
        # Save the modified XML to a new file
        tree.write(f'./GC_analysis_sanghwan/Model/GC_{component}_model{model}_test.xml', encoding='utf-8', xml_declaration=True)
    
    
    convol=''
    print(f'  [model={model}] running gtmodel for pion/bremss/ics (convol=yes)...', flush=True)
    for component in ['pion', 'bremss', 'ics']:
        gtmodel = GtApp('gtmodel', 'Likelihood')
        gtmodel['irfs']='P8R3_CLEAN_V3'
        gtmodel['outtype']='ccube'
        gtmodel['srcmdl']=f'./GC_analysis_sanghwan/Model/GC_{component}_model{model}_test.xml'
        gtmodel['outfile']=f'./GC_analysis_sanghwan/GC_{component}_model{model}_12yr{front}_clean{convol}.fits'
        gtmodel['expcube']=f'./GC_analysis_sanghwan/Allsky_ltcube_12yr{front}{"_clean"}.fits'
        gtmodel['bexpmap']=f'./GC_analysis_sanghwan/Allsky_expcube_edge_12yr{front}{"_clean"}.fits'
        gtmodel['convol']='yes'
        #gtmodel['resample']='no'
        gtmodel['evtype']=1
        gtmodel['srcmaps']=f'./GC_analysis_sanghwan/GC_Extended_srcmap_12yr{front}{"_clean"}_model_{model}{convol}.fits'
        gtmodel.run()
    
    
    convol='_no_convol'
    print(f'  [model={model}] running gtmodel for pion/bremss/ics (convol=no)...', flush=True)
    for component in ['pion', 'bremss', 'ics']:
        gtmodel = GtApp('gtmodel', 'Likelihood')
        gtmodel['irfs']='P8R3_CLEAN_V3'
        gtmodel['outtype']='ccube'
        gtmodel['srcmdl']=f'./GC_analysis_sanghwan/Model/GC_{component}_model{model}_test.xml'
        gtmodel['outfile']=f'./GC_analysis_sanghwan/GC_{component}_model{model}_12yr{front}_clean{convol}.fits'
        gtmodel['expcube']=f'./GC_analysis_sanghwan/Allsky_ltcube_12yr{front}{"_clean"}.fits'
        gtmodel['bexpmap']=f'./GC_analysis_sanghwan/Allsky_expcube_edge_12yr{front}{"_clean"}.fits'
        gtmodel['convol']='no'
        #gtmodel['resample']='no'
        gtmodel['evtype']=1
        gtmodel['srcmaps']=f'./GC_analysis_sanghwan/GC_Extended_srcmap_12yr{front}{"_clean"}_model_{model}{convol}.fits'
        gtmodel.run()


    # ===== Added: GCE / Fermi_bubble / isotropic component maps =====
    # These three components are model-INDEPENDENT (GCE NFW² template, Fermi Bubbles, IGRB
    # do not change with the 80 GDE-model grid index). Generate single-component XMLs and
    # run gtmodel to produce their model maps (both convolved and non-convolved).
    # We compute them for every model iteration for simplicity but skip if already exist.
    import os as _os

    # Single-component XML templates
    _gce_xml = f"""
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
    """
    _bubble_xml = """
    <source name="Fermi_bubble" type="DiffuseSource">
        <spectrum apply_edisp="true" file="./fermi_bubble_spectrum.txt" type="FileFunction">
          <parameter free="1" max="1e+10" min="0" name="Normalization" scale="1e-3" value="5" />
        </spectrum>
        <spatialModel file="./Fermi_Bubbles_template.fits" type="SpatialMap" map_based_integral="true">
        </spatialModel>
      </source>
    """
    _iso_xml = """
    <source name="isotropic" type="DiffuseSource">
        <spectrum apply_edisp="true" file="./isotropic_spectrum_ff.txt" type="FileFunction">
          <parameter free="1" max="10" min="1" name="Normalization" scale="1" value="1" />
        </spectrum>
        <spatialModel type="ConstantValue">
          <parameter free="0" max="10" min="0" name="Value" scale="1" value="1" />
        </spatialModel>
      </source>
    """

    # (source_name_in_xml, output_filename_prefix, xml_fragment)
    _extra_components = [
        ('GCE',          'GCE',          _gce_xml),
        ('Fermi_bubble', 'fermi_bubble', _bubble_xml),
        ('isotropic',    'isotropic',    _iso_xml),
    ]

    for _src_name, _fname, _xml_frag in _extra_components:
        _xml_path = f'./GC_analysis_sanghwan/Model/GC_{_fname}_singlecomp.xml'
        if not _os.path.exists(_xml_path):
            _new_tree = ET.ElementTree(ET.fromstring(f"<sources>{_xml_frag}</sources>"))
            _new_root = _new_tree.getroot()
            _base_tree = ET.parse('./GC_analysis_sanghwan/Model/empty_model.xml')
            _base_root = _base_tree.getroot()
            for _s in _new_root:
                _base_root.append(_s)
            _base_tree.write(_xml_path, encoding='utf-8', xml_declaration=True)

        for _convol_val, _convol_suffix in [('yes', ''), ('no', '_no_convol')]:
            _out = f'./GC_analysis_sanghwan/GC_{_fname}_model_12yr{front}_clean{_convol_suffix}.fits'
            if _os.path.exists(_out):
                print(f"[skip gtmodel {_fname}{_convol_suffix}] exists")
                continue
            _gtm = GtApp('gtmodel', 'Likelihood')
            _gtm['irfs']='P8R3_CLEAN_V3'
            _gtm['outtype']='ccube'
            _gtm['srcmdl']=_xml_path
            _gtm['outfile']=_out
            _gtm['expcube']=f'./GC_analysis_sanghwan/Allsky_ltcube_12yr{front}{"_clean"}.fits'
            _gtm['bexpmap']=f'./GC_analysis_sanghwan/Allsky_expcube_edge_12yr{front}{"_clean"}.fits'
            _gtm['convol']=_convol_val
            _gtm['evtype']=1
            _gtm['srcmaps']=f'./GC_analysis_sanghwan/GC_Extended_srcmap_12yr{front}{"_clean"}_model_{model}{_convol_suffix}.fits'
            _gtm.run()
    # ===== End added block =====





    print(f"  [model={model}] Stage 1 complete. gtsrcmaps + gtmodel outputs saved.", flush=True)

print("")
print("="*70)
print("Stage 1 complete. All gtsrcmaps/gtmodel outputs are ready.")
print("Now run Stage 2 (emcee MCMC) in a fresh Python process:")
print("  python run_mcmc_only.py X        # single model")
print("  python run_mcmc_only.py all      # all models")
print("="*70)