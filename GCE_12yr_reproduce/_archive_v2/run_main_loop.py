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





    ## Emcee running part
    
    def roi_solid_angle(delta_l_deg, delta_b_deg, b_deg):
        # Convert degrees to radians
        delta_l_rad = np.radians(delta_l_deg)
        delta_b_rad = np.radians(delta_b_deg)
        b_rad = np.radians(b_deg)
        
        # Calculate solid angle in steradians
        solid_angle = delta_l_rad * delta_b_rad * np.cos(b_rad)
        return solid_angle
    
    raw_map=fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits')
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
    disk_mask=np.load('./GC_analysis_sanghwan/Model/GC_disk_mask_60x60_definitions.npy')[100:500, 100:500]
    #psc_mask=np.load('./GC_analysis_sanghwan/Model/GC_mask_60x60_definitions_DR2.npy')[:, 100:500, 100:500]
    #psc_mask=np.load('./GC_analysis_sanghwan/Model/GC_mask_60x60_definitions_DR2.npy')[:, 100:500, 100:500]
    
    front='_front'
    convol='_no_convol'
    
    E_bounds=fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits')[1].data
    
    
    E=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        E[i] = np.sqrt(E_bounds[i][2]*E_bounds[i][1]*1e-6)*1e-3
    
    delta_E=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        delta_E[i] = (E_bounds[i][2] - E_bounds[i][1])*1e-6
    
    exp_cube = fits.open(f'./GC_analysis_sanghwan/GC_expcube_center_12yr_front_clean.fits')[0].data[:, 100:500, 100:500]*steradian_per_pixel[100:500, 100:500]
    
    
    
    file_name=f'./GC_analysis_sanghwan/GC_pion_model{model}_12yr_front_clean_no_convol.fits'
    pion=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        pion[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/a) )/np.sum(disk_mask)
    
    file_name=f'./GC_analysis_sanghwan/GC_bremss_model{model}_12yr_front_clean_no_convol.fits'
    bremss=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        bremss[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/a) )/np.sum(disk_mask)
        
    file_name=f'./GC_analysis_sanghwan/GC_ics_model{model}_12yr_front_clean_no_convol.fits'
    ics=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        ics[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/a) )/np.sum(disk_mask)
        
    file_name=f'./GC_analysis_sanghwan/GC_GCE_model_12yr_front_clean_no_convol.fits'
    GCE=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        GCE[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/exp_cube[i]) )/np.sum(disk_mask)
    
    
    file_name=f'./GC_analysis_sanghwan/GC_fermi_bubble_model_12yr_front_clean_no_convol.fits'
    bubble=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        bubble[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/a) )/np.sum(disk_mask)
        
    file_name=f'./GC_analysis_sanghwan/GC_isotropic_model_12yr_front_clean_no_convol.fits'
    isotropic=np.zeros(len(E_bounds))
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        isotropic[i] = np.sum( disk_mask*(fits.open(file_name)[0].data[i][100:500, 100:500]/a) )/np.sum(disk_mask)
        
    counts_per_exp=np.zeros(len(E_bounds))
    i=0
    for i in range(0, len(E_bounds), 1):
        a=exp_cube[i]
        counts_per_exp[i]=np.sum( disk_mask*( (fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits')[0].data[i][100:500, 100:500]) /a) )/np.sum(disk_mask)
    
    counts_per_exp_err=np.zeros(len(E_bounds))
    i=0
    for i in range(0, len(E_bounds), 1):
        counts_per_exp_err[i]= np.sqrt( np.sum( ( (np.sqrt(disk_mask*fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits')[0].data[i][100:500, 100:500]) /exp_cube[i] )**2)) )/np.sum(disk_mask)
        #counts_per_exp_err[i]=  np.sqrt(np.sum(disk_mask*fits.open(f'./GC_analysis_sanghwan/GC_all_time_60x60_ccube{front}_12yr.fits')[0].data[i][100:500, 100:500]))/np.sum(disk_mask*exp_cube[i]) 
    
        
    
    def log_factorial(O):
        return sum(np.log(i) for i in range(1, O+1))
    log_factorial=np.vectorize(log_factorial)
    #Constraints interpolated function
    #Contains constraints for bubble and isotropic as well
    #For isotropic, from https://arxiv.org/pdf/1410.3696.pdf Table 3
    #Correcting bubble template given from https://arxiv.org/pdf/1407.7905, Table 2
    
    bubble_constraints=np.loadtxt('./GC_analysis_sanghwan/Model/bubble_constraints.txt')
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
    
    
    iso_constraints=np.loadtxt('./GC_analysis_sanghwan/Model/iso_constraints_full_err.txt')
    #iso_constraints=np.loadtxt('./GC_analysis_sanghwan/Model/egb_constraints_full_err.txt')
    
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
    
    front='_front'
    class Likelihood:
        def __init__(self, model, energy_bin):
            import sys
            print(f"  [Likelihood __init__] model={model}, energy_bin={energy_bin}", flush=True)
            self.model=model
            self.energy_bin=energy_bin
            self.data=fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]
            self.pion_bremss=fits.open(f'./GC_analysis_sanghwan/GC_pion_model{model}_12yr_front_clean.fits')[0].data[self.energy_bin, 100:500, 100:500] + fits.open(f'./GC_analysis_sanghwan/GC_bremss_model{model}_12yr_front_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]  
            self.ics=fits.open(f'./GC_analysis_sanghwan/GC_ics_model{model}_12yr_front_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]
            self.GCE=fits.open(f'./GC_analysis_sanghwan/GC_GCE_model_12yr_front_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]
            self.bubble=fits.open(f'./GC_analysis_sanghwan/GC_fermi_bubble_model_12yr_front_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]
            self.iso=fits.open(f'./GC_analysis_sanghwan/GC_isotropic_model_12yr_front_clean.fits')[0].data[self.energy_bin, 100:500, 100:500]
            E_bounds=fits.open(f'./GC_analysis_sanghwan/GC_ccube_12yr_front_clean.fits')[1].data
    
            E=np.zeros(len(E_bounds))
            for i in range(0, len(E_bounds), 1):
                E[i] = np.sqrt(E_bounds[i][2]*E_bounds[i][1]*1e-6)*1e-3
            self.E = E
            delta_E=np.zeros(len(E_bounds))
            for i in range(0, len(E_bounds), 1):
                delta_E[i] = (E_bounds[i][2] - E_bounds[i][1])*1e-6
            self.delta_E = delta_E
            self.exp_cube = (fits.open(f'./GC_analysis_sanghwan/GC_expcube_center_12yr_front_clean.fits')[0].data[self.energy_bin]*steradian_per_pixel)[100:500, 100:500]
    
            
            psc_mask=np.load('./GC_analysis_sanghwan/Model/GC_mask_60x60_definitions_DR2.npy')[self.energy_bin, 100:500, 100:500]
            
            #psc_mask=np.load('./GC_analysis_sanghwan/Model/GC_mask_60x60_definitions_DR2_corrected.npy')[self.energy_bin, 100:500, 100:500]
    
            disk_mask=np.load('./GC_analysis_sanghwan/Model/GC_disk_mask_60x60_definitions.npy')[100:500, 100:500]
            full_mask=psc_mask*disk_mask
            self.disk_mask=disk_mask
            self.full_mask=full_mask
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
            lhd=2*( expected_pixel - observed_log_expected + log_factorial(observed_pixel.astype(int)) )
    
            
            file_name=f'./GC_analysis_sanghwan/GC_isotropic_model_12yr_front_clean_no_convol.fits'
            #isotropic = np.sum( self.full_mask*(fits.open(file_name)[0].data[self.energy_bin]/self.exp_cube) )*isotropic_param/np.sum(self.full_mask)
            #isotropic = np.sum( self.full_mask*(fits.open(file_name)[0].data[self.energy_bin]) )*isotropic_param/np.sum(self.full_mask*self.exp_cube)
            #isotropic = np.sum( self.full_mask*(fits.open(file_name)[0].data[self.energy_bin])/self.exp_cube )*isotropic_param/np.sum(self.full_mask)
            isotropic = np.sum( self.full_mask*(fits.open(file_name)[0].data[self.energy_bin, 100:500, 100:500])/self.exp_cube )*isotropic_param/np.sum(self.full_mask)
            #isotropic = np.sum( (fits.open(file_name)[0].data[self.energy_bin])/self.exp_cube )*isotropic_param/np.sum(600*600)
    
            isotropic_sed = (self.E[self.energy_bin]**2)*isotropic/(self.delta_E[self.energy_bin])
    
    
            file_name=f'./GC_analysis_sanghwan/GC_fermi_bubble_model_12yr_front_clean_no_convol.fits'
            #bubble = np.sum( self.full_mask*(fits.open(file_name)[0].data[self.energy_bin]/self.exp_cube) )*bubble_param/np.sum(self.full_mask)
            #bubble = np.sum( self.full_mask*(fits.open(file_name)[0].data[self.energy_bin]) )*bubble_param/np.sum(self.full_mask*self.exp_cube)
            #bubble = np.sum( self.full_mask*(fits.open(file_name)[0].data[self.energy_bin])/self.exp_cube )*bubble_param/np.sum(self.full_mask)
            bubble = np.sum( self.full_mask*(fits.open(file_name)[0].data[self.energy_bin, 100:500, 100:500])/self.exp_cube )*bubble_param/np.sum(self.full_mask)
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
                chi2_isotropic = ((isotropic_flux_data[i] - isotropic_sed)/isotropic_upper_error_data[self.energy_bin])**2
            if isotropic_flux_data[self.energy_bin] == isotropic_sed:
                chi2_isotropic = ((isotropic_flux_data[self.energy_bin] - isotropic_sed)/isotropic_larger_error)**2
            #print(chi2_bubble, chi2_isotropic)
            return (np.sum(lhd)  + chi2_bubble + chi2_isotropic)
    
    
    # [REMOVED] Sanghwan's sanity test: Likelihood('I', 0).likelihood_constrained(np.ones(5))
    
    # Define your likelihood function (with Likelihood object caching)
    _likelihood_cache = {}
    def log_likelihood(params, energy_bin):
        key = (model, energy_bin)
        if key not in _likelihood_cache:
            _likelihood_cache[key] = Likelihood(model, energy_bin)
        return -(1/2)*_likelihood_cache[key].likelihood_constrained(params)  # log likelihood -> Need to maximize
    
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
        # Memory check before starting Pool
        try:
            import resource
            mem_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024)
            print(f"  energy_bin={energy_bin}, parent RSS before Pool: {mem_gb:.2f} GB", flush=True)
        except Exception:
            print(f"  energy_bin={energy_bin}", flush=True)
        # === Run emcee SERIALLY (no Pool) ===
        # multiprocessing.Pool with fork-based workers crashes immediately when 
        # the parent has imported Fermi ScienceTools (GtApp). Running emcee serially
        # avoids fork issues entirely. Slower but reliable.
        if True:  # keep block-level scoping similar to original
            initial_params = np.vstack([
                np.random.uniform(0, 3, [nwalkers]),
                np.random.uniform(0, 3, [nwalkers]),
                np.random.uniform(0, 3, [nwalkers]),
                np.random.uniform(0, 10, [nwalkers]),
                np.random.uniform(0, 10, [nwalkers]),
            ]).T
            pos = initial_params
            from emcee.moves import DEMove, KDEMove
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(energy_bin,), pool=None)
            #for iteration in range(10):    
            print(f"  About to call sampler.run_mcmc(nsteps={nsteps}, walkers={nwalkers})...", flush=True)
            print(f"  Performing one test log_probability call to verify it works...", flush=True)
            _test_lp = log_probability(initial_params[0], energy_bin)
            print(f"  Test log_probability returned: {_test_lp}", flush=True)
            print("Running production... (chunked progress, no tqdm)", flush=True)
            
            # Run in chunks of 100 steps with explicit progress logging.
            # Avoids tqdm which doesn't flush properly in nohup/non-tty environments.
            chunk_size = 100
            current_pos = pos
            t_chunk_start = time.time()
            for chunk_start in range(0, nsteps, chunk_size):
                chunk_n = min(chunk_size, nsteps - chunk_start)
                try:
                    import resource
                    mem_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024)
                except Exception:
                    mem_gb = -1
                t0 = time.time()
                state_obj = sampler.run_mcmc(current_pos, chunk_n, progress=False)
                current_pos = state_obj.coords if hasattr(state_obj, 'coords') else state_obj[0]
                t_elapsed = time.time() - t0
                rate = chunk_n / t_elapsed if t_elapsed > 0 else 0
                cum_steps = chunk_start + chunk_n
                eta_sec = (nsteps - cum_steps) / rate if rate > 0 else 0
                print(f"    [bin={energy_bin}] step {cum_steps:>5}/{nsteps}  "
                      f"chunk_time={t_elapsed:.1f}s  rate={rate:.2f} it/s  "
                      f"ETA={eta_sec/60:.1f} min  RSS={mem_gb:.2f} GB", flush=True)
            
            # Reconstruct (pos, prob, state) for downstream code compatibility
            pos = current_pos
            prob = sampler.get_log_prob()[-1]
            state = state_obj
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
            plt.show()
        
            # Final Corner Plot with ChainConsumer
            flat_samples = sampler.get_chain(discard=burn_in_steps, thin=1, flat=True)  # Flatten for ChainConsumer
            c = ChainConsumer()
            c.add_chain(flat_samples, parameters=["param1", "param2", "param3", "param4", "param5"])
            fig = c.plotter.plot(figsize=(6, 6))
            axes=fig.axes
            #for ax in axes:
            #    ax.set_xlim(-1, 6)
            #    ax.set_ylim(-1, 6)
            plt.show()
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
    for i in range(0, n, 1):
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
        
        
        plt.show()
    
        print((fitted[n*2:n*3]*(GCE)*(E**2)/(delta_E*sr))[i])
        print(((fitted_params_upper[n*2:n*3] - fitted[n*2:n*3])*(GCE)*(E**2)/(delta_E*sr))[i])
        print(((fitted[n*2:n*3]- fitted_params_lower[n*2:n*3]  )*(GCE)*(E**2)/(delta_E*sr))[i])
    
                                                                
    
    
    
    max_likelihood
    
    np.sum(max_likelihood)
    
    np.savetxt(f'./GCE_model_{model}_12yr_cholis.dat', np.vstack([E, fitted[n*2:n*3]*(GCE)*(E**2)/(delta_E), (fitted_errors[n*2:n*3]*GCE)*(E**2)/(delta_E), (fitted_params_lower[n*2:n*3])*(GCE)*(E**2)/(delta_E), (fitted_params_upper[n*2:n*3])*(GCE)*(E**2)/(delta_E)]).T)
    np.savetxt(f'./GCE_model_{model}_12yr_cholis_likelihood_value', np.array((max_likelihood))) #Positive of log likelihood
    

