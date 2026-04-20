import os, sys, time, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    WORK_DIR, ANA_DIR, MAPCUBE_DIR, EMPTY_XML, PSC_XML,
    CCUBE, EXPCUBE_EDGE, LTCUBE, SCFILE,
    NFW2_TEMPLATE, BUBBLE_TEMPLATE, ISO_SPECTRUM, BUB_SPECTRUM,
    IRFS, EVTYPE, MODELS,
)

os.chdir(WORK_DIR)
from GtApp import GtApp

DIFFUSE_BLOCK = """
    <source name="bremss" type="DiffuseSource">
        <spectrum type="ConstantValue">
          <parameter free="1" max="100" min="1" name="Value" scale="1" value="1" />
        </spectrum>
        <spatialModel file="./MapCubes/bremss_mapcube_model{m}.fits" type="MapCubeFunction" map_based_integral="true">
          <parameter free="0" max="1000" min="0" name="Normalization" scale="1" value="1" />
        </spatialModel>
      </source>
      <source name="ics" type="DiffuseSource">
        <spectrum type="ConstantValue">
          <parameter free="1" max="100" min="1" name="Value" scale="1" value="1" />
        </spectrum>
        <spatialModel file="./MapCubes/ics_mapcube_model{m}.fits" type="MapCubeFunction" map_based_integral="true">
          <parameter free="0" max="1000" min="0" name="Normalization" scale="1" value="1" />
        </spatialModel>
      </source>
      <source name="pion" type="DiffuseSource">
        <spectrum type="ConstantValue">
          <parameter free="1" max="100" min="1" name="Value" scale="1" value="1" />
        </spectrum>
        <spatialModel file="./MapCubes/pion_mapcube_model{m}.fits" type="MapCubeFunction" map_based_integral="true">
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

SINGLE_DIFFUSE = """<source name="{c}" type="DiffuseSource">
        <spectrum type="ConstantValue">
          <parameter free="1" max="100" min="1" name="Value" scale="1" value="1" />
        </spectrum>
        <spatialModel file="./MapCubes/{c}_mapcube_model{m}.fits" type="MapCubeFunction" map_based_integral="true">
          <parameter free="0" max="1000" min="0" name="Normalization" scale="1" value="1" />
        </spatialModel>
      </source>"""

SINGLE_GCE = """<source name="GCE" type="DiffuseSource">
        <spectrum type="BrokenPowerLaw">
        <parameter free="0" max="1000.0" min="0.001" name="Prefactor" scale="1e-11" value="21"/>
        <parameter free="0" max="-1.0" min="-5." name="Index1" scale="1.0" value="-1.42"/>
        <parameter free="0" max="3000.0" min="30.0" name="BreakValue" scale="1.0" value="2006"/>
        <parameter free="0" max="-1.0" min="-5." name="Index2" scale="1.0" value="-2.63"/>
    </spectrum>
        <spatialModel file="./GCE_template_NFW2.fits" type="SpatialMap" map_based_integral="true">
        </spatialModel>
      </source>"""

SINGLE_BUBBLE = """<source name="Fermi_bubble" type="DiffuseSource">
        <spectrum apply_edisp="true" file="./fermi_bubble_spectrum.txt" type="FileFunction">
          <parameter free="1" max="1e+10" min="0" name="Normalization" scale="1e-3" value="5" />
        </spectrum>
        <spatialModel file="./Fermi_Bubbles_template.fits" type="SpatialMap" map_based_integral="true">
        </spatialModel>
      </source>"""

SINGLE_ISO = """<source name="isotropic" type="DiffuseSource">
        <spectrum apply_edisp="true" file="./isotropic_spectrum_ff.txt" type="FileFunction">
          <parameter free="1" max="10" min="1" name="Normalization" scale="1" value="1" />
        </spectrum>
        <spatialModel type="ConstantValue">
          <parameter free="0" max="10" min="0" name="Value" scale="1" value="1" />
        </spatialModel>
      </source>"""

def build_xml(base_xml_path, fragment, out_path):
    src_root = ET.fromstring(f"<sources>{fragment}</sources>")
    base = ET.parse(base_xml_path).getroot()
    for s in src_root:
        base.append(s)
    ET.ElementTree(base).write(out_path, encoding='utf-8', xml_declaration=True)

def run_gtsrcmaps(model, convol_flag):
    suffix = '' if convol_flag == 'yes' else '_no_convol'
    out = f'./GC_analysis_sanghwan/GC_Extended_srcmap_12yr_front_clean_model_{model}{suffix}.fits'
    if os.path.exists(out):
        print(f"  [skip gtsrcmaps {model} convol={convol_flag}]", flush=True)
        return
    src_xml = f'./GC_analysis_sanghwan/Model/GC_Extended_model{model}_test.xml'
    build_xml(EMPTY_XML, DIFFUSE_BLOCK.format(m=model), src_xml)
    g = GtApp('gtsrcmaps', 'Likelihood')
    g['scfile'] = SCFILE
    g['expcube'] = LTCUBE
    g['cmap'] = CCUBE
    g['bexpmap'] = EXPCUBE_EDGE
    g['srcmdl'] = src_xml
    g['outfile'] = out
    g['irfs'] = IRFS
    g['convol'] = convol_flag
    g['evtype'] = EVTYPE
    print(f"  running gtsrcmaps {model} convol={convol_flag}...", flush=True)
    t0 = time.time()
    g.run()
    print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)

def run_gtmodel(model, component, convol_flag, single_xml_template, out_basename):
    suffix = '' if convol_flag == 'yes' else '_no_convol'
    out = f'./GC_analysis_sanghwan/GC_{out_basename}_model{model if component in ("pion","bremss","ics") else ""}_12yr_front_clean{suffix}.fits'
    if component in ('pion', 'bremss', 'ics'):
        out = f'./GC_analysis_sanghwan/GC_{component}_model{model}_12yr_front_clean{suffix}.fits'
    else:
        out = f'./GC_analysis_sanghwan/GC_{out_basename}_model_12yr_front_clean{suffix}.fits'
    if os.path.exists(out):
        print(f"  [skip gtmodel {component} {model} convol={convol_flag}]", flush=True)
        return
    if component in ('pion', 'bremss', 'ics'):
        xml_path = f'./GC_analysis_sanghwan/Model/GC_{component}_model{model}_test.xml'
        build_xml(EMPTY_XML, single_xml_template.format(c=component, m=model), xml_path)
    else:
        xml_path = f'./GC_analysis_sanghwan/Model/GC_{out_basename}_singlecomp.xml'
        if not os.path.exists(xml_path):
            build_xml(EMPTY_XML, single_xml_template, xml_path)
    srcmaps = f'./GC_analysis_sanghwan/GC_Extended_srcmap_12yr_front_clean_model_{model}{suffix}.fits'
    g = GtApp('gtmodel', 'Likelihood')
    g['irfs'] = IRFS
    g['outtype'] = 'ccube'
    g['srcmdl'] = xml_path
    g['outfile'] = out
    g['expcube'] = LTCUBE
    g['bexpmap'] = EXPCUBE_EDGE
    g['convol'] = convol_flag
    g['evtype'] = EVTYPE
    g['srcmaps'] = srcmaps
    print(f"  running gtmodel {component} {model} convol={convol_flag}...", flush=True)
    t0 = time.time()
    g.run()
    print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)

def process_model(model):
    print(f"\n==== STAGE 1 :: MODEL {model} ====", flush=True)
    t_model = time.time()
    for cv in ('yes', 'no'):
        run_gtsrcmaps(model, cv)
    for cv in ('yes', 'no'):
        for comp in ('pion', 'bremss', 'ics'):
            run_gtmodel(model, comp, cv, SINGLE_DIFFUSE, comp)
    for cv in ('yes', 'no'):
        run_gtmodel(model, 'GCE', cv, SINGLE_GCE, 'GCE')
        run_gtmodel(model, 'Fermi_bubble', cv, SINGLE_BUBBLE, 'fermi_bubble')
        run_gtmodel(model, 'isotropic', cv, SINGLE_ISO, 'isotropic')
    print(f"==== MODEL {model} done in {(time.time()-t_model)/60:.1f} min ====", flush=True)

if __name__ == '__main__':
    targets = sys.argv[1:] if len(sys.argv) > 1 else MODELS
    for m in targets:
        process_model(m)
    print("\nALL STAGE 1 COMPLETE", flush=True)
