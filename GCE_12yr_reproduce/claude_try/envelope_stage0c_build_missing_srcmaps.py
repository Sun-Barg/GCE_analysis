"""
Envelope Stage 0c - Generate per-component srcmap FITS files for missing
models by running gtsrcmaps + gtmodel from MapCube inputs.

Strategy (matches Sanghwan pipeline exactly):
  For each missing model M:
    1. Build Extended XML with bremss+ics+pion+GCE+isotropic+Fermi_bubble
       using ./MapCubes/{component}_mapcube_model{M}.fits as spatial maps.
    2. Run gtsrcmaps twice:
         - convol=yes -> GC_Extended_srcmap_..._{M}.fits
         - convol=no  -> GC_Extended_srcmap_..._{M}_no_convol.fits
    3. Build single-component XMLs (pion, bremss, ics) and run gtmodel to
       split the Extended srcmap into per-component counts cubes.
    4. Output files:
         GC_pion_model{M}_12yr_front_clean.fits          (convol=yes)
         GC_bremss_model{M}_12yr_front_clean.fits        (convol=yes)
         GC_ics_model{M}_12yr_front_clean.fits           (convol=yes)
       These are the files stage2_fit.load_component_maps expects.

Parallelization:
  - Each gtsrcmaps takes ~5-15 minutes single-threaded.
  - Memory: ~5 GB per gtsrcmaps worker (as confirmed by previous sessions).
  - Safe concurrency: 12-16 gtsrcmaps workers in parallel.
  - Total work: 65 missing models x 2 gtsrcmaps + 6 gtmodel each -> ~15 hr
    single-threaded, ~1.5 hr at 12-parallel.

Usage:
  python envelope_stage0c_build_missing_srcmaps.py
  # Explicit model list:
  python envelope_stage0c_build_missing_srcmaps.py II III VIII
  # Control concurrency:
  N_PARALLEL_SRCMAPS=8 python envelope_stage0c_build_missing_srcmaps.py
"""
import os, sys, time, subprocess
import xml.etree.ElementTree as ET
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR

ANA = f'{WORK_DIR}/GC_analysis_sanghwan'
MAPCUBES = f'{WORK_DIR}/MapCubes'
NAMING_FILE = '/home/haebarg/GCE-Chi-square-fitting/GCE_TEMPLATES_FILES_v3/NAMING_CONVENTION_OF_DIFFUSE_EMISSION_MODELS.dat'

SC_FILE = '../GCE_allsky_data/lat_spacecraft_merged_12yr.fits'
LT_CUBE = f'{ANA}/Allsky_ltcube_12yr_front_clean.fits'
EXPCUBE_EDGE = f'{ANA}/Allsky_expcube_edge_12yr_front_clean.fits'
CCUBE = f'{ANA}/GC_ccube_12yr_front_clean.fits'
IRFS = 'P8R3_CLEAN_V3'
EVTYPE = 1

GCE_TEMPLATE = f'{WORK_DIR}/GCE_template_NFW2.fits'
BUBBLE_TEMPLATE = f'{WORK_DIR}/Fermi_Bubbles_template.fits'
ISO_SPEC_FF = f'{WORK_DIR}/isotropic_spectrum_ff.txt'
BUB_SPEC_FF = f'{WORK_DIR}/fermi_bubble_spectrum.txt'

MODEL_DIR = f'{ANA}/Model'
os.makedirs(MODEL_DIR, exist_ok=True)

N_PARALLEL = int(os.environ.get('N_PARALLEL_SRCMAPS', '12'))


def load_all_model_names():
    names = []
    with open(NAMING_FILE) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if parts:
                names.append(parts[0])
    return names


def output_exists(model):
    files = [
        f'{ANA}/GC_pion_model{model}_12yr_front_clean.fits',
        f'{ANA}/GC_bremss_model{model}_12yr_front_clean.fits',
        f'{ANA}/GC_ics_model{model}_12yr_front_clean.fits',
    ]
    return all(os.path.exists(f) and os.path.getsize(f) > 1e6 for f in files)


def mapcube_exists(model):
    files = [
        f'{MAPCUBES}/bremss_mapcube_model{model}.fits',
        f'{MAPCUBES}/ics_mapcube_model{model}.fits',
        f'{MAPCUBES}/pion_mapcube_model{model}.fits',
    ]
    return all(os.path.exists(f) for f in files)


def build_extended_xml(model, xml_path):
    xml_text = f"""<?xml version="1.0" encoding="utf-8"?>
<source_library title="source library">
  <source name="bremss" type="DiffuseSource">
    <spectrum type="ConstantValue">
      <parameter error="0.02899312444" free="1" max="100" min="1" name="Value" scale="1" value="1" />
    </spectrum>
    <spatialModel file="{MAPCUBES}/bremss_mapcube_model{model}.fits" type="MapCubeFunction" map_based_integral="true">
      <parameter free="0" max="1000" min="0" name="Normalization" scale="1" value="1" />
    </spatialModel>
  </source>
  <source name="ics" type="DiffuseSource">
    <spectrum type="ConstantValue">
      <parameter error="0.04073673429" free="1" max="100" min="1" name="Value" scale="1" value="1" />
    </spectrum>
    <spatialModel file="{MAPCUBES}/ics_mapcube_model{model}.fits" type="MapCubeFunction" map_based_integral="true">
      <parameter free="0" max="1000" min="0" name="Normalization" scale="1" value="1" />
    </spatialModel>
  </source>
  <source name="pion" type="DiffuseSource">
    <spectrum type="ConstantValue">
      <parameter error="0.02899312444" free="1" max="100" min="1" name="Value" scale="1" value="1" />
    </spectrum>
    <spatialModel file="{MAPCUBES}/pion_mapcube_model{model}.fits" type="MapCubeFunction" map_based_integral="true">
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
    <spatialModel file="{GCE_TEMPLATE}" type="SpatialMap" map_based_integral="true"/>
  </source>
  <source name="isotropic" type="DiffuseSource">
    <spectrum apply_edisp="true" file="{ISO_SPEC_FF}" type="FileFunction">
      <parameter free="1" max="10" min="1" name="Normalization" scale="1" value="1" />
    </spectrum>
    <spatialModel type="ConstantValue">
      <parameter free="0" max="10" min="0" name="Value" scale="1" value="1" />
    </spatialModel>
  </source>
  <source name="Fermi_bubble" type="DiffuseSource">
    <spectrum apply_edisp="true" file="{BUB_SPEC_FF}" type="FileFunction">
      <parameter free="1" max="1e+10" min="0" name="Normalization" scale="1e-3" value="5" />
    </spectrum>
    <spatialModel file="{BUBBLE_TEMPLATE}" type="SpatialMap" map_based_integral="true"/>
  </source>
</source_library>
"""
    with open(xml_path, 'w') as f:
        f.write(xml_text)


def build_single_component_xml(model, component, xml_path):
    xml_text = f"""<?xml version="1.0" encoding="utf-8"?>
<source_library title="source library">
  <source name="{component}" type="DiffuseSource">
    <spectrum type="ConstantValue">
      <parameter error="0.02899312444" free="1" max="100" min="1" name="Value" scale="1" value="1" />
    </spectrum>
    <spatialModel file="{MAPCUBES}/{component}_mapcube_model{model}.fits" type="MapCubeFunction" map_based_integral="true">
      <parameter free="0" max="1000" min="0" name="Normalization" scale="1" value="1" />
    </spatialModel>
  </source>
</source_library>
"""
    with open(xml_path, 'w') as f:
        f.write(xml_text)


def run_cmd(cmd, logf):
    with open(logf, 'a') as lf:
        lf.write(f'\n=== {time.strftime("%H:%M:%S")} CMD ===\n{cmd}\n')
        lf.flush()
        rc = subprocess.call(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                              cwd=WORK_DIR)
    return rc


def build_one_model(model):
    logf = os.path.join(OUT_DIR, f'log_build_srcmap_{model}.txt')
    with open(logf, 'w') as lf:
        lf.write(f'Build srcmap for model {model}  start {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    if output_exists(model):
        with open(logf, 'a') as lf:
            lf.write('All output files already exist, skipping\n')
        return (model, 'skip', 0.0)

    if not mapcube_exists(model):
        with open(logf, 'a') as lf:
            lf.write(f'ERROR: MapCube files missing for model {model}\n')
        return (model, 'no_mapcube', 0.0)

    t0 = time.time()

    ext_xml = f'{MODEL_DIR}/GC_Extended_model{model}_test.xml'
    build_extended_xml(model, ext_xml)

    ext_srcmap_c = f'{ANA}/GC_Extended_srcmap_12yr_front_clean_model_{model}.fits'
    ext_srcmap_nc = f'{ANA}/GC_Extended_srcmap_12yr_front_clean_model_{model}_no_convol.fits'

    cmd_c = (
        f'gtsrcmaps scfile={SC_FILE} expcube={LT_CUBE} cmap={CCUBE} '
        f'bexpmap={EXPCUBE_EDGE} srcmdl={ext_xml} outfile={ext_srcmap_c} '
        f'irfs={IRFS} evtype={EVTYPE} convol=yes '
        f'resample=yes rfactor=2 minbinsz=0.1 ptsrc=no psfcorr=yes emapbnds=yes '
        f'chatter=2 clobber=yes mode=ql'
    )
    rc = run_cmd(cmd_c, logf)
    if rc != 0 or not os.path.exists(ext_srcmap_c):
        return (model, f'gtsrcmaps_convol_fail rc={rc}', (time.time()-t0)/60)

    cmd_nc = (
        f'gtsrcmaps scfile={SC_FILE} expcube={LT_CUBE} cmap={CCUBE} '
        f'bexpmap={EXPCUBE_EDGE} srcmdl={ext_xml} outfile={ext_srcmap_nc} '
        f'irfs={IRFS} evtype={EVTYPE} convol=no '
        f'resample=yes rfactor=2 minbinsz=0.1 ptsrc=no psfcorr=no emapbnds=yes '
        f'chatter=2 clobber=yes mode=ql'
    )
    rc = run_cmd(cmd_nc, logf)
    if rc != 0 or not os.path.exists(ext_srcmap_nc):
        return (model, f'gtsrcmaps_noconvol_fail rc={rc}', (time.time()-t0)/60)

    for component in ['pion', 'bremss', 'ics']:
        comp_xml = f'{MODEL_DIR}/GC_{component}_model{model}_test.xml'
        build_single_component_xml(model, component, comp_xml)

        out_c = f'{ANA}/GC_{component}_model{model}_12yr_front_clean.fits'
        cmd_mc = (
            f'gtmodel srcmaps={ext_srcmap_c} srcmdl={comp_xml} '
            f'outfile={out_c} expcube={LT_CUBE} bexpmap={EXPCUBE_EDGE} '
            f'irfs={IRFS} outtype=ccube convol=yes evtype={EVTYPE} '
            f'chatter=0 clobber=yes mode=ql'
        )
        rc = run_cmd(cmd_mc, logf)
        if rc != 0 or not os.path.exists(out_c):
            return (model, f'gtmodel_{component}_convol_fail rc={rc}',
                    (time.time()-t0)/60)

        out_nc = f'{ANA}/GC_{component}_model{model}_12yr_front_clean_no_convol.fits'
        cmd_mnc = (
            f'gtmodel srcmaps={ext_srcmap_nc} srcmdl={comp_xml} '
            f'outfile={out_nc} expcube={LT_CUBE} bexpmap={EXPCUBE_EDGE} '
            f'irfs={IRFS} outtype=ccube convol=no evtype={EVTYPE} '
            f'chatter=0 clobber=yes mode=ql'
        )
        rc = run_cmd(cmd_mnc, logf)
        if rc != 0 or not os.path.exists(out_nc):
            return (model, f'gtmodel_{component}_noconvol_fail rc={rc}',
                    (time.time()-t0)/60)

    if os.path.exists(ext_srcmap_c):
        try:
            os.remove(ext_srcmap_c)
            os.remove(ext_srcmap_nc)
        except Exception:
            pass

    return (model, 'OK', (time.time()-t0)/60)


def main():
    os.chdir(WORK_DIR)
    print("=" * 90)
    print(f"Envelope Stage 0c : build per-component srcmaps from MapCube inputs")
    print("=" * 90)

    if sys.argv[1:]:
        models = sys.argv[1:]
    else:
        all_models = load_all_model_names()
        missing = [m for m in all_models
                    if mapcube_exists(m) and not output_exists(m)]
        models = missing

    if not models:
        print("Nothing to build - all targeted models have output files already.")
        return

    print(f"  models to build : {len(models)}")
    print(f"  N_PARALLEL      : {N_PARALLEL}")
    print(f"  per-model time  : ~15 min single-threaded")
    print(f"  wall-clock est  : {len(models) * 15 / max(1, N_PARALLEL):.0f} min")
    print(f"  RAM per worker  : ~5 GB  ->  estimated peak {N_PARALLEL * 5} GB + 37 Indico")
    print()

    t_start = time.time()
    done = []
    with Pool(N_PARALLEL) as pool:
        for i, (m, status, dur) in enumerate(pool.imap_unordered(build_one_model, models)):
            elapsed = (time.time() - t_start) / 60
            symbol = '  OK' if status == 'OK' else ('SKIP' if status == 'skip' else 'FAIL')
            print(f"  [{i+1:3d}/{len(models)}] {symbol} {m:<10} "
                  f"{status:<35} model={dur:5.1f}m  total={elapsed:5.1f}m",
                  flush=True)
            done.append((m, status, dur))

    total_min = (time.time() - t_start) / 60
    n_ok = sum(1 for _, s, _ in done if s == 'OK')
    n_skip = sum(1 for _, s, _ in done if s == 'skip')
    n_fail = sum(1 for _, s, _ in done if s not in ('OK', 'skip'))

    print()
    print(f"Stage 0c done in {total_min:.1f} min: OK={n_ok} skip={n_skip} fail={n_fail}")

    if n_fail:
        print("Failed models:")
        for m, s, _ in done:
            if s not in ('OK', 'skip'):
                print(f"  {m:<10} {s}")


if __name__ == '__main__':
    main()
