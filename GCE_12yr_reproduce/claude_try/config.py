import os

WORK_DIR = '/home/haebarg/GCE-Chi-square-fitting/GCE_12yr_reproduce'
ANA_DIR = os.path.join(WORK_DIR, 'GC_analysis_sanghwan')
MAPCUBE_DIR = os.path.join(WORK_DIR, 'MapCubes')
ALLSKY_DIR = '/home/haebarg/GCE-Chi-square-fitting/GCE_allsky_data'
TEMPLATES_DIR = '/home/haebarg/GCE-Chi-square-fitting/GCE_TEMPLATES_FILES_v3'
ZENODO_FIG_DIR = os.path.join(TEMPLATES_DIR, 'Figures_12_and_14_GCE_Spectra')

OUT_DIR = os.path.join(WORK_DIR, 'haebarg_v_claude')
os.makedirs(OUT_DIR, exist_ok=True)

CCUBE = os.path.join(ANA_DIR, 'GC_ccube_12yr_front_clean.fits')
EXPCUBE_CENTER = os.path.join(ANA_DIR, 'GC_expcube_center_12yr_front_clean.fits')
LTCUBE = os.path.join(ANA_DIR, 'Allsky_ltcube_12yr_front_clean.fits')
EXPCUBE_EDGE = os.path.join(ANA_DIR, 'Allsky_expcube_edge_12yr_front_clean.fits')
SCFILE = os.path.join(ALLSKY_DIR, 'lat_spacecraft_merged_12yr.fits')

EMPTY_XML = os.path.join(ANA_DIR, 'Model', 'empty_model.xml')
PSC_XML = os.path.join(ANA_DIR, 'Model', 'GC_psc_model_DR2.xml')

NFW2_TEMPLATE = os.path.join(WORK_DIR, 'GCE_template_NFW2.fits')
BUBBLE_TEMPLATE = os.path.join(WORK_DIR, 'Fermi_Bubbles_template.fits')
ISO_SPECTRUM = os.path.join(WORK_DIR, 'isotropic_spectrum_ff.txt')
BUB_SPECTRUM = os.path.join(WORK_DIR, 'fermi_bubble_spectrum.txt')

DISK_MASK_NPY = os.path.join(ANA_DIR, 'Model', 'GC_disk_mask_60x60_definitions.npy')
PSC_MASK_SANGHWAN = os.path.join(ANA_DIR, 'Model', 'GC_mask_60x60_definitions_DR2.npy')
PSC_MASK_CHOLIS_RAW = os.path.join(WORK_DIR, 'zhong_mask_loaded.npy')

BUBBLE_CONSTRAINTS = os.path.join(ANA_DIR, 'Model', 'bubble_constraints.txt')
ISO_CONSTRAINTS = os.path.join(ANA_DIR, 'Model', 'iso_constraints_full_err.txt')

SR_PER_PIXEL_NPY = os.path.join(WORK_DIR, 'sr_per_pixel.npy')

TOTAL_CORES = 64
STAGE2_WORKERS = 64
STAGE1_PARALLEL_MODELS = 5

MODELS = ['X', 'XV', 'XLVIII', 'XLIX', 'LIII']

ROI_SLICE = (slice(100, 500), slice(100, 500))
N_PIX_FULL = 600
N_PIX_FIT = 400
PIXEL_SIZE_DEG = 0.1

IRFS = 'P8R3_CLEAN_V3'
EVTYPE = 1

NWALKERS = 100
NSTEPS = 1500
NBURN = 500
NDIM = 5

MIN_REL_ERROR_FLOOR = 0.05

CHOLIS_REF_FILES = {
    'X': 'GCE_ModelX_flux_Inner40x40_masked_disk.dat',
    'XV': 'GCE_ModelXV_flux_Inner40x40_masked_disk.dat',
    'XLVIII': 'GCE_ModelXLVIII_flux_Inner40x40_masked_disk.dat',
    'XLIX': 'GCE_ModelXLIX_flux_Inner40x40_masked_disk.dat',
    'LIII': 'GCE_ModelLIII_flux_Inner40x40_masked_disk.dat',
}
