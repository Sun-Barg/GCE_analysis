"""
Diagnostic F2 — Apply approximate energy-dispersion to GDE maps and re-fit.

Motivation
----------
Cholis+2022 applies PSF to the GDE model (Eq. 9) but says nothing about
EDISP. Sanghwan XML applies EDISP only to isotropic/bubble. Neither applies
EDISP to Pi0 / Bremss / ICS templates, even though these have steep
low-energy power laws that suffer bin-migration.

This diagnostic:
  1. Applies a log-Gaussian energy kernel to convolve Pi0, Bremss, ICS, GCE
     maps in the energy axis only (spatial axes untouched).
  2. Saves the smeared cubes to OUT_DIR/edisp_smeared/.
  3. Patches stage2_fit.load_component_maps via monkey-patch to read the
     smeared cubes.
  4. Runs a short global-c_gce fit for ONE model (arg) to measure the
     ratio change vs Cholis.

Resolution used
---------------
Fermi-LAT P8R3_CLEAN_V3 Front EDISP 68% containment is typically 10-15%
at 1 GeV and 8-10% at 10 GeV. We use a log-normal kernel with sigma
varying from 0.15 (~15%) at low-E to 0.08 (~8%) at high-E, matching the
published LAT performance curves.

This is a proxy, not a rigorous gtpsf/gtedisp application. Purpose is
to see if EDISP effects are large enough to matter, not to be precise.

Runtime: map smoothing ~30s, single-model MCMC ~180min.
Memory: keeps ~4 GB peak for map copies.
"""
import os, sys, time, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
from astropy.io import fits
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR, CCUBE, EXPCUBE_CENTER, MODELS, ROI_SLICE

ANA = f'{WORK_DIR}/GC_analysis_sanghwan'
SMEARED_DIR = f'{OUT_DIR}/edisp_smeared'
os.makedirs(SMEARED_DIR, exist_ok=True)


def load_E_array():
    with fits.open(CCUBE) as h:
        eb = h[1].data
    emin = np.asarray(eb['E_MIN']) / 1000.0
    emax = np.asarray(eb['E_MAX']) / 1000.0
    E = np.sqrt(emin * emax)
    dE = emax - emin
    return E, dE


def edisp_sigma_log(E_gev):
    """Approx Pass-8 FRONT CLEAN EDISP 68% containment in log10(E).
    Low-E: 15%  -> sigma_log ~ 0.065
    High-E: 8%  -> sigma_log ~ 0.035
    Smooth transition around 1 GeV.
    """
    sigma_pct = 0.15 - 0.07 / (1.0 + (1.0 / np.maximum(E_gev, 1e-3)))
    sigma_pct = np.clip(sigma_pct, 0.08, 0.15)
    sigma_log = sigma_pct / np.log(10)
    return sigma_log


def build_edisp_kernel_matrix(E_centers):
    n = len(E_centers)
    logE = np.log10(E_centers)
    sigma_log = edisp_sigma_log(E_centers)
    K = np.zeros((n, n))
    for j in range(n):
        d = logE - logE[j]
        k = np.exp(-0.5 * (d / sigma_log[j]) ** 2)
        K[:, j] = k / k.sum()
    return K


def smear_cube_energy(cube, K):
    """cube shape (n_e, ny, nx). Apply energy-mixing matrix K along axis 0."""
    n_e, ny, nx = cube.shape
    flat = cube.reshape(n_e, ny * nx)
    out = K @ flat
    return out.reshape(n_e, ny, nx)


def smear_and_save_component_maps(model, K):
    files = [
        (f'GC_pion_model{model}_12yr_front_clean.fits',
         f'GC_pion_model{model}_12yr_front_clean_edisp.fits'),
        (f'GC_bremss_model{model}_12yr_front_clean.fits',
         f'GC_bremss_model{model}_12yr_front_clean_edisp.fits'),
        (f'GC_ics_model{model}_12yr_front_clean.fits',
         f'GC_ics_model{model}_12yr_front_clean_edisp.fits'),
        ('GC_GCE_model_12yr_front_clean.fits',
         'GC_GCE_model_12yr_front_clean_edisp.fits'),
    ]
    for in_name, out_name in files:
        in_path = f'{ANA}/{in_name}'
        out_path = f'{SMEARED_DIR}/{out_name}'
        if os.path.exists(out_path):
            print(f"  [skip] {out_name} already exists", flush=True)
            continue
        print(f"  smearing {in_name} ...", flush=True)
        with fits.open(in_path) as h:
            data = h[0].data.astype(np.float64)
            hdr = h[0].header.copy()
        smeared = smear_cube_energy(data, K)
        hdr['EDISP_PR'] = ('lognormal', 'proxy EDISP applied')
        hdr['EDISP_SG'] = ('0.08-0.15', 'sigma_pct log10 range')
        fits.writeto(out_path, smeared.astype(np.float32),
                      header=hdr, overwrite=True)
        print(f"    -> saved {out_name}", flush=True)


def patch_stage2_for_smeared(model_focus):
    """Monkey-patch load_component_maps so it reads smeared maps for
    Pi0+Bremss+ICS+GCE while keeping iso/bubble as in Sanghwan srcmap.
    """
    import stage2_fit

    orig = stage2_fit.load_component_maps

    def load_component_maps_smeared(model, ebin):
        yroi, xroi = ROI_SLICE
        def _open(path):
            return fits.open(path)[0].data[ebin, yroi, xroi].astype(np.float64)
        pion_c = _open(f'{SMEARED_DIR}/GC_pion_model{model}_12yr_front_clean_edisp.fits')
        brem_c = _open(f'{SMEARED_DIR}/GC_bremss_model{model}_12yr_front_clean_edisp.fits')
        ics_c  = _open(f'{SMEARED_DIR}/GC_ics_model{model}_12yr_front_clean_edisp.fits')
        gce_c  = _open(f'{SMEARED_DIR}/GC_GCE_model_12yr_front_clean_edisp.fits')
        bub_c  = _open(f'{ANA}/GC_fermi_bubble_model_12yr_front_clean.fits')
        iso_c  = _open(f'{ANA}/GC_isotropic_model_12yr_front_clean.fits')
        iso_nc = _open(f'{ANA}/GC_isotropic_model_12yr_front_clean_no_convol.fits')
        bub_nc = _open(f'{ANA}/GC_fermi_bubble_model_12yr_front_clean_no_convol.fits')
        return pion_c + brem_c, ics_c, gce_c, bub_c, iso_c, iso_nc, bub_nc

    stage2_fit.load_component_maps = load_component_maps_smeared
    return orig


def main():
    if len(sys.argv) < 2:
        print("usage: diagnostic_F2_edisp_smeared_fit.py MODEL", file=sys.stderr)
        print("       (e.g. X or XLIX)", file=sys.stderr)
        sys.exit(1)

    model = sys.argv[1]
    os.chdir(WORK_DIR)

    print("=" * 90)
    print(f"Diagnostic F2 : EDISP-smeared GDE/GCE fit   model={model}")
    print("=" * 90)

    print()
    print("[1/4] Build energy-mixing kernel from approximate EDISP 68% curve")
    E, dE = load_E_array()
    K = build_edisp_kernel_matrix(E)
    sigmas = edisp_sigma_log(E) * np.log(10)
    print(f"{'bin':>3} {'E':>8}  {'sigma_pct':>10}  "
          f"{'kernel self-weight':>18}  {'kernel-nn':>12}")
    for i in range(len(E)):
        nn = K[i-1, i] if i > 0 else 0.0
        print(f"{i:>3} {E[i]:>8.3f}  {sigmas[i]:>10.3f}  "
              f"{K[i,i]:>18.3f}  {nn:>12.3f}")

    print()
    print(f"[2/4] Smear and save GDE/GCE cubes for model {model}")
    smear_and_save_component_maps(model, K)

    print()
    print(f"[3/4] Patch stage2 to read smeared maps; run global c_gce fit")
    _orig = patch_stage2_for_smeared(model)

    os.environ['FIT_SUFFIX'] = os.environ.get('FIT_SUFFIX', 'edispSmeared')

    from diagnostic_C_global_fit import fit_one_model

    out_pkl = fit_one_model(model)

    print()
    print(f"[4/4] Done. Result: {out_pkl}")
    print("      Compare with baseline via diagnostic_C_compare.py")


if __name__ == '__main__':
    main()
