"""
Diagnostic F1 — What PSF/EDISP state is our pipeline in?

Verifies which corrections are *already* baked into the pre-computed
Sanghwan srcmap FITS files we consume. Prints header keywords and does
quick sanity checks against Cholis+2022 Eq. 8-10.

Runs in seconds. No MCMC.
"""
import os, sys, numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR, CCUBE, EXPCUBE_CENTER
from stage2_fit import load_energy_axis

ANA = f'{WORK_DIR}/GC_analysis_sanghwan'


def dump_header_keywords(path, keys, label):
    if not os.path.exists(path):
        print(f"  [{label}] FILE MISSING: {path}")
        return
    with fits.open(path) as h:
        hdr = h[0].header
        print(f"\n  === {label} ===")
        print(f"    path: {os.path.basename(path)}")
        print(f"    shape: {h[0].data.shape}  dtype: {h[0].data.dtype}")
        for k in keys:
            v = hdr.get(k, '(not set)')
            print(f"    {k:<20} = {v}")


def sanity_pairs_convolved_vs_not(model):
    ics_c  = fits.open(f'{ANA}/GC_ics_model{model}_12yr_front_clean.fits')[0].data
    gce_c  = fits.open(f'{ANA}/GC_GCE_model_12yr_front_clean.fits')[0].data
    iso_c  = fits.open(f'{ANA}/GC_isotropic_model_12yr_front_clean.fits')[0].data
    iso_nc = fits.open(f'{ANA}/GC_isotropic_model_12yr_front_clean_no_convol.fits')[0].data
    bub_c  = fits.open(f'{ANA}/GC_fermi_bubble_model_12yr_front_clean.fits')[0].data
    bub_nc = fits.open(f'{ANA}/GC_fermi_bubble_model_12yr_front_clean_no_convol.fits')[0].data

    print(f"\n  Sanity: integrated counts per energy bin (entire 600x600 map):")
    print(f"  {'bin':>3} {'ICS_c':>12} {'GCE_c':>12} "
          f"{'iso_c':>12} {'iso_nc':>12} {'iso_c/nc':>10} "
          f"{'bub_c':>12} {'bub_nc':>12} {'bub_c/nc':>10}")
    for eb in range(gce_c.shape[0]):
        ic_sum  = float(ics_c[eb].sum())
        gc_sum  = float(gce_c[eb].sum())
        iso_c_s = float(iso_c[eb].sum())
        iso_nc_s = float(iso_nc[eb].sum())
        bub_c_s = float(bub_c[eb].sum())
        bub_nc_s = float(bub_nc[eb].sum())
        iso_ratio = iso_c_s / iso_nc_s if iso_nc_s else float('nan')
        bub_ratio = bub_c_s / bub_nc_s if bub_nc_s else float('nan')
        print(f"  {eb:>3d} {ic_sum:>12.3e} {gc_sum:>12.3e} "
              f"{iso_c_s:>12.3e} {iso_nc_s:>12.3e} {iso_ratio:>10.3f} "
              f"{bub_c_s:>12.3e} {bub_nc_s:>12.3e} {bub_ratio:>10.3f}")
    print()
    print("  Interpretation:")
    print("  - iso_c/nc == 1.0  -> isotropic srcmap was NOT PSF-convolved (flat template: PSF is identity)")
    print("  - iso_c/nc != 1.0  -> some convolution OR EDISP migration applied")
    print("  - If the ratio drifts with energy, that is the energy-redistribution signature of EDISP.")


def check_psf_convolution_signature(model):
    gce_c = fits.open(f'{ANA}/GC_GCE_model_12yr_front_clean.fits')[0].data
    print("\n  Is GCE PSF-convolved? Test: peak-to-edge ratio vs energy.")
    print("  If PSF-convolved: peak should fall with decreasing energy (broader PSF).")
    print("  If NOT PSF-convolved: peak remains sharp at all energies.")
    h, w = gce_c.shape[1], gce_c.shape[2]
    cy, cx = h // 2, w // 2
    print(f"  {'bin':>3} {'peak':>12} {'2deg_avg':>12} {'peak/ring':>12}")
    for eb in range(gce_c.shape[0]):
        img = gce_c[eb]
        peak = img[cy-2:cy+3, cx-2:cx+3].mean()
        yy, xx = np.indices(img.shape)
        r_pix = np.sqrt((yy-cy)**2 + (xx-cx)**2)
        ring = img[(r_pix > 15) & (r_pix < 25)].mean()
        ratio = peak / max(ring, 1e-30)
        print(f"  {eb:>3d} {peak:>12.3e} {ring:>12.3e} {ratio:>12.2f}")


def main():
    os.chdir(WORK_DIR)
    print("=" * 90)
    print("Diagnostic F1 : inspect PSF / EDISP state of pre-computed srcmap files")
    print("=" * 90)

    print()
    print("[1] CCUBE header")
    dump_header_keywords(CCUBE, [
        'NAXIS1', 'NAXIS2', 'NAXIS3', 'CTYPE1', 'CTYPE2', 'CTYPE3',
        'CDELT1', 'CDELT2', 'CDELT3', 'EVTYPE', 'IRFS',
    ], 'CCUBE')

    print()
    print("[2] EXPCUBE header")
    dump_header_keywords(EXPCUBE_CENTER, [
        'NAXIS1', 'NAXIS2', 'NAXIS3', 'CTYPE3', 'CDELT3', 'IRFS',
    ], 'EXPCUBE_CENTER')

    print()
    print("[3] GDE srcmap headers (check for gtsrcmaps-specific keywords)")
    for name, fn in [
        ('Pi0', 'GC_pion_modelX_12yr_front_clean.fits'),
        ('Bremss', 'GC_bremss_modelX_12yr_front_clean.fits'),
        ('ICS', 'GC_ics_modelX_12yr_front_clean.fits'),
        ('GCE', 'GC_GCE_model_12yr_front_clean.fits'),
        ('iso', 'GC_isotropic_model_12yr_front_clean.fits'),
        ('iso_no_convol', 'GC_isotropic_model_12yr_front_clean_no_convol.fits'),
        ('bubble', 'GC_fermi_bubble_model_12yr_front_clean.fits'),
        ('bubble_no_convol', 'GC_fermi_bubble_model_12yr_front_clean_no_convol.fits'),
    ]:
        dump_header_keywords(f'{ANA}/{fn}', [
            'EVTYPE', 'IRFS', 'APPLY_ED', 'CONVOL',
            'PSFCORR', 'RFACTOR', 'EDISP_BI', 'EDISPBIN',
        ], name)

    print()
    print("[4] Integrated-counts sanity check on model X")
    sanity_pairs_convolved_vs_not('X')

    print()
    print("[5] PSF convolution signature on GCE template (model X)")
    check_psf_convolution_signature('X')

    print()
    print("=" * 90)
    print("Conclusion (for the human to interpret):")
    print("  - If isotropic 'c/nc' ratio is ~ 1.0 at high E and drops at low E, EDISP is applied.")
    print("  - If GCE peak/ring ratio drops at low E, PSF convolution is applied.")
    print("  - Both positive -> PSF+EDISP both already in the srcmap.")
    print("  - Both negative -> need to re-generate srcmaps with convol=yes + apply_edisp.")
    print("=" * 90)


if __name__ == '__main__':
    main()
