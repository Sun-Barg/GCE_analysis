"""
Diagnostic G : Compare Zenodo original GDE maps vs Sanghwan pre-computed srcmaps

Purpose
-------
Check whether the Pi0, Bremss, and ICS template fluxes we fit against
(from Sanghwan's 0.1 deg gtsrcmaps output) match the original Zenodo
Cholis maps (0.25 deg, pre-gtsrcmaps) that the paper's GCE spectra were
actually derived from.

Any systematic mismatch here would explain Group A's residual low-energy
deficit that mask calibration + EDISP testing could not eliminate.

Method
------
- Zenodo maps: units E^2 dPhi/dE [GeV cm-2 s-1 sr-1], 0.25 deg pix, 38 bins
- Sanghwan maps: units counts per 0.1 deg pixel, 14 bins (gtsrcmaps output).
  To compare to Zenodo flux, divide by exposure cube and pixel solid angle.

- Bin mapping (Zenodo 38 -> analysis 14) as per Cholis_ZENODO_README.md:
    0->7, 1->8, 2->9, 3->10, 4->11, 5->12, 6->13, 7->14, 8->15, 9->16, 10->17,
    11->[18,19,20] (mean), 12->[21,22,23] (mean), 13->[24,25,26] (mean).

- Compute mean flux inside inner 40x40 deg (following paper convention).

- Models: 5 we're fitting (X, XV, XLVIII, XLIX, LIII) via their Zenodo code
  mapping (ch, c4, 8l, 8t, bf).

Outputs
-------
- diagnostic_G_gde_flux_comparison.png: Pi0+Bremss, ICS per model, 14-bin
  flux curves with Zenodo-vs-Sanghwan overlay.
- diagnostic_G_gde_flux_comparison.txt: tabular per-bin ratios.

Runtime: ~1 minute (I/O dominated).
"""
import os, sys, numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR, CCUBE, EXPCUBE_CENTER, TEMPLATES_DIR, ROI_SLICE
from stage2_fit import load_energy_axis, solid_angle_per_pixel
from astropy.wcs import WCS

ANA = f'{WORK_DIR}/GC_analysis_sanghwan'
ZENODO_GDE_DIR = f'{TEMPLATES_DIR}/GALACTIC_DIFFUSE_EMISSION_MAPS_0p25deg'

MODEL_CODES = {
    'X': 'ch',
    'XV': 'c4',
    'XLVIII': '8l',
    'XLIX': '8t',
    'LIII': 'bf',
}

ZENODO_38TO14 = {
    0: [7], 1: [8], 2: [9], 3: [10], 4: [11], 5: [12], 6: [13],
    7: [14], 8: [15], 9: [16], 10: [17],
    11: [18, 19, 20], 12: [21, 22, 23], 13: [24, 25, 26],
}

ZENODO_EBINS_MEV = [
    (43.8587, 57.0013), (57.0013, 74.082), (74.082, 96.2812),
    (96.2812, 125.133), (125.133, 162.629), (162.629, 211.362),
    (211.362, 274.698), (274.698, 357.014), (357.014, 463.995),
    (463.995, 603.034), (603.034, 783.737), (783.737, 1018.59),
    (1018.59, 1323.82), (1323.82, 1720.51), (1720.51, 2236.07),
    (2236.07, 2906.12), (2906.12, 3776.96), (3776.96, 4908.75),
    (4908.75, 6379.69), (6379.69, 8291.4), (8291.4, 10776.0),
    (10776.0, 14005.1), (14005.1, 18201.8), (18201.8, 23656.1),
    (23656.1, 30744.8), (30744.8, 39957.6), (39957.6, 51931.2),
    (51931.2, 67492.7), (67492.7, 87717.4), (87717.4, 114002.0),
    (114002.0, 148164.0), (148164.0, 192562.0), (192562.0, 250265.0),
    (250265.0, 325258.0), (325258.0, 422724.0), (422724.0, 549396.0),
    (549396.0, 714027.0), (714027.0, 927989.0),
]

INNER_40x40_PIXELS = slice(100, 500), slice(100, 500)


def load_zenodo_cube(component, model_code):
    path = f'{ZENODO_GDE_DIR}/{component}_{model_code}_Map_flux_E_50-814008_MeV_InnerGalaxy_60x60.fits'
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with fits.open(path) as h:
        data = h[0].data.astype(np.float64)
        header = h[0].header
    return data, header


def zenodo_e2flux_to_14bin_flux(zenodo_cube):
    """Convert Zenodo 38-bin E^2 dPhi/dE [GeV cm-2 s-1 sr-1]
    to 14-bin E^2 dPhi/dE (group average for bins 11-13)."""
    n_y, n_x = zenodo_cube.shape[1], zenodo_cube.shape[2]
    out_14 = np.zeros((14, n_y, n_x))
    for eb_14, eb_list in ZENODO_38TO14.items():
        if len(eb_list) == 1:
            out_14[eb_14] = zenodo_cube[eb_list[0]]
        else:
            out_14[eb_14] = zenodo_cube[eb_list].mean(axis=0)
    return out_14


def mean_inside_inner40x40(zenodo_cube_14):
    """Mean of E^2 dPhi/dE over the Inner 40x40 deg window at 0.25 deg res.
    Zenodo map is 240x240 (60 deg / 0.25 deg). Inner 40x40 = central 160x160."""
    ny, nx = zenodo_cube_14.shape[1], zenodo_cube_14.shape[2]
    cy, cx = ny // 2, nx // 2
    half = int(20.0 / 0.25)
    sub = zenodo_cube_14[:, cy-half:cy+half, cx-half:cx+half]
    return sub.mean(axis=(1, 2))


def sanghwan_counts_to_e2flux(sanghwan_cube, exp_cube_pix_sr, E_centers_gev, dE_gev):
    """Convert Sanghwan counts/pixel to E^2 dPhi/dE [GeV cm-2 s-1 sr-1]
    for direct comparison with Zenodo.

    gtsrcmaps output:  counts = dPhi/dE * dE * Exposure * dOmega
    => dPhi/dE = counts / (dE * Exposure * dOmega)
    => E^2 dPhi/dE = counts * E^2 / (dE * Exposure * dOmega)
    """
    n_e = sanghwan_cube.shape[0]
    out = np.zeros_like(sanghwan_cube, dtype=np.float64)
    for eb in range(n_e):
        denom = exp_cube_pix_sr[eb] * dE_gev[eb]
        with np.errstate(divide='ignore', invalid='ignore'):
            dphi_de = np.where(denom > 0, sanghwan_cube[eb] / denom, 0.0)
        out[eb] = dphi_de * (E_centers_gev[eb] ** 2)
    return out


def mean_inside_inner40x40_sanghwan(sang_e2flux_14):
    """Mean of E^2 dPhi/dE over Inner 40x40 deg at 0.1 deg res.
    Sanghwan ROI is 400x400 pixels = 40 deg on a side -> entire ROI is Inner 40x40."""
    return sang_e2flux_14.mean(axis=(1, 2))


def analyze_model(model_name, model_code, E_gev, dE_gev, exp_cube_pix_sr):
    print(f"\n--- Model {model_name} (Zenodo code: {model_code}) ---")

    try:
        zen_pi0, _ = load_zenodo_cube('pi0', model_code)
        zen_brm, _ = load_zenodo_cube('bremss', model_code)
        zen_ics, _ = load_zenodo_cube('ICS', model_code)
    except FileNotFoundError as e:
        print(f"  Zenodo files missing: {e}")
        return None

    zen_pi0_14 = zenodo_e2flux_to_14bin_flux(zen_pi0)
    zen_brm_14 = zenodo_e2flux_to_14bin_flux(zen_brm)
    zen_ics_14 = zenodo_e2flux_to_14bin_flux(zen_ics)

    zen_pb_40 = mean_inside_inner40x40(zen_pi0_14 + zen_brm_14)
    zen_ics_40 = mean_inside_inner40x40(zen_ics_14)

    sang_pb_path = f'{ANA}/GC_pion_model{model_name}_12yr_front_clean.fits'
    sang_brm_path = f'{ANA}/GC_bremss_model{model_name}_12yr_front_clean.fits'
    sang_ics_path = f'{ANA}/GC_ics_model{model_name}_12yr_front_clean.fits'

    with fits.open(sang_pb_path) as h:
        sang_pi0 = h[0].data.astype(np.float64)
    with fits.open(sang_brm_path) as h:
        sang_brm = h[0].data.astype(np.float64)
    with fits.open(sang_ics_path) as h:
        sang_ics = h[0].data.astype(np.float64)

    yroi, xroi = ROI_SLICE
    sang_pi0_roi = sang_pi0[:, yroi, xroi]
    sang_brm_roi = sang_brm[:, yroi, xroi]
    sang_ics_roi = sang_ics[:, yroi, xroi]

    sang_pb_e2 = sanghwan_counts_to_e2flux(
        sang_pi0_roi + sang_brm_roi, exp_cube_pix_sr, E_gev, dE_gev)
    sang_ics_e2 = sanghwan_counts_to_e2flux(
        sang_ics_roi, exp_cube_pix_sr, E_gev, dE_gev)

    sang_pb_40 = mean_inside_inner40x40_sanghwan(sang_pb_e2)
    sang_ics_40 = mean_inside_inner40x40_sanghwan(sang_ics_e2)

    return {
        'E': E_gev,
        'zen_pb': zen_pb_40, 'zen_ics': zen_ics_40,
        'sang_pb': sang_pb_40, 'sang_ics': sang_ics_40,
    }


def main():
    os.chdir(WORK_DIR)
    print("=" * 90)
    print("Diagnostic G : Zenodo vs Sanghwan GDE template flux comparison")
    print("=" * 90)

    E, dE = load_energy_axis(CCUBE)

    print(f"\n[prep] Building exposure cube and pixel solid angle...")
    with fits.open(CCUBE) as h:
        header = h[0].header
        wcs = WCS(header).dropaxis(2)
        cube_shape = h[0].data.shape
    sap_full = solid_angle_per_pixel(cube_shape[2], cube_shape[1], wcs)
    with fits.open(EXPCUBE_CENTER) as h:
        expcube_full = h[0].data.astype(np.float64)
    yroi, xroi = ROI_SLICE
    exp_cube_pix_sr = expcube_full[:, yroi, xroi] * sap_full[yroi, xroi][None, :, :]

    print(f"  ROI {yroi.stop-yroi.start} x {xroi.stop-xroi.start} pixels")
    print(f"  E range: {E[0]:.3f} - {E[-1]:.3f} GeV  (14 bins)")

    results = {}
    for name, code in MODEL_CODES.items():
        r = analyze_model(name, code, E, dE, exp_cube_pix_sr)
        if r is not None:
            results[name] = r

    if not results:
        print("No models could be loaded. Check Zenodo file paths.")
        return

    out_txt = f'{OUT_DIR}/diagnostic_G_gde_flux_comparison.txt'
    with open(out_txt, 'w') as f:
        f.write("Zenodo vs Sanghwan GDE template flux comparison (Inner 40x40 deg average)\n")
        f.write("=" * 95 + "\n\n")
        f.write("Units: E^2 dPhi/dE  [GeV cm^-2 s^-1 sr^-1]\n")
        f.write("ratio = Sanghwan / Zenodo\n\n")

        for name, r in results.items():
            f.write(f"\n=== Model {name} ===\n")
            f.write(f"{'bin':>3} {'E_gev':>8} "
                    f"{'zen_pi0+bremss':>16} {'sang_pi0+bremss':>16} {'ratio_pb':>10} "
                    f"{'zen_ics':>14} {'sang_ics':>14} {'ratio_ics':>10}\n")
            for eb in range(14):
                r_pb = r['sang_pb'][eb] / r['zen_pb'][eb] if r['zen_pb'][eb] > 0 else float('nan')
                r_ics = r['sang_ics'][eb] / r['zen_ics'][eb] if r['zen_ics'][eb] > 0 else float('nan')
                f.write(f"{eb:>3d} {r['E'][eb]:>8.3f} "
                        f"{r['zen_pb'][eb]:>16.4e} {r['sang_pb'][eb]:>16.4e} {r_pb:>10.3f} "
                        f"{r['zen_ics'][eb]:>14.4e} {r['sang_ics'][eb]:>14.4e} {r_ics:>10.3f}\n")

        f.write("\n\nSummary (mean ratio Sanghwan/Zenodo, low / mid / high energy):\n")
        f.write(f"{'Model':<8} {'Pi0+Brem_low':>14} {'Pi0+Brem_mid':>14} {'Pi0+Brem_hi':>14}"
                f" {'ICS_low':>10} {'ICS_mid':>10} {'ICS_hi':>10}\n")
        for name, r in results.items():
            r_pb = r['sang_pb'] / np.maximum(r['zen_pb'], 1e-30)
            r_ic = r['sang_ics'] / np.maximum(r['zen_ics'], 1e-30)
            low = r['E'] < 1.0; mid = (r['E'] >= 1.0) & (r['E'] <= 10.0); hi = r['E'] > 10.0
            f.write(f"{name:<8} {np.mean(r_pb[low]):>14.3f} {np.mean(r_pb[mid]):>14.3f} "
                    f"{np.mean(r_pb[hi]):>14.3f} "
                    f"{np.mean(r_ic[low]):>10.3f} {np.mean(r_ic[mid]):>10.3f} "
                    f"{np.mean(r_ic[hi]):>10.3f}\n")

    print(f"\nSaved: {out_txt}")

    fig, axes = plt.subplots(2, len(results), figsize=(4.3*len(results), 9))
    if len(results) == 1:
        axes = axes.reshape(2, 1)
    for i, (name, r) in enumerate(results.items()):
        ax = axes[0, i]
        ax.loglog(r['E'], r['zen_pb'], 'o-', color='C3',
                   label='Zenodo Pi0+Bremss', alpha=0.8)
        ax.loglog(r['E'], r['sang_pb'], 's--', color='C0',
                   label='Sanghwan Pi0+Bremss', alpha=0.8)
        ax.loglog(r['E'], r['zen_ics'], '^-', color='C1',
                   label='Zenodo ICS', alpha=0.8)
        ax.loglog(r['E'], r['sang_ics'], 'v--', color='C2',
                   label='Sanghwan ICS', alpha=0.8)
        ax.set_xlabel('E [GeV]')
        ax.set_title(f'Model {name}')
        if i == 0:
            ax.set_ylabel(r'$E^2\, d\Phi/dE$  [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
            ax.legend(fontsize=8, loc='lower left')
        ax.grid(True, which='both', alpha=0.3)

        ax_r = axes[1, i]
        ax_r.axhline(1.0, color='k', linestyle=':', alpha=0.5)
        ax_r.semilogx(r['E'], r['sang_pb'] / np.maximum(r['zen_pb'], 1e-30),
                       'o-', color='C0', label='Pi0+Brem ratio', alpha=0.85)
        ax_r.semilogx(r['E'], r['sang_ics'] / np.maximum(r['zen_ics'], 1e-30),
                       's--', color='C2', label='ICS ratio', alpha=0.85)
        ax_r.set_xlabel('E [GeV]')
        ax_r.set_ylim(0.3, 3.0)
        ax_r.grid(True, which='both', alpha=0.3)
        if i == 0:
            ax_r.set_ylabel('Sanghwan / Zenodo')
            ax_r.legend(fontsize=8)

    plt.suptitle('GDE template flux: Sanghwan gtsrcmaps output vs Zenodo original', fontsize=13)
    plt.tight_layout()
    out_png = f'{OUT_DIR}/diagnostic_G_gde_flux_comparison.png'
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")

    print("\n=== Quick summary ===")
    for name, r in results.items():
        r_pb = r['sang_pb'] / np.maximum(r['zen_pb'], 1e-30)
        r_ic = r['sang_ics'] / np.maximum(r['zen_ics'], 1e-30)
        low = r['E'] < 1.0
        mid = (r['E'] >= 1.0) & (r['E'] <= 10.0)
        print(f"  {name:<8} Pi0+Brem low/mid = {np.mean(r_pb[low]):.3f}/{np.mean(r_pb[mid]):.3f}  "
              f"ICS low/mid = {np.mean(r_ic[low]):.3f}/{np.mean(r_ic[mid]):.3f}")


if __name__ == '__main__':
    main()
