import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    WORK_DIR, OUT_DIR, CCUBE, EXPCUBE_CENTER, MODELS, ROI_SLICE,
)
from prepare_masks import FULL_MASK_PATH, prepare_masks
from stage2_fit import (
    load_energy_axis, solid_angle_per_pixel,
)

GROUP = {'X': 'A', 'XV': 'B', 'XLVIII': 'A', 'XLIX': 'B', 'LIII': 'A'}
BINS_TO_PLOT = [0, 3, 6, 10]


def load_gce_cube(model):
    path = f'{WORK_DIR}/GC_analysis_sanghwan/GC_GCE_model_12yr_front_clean.fits'
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return fits.open(path)[0].data.astype(np.float64)


def load_gce_cube_noconv(model):
    path = f'{WORK_DIR}/GC_analysis_sanghwan/GC_GCE_model_12yr_front_clean_no_convol.fits'
    if not os.path.exists(path):
        return None
    return fits.open(path)[0].data.astype(np.float64)


def radial_profile(img_2d, center_pix):
    h, w = img_2d.shape
    y, x = np.indices((h, w))
    cy, cx = center_pix
    r_pix = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_deg = r_pix * 0.1

    r_edges = np.arange(0, 21, 0.5)
    r_ctr = 0.5 * (r_edges[:-1] + r_edges[1:])
    prof = np.zeros(len(r_ctr))
    for i in range(len(r_ctr)):
        m = (r_deg >= r_edges[i]) & (r_deg < r_edges[i+1])
        if m.any():
            prof[i] = img_2d[m].mean()
    return r_ctr, prof


def main():
    os.chdir(WORK_DIR)
    if not os.path.exists(FULL_MASK_PATH):
        prepare_masks()
    full_mask = np.load(FULL_MASK_PATH).astype(np.float32)
    yroi, xroi = ROI_SLICE

    E, dE = load_energy_axis(CCUBE)

    ccube_hdu = fits.open(CCUBE)
    wcs = WCS(ccube_hdu[0].header).dropaxis(2)
    sap_full = solid_angle_per_pixel(
        ccube_hdu[0].data.shape[2], ccube_hdu[0].data.shape[1], wcs)
    expcube_full = fits.open(EXPCUBE_CENTER)[0].data.astype(np.float64)
    exp_cube_pix_sr = expcube_full[:, yroi, xroi] * sap_full[yroi, xroi][None, :, :]

    note = (
        "GCE map is written as GC_GCE_model_12yr_front_clean.fits without {model}\n"
        "suffix. We verify here that it IS identical across models by checking\n"
        "file existence / sanity. Per-model c_gce_global values differ due to\n"
        "differing c_ics, c_gas values absorbing counts at low-E.\n"
    )

    print('=' * 100)
    print('GCE map cross-model verification')
    print('=' * 100)
    gce_path_default = f'{WORK_DIR}/GC_analysis_sanghwan/GC_GCE_model_12yr_front_clean.fits'
    for m in MODELS:
        alt_path = f'{WORK_DIR}/GC_analysis_sanghwan/GC_GCE_model{m}_12yr_front_clean.fits'
        if os.path.exists(alt_path):
            print(f"  [{m}] per-model GCE file EXISTS: {alt_path}")
            arr = fits.open(alt_path)[0].data
            arr_def = fits.open(gce_path_default)[0].data
            if arr.shape == arr_def.shape:
                rel_diff = np.abs(arr - arr_def).sum() / (np.abs(arr_def).sum() + 1e-30)
                print(f"      rel abs diff vs default = {rel_diff:.3e}")
        else:
            print(f"  [{m}] uses default (model-independent) GCE file")
    print(f"  Default: {gce_path_default}")
    print()

    gc_cube = fits.open(gce_path_default)[0].data.astype(np.float64)
    gc_cube_nc = None
    nc_path = f'{WORK_DIR}/GC_analysis_sanghwan/GC_GCE_model_12yr_front_clean_no_convol.fits'
    if os.path.exists(nc_path):
        gc_cube_nc = fits.open(nc_path)[0].data.astype(np.float64)

    print('=' * 100)
    print('GCE map spatial analysis (model-independent shape)')
    print('=' * 100)
    print(note)

    n_bins = len(E)
    center_pix = (gc_cube.shape[1] // 2, gc_cube.shape[2] // 2)
    cy_roi = (yroi.stop - yroi.start) // 2
    cx_roi = (xroi.stop - xroi.start) // 2

    fig1, axes1 = plt.subplots(2, len(BINS_TO_PLOT),
                                figsize=(4.3*len(BINS_TO_PLOT), 7.5))
    extent_roi = [30, -30, -30, 30]

    for col, eb in enumerate(BINS_TO_PLOT):
        ax = axes1[0, col]
        img = gc_cube[eb, yroi, xroi]
        with np.errstate(divide='ignore', invalid='ignore'):
            img_norm = np.where(exp_cube_pix_sr[eb] > 0,
                                 img / exp_cube_pix_sr[eb], 0.0)
        vmax = np.percentile(img_norm[img_norm > 0], 99) if (img_norm > 0).any() else 1.0
        im = ax.imshow(img_norm, origin='lower', extent=extent_roi,
                        cmap='viridis', vmin=0, vmax=vmax)
        ax.set_title(f'GCE / exp / sr\nbin {eb}  E={E[eb]:.2f} GeV', fontsize=10)
        if col == 0:
            ax.set_ylabel('b [deg]', fontsize=10)
        ax.set_xlabel('l [deg]', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes1[1, col]
        masked = img_norm * full_mask[eb]
        im = ax.imshow(masked, origin='lower', extent=extent_roi,
                        cmap='viridis', vmin=0, vmax=vmax)
        kept = full_mask[eb].sum() / full_mask[eb].size * 100
        ax.set_title(f'After mask ({kept:.0f}% kept)', fontsize=10)
        if col == 0:
            ax.set_ylabel('b [deg]', fontsize=10)
        ax.set_xlabel('l [deg]', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle('GCE template (convolved) / exposure / solid_angle  per bin', fontsize=13)
    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, 'diagnostic_D2_gce_map_per_bin.png')
    plt.savefig(out1, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out1}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    gce_avg = np.zeros(n_bins)
    gce_avg_unmasked = np.zeros(n_bins)
    gce_center_peak = np.zeros(n_bins)
    for eb in range(n_bins):
        img_full = gc_cube[eb, yroi, xroi]
        with np.errstate(divide='ignore', invalid='ignore'):
            img_n = np.where(exp_cube_pix_sr[eb] > 0,
                              img_full / exp_cube_pix_sr[eb], 0.0)
        mask_eb = full_mask[eb]
        ms = mask_eb.sum()
        gce_avg[eb] = (mask_eb * img_n).sum() / ms if ms > 0 else 0
        gce_avg_unmasked[eb] = img_n.mean()
        yy, xx = np.indices(img_n.shape)
        center_region = (np.abs(yy - cy_roi) < 10) & (np.abs(xx - cx_roi) < 10)
        gce_center_peak[eb] = img_n[center_region].mean()

    e2_dE = (E ** 2) / dE
    ax = axes2[0]
    ax.loglog(E, gce_avg * e2_dE, 'o-', color='C0', label='Masked avg (analysis)')
    ax.loglog(E, gce_avg_unmasked * e2_dE, 's--', color='C1', alpha=0.7,
              label='Unmasked avg (all ROI)')
    ax.loglog(E, gce_center_peak * e2_dE, '^:', color='C3', alpha=0.7,
              label='Central 2x2 deg peak')
    ax.set_xlabel('E [GeV]', fontsize=11)
    ax.set_ylabel('GCE_avg × E² / ΔE\n(per-pixel GCE flux in GeV/cm²/s/sr)', fontsize=11)
    ax.set_title('GCE template shape (fixed across models)', fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)

    ax = axes2[1]
    for eb in BINS_TO_PLOT:
        img_full = gc_cube[eb, yroi, xroi]
        with np.errstate(divide='ignore', invalid='ignore'):
            img_n = np.where(exp_cube_pix_sr[eb] > 0,
                              img_full / exp_cube_pix_sr[eb], 0.0)
        r_ctr, prof = radial_profile(img_n, (cy_roi, cx_roi))
        ax.semilogy(r_ctr, prof, 'o-', alpha=0.8,
                     label=f'bin {eb}  E={E[eb]:.2f} GeV')
    ax.set_xlabel('Radius from (l=0, b=0) [deg]', fontsize=11)
    ax.set_ylabel('<GCE / exp / sr>', fontsize=11)
    ax.set_title('GCE radial profile (un-masked)', fontsize=11)
    ax.set_xlim(0, 20)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, 'diagnostic_D2_gce_spectrum_and_profile.png')
    plt.savefig(out2, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out2}")

    import pickle
    rows = []
    for m in MODELS:
        for suffix, label in [('', 'orig'), ('_globalCgce', 'global')]:
            p = os.path.join(OUT_DIR,
                              f'GCE_model_{m}_haebarg_v_claude{suffix}.pkl')
            if not os.path.exists(p):
                continue
            d = pickle.load(open(p, 'rb'))
            rows.append({'model': m, 'suffix': label, 'data': d})

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    for r in rows:
        if r['suffix'] != 'orig':
            continue
        E_r = r['data']['E']
        c_gce_per_bin = r['data']['coef_best'][:, 2]
        ax = axes3[0]
        ax.semilogx(E_r, c_gce_per_bin, 'o-',
                     label=f"{r['model']} ({GROUP[r['model']]})", alpha=0.8)

    axes3[0].set_xlabel('E [GeV]', fontsize=11)
    axes3[0].set_ylabel('c_gce (original fit, 14 free)', fontsize=11)
    axes3[0].set_title('Fitted c_gce per bin — original fit', fontsize=11)
    axes3[0].grid(True, which='both', alpha=0.3)
    axes3[0].legend(fontsize=10)

    ax = axes3[1]
    for r in rows:
        if r['suffix'] != 'orig':
            continue
        d = r['data']
        E_r = d['E']; dE_r = d['dE']
        gce_avg_here = d.get('gce_avg_per_bin', None)
        if gce_avg_here is None:
            continue
        contrib = d['coef_best'][:, 2] * gce_avg_here * (E_r ** 2) / dE_r
        ax.loglog(E_r, contrib, 'o-',
                   label=f"{r['model']} ({GROUP[r['model']]})", alpha=0.8)

    ax.set_xlabel('E [GeV]', fontsize=11)
    ax.set_ylabel('c_gce × gce_avg × E² / ΔE', fontsize=11)
    ax.set_title('GCE flux contribution per bin — original fit', fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out3 = os.path.join(OUT_DIR, 'diagnostic_D2_cgce_per_model.png')
    plt.savefig(out3, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out3}")

    out_txt = os.path.join(OUT_DIR, 'diagnostic_D2_gce_map_analysis.txt')
    with open(out_txt, 'w') as f:
        f.write('GCE map spatial analysis\n' + '='*100 + '\n\n')
        f.write(note + '\n')
        f.write("Per-bin GCE template statistics (model-independent):\n")
        f.write("-"*100 + '\n')
        f.write(f"{'Bin':>4} {'E[GeV]':>8} "
                f"{'gce_avg_masked':>16} {'gce_avg_full':>14} {'gce_center_peak':>18}\n")
        for eb in range(n_bins):
            f.write(f"{eb:>4} {E[eb]:>8.3f} "
                    f"{gce_avg[eb]:>16.3e} {gce_avg_unmasked[eb]:>14.3e} "
                    f"{gce_center_peak[eb]:>18.3e}\n")

        f.write("\n\nPer-model c_gce × gce_avg comparison at low-E:\n")
        f.write("-"*100 + '\n')
        f.write(f"{'Model':<8} {'grp':<4} {'fit':<8} "
                f"{'c_gce[0]':>10} {'c_gce[3]':>10} {'c_gce[6]':>10} "
                f"{'ratio_low avg':>14}\n")
        for r in rows:
            d = r['data']
            cg = d['coef_best'][:, 2]
            f.write(f"{r['model']:<8} {GROUP[r['model']]:<4} {r['suffix']:<8} "
                    f"{cg[0]:>10.3f} {cg[3]:>10.3f} {cg[6]:>10.3f}")
            f.write('\n')

    print(f"Saved: {out_txt}")


if __name__ == '__main__':
    main()
