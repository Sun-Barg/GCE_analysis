import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    WORK_DIR, OUT_DIR, CCUBE, ROI_SLICE,
    DISK_MASK_NPY, PSC_MASK_SANGHWAN, PSC_MASK_CHOLIS_RAW,
)
from prepare_masks import FULL_MASK_PATH, prepare_masks
from stage2_fit import load_energy_axis


def load_three_masks():
    yroi, xroi = ROI_SLICE

    if not os.path.exists(FULL_MASK_PATH):
        prepare_masks()
    ours = np.load(FULL_MASK_PATH).astype(np.float32)

    psc_sh = np.load(PSC_MASK_SANGHWAN).astype(np.float32)
    disk_full = np.load(DISK_MASK_NPY).astype(np.float32)
    if psc_sh.shape[1] >= 600 and psc_sh.shape[2] >= 600:
        psc_sh = psc_sh[:, yroi, xroi]
    if disk_full.shape == (600, 600):
        disk = disk_full[yroi, xroi]
    else:
        disk = disk_full
    sanghwan = psc_sh * disk[None, :, :]

    raw = np.load(PSC_MASK_CHOLIS_RAW).astype(np.float32)
    if raw.shape[1] >= 600 and raw.shape[2] >= 600:
        raw_400 = raw[:, yroi, xroi]
    else:
        raw_400 = raw
    raw_disk = raw_400 * disk[None, :, :]

    return {'Ours (flipped+disk)': ours,
            'Sanghwan (DR2+disk)': sanghwan,
            'Raw Zhong (unflipped+disk)': raw_disk}, disk


def fraction_summary(masks, E):
    n = len(E)
    lines = []
    lines.append(f"{'Bin':>4} {'E[GeV]':>8}  "
                  + "  ".join(f"{name:>24}" for name in masks.keys())
                  + "  " + f"{'Cholis target':>14}")

    targets = [71.8, 62.9, 52.2, 38.5, 29.2, 23.4, 19.0, 16.3,
                13.0, 12.9, 11.6, 11.5, 10.3, 10.3]
    for eb in range(n):
        fracs = []
        for m in masks.values():
            kept = m[eb].sum() / (m.shape[1] * m.shape[2])
            fracs.append((1 - kept) * 100)
        frag = "  ".join(f"{f:>23.1f}%" for f in fracs)
        lines.append(f"{eb:>4} {E[eb]:>8.3f}  {frag}  {targets[eb]:>13.1f}%")
    return "\n".join(lines)


def pairwise_agreement(masks, E):
    names = list(masks.keys())
    n = len(E)
    out_lines = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = masks[names[i]]
            b = masks[names[j]]
            out_lines.append(f"\n=== {names[i]} vs {names[j]} ===")
            out_lines.append(f"{'Bin':>4} {'E':>8} {'agree%':>8} "
                              f"{'only_A':>10} {'only_B':>10} {'both_1':>10} {'both_0':>10}")
            for eb in range(n):
                a_b = a[eb]; b_b = b[eb]
                agree = (a_b == b_b).mean() * 100
                only_a = ((a_b == 1) & (b_b == 0)).sum()
                only_b = ((a_b == 0) & (b_b == 1)).sum()
                both_1 = ((a_b == 1) & (b_b == 1)).sum()
                both_0 = ((a_b == 0) & (b_b == 0)).sum()
                out_lines.append(f"{eb:>4} {E[eb]:>8.3f} {agree:>7.2f}% "
                                  f"{only_a:>10d} {only_b:>10d} {both_1:>10d} {both_0:>10d}")
    return "\n".join(out_lines)


def plot_overlay(masks, E, bins_to_plot, outpath):
    names = list(masks.keys())
    nm = len(names)
    nb = len(bins_to_plot)

    extent = [30, -30, -30, 30]

    fig, axes = plt.subplots(nm + 1, nb, figsize=(4.3*nb, 3.7*(nm+1)),
                              squeeze=False)
    for col, eb in enumerate(bins_to_plot):
        for row, name in enumerate(names):
            ax = axes[row, col]
            m = masks[name][eb]
            kept_frac = m.sum() / (m.shape[0] * m.shape[1])
            ax.imshow(m, origin='lower', extent=extent,
                       cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
            title = f"{name}\nbin {eb} E={E[eb]:.2f} GeV" if col == 0 else \
                     f"bin {eb} E={E[eb]:.2f} GeV"
            ax.set_title(title + f"  kept={kept_frac*100:.1f}%", fontsize=10)
            if col == 0:
                ax.set_ylabel('b [deg]', fontsize=9)
            if row == nm - 1:
                ax.set_xlabel('l [deg]', fontsize=9)
            ax.set_xticks([-20, -10, 0, 10, 20])
            ax.set_yticks([-20, -10, 0, 10, 20])

    for col, eb in enumerate(bins_to_plot):
        ax = axes[nm, col]
        m0 = masks[names[0]][eb]
        m1 = masks[names[1]][eb]
        diff = np.full(m0.shape, 0.5)
        diff[(m0 == 1) & (m1 == 1)] = 1.0
        diff[(m0 == 0) & (m1 == 0)] = 0.0
        diff[(m0 == 1) & (m1 == 0)] = 0.75
        diff[(m0 == 0) & (m1 == 1)] = 0.25

        cmap = ListedColormap(['#000000', '#ff6666', '#ffffff', '#6666ff'])
        ax.imshow(diff, origin='lower', extent=extent,
                   cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        only_ours = ((m0 == 1) & (m1 == 0)).sum()
        only_sh = ((m0 == 0) & (m1 == 1)).sum()
        ax.set_title(f"Ours vs Sanghwan diff\nbin {eb}  "
                      f"only_Ours={only_ours}  only_Sh={only_sh}",
                      fontsize=10)
        ax.set_xlabel('l [deg]', fontsize=9)
        if col == 0:
            ax.set_ylabel('b [deg]', fontsize=9)
        ax.set_xticks([-20, -10, 0, 10, 20])
        ax.set_yticks([-20, -10, 0, 10, 20])

    plt.suptitle('Mask overlay comparison  (black=masked out, white=kept)', fontsize=13)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close()


def plot_fraction_curve(masks, E, outpath):
    targets = np.array([71.8, 62.9, 52.2, 38.5, 29.2, 23.4, 19.0, 16.3,
                         13.0, 12.9, 11.6, 11.5, 10.3, 10.3])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(E, targets, 'k*', markersize=12, label='Cholis+2022 Table III')
    colors = ['C0', 'C3', 'C2']
    for (name, m), col in zip(masks.items(), colors):
        fracs_masked = (1 - m.sum(axis=(1, 2)) / (m.shape[1] * m.shape[2])) * 100
        ax.plot(E, fracs_masked, 'o-', color=col, label=name, alpha=0.85)
    ax.set_xscale('log')
    ax.set_xlabel('E [GeV]', fontsize=12)
    ax.set_ylabel('Masked fraction [%]', fontsize=12)
    ax.set_title('Masked fraction per energy bin vs Cholis target', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close()


def main():
    os.chdir(WORK_DIR)
    E, dE = load_energy_axis(CCUBE)
    masks, disk = load_three_masks()

    print()
    print('=' * 110)
    print('Mask fraction summary (% of ROI masked OUT)')
    print('=' * 110)
    summary = fraction_summary(masks, E)
    print(summary)

    print()
    print('=' * 110)
    print('Pairwise pixel-level agreement')
    print('=' * 110)
    pairwise = pairwise_agreement(masks, E)
    print(pairwise)

    out_txt = os.path.join(OUT_DIR, 'diagnostic_D1_mask_overlay.txt')
    with open(out_txt, 'w') as f:
        f.write('Mask comparison\n' + '='*110 + '\n\n')
        f.write('Per-bin fractions vs Cholis+2022 Table III target\n')
        f.write(summary + '\n\n')
        f.write('Pairwise pixel agreement\n')
        f.write(pairwise + '\n')
    print(f"\nSaved: {out_txt}")

    bins_to_plot = [0, 3, 6, 10, 13]
    png_overlay = os.path.join(OUT_DIR, 'diagnostic_D1_mask_overlay.png')
    plot_overlay(masks, E, bins_to_plot, png_overlay)
    print(f"Saved: {png_overlay}")

    png_curve = os.path.join(OUT_DIR, 'diagnostic_D1_mask_fraction_curve.png')
    plot_fraction_curve(masks, E, png_curve)
    print(f"Saved: {png_curve}")


if __name__ == '__main__':
    main()
