import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR, CCUBE, ROI_SLICE
from stage2_fit import load_energy_axis
from prepare_masks import FULL_MASK_PATH
from build_calibrated_mask import MASK_CALIBRATED_PATH, CALIBRATION_META_PATH


def main():
    os.chdir(WORK_DIR)
    if not os.path.exists(MASK_CALIBRATED_PATH):
        print(f"ERROR: {MASK_CALIBRATED_PATH} not found. Run build_calibrated_mask.py first.")
        sys.exit(1)
    if not os.path.exists(FULL_MASK_PATH):
        print(f"ERROR: {FULL_MASK_PATH} not found. Run prepare_masks.py first.")
        sys.exit(1)

    cal = np.load(MASK_CALIBRATED_PATH)
    ours = np.load(FULL_MASK_PATH)
    E, dE = load_energy_axis(CCUBE)

    meta = np.load(CALIBRATION_META_PATH)
    scales = meta['scales']
    achieved = meta['achieved']
    target = meta['target']

    targets_tbl = np.array([71.8, 62.9, 52.2, 38.5, 29.2, 23.4, 19.0, 16.3,
                             13.0, 12.9, 11.6, 11.5, 10.3, 10.3])

    cal_fracs = (1 - cal.mean(axis=(1, 2))) * 100
    ours_fracs = (1 - ours.mean(axis=(1, 2))) * 100

    print()
    print("=" * 95)
    print("Calibrated mask verification")
    print("=" * 95)
    print(f"{'Bin':>3} {'E':>8} {'target':>8} {'Calibrated':>12} "
          f"{'Ours (old)':>12} {'scale':>7} {'cal-tgt':>9} {'ours-tgt':>10}")
    print("-" * 95)
    for eb in range(14):
        e_cal = cal_fracs[eb] - targets_tbl[eb]
        e_our = ours_fracs[eb] - targets_tbl[eb]
        print(f"{eb:>3} {E[eb]:>8.3f} {targets_tbl[eb]:>7.1f}% "
              f"{cal_fracs[eb]:>11.1f}% {ours_fracs[eb]:>11.1f}% "
              f"{scales[eb]:>7.3f} {e_cal:>+8.2f}% {e_our:>+9.2f}%")

    print("-" * 95)
    cal_rms = np.sqrt(np.mean((cal_fracs - targets_tbl) ** 2))
    ours_rms = np.sqrt(np.mean((ours_fracs - targets_tbl) ** 2))
    cal_max = np.max(np.abs(cal_fracs - targets_tbl))
    ours_max = np.max(np.abs(ours_fracs - targets_tbl))
    print(f"{'':>3} {'':>8} {'':>8} {'rms':>11}%p "
          f"{'rms':>11}%p  {'':>7} {cal_rms:>+8.2f}  {ours_rms:>+9.2f}")
    print(f"{'':>3} {'':>8} {'':>8} {'max':>11}%p "
          f"{'max':>11}%p  {'':>7} {cal_max:>+8.2f}  {ours_max:>+9.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    ax.plot(E, targets_tbl, 'k*', markersize=14, label='Cholis+2022 target')
    ax.plot(E, ours_fracs, 'o-', color='C0', alpha=0.8, label='Ours (Zhong flipped)')
    ax.plot(E, cal_fracs, 's-', color='C2', alpha=0.9, label='Calibrated')
    ax.set_xscale('log')
    ax.set_xlabel('E [GeV]', fontsize=12)
    ax.set_ylabel('Masked fraction [%]', fontsize=12)
    ax.set_title('Masked fraction vs target', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)

    ax = axes[1]
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax.plot(E, ours_fracs - targets_tbl, 'o-', color='C0',
            alpha=0.8, label=f'Ours - target  (rms={ours_rms:.2f})')
    ax.plot(E, cal_fracs - targets_tbl, 's-', color='C2',
            alpha=0.9, label=f'Calibrated - target  (rms={cal_rms:.2f})')
    ax.set_xscale('log')
    ax.set_xlabel('E [GeV]', fontsize=12)
    ax.set_ylabel('Masked fraction error [%p]', fontsize=12)
    ax.set_title('Error vs target', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)

    plt.suptitle('Calibrated mask performance', fontsize=14)
    plt.tight_layout()
    outp1 = os.path.join(OUT_DIR, 'diagnostic_E1_calibrated_fraction.png')
    plt.savefig(outp1, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {outp1}")

    bins_to_plot = [0, 3, 6, 10, 13]
    nb = len(bins_to_plot)
    fig2, axes2 = plt.subplots(3, nb, figsize=(4.3*nb, 10))
    extent = [30, -30, -30, 30]
    for col, eb in enumerate(bins_to_plot):
        ax = axes2[0, col]
        ax.imshow(ours[eb], origin='lower', extent=extent,
                   cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'Ours\nbin {eb}  E={E[eb]:.2f} GeV\n'
                     f'kept={(1-ours_fracs[eb]/100)*100:.1f}%', fontsize=10)
        if col == 0: ax.set_ylabel('b [deg]')

        ax = axes2[1, col]
        ax.imshow(cal[eb], origin='lower', extent=extent,
                   cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'Calibrated  scale={scales[eb]:.2f}\n'
                     f'kept={(1-cal_fracs[eb]/100)*100:.1f}%  '
                     f'target kept={100-targets_tbl[eb]:.1f}%',
                     fontsize=10)
        if col == 0: ax.set_ylabel('b [deg]')

        ax = axes2[2, col]
        from matplotlib.colors import ListedColormap
        a = ours[eb]; b = cal[eb]
        diff = np.full(a.shape, 0.5)
        diff[(a == 1) & (b == 1)] = 1.0
        diff[(a == 0) & (b == 0)] = 0.0
        diff[(a == 1) & (b == 0)] = 0.75
        diff[(a == 0) & (b == 1)] = 0.25
        cmap = ListedColormap(['#000000', '#ff6666', '#ffffff', '#6666ff'])
        ax.imshow(diff, origin='lower', extent=extent,
                   cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        only_ours = ((a == 1) & (b == 0)).sum()
        only_cal = ((a == 0) & (b == 1)).sum()
        ax.set_title(f'diff: only_Ours={only_ours}  only_Cal={only_cal}',
                     fontsize=10)
        ax.set_xlabel('l [deg]')
        if col == 0: ax.set_ylabel('b [deg]')

    plt.suptitle('Calibrated vs Ours (top:Ours, mid:Calibrated, '
                 'bottom:diff [red=only_Ours, blue=only_Cal])',
                 fontsize=13)
    plt.tight_layout()
    outp2 = os.path.join(OUT_DIR, 'diagnostic_E1_calibrated_overlay.png')
    plt.savefig(outp2, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outp2}")


if __name__ == '__main__':
    main()
