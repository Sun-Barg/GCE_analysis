import os, sys, pickle, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR, MODELS, ROI_SLICE
from prepare_masks import FULL_MASK_PATH, prepare_masks

GROUP_A = ['X', 'XLVIII', 'LIII']
GROUP_B = ['XV', 'XLIX']


def run():
    os.chdir(WORK_DIR)
    if not os.path.exists(FULL_MASK_PATH):
        prepare_masks()
    full_mask = np.load(FULL_MASK_PATH)
    yroi, xroi = ROI_SLICE

    results = {}
    for m in MODELS:
        pkl_path = os.path.join(OUT_DIR, f'GCE_model_{m}_haebarg_v_claude.pkl')
        if not os.path.exists(pkl_path):
            print(f"[skip {m}] pkl missing"); continue
        d = pickle.load(open(pkl_path, 'rb'))
        E = d['E']; dE = d['dE']
        c_gas = d['coef_best'][:, 0]
        c_ics = d['coef_best'][:, 1]
        c_gce = d['coef_best'][:, 2]

        base = f'{WORK_DIR}/GC_analysis_sanghwan'
        ics_cube = fits.open(f'{base}/GC_ics_model{m}_12yr_front_clean.fits')[0].data[:, yroi, xroi]
        pion_cube = fits.open(f'{base}/GC_pion_model{m}_12yr_front_clean.fits')[0].data[:, yroi, xroi]
        brem_cube = fits.open(f'{base}/GC_bremss_model{m}_12yr_front_clean.fits')[0].data[:, yroi, xroi]
        gce_cube = fits.open(f'{base}/GC_GCE_model_12yr_front_clean.fits')[0].data[:, yroi, xroi]

        ics_masked_sum = np.array([(full_mask[eb] * ics_cube[eb]).sum() for eb in range(len(E))])
        pb_masked_sum = np.array([(full_mask[eb] * (pion_cube[eb] + brem_cube[eb])).sum()
                                   for eb in range(len(E))])
        gce_masked_sum = np.array([(full_mask[eb] * gce_cube[eb]).sum() for eb in range(len(E))])

        ics_contrib = c_ics * ics_masked_sum
        pb_contrib = c_gas * pb_masked_sum
        gce_contrib = c_gce * gce_masked_sum

        total_fit_counts = ics_contrib + pb_contrib + gce_contrib
        ics_frac = ics_contrib / np.maximum(total_fit_counts, 1e-30)
        gce_frac = gce_contrib / np.maximum(total_fit_counts, 1e-30)
        pb_frac = pb_contrib / np.maximum(total_fit_counts, 1e-30)

        results[m] = {
            'E': E, 'dE': dE, 'group': 'A' if m in GROUP_A else 'B',
            'c_gas': c_gas, 'c_ics': c_ics, 'c_gce': c_gce,
            'ics_map_sum': ics_masked_sum, 'pb_map_sum': pb_masked_sum,
            'gce_map_sum': gce_masked_sum,
            'ics_contrib': ics_contrib, 'pb_contrib': pb_contrib,
            'gce_contrib': gce_contrib,
            'ics_frac': ics_frac, 'gce_frac': gce_frac, 'pb_frac': pb_frac,
        }

    out_txt = os.path.join(OUT_DIR, 'diagnostic_C0_ics_contribution.txt')
    with open(out_txt, 'w') as f:
        f.write("Experiment C0 : c_ics x ICS_map contribution decomposition\n")
        f.write("=" * 110 + "\n")
        f.write("Shows whether Group B (XV, XLIX) low-E ratio comes from:\n")
        f.write("  (1) small c_ics coefficient [fit artifact]  OR\n")
        f.write("  (2) small ICS_map_sum (weak ICS model prediction itself) [physics]\n\n")

        f.write("\n--- Per-bin ICS map sum (mask-integrated, no coefficient) ---\n")
        f.write(f"{'E[GeV]':>10}  " + "  ".join(f"{m:>12}" for m in MODELS) + "\n")
        for eb in range(len(results[MODELS[0]]['E'])):
            row = "  ".join(f"{results[m]['ics_map_sum'][eb]:>12.3e}" for m in MODELS)
            f.write(f"{results[MODELS[0]]['E'][eb]:>10.3f}  " + row + "\n")

        f.write("\n--- Per-bin c_ics x ICS_map_sum (fitted ICS contribution, counts) ---\n")
        f.write(f"{'E[GeV]':>10}  " + "  ".join(f"{m:>12}" for m in MODELS) + "\n")
        for eb in range(len(results[MODELS[0]]['E'])):
            row = "  ".join(f"{results[m]['ics_contrib'][eb]:>12.3e}" for m in MODELS)
            f.write(f"{results[MODELS[0]]['E'][eb]:>10.3f}  " + row + "\n")

        f.write("\n--- Per-bin ICS / (ICS + gas + GCE) fraction ---\n")
        f.write(f"{'E[GeV]':>10}  " + "  ".join(f"{m:>10}" for m in MODELS) + "\n")
        for eb in range(len(results[MODELS[0]]['E'])):
            row = "  ".join(f"{results[m]['ics_frac'][eb]:>10.3f}" for m in MODELS)
            f.write(f"{results[MODELS[0]]['E'][eb]:>10.3f}  " + row + "\n")

        f.write("\n\n=== Group A vs B comparison ===\n")
        E = results[MODELS[0]]['E']
        low = E < 1.0
        idx_a = [m for m in MODELS if results[m]['group'] == 'A']
        idx_b = [m for m in MODELS if results[m]['group'] == 'B']

        def mean_ratio(key, mask):
            va = np.mean([np.mean(results[m][key][mask]) for m in idx_a])
            vb = np.mean([np.mean(results[m][key][mask]) for m in idx_b])
            return va, vb

        f.write(f"{'metric':<28} {'Group A':>14} {'Group B':>14} {'B/A':>8}\n")
        f.write("-" * 66 + "\n")
        for label, key in [('ICS_map_sum (E<1 GeV)', 'ics_map_sum'),
                            ('c_ics (E<1 GeV)', 'c_ics'),
                            ('c_ics x ICS_map (E<1 GeV)', 'ics_contrib'),
                            ('ICS fraction (E<1 GeV)', 'ics_frac'),
                            ('GCE fraction (E<1 GeV)', 'gce_frac'),
                            ('c_gce (E<1 GeV)', 'c_gce'),
                            ('c_gce x GCE_map (E<1 GeV)', 'gce_contrib')]:
            va, vb = mean_ratio(key, low)
            ratio_ba = vb / va if va != 0 else np.nan
            f.write(f"{label:<28} {va:>14.3e} {vb:>14.3e} {ratio_ba:>8.3f}\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'A': 'C0', 'B': 'C3'}

    ax = axes[0, 0]
    for m in MODELS:
        r = results[m]
        ax.loglog(r['E'], r['ics_map_sum'], 'o-',
                   color=colors[r['group']], alpha=0.8,
                   label=f"{m} ({r['group']})")
    ax.set_xlabel('E [GeV]'); ax.set_ylabel('Σ(ICS_map) within mask [counts/unit_c_ics]')
    ax.set_title('ICS map integrated flux (BEFORE coefficient)')
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=9)

    ax = axes[0, 1]
    for m in MODELS:
        r = results[m]
        ax.semilogx(r['E'], r['c_ics'], 'o-',
                     color=colors[r['group']], alpha=0.8,
                     label=f"{m} ({r['group']})")
    ax.set_xlabel('E [GeV]'); ax.set_ylabel('c_ics (fitted coefficient)')
    ax.set_title('Fitted c_ics coefficient')
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1, 0]
    for m in MODELS:
        r = results[m]
        ax.loglog(r['E'], r['ics_contrib'], 'o-',
                   color=colors[r['group']], alpha=0.8,
                   label=f"{m} ({r['group']})")
    ax.set_xlabel('E [GeV]'); ax.set_ylabel('c_ics × Σ(ICS_map) [counts]')
    ax.set_title('Fitted ICS contribution (coefficient × map)')
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1, 1]
    for m in MODELS:
        r = results[m]
        ax.semilogx(r['E'], r['gce_frac'], 'o-',
                     color=colors[r['group']], alpha=0.8,
                     label=f"{m} ({r['group']})")
    ax.set_xlabel('E [GeV]'); ax.set_ylabel('GCE counts / (ICS + gas + GCE) counts')
    ax.set_title('GCE counts share')
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=9)

    plt.suptitle('Experiment C0 : ICS contribution decomposition', fontsize=13)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'diagnostic_C0_ics_contribution.png')
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close()

    with open(os.path.join(OUT_DIR, 'diagnostic_C0_ics_contribution.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved: {out_txt}")
    print(f"Saved: {out_png}")

    print("\n=== KEY SUMMARY ===")
    E = results[MODELS[0]]['E']
    low = E < 1.0
    print(f"{'Model':<8} {'grp':<4} {'<ICS_map>_lowE':>16} {'<c_ics>_lowE':>14} "
          f"{'<c_ics·ICS>_lowE':>18} {'<GCE_frac>_lowE':>16}")
    for m in MODELS:
        r = results[m]
        print(f"{m:<8} {r['group']:<4} "
              f"{np.mean(r['ics_map_sum'][low]):>16.3e} "
              f"{np.mean(r['c_ics'][low]):>14.3f} "
              f"{np.mean(r['ics_contrib'][low]):>18.3e} "
              f"{np.mean(r['gce_frac'][low]):>16.3f}")


if __name__ == '__main__':
    run()
