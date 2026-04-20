import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, ZENODO_FIG_DIR, CHOLIS_REF_FILES, MODELS


def load_ours(model):
    p = os.path.join(OUT_DIR, f'GCE_model_{model}_haebarg_v_claude.pkl')
    with open(p, 'rb') as f:
        d = pickle.load(f)
    return d


def load_cholis(model):
    p = os.path.join(ZENODO_FIG_DIR, CHOLIS_REF_FILES[model])
    arr = np.loadtxt(p)
    return {'E': arr[:, 0], 'flux': arr[:, 1], 'lo': arr[:, 2], 'hi': arr[:, 3]}


def make_comparison_table():
    rows = []
    for m in MODELS:
        ours = load_ours(m)
        cho = load_cholis(m)
        E = ours['E']
        ratio = ours['flux_best'] / np.maximum(cho['flux'], 1e-30)
        mask_1_10 = (E >= 1.0) & (E <= 10.0)
        mask_03_1 = (E >= 0.3) & (E < 1.0)
        mask_10_50 = (E > 10.0) & (E <= 60.0)
        rows.append({
            'model': m,
            'mean_ratio_1_10': float(np.mean(ratio[mask_1_10])),
            'mean_ratio_03_1': float(np.mean(ratio[mask_03_1])),
            'mean_ratio_10_50': float(np.mean(ratio[mask_10_50])),
            'mean_ratio_all': float(np.mean(ratio)),
            'total_neg2_logL': ours['total_neg2_logL'],
            'flux_best': ours['flux_best'],
            'flux_lo': ours['flux_lo'],
            'flux_hi': ours['flux_hi'],
            'cholis_E': cho['E'],
            'cholis_flux': cho['flux'],
            'cholis_lo': cho['lo'],
            'cholis_hi': cho['hi'],
            'E': E,
            'ratio': ratio,
        })

    print()
    print('=' * 90)
    print(f"{'Model':<8} {'<ratio>(1-10GeV)':>18} {'<ratio>(0.3-1)':>16} "
          f"{'<ratio>(10-50)':>16} {'-2 log L':>14}")
    print('=' * 90)
    for r in rows:
        print(f"{r['model']:<8} {r['mean_ratio_1_10']:>18.3f} {r['mean_ratio_03_1']:>16.3f} "
              f"{r['mean_ratio_10_50']:>16.3f} {r['total_neg2_logL']:>14.2f}")
    print('=' * 90)
    return rows


def plot_comparison(rows, outpath):
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})
    for i, r in enumerate(rows):
        ax = axes[0, i]
        ax_r = axes[1, i]
        ax.errorbar(r['cholis_E'], r['cholis_flux'],
                    yerr=[r['cholis_flux']-r['cholis_lo'], r['cholis_hi']-r['cholis_flux']],
                    fmt='s', color='C3', label='Cholis+2022', capsize=3, markersize=5)
        ax.errorbar(r['E'], r['flux_best'],
                    yerr=[np.maximum(r['flux_best']-r['flux_lo'], 1e-30),
                          np.maximum(r['flux_hi']-r['flux_best'], 1e-30)],
                    fmt='o', color='C0', label='Claude v0', capsize=3, markersize=5)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(0.25, 60); ax.set_ylim(3e-8, 3e-6)
        ax.set_title(f"Model {r['model']}  (-2lnL = {r['total_neg2_logL']:.0f})", fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        if i == 0:
            ax.set_ylabel(r'$E^2 dN/dE$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]', fontsize=11)
            ax.legend(loc='lower center', fontsize=9)

        ax_r.axhline(1.0, color='k', linestyle=':', alpha=0.5)
        ax_r.axhline(0.87, color='C2', linestyle='--', alpha=0.5,
                     label='prior pipeline conv. (~0.87)')
        ax_r.semilogx(r['E'], r['ratio'], 'o-', color='C0')
        ax_r.set_ylim(0.0, 1.6); ax_r.set_xlim(0.25, 60)
        ax_r.set_xlabel('E [GeV]', fontsize=10)
        ax_r.grid(True, which='both', alpha=0.3)
        if i == 0:
            ax_r.set_ylabel('Claude / Cholis', fontsize=10)
            ax_r.legend(loc='lower right', fontsize=8)

    plt.suptitle('Claude v0 pipeline vs Cholis+2022 (5 best-fit models)', fontsize=13)
    plt.tight_layout()
    plt.savefig(outpath, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {outpath}")


def save_summary_text(rows, outpath):
    with open(outpath, 'w') as f:
        f.write("Claude v0 pipeline vs Cholis+2022 — comparison summary\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"{'Model':<8} {'<ratio>(1-10GeV)':>18} {'<ratio>(0.3-1)':>16} "
                f"{'<ratio>(10-50)':>16} {'-2 log L':>14}\n")
        f.write("-" * 90 + "\n")
        for r in rows:
            f.write(f"{r['model']:<8} {r['mean_ratio_1_10']:>18.3f} {r['mean_ratio_03_1']:>16.3f} "
                    f"{r['mean_ratio_10_50']:>16.3f} {r['total_neg2_logL']:>14.2f}\n")
        f.write("\n\nPer-bin ratio (Claude / Cholis):\n")
        f.write("-" * 90 + "\n")
        e_template = rows[0]['E']
        f.write(f"{'E[GeV]':>10}  " + "  ".join(f"{r['model']:>10}" for r in rows) + "\n")
        for i, e in enumerate(e_template):
            f.write(f"{e:>10.3f}  " + "  ".join(f"{r['ratio'][i]:>10.3f}" for r in rows) + "\n")
        f.write("\n\nPer-bin Claude flux [GeV cm^-2 s^-1 sr^-1]:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'E[GeV]':>10}  " + "  ".join(f"{r['model']:>11}" for r in rows) + "\n")
        for i, e in enumerate(e_template):
            f.write(f"{e:>10.3f}  " + "  ".join(f"{r['flux_best'][i]:>11.4e}" for r in rows) + "\n")
        f.write("\n\nPer-bin Cholis flux [GeV cm^-2 s^-1 sr^-1]:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'E[GeV]':>10}  " + "  ".join(f"{r['model']:>11}" for r in rows) + "\n")
        for i, e in enumerate(e_template):
            f.write(f"{e:>10.3f}  " + "  ".join(f"{r['cholis_flux'][i]:>11.4e}" for r in rows) + "\n")
    print(f"Saved: {outpath}")


if __name__ == '__main__':
    rows = make_comparison_table()
    plot_comparison(rows, os.path.join(OUT_DIR, 'compare_5models_vs_cholis.png'))
    save_summary_text(rows, os.path.join(OUT_DIR, 'compare_5models_summary.txt'))
