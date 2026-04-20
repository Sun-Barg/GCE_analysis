import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, ZENODO_FIG_DIR, CHOLIS_REF_FILES, MODELS

SUFFIX = os.environ.get('FIT_SUFFIX', 'globalCgce')
GROUP_A = ['X', 'XLVIII', 'LIII']
GROUP_B = ['XV', 'XLIX']


def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def load_cholis(m):
    arr = np.loadtxt(os.path.join(ZENODO_FIG_DIR, CHOLIS_REF_FILES[m]))
    return {'E': arr[:, 0], 'flux': arr[:, 1], 'lo': arr[:, 2], 'hi': arr[:, 3]}


def run():
    rows = []
    for m in MODELS:
        orig_pkl = os.path.join(OUT_DIR, f'GCE_model_{m}_haebarg_v_claude.pkl')
        new_pkl = os.path.join(OUT_DIR, f'GCE_model_{m}_haebarg_v_claude_{SUFFIX}.pkl')
        if not (os.path.exists(orig_pkl) and os.path.exists(new_pkl)):
            print(f"[skip {m}] orig={os.path.exists(orig_pkl)} new={os.path.exists(new_pkl)}")
            continue
        orig = load_pkl(orig_pkl); new = load_pkl(new_pkl); cho = load_cholis(m)
        E = orig['E']
        r_orig = orig['flux_best'] / np.maximum(cho['flux'], 1e-30)
        r_new = new['flux_best'] / np.maximum(cho['flux'], 1e-30)
        low = E < 1.0; mid = (E >= 1.0) & (E <= 10.0); hi = (E > 10.0) & (E <= 60.0)
        rows.append({
            'model': m, 'group': 'A' if m in GROUP_A else ('B' if m in GROUP_B else '?'),
            'E': E,
            'flux_orig': orig['flux_best'], 'flux_new': new['flux_best'],
            'flux_new_lo': new['flux_lo'], 'flux_new_hi': new['flux_hi'],
            'coef_orig': orig['coef_best'], 'coef_new': new['coef_best'],
            'c_gce_global': new['c_gce_global'],
            'c_gce_global_std': new['c_gce_global_std'],
            'cholis_E': cho['E'], 'cholis_flux': cho['flux'],
            'cholis_lo': cho['lo'], 'cholis_hi': cho['hi'],
            'ratio_orig': r_orig, 'ratio_new': r_new,
            'mean_r_low_orig': float(np.mean(r_orig[low])),
            'mean_r_low_new': float(np.mean(r_new[low])),
            'mean_r_mid_orig': float(np.mean(r_orig[mid])),
            'mean_r_mid_new': float(np.mean(r_new[mid])),
            'mean_r_hi_orig': float(np.mean(r_orig[hi])),
            'mean_r_hi_new': float(np.mean(r_new[hi])),
            'neg2logL_orig': orig['total_neg2_logL'],
            'neg2logL_new': new['total_neg2_logL'],
        })

    print()
    print('=' * 115)
    print(f"Experiment C :: global c_gce (shared across 14 bins)   suffix={SUFFIX}")
    print('=' * 115)

    if not rows:
        print(f"\n[diagnostic_C_compare] No Experiment C results found with suffix '{SUFFIX}'.")
        print("Expected files: GCE_model_<M>_haebarg_v_claude_{SUFFIX}.pkl")
        print("Check that diagnostic_C_parallel.py finished successfully.")
        print("If results exist under a different suffix, set:  export FIT_SUFFIX=<suffix>")
        print(f"\nLooked in: {OUT_DIR}")
        print("Found these pkl files:")
        for f in sorted(os.listdir(OUT_DIR)):
            if f.startswith('GCE_model_') and f.endswith('.pkl'):
                print(f"  {f}")
        return

    print(f"{'Model':<7} {'grp':<4} {'<r>_low orig→new':>22} {'<r>_mid orig→new':>22} "
          f"{'<r>_hi orig→new':>22} {'Δ(-2lnL)':>12}  {'c_gce_global':>14}")
    print('-' * 120)
    for r in rows:
        dlogL = r['neg2logL_new'] - r['neg2logL_orig']
        print(f"{r['model']:<7} {r['group']:<4} "
              f"{r['mean_r_low_orig']:>10.3f} →{r['mean_r_low_new']:>9.3f}  "
              f"{r['mean_r_mid_orig']:>10.3f} →{r['mean_r_mid_new']:>9.3f}  "
              f"{r['mean_r_hi_orig']:>10.3f} →{r['mean_r_hi_new']:>9.3f}  "
              f"{dlogL:>+12.2f}  {r['c_gce_global']:>8.3f} ± {r['c_gce_global_std']:>4.3f}")

    fig, axes = plt.subplots(2, len(rows), figsize=(4.5*len(rows), 9),
                              gridspec_kw={'height_ratios': [3, 1.2]})
    if len(rows) == 1:
        axes = axes.reshape(2, 1)
    for i, r in enumerate(rows):
        ax = axes[0, i]
        ax.errorbar(r['cholis_E'], r['cholis_flux'],
                    yerr=[r['cholis_flux']-r['cholis_lo'], r['cholis_hi']-r['cholis_flux']],
                    fmt='s', color='C3', label='Cholis+2022', capsize=3, markersize=5)
        ax.plot(r['E'], r['flux_orig'], 'o-', color='C0',
                label='Original (14 free c_gce)', markersize=5)
        ax.fill_between(r['E'], np.maximum(r['flux_new_lo'], 1e-12),
                         np.maximum(r['flux_new_hi'], 1e-12),
                         color='C2', alpha=0.25)
        ax.plot(r['E'], r['flux_new'], '^-', color='C2',
                label='Global c_gce (1 free)', markersize=5)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(0.25, 60); ax.set_ylim(3e-8, 5e-6)
        ax.set_title(f"Model {r['model']} (grp {r['group']})   "
                     f"c_gce={r['c_gce_global']:.2f}", fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        if i == 0:
            ax.set_ylabel(r'$E^2 dN/dE$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]', fontsize=10)
            ax.legend(loc='lower center', fontsize=8)

        ax_r = axes[1, i]
        ax_r.axhline(1.0, color='k', linestyle=':', alpha=0.5)
        ax_r.semilogx(r['E'], r['ratio_orig'], 'o-', color='C0',
                       label='orig', markersize=4)
        ax_r.semilogx(r['E'], r['ratio_new'], '^-', color='C2',
                       label='global', markersize=4)
        ax_r.set_ylim(0.0, 2.0); ax_r.set_xlim(0.25, 60)
        ax_r.set_xlabel('E [GeV]', fontsize=10); ax_r.grid(True, which='both', alpha=0.3)
        if i == 0:
            ax_r.set_ylabel('Claude / Cholis', fontsize=10)
            ax_r.legend(loc='upper right', fontsize=8)

    plt.suptitle(f"Experiment C : global c_gce across 14 bins  ({SUFFIX})", fontsize=13)
    plt.tight_layout()
    outp = os.path.join(OUT_DIR, f'diagnostic_C_comparison_{SUFFIX}.png')
    plt.savefig(outp, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved: {outp}")

    out_txt = os.path.join(OUT_DIR, f'diagnostic_C_comparison_{SUFFIX}.txt')
    with open(out_txt, 'w') as f:
        f.write(f"Experiment C : global c_gce  (suffix={SUFFIX})\n")
        f.write('=' * 115 + '\n')
        f.write("A single c_gce shared by all 14 energy bins — the GCE template's\n")
        f.write("built-in BPL spectral shape (from NFW²+BPL srcmap) is the only GCE spectrum allowed.\n\n")
        f.write("If orig-vs-new agrees well → the original 14-param fit was already tracking BPL shape\n")
        f.write("If new SED becomes flatter → bin-by-bin fit was absorbing counts at low-E that BPL\n")
        f.write("would forbid. Cross-check with Cholis: if global-fit low-E flux moves closer to Cholis\n")
        f.write("then Cholis indeed fixed spectrum; if it diverges further, systematic is elsewhere.\n\n")
        f.write(f"{'Model':<7} {'grp':<4} {'<r>_low orig→new':>22} {'<r>_mid orig→new':>22} "
                f"{'<r>_hi orig→new':>22} {'Δ(-2lnL)':>12}  {'c_gce_global':>18}\n")
        f.write('-' * 120 + '\n')
        for r in rows:
            dlogL = r['neg2logL_new'] - r['neg2logL_orig']
            f.write(f"{r['model']:<7} {r['group']:<4} "
                    f"{r['mean_r_low_orig']:>10.3f} →{r['mean_r_low_new']:>9.3f}  "
                    f"{r['mean_r_mid_orig']:>10.3f} →{r['mean_r_mid_new']:>9.3f}  "
                    f"{r['mean_r_hi_orig']:>10.3f} →{r['mean_r_hi_new']:>9.3f}  "
                    f"{dlogL:>+12.2f}  {r['c_gce_global']:>8.3f} ± {r['c_gce_global_std']:>6.3f}\n")
        f.write("\nPer-bin flux (orig vs global vs Cholis) [GeV cm^-2 s^-1 sr^-1]:\n")
        f.write('-' * 115 + '\n')
        for r in rows:
            f.write(f"\n=== {r['model']} (grp {r['group']}) ===\n")
            f.write(f"{'E':>8} {'flux_orig':>12} {'flux_global':>12} {'cholis':>12} "
                    f"{'r_orig':>8} {'r_global':>10}\n")
            for eb in range(len(r['E'])):
                f.write(f"{r['E'][eb]:>8.3f} {r['flux_orig'][eb]:>12.4e} "
                        f"{r['flux_new'][eb]:>12.4e} {r['cholis_flux'][eb]:>12.4e} "
                        f"{r['ratio_orig'][eb]:>8.3f} {r['ratio_new'][eb]:>10.3f}\n")
    print(f"Saved: {out_txt}")


if __name__ == '__main__':
    run()
