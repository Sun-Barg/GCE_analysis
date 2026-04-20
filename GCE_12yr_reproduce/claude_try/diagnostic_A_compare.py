import os, sys, pickle, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, ZENODO_FIG_DIR, CHOLIS_REF_FILES, MODELS


SUFFIX = os.environ.get('FIT_SUFFIX', 'icsfloor0p10')

GROUP_A = ['X', 'XLVIII', 'LIII']
GROUP_B = ['XV', 'XLIX']


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_cholis(model):
    arr = np.loadtxt(os.path.join(ZENODO_FIG_DIR, CHOLIS_REF_FILES[model]))
    return {'E': arr[:, 0], 'flux': arr[:, 1], 'lo': arr[:, 2], 'hi': arr[:, 3]}


def summary():
    rows = []
    for m in MODELS:
        orig_pkl = os.path.join(OUT_DIR, f'GCE_model_{m}_haebarg_v_claude.pkl')
        new_pkl = os.path.join(OUT_DIR, f'GCE_model_{m}_haebarg_v_claude_{SUFFIX}.pkl')
        if not (os.path.exists(orig_pkl) and os.path.exists(new_pkl)):
            print(f"[skip {m}] missing: orig={os.path.exists(orig_pkl)}, new={os.path.exists(new_pkl)}")
            continue
        orig = load_pkl(orig_pkl)
        new = load_pkl(new_pkl)
        cho = load_cholis(m)

        E = orig['E']
        r_orig = orig['flux_best'] / np.maximum(cho['flux'], 1e-30)
        r_new = new['flux_best'] / np.maximum(cho['flux'], 1e-30)

        low = E < 1.0
        mid = (E >= 1.0) & (E <= 10.0)
        hi = (E > 10.0) & (E <= 60.0)

        group = 'A' if m in GROUP_A else ('B' if m in GROUP_B else '?')
        rows.append({
            'model': m, 'group': group, 'E': E,
            'flux_orig': orig['flux_best'], 'flux_new': new['flux_best'],
            'coef_orig': orig['coef_best'], 'coef_new': new['coef_best'],
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
    print("=" * 110)
    print(f"Experiment A :: c_ics floor applied  (suffix = {SUFFIX})")
    print("=" * 110)
    header = f"{'Model':<7} {'grp':<4} {'<r>_low orig→new':>22} {'<r>_mid orig→new':>22} {'<r>_hi orig→new':>22} {'Δ(-2lnL)':>12}"
    print(header)
    print("-" * 110)
    for r in rows:
        dlogL = r['neg2logL_new'] - r['neg2logL_orig']
        print(f"{r['model']:<7} {r['group']:<4} "
              f"{r['mean_r_low_orig']:>10.3f} →{r['mean_r_low_new']:>9.3f}  "
              f"{r['mean_r_mid_orig']:>10.3f} →{r['mean_r_mid_new']:>9.3f}  "
              f"{r['mean_r_hi_orig']:>10.3f} →{r['mean_r_hi_new']:>9.3f}  "
              f"{dlogL:>+12.2f}")

    for r in rows:
        orig_c_ics = r['coef_orig'][:, 1]
        new_c_ics = r['coef_new'][:, 1]
        orig_c_gce = r['coef_orig'][:, 2]
        new_c_gce = r['coef_new'][:, 2]
        print(f"\n[{r['model']} grp {r['group']}] per-bin c_ics  (orig → new)")
        for eb in range(len(r['E'])):
            flag = '  *' if (r['E'][eb] < 1.0) else ''
            print(f"  E={r['E'][eb]:>6.3f}  c_ics: {orig_c_ics[eb]:>6.3f} → {new_c_ics[eb]:>6.3f}   "
                  f"c_gce: {orig_c_gce[eb]:>6.3f} → {new_c_gce[eb]:>6.3f}{flag}")

    return rows


def plot(rows, out_png):
    fig, axes = plt.subplots(3, len(rows), figsize=(4.5*len(rows), 11),
                              gridspec_kw={'height_ratios': [3, 1.2, 1.2]})
    if len(rows) == 1:
        axes = axes.reshape(3, 1)

    for i, r in enumerate(rows):
        ax = axes[0, i]
        ax.errorbar(r['cholis_E'], r['cholis_flux'],
                    yerr=[r['cholis_flux']-r['cholis_lo'], r['cholis_hi']-r['cholis_flux']],
                    fmt='s', color='C3', label='Cholis+2022', capsize=3, markersize=5)
        ax.plot(r['E'], r['flux_orig'], 'o-', color='C0', label='Original', markersize=5)
        ax.plot(r['E'], r['flux_new'], '^-', color='C2',
                label=f'c_ics ≥ floor', markersize=5)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(0.25, 60); ax.set_ylim(3e-8, 5e-6)
        ax.set_title(f"Model {r['model']} (grp {r['group']})", fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        if i == 0:
            ax.set_ylabel(r'$E^2 dN/dE$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]', fontsize=10)
            ax.legend(loc='lower center', fontsize=8)

        ax_r = axes[1, i]
        ax_r.axhline(1.0, color='k', linestyle=':', alpha=0.5)
        ax_r.semilogx(r['E'], r['ratio_orig'], 'o-', color='C0', label='orig', markersize=4)
        ax_r.semilogx(r['E'], r['ratio_new'], '^-', color='C2', label='floor', markersize=4)
        ax_r.set_ylim(0.0, 2.0); ax_r.set_xlim(0.25, 60)
        ax_r.grid(True, which='both', alpha=0.3)
        if i == 0:
            ax_r.set_ylabel('Claude / Cholis', fontsize=10)
            ax_r.legend(loc='upper right', fontsize=8)

        ax_c = axes[2, i]
        ax_c.semilogx(r['E'], r['coef_orig'][:, 1], 'o-', color='C0',
                       label='c_ics orig', markersize=4)
        ax_c.semilogx(r['E'], r['coef_new'][:, 1], '^-', color='C2',
                       label='c_ics floor', markersize=4)
        ax_c.set_ylim(-0.1, 3.0); ax_c.set_xlim(0.25, 60)
        ax_c.set_xlabel('E [GeV]', fontsize=10); ax_c.grid(True, which='both', alpha=0.3)
        if i == 0:
            ax_c.set_ylabel('c_ics (ICS coeff)', fontsize=10)
            ax_c.legend(loc='upper right', fontsize=8)

    plt.suptitle(f'Experiment A : effect of c_ics floor on GCE SED  ({SUFFIX})', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")


def save_text(rows, outpath):
    with open(outpath, 'w') as f:
        f.write(f"Experiment A : c_ics floor analysis  (suffix={SUFFIX})\n")
        f.write("=" * 110 + "\n\n")
        f.write("Hypothesis: Group B (XV, XLIX) shows high low-E ratio because fit\n")
        f.write("drives c_ics to ~0 (ICS-GCE degeneracy) and the excess is absorbed into GCE.\n\n")
        f.write("Prediction: with c_ics > 0.1 floor, Group B low-E ratio should drop toward Group A.\n\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Model':<7} {'grp':<4} {'<r>_low(orig→new)':>22} {'<r>_mid(orig→new)':>22} "
                f"{'<r>_hi(orig→new)':>22} {'Δ(-2lnL)':>12}\n")
        f.write("-" * 110 + "\n")
        for r in rows:
            dlogL = r['neg2logL_new'] - r['neg2logL_orig']
            f.write(f"{r['model']:<7} {r['group']:<4} "
                    f"{r['mean_r_low_orig']:>10.3f} →{r['mean_r_low_new']:>9.3f}  "
                    f"{r['mean_r_mid_orig']:>10.3f} →{r['mean_r_mid_new']:>9.3f}  "
                    f"{r['mean_r_hi_orig']:>10.3f} →{r['mean_r_hi_new']:>9.3f}  "
                    f"{dlogL:>+12.2f}\n")

        f.write("\n\nPer-bin c_ics and c_gce comparison:\n")
        f.write("-" * 110 + "\n")
        for r in rows:
            f.write(f"\n=== Model {r['model']} (group {r['group']}) ===\n")
            f.write(f"{'E[GeV]':>8} {'c_ics_orig':>10} {'c_ics_new':>10} "
                    f"{'c_gce_orig':>10} {'c_gce_new':>10}  {'flag':>6}\n")
            for eb in range(len(r['E'])):
                flag = 'low-E' if r['E'][eb] < 1.0 else ''
                f.write(f"{r['E'][eb]:>8.3f} "
                        f"{r['coef_orig'][eb,1]:>10.3f} {r['coef_new'][eb,1]:>10.3f} "
                        f"{r['coef_orig'][eb,2]:>10.3f} {r['coef_new'][eb,2]:>10.3f}  "
                        f"{flag:>6}\n")
    print(f"Saved: {outpath}")


if __name__ == '__main__':
    rows = summary()
    if rows:
        plot(rows, os.path.join(OUT_DIR, f'diagnostic_A_comparison_{SUFFIX}.png'))
        save_text(rows, os.path.join(OUT_DIR, f'diagnostic_A_comparison_{SUFFIX}.txt'))
