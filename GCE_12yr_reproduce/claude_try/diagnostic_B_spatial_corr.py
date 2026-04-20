import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR, CCUBE, MODELS, ROI_SLICE
from prepare_masks import FULL_MASK_PATH, prepare_masks
from stage2_fit import load_energy_axis


def load_component_flat(model, component, ebin, mask_bool):
    yroi, xroi = ROI_SLICE
    path = f'{WORK_DIR}/GC_analysis_sanghwan/GC_{component}_model{model}_12yr_front_clean.fits'
    if component in ('GCE', 'fermi_bubble', 'isotropic'):
        path = f'{WORK_DIR}/GC_analysis_sanghwan/GC_{component}_model_12yr_front_clean.fits'
    arr = fits.open(path)[0].data[ebin, yroi, xroi].astype(np.float64)
    return arr[mask_bool]


def pair_correlation(a, b):
    a = a - a.mean(); b = b - b.mean()
    na = np.sqrt((a * a).sum()); nb = np.sqrt((b * b).sum())
    if na < 1e-30 or nb < 1e-30:
        return np.nan
    return float((a * b).sum() / (na * nb))


def run():
    os.chdir(WORK_DIR)
    if not os.path.exists(FULL_MASK_PATH):
        prepare_masks()
    full_mask = np.load(FULL_MASK_PATH)
    E, dE = load_energy_axis(CCUBE)
    n_ebin = len(E)

    pairs = [
        ('GCE', 'ics'),
        ('GCE', 'pion'),
        ('GCE', 'bremss'),
        ('GCE', 'fermi_bubble'),
        ('ics', 'pion'),
    ]

    corrs = {p: np.full((len(MODELS), n_ebin), np.nan) for p in pairs}

    for mi, m in enumerate(MODELS):
        for eb in range(n_ebin):
            mb = full_mask[eb] > 0.5
            if mb.sum() < 10:
                continue
            cache = {}
            for (a, b) in pairs:
                for cname in (a, b):
                    if cname not in cache:
                        cache[cname] = load_component_flat(m, cname, eb, mb)
                corrs[(a, b)][mi, eb] = pair_correlation(cache[a], cache[b])

    out_txt = os.path.join(OUT_DIR, 'diagnostic_B_spatial_correlation.txt')
    with open(out_txt, 'w') as f:
        f.write("Experiment B : Spatial correlation of component maps within fit mask\n")
        f.write("=" * 88 + "\n")
        f.write("Pearson correlation between two component maps (convol=yes, masked pixels only)\n")
        f.write("High correlation (>0.9) at low E indicates strong degeneracy\n")
        f.write("between the two components in the fit.\n\n")
        for (a, b) in pairs:
            f.write(f"\n--- corr({a}, {b}) ---\n")
            f.write(f"{'E[GeV]':>10}  " + "  ".join(f"{m:>10}" for m in MODELS) + "\n")
            for eb in range(n_ebin):
                row = [f"{corrs[(a,b)][mi, eb]:>10.4f}" for mi in range(len(MODELS))]
                f.write(f"{E[eb]:>10.3f}  " + "  ".join(row) + "\n")

        f.write("\n\n=== Group A vs B comparison (corr(GCE, ICS), 0.3-1 GeV) ===\n")
        low_mask = E < 1.0
        group_a = ['X', 'XLVIII', 'LIII']
        group_b = ['XV', 'XLIX']
        idx_a = [MODELS.index(m) for m in group_a]
        idx_b = [MODELS.index(m) for m in group_b]
        gi = corrs[('GCE', 'ics')]
        f.write(f"Group A (X, XLVIII, LIII)  mean corr(GCE,ICS) at E<1 GeV: "
                f"{np.nanmean(gi[idx_a][:, low_mask]):.4f}\n")
        f.write(f"Group B (XV, XLIX)         mean corr(GCE,ICS) at E<1 GeV: "
                f"{np.nanmean(gi[idx_b][:, low_mask]):.4f}\n")
        f.write(f"Difference (B - A): {np.nanmean(gi[idx_b][:, low_mask]) - np.nanmean(gi[idx_a][:, low_mask]):+.4f}\n")

        high_mask = (E >= 1.0) & (E <= 10.0)
        f.write(f"\nAt 1-10 GeV (control):\n")
        f.write(f"Group A  mean corr(GCE,ICS): {np.nanmean(gi[idx_a][:, high_mask]):.4f}\n")
        f.write(f"Group B  mean corr(GCE,ICS): {np.nanmean(gi[idx_b][:, high_mask]):.4f}\n")

    fig, axes = plt.subplots(1, len(pairs), figsize=(5*len(pairs), 5), sharey=True)
    for i, (a, b) in enumerate(pairs):
        ax = axes[i]
        im = ax.imshow(corrs[(a, b)], aspect='auto', origin='lower',
                        cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        ax.set_xticks(range(n_ebin))
        ax.set_xticklabels([f'{E[e]:.2f}' for e in range(n_ebin)], rotation=60, fontsize=8)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(MODELS, fontsize=10)
        ax.set_title(f"corr({a}, {b})", fontsize=11)
        ax.set_xlabel('E [GeV]', fontsize=10)
        for mi in range(len(MODELS)):
            for eb in range(n_ebin):
                v = corrs[(a, b)][mi, eb]
                if not np.isnan(v):
                    ax.text(eb, mi, f'{v:.2f}', ha='center', va='center',
                            fontsize=6.5, color='black' if abs(v) < 0.5 else 'white')
        plt.colorbar(im, ax=ax, fraction=0.04)
    plt.suptitle('Experiment B : spatial correlation of component maps (within fit mask)', fontsize=13)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'diagnostic_B_spatial_correlation.png')
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close()

    np.savez(os.path.join(OUT_DIR, 'diagnostic_B_spatial_correlation.npz'),
             E=E, models=MODELS, **{f'corr_{a}_{b}': corrs[(a,b)] for (a,b) in pairs})

    print(f"\nSaved: {out_txt}")
    print(f"Saved: {out_png}")
    print("\n=== Summary ===")
    gi = corrs[('GCE', 'ics')]
    low_mask = E < 1.0
    print(f"{'Model':<8} {'corr(GCE,ICS) E<1GeV':>22} {'corr(GCE,ICS) 1-10GeV':>24}")
    for mi, m in enumerate(MODELS):
        low = np.nanmean(gi[mi, low_mask])
        hi = np.nanmean(gi[mi, (E >= 1.0) & (E <= 10.0)])
        print(f"{m:<8} {low:>22.4f} {hi:>24.4f}")


if __name__ == '__main__':
    run()
