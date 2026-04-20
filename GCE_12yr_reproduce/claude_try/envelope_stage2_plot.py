"""
Envelope Stage 2 - aggregate per-model fits, compute envelope band,
compare to Cholis+2022 Figure 12/14.

Reads all pkl files matching GCE_model_*_haebarg_v_claude_{SUFFIX}.pkl,
computes per-bin min/max/median/percentiles, and overlays on Cholis
reference files from Zenodo Figures_12_and_14_GCE_Spectra/.

Outputs
-------
- envelope_band_{SUFFIX}.png   : Figure 12 style, all-model envelope
- envelope_bestfit_{SUFFIX}.png: Figure 14 style, top-N best-fit band
- envelope_band_{SUFFIX}.txt   : per-bin numeric table

Usage
-----
    python envelope_stage2_plot.py
    # Or pick another suffix
    FIT_SUFFIX=globalCgce_envelope_full python envelope_stage2_plot.py
"""
import os, sys, pickle, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT_DIR, ZENODO_FIG_DIR

SUFFIX = os.environ.get('FIT_SUFFIX', 'globalCgce_envelope')
TOP_N_BEST_FIT = int(os.environ.get('TOP_N_BEST_FIT', '10'))


def collect_results():
    pattern = os.path.join(OUT_DIR, f'GCE_model_*_haebarg_v_claude_{SUFFIX}.pkl')
    files = sorted(glob.glob(pattern))
    results = {}
    for fp in files:
        base = os.path.basename(fp)
        model = base.replace('GCE_model_', '').replace(
            f'_haebarg_v_claude_{SUFFIX}.pkl', '')
        try:
            with open(fp, 'rb') as f:
                d = pickle.load(f)
        except Exception as e:
            print(f"  [skip {model}] pickle load failed: {e}")
            continue
        results[model] = d
    return results


def load_cholis_reference_band():
    """Load all individual model flux files from Zenodo Figure 12 inputs."""
    pattern = os.path.join(ZENODO_FIG_DIR, 'GCE_Model*_flux_Inner40x40_masked_disk.dat')
    files = sorted(glob.glob(pattern))
    all_fluxes = []
    E_common = None
    for fp in files:
        if 'BestFit' in os.path.basename(fp):
            continue
        arr = np.loadtxt(fp)
        E = arr[:, 0]
        flux = arr[:, 1]
        if E_common is None:
            E_common = E
        if np.allclose(E, E_common, rtol=1e-3):
            all_fluxes.append(flux)
    if not all_fluxes:
        return None, None
    return E_common, np.array(all_fluxes)


def load_cholis_bestfit_series():
    """Load the Best / 2nd / 3rd / 4th / 5th best-fit (all-sky) Cholis files."""
    names = ['GCE_BestFitModel', 'GCE_2ndBestFitModel', 'GCE_3rdBestFitModel',
             'GCE_4thBestFitModel', 'GCE_5thBestFitModel']
    series = {}
    for n in names:
        fp = os.path.join(ZENODO_FIG_DIR, f'{n}_flux_Inner40x40_masked_disk.dat')
        if os.path.exists(fp):
            arr = np.loadtxt(fp)
            series[n] = {'E': arr[:, 0], 'flux': arr[:, 1],
                         'lo': arr[:, 2], 'hi': arr[:, 3]}
    return series


def build_envelope(results):
    if not results:
        return None
    Es = []
    fluxes = []
    neg2lnLs = []
    c_gce_globals = []
    model_names = []
    for m, d in results.items():
        if 'flux_best' not in d or 'E' not in d:
            continue
        Es.append(d['E'])
        fluxes.append(d['flux_best'])
        neg2lnLs.append(float(d.get('total_neg2_logL', np.inf)))
        c_gce_globals.append(float(d.get('c_gce_global', np.nan)))
        model_names.append(m)
    if not fluxes:
        return None
    E = np.mean(np.array(Es), axis=0)
    fluxes = np.array(fluxes)
    neg2lnLs = np.array(neg2lnLs)
    return {
        'E': E,
        'flux_per_model': fluxes,
        'models': model_names,
        'neg2lnL_per_model': neg2lnLs,
        'c_gce_globals': np.array(c_gce_globals),
        'min_band': fluxes.min(axis=0),
        'max_band': fluxes.max(axis=0),
        'median': np.median(fluxes, axis=0),
        'p16': np.percentile(fluxes, 16, axis=0),
        'p84': np.percentile(fluxes, 84, axis=0),
    }


def plot_figure12_style(env, out_png):
    E_ref, cholis_all = load_cholis_reference_band()
    cholis_min = cholis_all.min(axis=0) if cholis_all is not None else None
    cholis_max = cholis_all.max(axis=0) if cholis_all is not None else None

    fig, axes = plt.subplots(2, 1, figsize=(9, 10),
                              gridspec_kw={'height_ratios': [3, 1.5]})

    ax = axes[0]
    ax.fill_between(env['E'], env['min_band'], env['max_band'],
                     color='C0', alpha=0.25, label=f"Claude-v0 envelope "
                     f"({env['flux_per_model'].shape[0]} models)")
    ax.plot(env['E'], env['median'], 'o-', color='C0', alpha=0.9,
            label='Claude-v0 median')

    if cholis_all is not None:
        ax.fill_between(E_ref, cholis_min, cholis_max,
                         color='C3', alpha=0.2,
                         label=f'Cholis+2022 envelope ({cholis_all.shape[0]} models)')
        ax.plot(E_ref, np.median(cholis_all, axis=0), 's-', color='C3',
                 alpha=0.7, label='Cholis+2022 median')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.25, 60)
    ax.set_ylim(5e-8, 5e-6)
    ax.set_ylabel(r'$E^2\, dN/dE$  [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]', fontsize=12)
    ax.set_title(f'GCE envelope across GDE models  ({SUFFIX})', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10, loc='lower center')

    ax_r = axes[1]
    if cholis_all is not None and np.allclose(env['E'], E_ref, rtol=1e-3):
        c_med = np.median(cholis_all, axis=0)
        ax_r.plot(env['E'], env['median'] / np.maximum(c_med, 1e-30),
                   'o-', color='C0', label='median ratio')
        ax_r.plot(env['E'], env['min_band'] / np.maximum(c_med, 1e-30),
                   '^--', color='C0', alpha=0.5, label='min ratio')
        ax_r.plot(env['E'], env['max_band'] / np.maximum(c_med, 1e-30),
                   'v--', color='C0', alpha=0.5, label='max ratio')
        ax_r.axhline(1.0, color='k', linestyle=':', alpha=0.5)
        ax_r.set_xscale('log')
        ax_r.set_ylim(0.3, 2.0)
        ax_r.set_xlim(0.25, 60)
        ax_r.set_xlabel('E [GeV]', fontsize=12)
        ax_r.set_ylabel('Claude / Cholis median', fontsize=11)
        ax_r.grid(True, which='both', alpha=0.3)
        ax_r.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_png}")


def plot_figure14_style(env, top_n, out_png):
    neg2lnL = env['neg2lnL_per_model']
    order = np.argsort(neg2lnL)
    best_idx = order[:top_n]
    best_fluxes = env['flux_per_model'][best_idx]
    best_models = [env['models'][i] for i in best_idx]

    print(f"  Best {top_n} models by -2lnL:")
    for rank, i in enumerate(best_idx):
        print(f"    {rank+1:>2}. {env['models'][i]:<10} "
              f"-2lnL={neg2lnL[i]:.2f}  c_gce={env['c_gce_globals'][i]:.3f}")

    bestfit_series = load_cholis_bestfit_series()

    fig, axes = plt.subplots(2, 1, figsize=(9, 10),
                              gridspec_kw={'height_ratios': [3, 1.5]})

    ax = axes[0]
    ax.fill_between(env['E'], best_fluxes.min(axis=0), best_fluxes.max(axis=0),
                     color='C2', alpha=0.3, label=f'Claude-v0 top {top_n}')
    ax.plot(env['E'], np.median(best_fluxes, axis=0), 'o-', color='C2',
            alpha=0.9, label='Claude-v0 top N median')

    cholis_colors = ['C3', 'C1', 'C4', 'C5', 'C6']
    for (name, series), col in zip(bestfit_series.items(), cholis_colors):
        ax.errorbar(series['E'], series['flux'],
                    yerr=[series['flux']-series['lo'], series['hi']-series['flux']],
                    fmt='s-', color=col, alpha=0.7, markersize=4,
                    label=name.replace('GCE_', '').replace('Model', ''))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.25, 60)
    ax.set_ylim(5e-8, 5e-6)
    ax.set_ylabel(r'$E^2\, dN/dE$  [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]', fontsize=12)
    ax.set_title(f'Best-fit models  Claude-v0 top {top_n} vs Cholis best 5', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='lower center')

    ax_r = axes[1]
    if 'GCE_BestFitModel' in bestfit_series:
        s = bestfit_series['GCE_BestFitModel']
        if np.allclose(env['E'], s['E'], rtol=1e-3):
            med_claude = np.median(best_fluxes, axis=0)
            ax_r.plot(env['E'], med_claude / np.maximum(s['flux'], 1e-30),
                       'o-', color='C2', label='top-N median / Cholis BestFit')
            ax_r.axhline(1.0, color='k', linestyle=':', alpha=0.5)
            ax_r.set_xscale('log')
            ax_r.set_ylim(0.3, 2.0)
            ax_r.set_xlim(0.25, 60)
            ax_r.set_xlabel('E [GeV]', fontsize=12)
            ax_r.set_ylabel('Claude / Cholis ratio', fontsize=11)
            ax_r.grid(True, which='both', alpha=0.3)
            ax_r.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_png}")


def save_band_table(env, out_txt):
    with open(out_txt, 'w') as f:
        f.write(f"Envelope band  suffix={SUFFIX}  "
                f"N_models={env['flux_per_model'].shape[0]}\n")
        f.write("=" * 95 + "\n\n")
        f.write(f"{'E[GeV]':>8} {'min':>12} {'p16':>12} {'median':>12} "
                f"{'p84':>12} {'max':>12} {'spread_log10':>14}\n")
        for eb in range(len(env['E'])):
            spread = (np.log10(env['max_band'][eb])
                       - np.log10(max(env['min_band'][eb], 1e-30)))
            f.write(f"{env['E'][eb]:>8.3f} "
                    f"{env['min_band'][eb]:>12.4e} "
                    f"{env['p16'][eb]:>12.4e} "
                    f"{env['median'][eb]:>12.4e} "
                    f"{env['p84'][eb]:>12.4e} "
                    f"{env['max_band'][eb]:>12.4e} "
                    f"{spread:>14.3f}\n")
        f.write("\n\nModels included (ordered by -2lnL best first):\n")
        order = np.argsort(env['neg2lnL_per_model'])
        for rank, i in enumerate(order):
            f.write(f"  {rank+1:>3}. {env['models'][i]:<10} "
                    f"-2lnL={env['neg2lnL_per_model'][i]:.2f}  "
                    f"c_gce={env['c_gce_globals'][i]:.3f}\n")
    print(f"  Saved: {out_txt}")


def main():
    print(f"=== Envelope Stage 2 - aggregate and plot  suffix={SUFFIX} ===")
    print()

    results = collect_results()
    print(f"  Loaded {len(results)} model pkl files")
    if not results:
        print("  Nothing to aggregate.")
        sys.exit(1)

    env = build_envelope(results)
    if env is None:
        print("  Build envelope failed.")
        sys.exit(1)

    out_png_12 = os.path.join(OUT_DIR, f'envelope_band_{SUFFIX}.png')
    plot_figure12_style(env, out_png_12)

    if len(results) >= TOP_N_BEST_FIT:
        out_png_14 = os.path.join(OUT_DIR, f'envelope_bestfit_{SUFFIX}.png')
        plot_figure14_style(env, TOP_N_BEST_FIT, out_png_14)

    out_txt = os.path.join(OUT_DIR, f'envelope_band_{SUFFIX}.txt')
    save_band_table(env, out_txt)


if __name__ == '__main__':
    main()
