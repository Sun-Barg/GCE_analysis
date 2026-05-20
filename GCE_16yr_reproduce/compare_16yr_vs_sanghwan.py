#!/usr/bin/env python3
"""
compare_16yr_vs_sanghwan.py

Standalone porting-diagnostic comparator. Does NOT import or depend on the
17yr pipeline internals. Operates purely on .dat outputs, plus an optional
static text scan of run_one_model.py for the two logic points most likely to
break the overlay.

.dat format (both sides): col0=E[GeV] col1=E^2 dN/dE  col2=err col3=lower col4=upper
  - reproduced GCE_16yr : 14 rows (Cholis-exact 14-bin)
  - Sanghwan front      : 17 rows (extended); only first 14 used (bins 0-13)

Usage:
  python3 compare_16yr_vs_sanghwan.py \
      --our-dir   results_16yr \
      --sanghwan  GCE_models_16yr_cholis_sanghwan_approach_tar.gz \
      --models    I,X,XLIX  (or  all) \
      --out       overlay_16yr \
      --run-one-model run_one_model.py     # optional static logic scan
"""
import argparse, io, os, re, sys, tarfile
import numpy as np

# ---- Sanghwan reference (from his 16yr notebook, for human diff of logic) ----
SW_GCE_TEMPLATE_REF = (
    "Sanghwan .dat GCE template integral (notebook cell 59 preamble):\n"
    "  GCE[i] = sum( disk_mask * (GCE_no_convol[i]/exp_cube[i]) ) / sum(disk_mask)\n"
    "  -> disk_mask ONLY (NOT full_mask), '_no_convol' map, /exp_cube,\n"
    "     inner 40x40 crop [100:500,100:500]\n"
    "  .dat col1 = fitted[2n:3n] * GCE * E**2 / delta_E"
)
SW_CENTRAL_REF = (
    "Sanghwan central value (notebook cell 59 run_mcmc_for_bin):\n"
    "  log_prob_samples = sampler.get_log_prob(discard=burn_in, flat=True)\n"
    "  best_fit_params  = get_chain(discard=burn_in, flat=True)[argmax(log_prob)]\n"
    "  fitted_param = best_fit_params   # MAP (argmax), NOT median\n"
    "  std = std(flat_samples, ddof=1); lower/upper = 16/84 percentile"
)

ROMAN = None  # not needed; models are taken from filenames/args


def load_dat(text_or_path, is_text=False):
    arr = np.loadtxt(io.StringIO(text_or_path) if is_text else text_or_path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr  # (nrows, >=5)


def read_sanghwan(src, model):
    """Return Sanghwan front .dat array for `model` from a tar.gz or a dir."""
    fname = f"GCE_model_{model}_front_16yr_cholis.dat"
    if os.path.isdir(src):
        p = os.path.join(src, fname)
        return load_dat(p) if os.path.isfile(p) else None
    if tarfile.is_tarfile(src):
        with tarfile.open(src) as tf:
            members = {os.path.basename(m.name): m for m in tf.getmembers()}
            if fname not in members:
                return None
            data = tf.extractfile(members[fname]).read().decode()
            return load_dat(data, is_text=True)
    return None


def read_ours(our_dir, model):
    for cand in (f"GCE_model_{model}_front_16yr_cholis.dat",
                 f"GCE_model_{model}_16yr.dat",
                 f"GCE_model_{model}_front_16yr.dat"):
        p = os.path.join(our_dir, cand)
        if os.path.isfile(p):
            return load_dat(p), cand
    return None, None


def align_bins(ours, sw, rtol=2e-3):
    """Match our 14 rows to Sanghwan's first rows by E (rtol absorbs
    Cholis 0.274698 vs Sanghwan 0.275 edge rounding). Returns matched index
    pairs (i_our, j_sw)."""
    pairs = []
    for i in range(ours.shape[0]):
        e = ours[i, 0]
        j = int(np.argmin(np.abs(sw[:, 0] - e)))
        if np.isclose(sw[j, 0], e, rtol=rtol):
            pairs.append((i, j))
    return pairs


def compare_model(model, ours, sw):
    pairs = align_bins(ours, sw)
    if not pairs:
        return None
    rows, ratios = [], []
    for i, j in pairs:
        eo, fo = ours[i, 0], ours[i, 1]
        es, fs = sw[j, 0], sw[j, 1]
        r = fo / fs if fs != 0 and np.isfinite(fs) else np.nan
        ratios.append(r)
        rows.append((i, eo, es, fo, fs, r))
    ratios = np.array(ratios, dtype=float)
    finite = ratios[np.isfinite(ratios)]
    summary = dict(
        model=model, nbin=len(pairs),
        med_ratio=float(np.median(finite)) if finite.size else np.nan,
        max_dev=float(np.nanmax(np.abs(finite - 1.0))) if finite.size else np.nan,
        rows=rows,
    )
    return summary


def print_table(s):
    print(f"\n=== Model {s['model']}  (matched {s['nbin']} bins) "
          f"median(our/sw)={s['med_ratio']:.4f}  "
          f"max|ratio-1|={s['max_dev']:.4f} ===")
    print(f"{'bin':>3} {'E_our[GeV]':>12} {'E_sw[GeV]':>12} "
          f"{'flux_our':>13} {'flux_sw':>13} {'our/sw':>9}")
    for (i, eo, es, fo, fs, r) in s['rows']:
        print(f"{i:>3} {eo:>12.5f} {es:>12.5f} {fo:>13.4e} {fs:>13.4e} {r:>9.4f}")


def make_plot(summaries, ours_map, sw_map, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped (matplotlib unavailable: {e})")
        return
    os.makedirs(outdir, exist_ok=True)
    for s in summaries:
        m = s['model']
        o, sw = ours_map[m], sw_map[m]
        pairs = align_bins(o, sw)
        io_, jo_ = zip(*pairs)
        io_, jo_ = list(io_), list(jo_)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                       gridspec_kw=dict(height_ratios=[3, 1]))
        ax1.errorbar(o[io_, 0], o[io_, 1],
                     yerr=o[io_, 2], fmt='o-', label='reproduced 16yr (DR4, mask1.0)')
        ax1.errorbar(sw[jo_, 0], sw[jo_, 1],
                     yerr=sw[jo_, 2], fmt='s--', label='Sanghwan 16yr front')
        ax1.set_xscale('log'); ax1.set_yscale('log')
        ax1.set_ylabel(r'$E^2\,dN/dE$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
        ax1.set_title(f"Model {m}  bins 0-{len(pairs)-1}  "
                      f"med(our/sw)={s['med_ratio']:.3f}")
        ax1.legend(); ax1.grid(True, which='both', alpha=.3)
        ratio = o[io_, 1] / sw[jo_, 1]
        ax2.axhline(1.0, color='k', lw=.8)
        ax2.plot(o[io_, 0], ratio, 'o-')
        ax2.set_xscale('log'); ax2.set_ylabel('our / sw')
        ax2.set_xlabel('E [GeV]'); ax2.grid(True, which='both', alpha=.3)
        fig.tight_layout()
        p = os.path.join(outdir, f"overlay_model_{m}.png")
        fig.savefig(p, dpi=130); plt.close(fig)
        print(f"[plot] {p}")


def static_scan(path):
    if not path or not os.path.isfile(path):
        print(f"[static] run_one_model.py not given/found: {path}")
        return
    src = open(path, encoding='utf-8', errors='replace').read()
    lines = src.splitlines()
    print("\n" + "=" * 60)
    print("STATIC LOGIC SCAN of", path)
    print("=" * 60)

    print("\n--- (1) GCE template integral construction ---")
    print(SW_GCE_TEMPLATE_REF)
    print("\n  >> matching lines in run_one_model.py:")
    pat1 = re.compile(r'GCE|disk_mask|full_mask|no_convol|exp_cube|\[100:500|delta_E|fitted_params\[2\]')
    for n, ln in enumerate(lines, 1):
        if pat1.search(ln) and ('GCE' in ln or 'mask' in ln or 'convol' in ln
                                or 'exp_cube' in ln or '100:500' in ln):
            print(f"  {n:>5}: {ln.strip()[:140]}")

    print("\n--- (2) central-value definition (MAP argmax vs median) ---")
    print(SW_CENTRAL_REF)
    print("\n  >> matching lines in run_one_model.py:")
    pat2 = re.compile(r'argmax|get_log_prob|get_chain|percentile|np\.median|median|best_fit|fitted_param')
    for n, ln in enumerate(lines, 1):
        if pat2.search(ln):
            print(f"  {n:>5}: {ln.strip()[:140]}")
    print("\n[static] Eyeball the above against the Sanghwan reference blocks.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--our-dir', default='results_16yr')
    ap.add_argument('--sanghwan', required=True,
                    help='tar.gz or dir with GCE_model_*_front_16yr_cholis.dat')
    ap.add_argument('--models', default='all',
                    help='comma list (e.g. I,X,XLIX) or "all"')
    ap.add_argument('--out', default='overlay_16yr')
    ap.add_argument('--run-one-model', default=None,
                    help='path to copied run_one_model.py for static logic scan')
    a = ap.parse_args()

    if a.models.strip().lower() == 'all':
        if tarfile.is_tarfile(a.sanghwan):
            with tarfile.open(a.sanghwan) as tf:
                names = [os.path.basename(m.name) for m in tf.getmembers()]
        else:
            names = os.listdir(a.sanghwan)
        models = sorted({
            re.match(r'GCE_model_(.+?)_front_16yr_cholis\.dat$', n).group(1)
            for n in names
            if re.match(r'GCE_model_(.+?)_front_16yr_cholis\.dat$', n)
        })
    else:
        models = [m.strip() for m in a.models.split(',') if m.strip()]

    summaries, ours_map, sw_map, missing = [], {}, {}, []
    for m in models:
        o, used = read_ours(a.our_dir, m)
        s = read_sanghwan(a.sanghwan, m)
        if o is None or s is None:
            missing.append((m, o is None, s is None))
            continue
        ours_map[m], sw_map[m] = o, s
        res = compare_model(m, o, s)
        if res:
            summaries.append(res)
            print_table(res)

    if summaries:
        meds = np.array([s['med_ratio'] for s in summaries], dtype=float)
        print("\n" + "=" * 60)
        print(f"OVERALL: {len(summaries)} models compared | "
              f"median of per-model median(our/sw) = "
              f"{np.nanmedian(meds):.4f} | "
              f"spread [{np.nanmin(meds):.3f}, {np.nanmax(meds):.3f}]")
        print("Reminder: mask 1.0-vs-0.9 and 14-vs-17 bin edges produce a")
        print("small EXPECTED offset; a clean port should overlay to within")
        print("that, not exactly. Large per-bin structure -> see static scan.")
        make_plot(summaries, ours_map, sw_map, a.out)
    if missing:
        print("\n[skipped] (model, our_missing, sw_missing):")
        for x in missing:
            print("  ", x)

    static_scan(a.run_one_model)


if __name__ == '__main__':
    sys.exit(main())
