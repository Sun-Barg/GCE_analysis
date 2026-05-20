# ============================================================
# validate_cov_matrix.py
# ============================================================
#!/usr/bin/env python3
"""validate_cov_matrix.py — 17yr cov matrix sanity checks.

Verifies build_cov_matrix.py output GCE_systematic_covariance_matrix_17yr.npz:
  - 8 required keys present (cov_matrix, E, delta_E, mean_GCE, sigma_sys,
    GCE_flux_per_roi, c_gce_per_roi, GCE_tmpl_per_roi)
  - cov_matrix shape (14,14), symmetric (rel < 1e-10), PSD (eig_min >= 0)
  - condition number (17yr reference ~2.35e5; flag if >1e10)
  - sigma_sys == sqrt(diag(cov)) (max rel diff < 1e-6)
  - per-bin sigma_sys SED (Calore+ 1409.0042 Fig 2 pattern: peak at low-mid E
    ~0.5 GeV, magnitudes 1e-6 .. 1e-7 GeV/cm^2/s/sr)
  - per-ROI dispersion summary (22 ROIs per Cholis L1637)

CLI:
  python validate_cov_matrix.py
  python validate_cov_matrix.py --npz <path>
  python validate_cov_matrix.py --plot
"""
import os
import sys
import argparse
import numpy as np

DEFAULT_NPZ = 'results_cov_17yr/GCE_systematic_covariance_matrix_17yr.npz'
REQUIRED_KEYS = ['cov_matrix', 'E', 'delta_E', 'mean_GCE', 'sigma_sys',
                 'GCE_flux_per_roi', 'c_gce_per_roi', 'GCE_tmpl_per_roi']

# Calore+ 2015 (1409.0042) Fig 2 approximate scale (orientation-of-magnitude check only)
CALORE_PEAK_GEV  = 0.5
CALORE_PEAK_REF  = 5e-7        # GeV/cm^2/s/sr, ballpark
CALORE_HIGH_REF  = 1e-7


def _check_cov(cov):
    msgs, ok = [], True
    if cov.shape != (14, 14):
        msgs.append(f'  [FAIL] shape={cov.shape} (want (14,14))'); ok = False
    asym = float(np.max(np.abs(cov - cov.T)))
    scale = max(float(np.max(np.abs(cov))), 1e-30)
    rel_asym = asym / scale
    if rel_asym > 1e-10:
        msgs.append(f'  [WARN] asymmetric: max|C-C.T|/max|C| = {rel_asym:.2e}')
    else:
        msgs.append(f'  [ ok ] symmetric  (max|C-C.T|/max|C| = {rel_asym:.2e})')
    cov_sym = 0.5 * (cov + cov.T)
    eig = np.linalg.eigvalsh(cov_sym)
    e_min, e_max = float(eig.min()), float(eig.max())
    if e_min < -1e-12 * abs(e_max):
        msgs.append(f'  [FAIL] not PSD: eig_min={e_min:.3e}  eig_max={e_max:.3e}'); ok = False
    else:
        msgs.append(f'  [ ok ] PSD  eig in [{e_min:.3e}, {e_max:.3e}]')
    cond = e_max / max(e_min, 1e-30) if e_min > 0 else float('inf')
    msgs.append(f'  [info] condition number = {cond:.3e}   '
                f'(17yr reference ~2.35e5)')
    if cond > 1e10:
        msgs.append(f'  [WARN] condition number very large; matrix near-degenerate')
    return ok, msgs, eig, cond


def _check_sigma(cov, sigma_stored, E):
    msgs = []
    sigma_diag = np.sqrt(np.maximum(np.diag(cov), 0))
    rel = np.abs(sigma_diag - sigma_stored) / np.maximum(sigma_stored, 1e-30)
    max_rel = float(rel.max())
    ok = max_rel <= 1e-6
    tag = '[ ok ]' if ok else '[FAIL]'
    msgs.append(f'  {tag} sigma_sys == sqrt(diag(cov))  (max rel diff {max_rel:.2e})')
    msgs.append(f'  [info] per-bin sigma_sys [GeV/cm^2/s/sr]:')
    for i, (e, s) in enumerate(zip(E, sigma_stored)):
        msgs.append(f'         bin {i:2d}  E={e:7.3f} GeV   sigma_sys={s:.3e}')
    peak_bin = int(np.argmax(sigma_stored))
    msgs.append(f'  [info] sigma_sys peaks at bin {peak_bin} (E={E[peak_bin]:.3f} GeV); '
                f'Calore+ Fig 2 peak ~{CALORE_PEAK_GEV} GeV')
    return ok, msgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', default=DEFAULT_NPZ)
    p.add_argument('--plot', action='store_true',
                   help='save validate_cov_matrix.png (sigma_sys SED + cov heatmap)')
    args = p.parse_args()

    print('=' * 70)
    print(f'validate_cov_matrix.py   npz={args.npz}')
    print('=' * 70)

    if not os.path.exists(args.npz):
        print(f'[FATAL] file not found: {args.npz}', file=sys.stderr); sys.exit(2)

    d = np.load(args.npz)
    missing = [k for k in REQUIRED_KEYS if k not in d.files]
    if missing:
        print(f'\n[FAIL] missing keys: {missing}', file=sys.stderr); sys.exit(1)
    print(f'\n[keys] all required keys present ({len(REQUIRED_KEYS)})')

    cov = np.asarray(d['cov_matrix'])
    E   = np.asarray(d['E'])
    sig = np.asarray(d['sigma_sys'])
    gce_flux_per_roi = np.asarray(d['GCE_flux_per_roi'])
    c_gce_per_roi    = np.asarray(d['c_gce_per_roi'])
    mean_GCE         = np.asarray(d['mean_GCE'])

    print(f'\n[shapes]')
    print(f'  cov_matrix         = {cov.shape}     (want (14,14))')
    print(f'  E                  = {E.shape}        (want (14,))')
    print(f'  sigma_sys          = {sig.shape}        (want (14,))')
    print(f'  GCE_flux_per_roi   = {gce_flux_per_roi.shape}   (want (22,14))')
    print(f'  c_gce_per_roi      = {c_gce_per_roi.shape}      (want (22,14))')
    print(f'  mean_GCE           = {mean_GCE.shape}        (want (14,))')

    n_roi = gce_flux_per_roi.shape[0]
    if n_roi != 22:
        print(f'\n[WARN] GCE_flux_per_roi first dim = {n_roi} (expected 22; '
              f'Cholis L1637)', file=sys.stderr)

    print(f'\n[cov matrix tests]')
    cov_ok, cov_msgs, eig, cond = _check_cov(cov)
    for m in cov_msgs: print(m)

    print(f'\n[sigma_sys consistency + SED]')
    s_ok, s_msgs = _check_sigma(cov, sig, E)
    for m in s_msgs: print(m)

    print(f'\n[per-ROI GCE flux summary]  (want 22 ROIs)')
    print(f'  flux_per_roi range: [{gce_flux_per_roi.min():.3e}, '
          f'{gce_flux_per_roi.max():.3e}]')
    for i in range(len(mean_GCE)):
        per_roi = gce_flux_per_roi[:, i]
        print(f'    bin {i:2d}  E={E[i]:7.3f}  mean={mean_GCE[i]:.3e}  '
              f'roi_std={per_roi.std(ddof=1):.3e}  '
              f'roi_range=[{per_roi.min():.3e}, {per_roi.max():.3e}]')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].loglog(E, sig, 'o-', lw=1.5, label='17yr sigma_sys')
        ax[0].axhline(CALORE_PEAK_REF, ls=':', color='gray',
                      label=f'Calore+ peak ~{CALORE_PEAK_REF:.0e}')
        ax[0].axhline(CALORE_HIGH_REF, ls=':', color='lightgray',
                      label=f'Calore+ tail ~{CALORE_HIGH_REF:.0e}')
        ax[0].set_xlabel('E [GeV]')
        ax[0].set_ylabel(r'$\sigma_{sys}$ [GeV/cm$^2$/s/sr]')
        ax[0].set_title('Systematic uncertainty SED (17yr)')
        ax[0].legend(fontsize=8); ax[0].grid(which='both', alpha=0.3)
        vmax = float(np.abs(cov).max())
        im = ax[1].imshow(cov, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax[1].set_title(f'cov_matrix (14x14)   cond={cond:.2e}')
        ax[1].set_xlabel('bin'); ax[1].set_ylabel('bin')
        plt.colorbar(im, ax=ax[1], fraction=0.04)
        plt.tight_layout()
        plt.savefig('validate_cov_matrix.png', dpi=100, bbox_inches='tight')
        print(f'\n[saved] validate_cov_matrix.png')

    print('\n' + '=' * 70)
    if cov_ok and s_ok:
        print('VERDICT: cov matrix passes all sanity checks')
        sys.exit(0)
    else:
        print('VERDICT: cov matrix has FAILURES (see above)', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
