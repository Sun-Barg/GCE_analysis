#!/usr/bin/env python3
"""
make_phase2_overlay.py — 16yr_reproduce phase2 best-5 vs Sanghwan 16yr overlay.

5 best-fit GDE models (Cholis 2022 Fig 12 caption):
    X, XV, XLVIII, XLIX, LIII

각 모델: 우리 .dat (5 col: E, flux, std, lower16, upper84) vs Sanghwan .dat.
구성: 2 row × 5 col, 위는 SED overlay (log-log), 아래는 ratio our/sw (linear).
출력: overlay_16yr/phase2_best5_overlay.png

데이터:
  우리:        ./GCE_model_{M}_front_16yr_cholis.dat
  Sanghwan:    ../GCE_16yr_data/Sanghwan_result/GCE_model_{M}_front_16yr_cholis.dat
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

MODELS  = ['X', 'XV', 'XLVIII', 'XLIX', 'LIII']
OUR_DIR = '.'
SW_DIR  = '../GCE_16yr_data/Sanghwan_result'
OUT_PNG = 'overlay_16yr/phase2_best5_overlay.png'


def _load(path):
    d = np.loadtxt(path)
    if d.ndim != 2 or d.shape[1] < 3:
        raise ValueError(f'unexpected shape {d.shape} for {path}')
    E    = d[:, 0]
    flux = d[:, 1]
    std  = d[:, 2]
    lo   = d[:, 3] if d.shape[1] >= 5 else flux - std
    hi   = d[:, 4] if d.shape[1] >= 5 else flux + std
    return E, flux, std, lo, hi


def main():
    os.makedirs('overlay_16yr', exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(20, 7),
                             sharex='col',
                             gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    medians = []
    for col, M in enumerate(MODELS):
        our_p = os.path.join(OUR_DIR, f'GCE_model_{M}_front_16yr_cholis.dat')
        sw_p  = os.path.join(SW_DIR,  f'GCE_model_{M}_front_16yr_cholis.dat')

        if not os.path.exists(our_p):
            print(f'[MISS] our: {our_p}'); sys.exit(1)
        if not os.path.exists(sw_p):
            print(f'[MISS] sw : {sw_p}');  sys.exit(1)

        Eo, fo, so, lo_o, hi_o = _load(our_p)
        Es, fs, _,  _,    _    = _load(sw_p)

        # 두 .dat의 E grid가 다를 수 있어 (14 vs 17 bin) — 짧은 쪽으로 매칭
        n = min(len(Eo), len(Es))
        Em = Eo[:n]
        fo, lo_o, hi_o = fo[:n], lo_o[:n], hi_o[:n]
        fs = fs[:n]
        # 비율: 데이터 보존, 0 division 방지
        ratio = np.where(fs > 0, fo / fs, np.nan)
        med = float(np.nanmedian(ratio))
        medians.append((M, med))

        # SED 상단
        ax = axes[0, col]
        # 우리: errorbar (lo, hi가 16/84 percentile) — asymmetric
        yerr_lo = np.clip(fo - lo_o, 0, None)
        yerr_hi = np.clip(hi_o - fo, 0, None)
        ax.errorbar(Em, fo, yerr=[yerr_lo, yerr_hi], fmt='o-',
                    color='C0', label='reproduced 16yr (DR4, mask1.0)',
                    markersize=4, linewidth=1.2, capsize=2, alpha=0.9)
        ax.plot(Em, fs, 's--', color='C1', label='Sanghwan 16yr front',
                markersize=4, linewidth=1.2, alpha=0.9)

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_ylim(5e-8, 2e-6)
        ax.set_title(f'Model {M}   med(our/sw)={med:.3f}', fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        if col == 0:
            ax.set_ylabel(r'$E^2\, dN/dE$  [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
            ax.legend(loc='lower center', fontsize=8, framealpha=0.9)

        # Ratio 하단
        axr = axes[1, col]
        axr.plot(Em, ratio, 'o-', color='C0', markersize=4, linewidth=1.2)
        axr.axhline(1.0, color='k', lw=0.8, alpha=0.5)
        axr.axhline(med, color='C3', lw=0.8, ls=':',
                    alpha=0.7, label=f'median={med:.3f}')
        axr.set_xscale('log')
        axr.set_ylim(0.2, 1.3)
        axr.set_xlabel('E [GeV]')
        axr.grid(True, which='both', alpha=0.3)
        if col == 0:
            axr.set_ylabel('our / sw')
        axr.legend(loc='lower right', fontsize=8)

    fig.suptitle('16yr reproduce — phase 2 (MapCube flip + bubble 1/1 + iso CLEAN) '
                 'vs Sanghwan 16yr  |  best-5 GDE models',
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    print(f'[done] {OUT_PNG}')

    # 요약 표
    print('\n=== median(our/sw) per model ===')
    for M, m in medians:
        print(f'  {M:>7s}  {m:.4f}')
    print(f'  ---\n  overall median = {np.median([m for _, m in medians]):.4f}')


if __name__ == '__main__':
    main()
