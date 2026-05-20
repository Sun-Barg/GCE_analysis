#!/usr/bin/env python3
# =============================================================================
# interp_offshell_spectra.py  (v1, 2026-07-10)
#   off-shell production 10노드(Δr=0.1)의 photon 스펙트럼을 r-방향으로 보간해
#   소수-r 스펙트럼 "파일"을 생성한다. 목적: ④⑤의 σv*/TS가 파생량(TS) 보간이
#   아니라 보간된 스펙트럼 위에서 계산되도록 하여, 전역 best fit이 소수 r에
#   앉을 수 있게 함 (해밝 제안 2026-07-10).
#
#   설계 (세션 합의):
#   · 보간량 = dN/dx (x = E/m_χ), 공통 x-빈 위 고정-x 선형 보간.
#     ① 브래킷 폴더 안 질량 정렬(raw x-빈 위 질량 선형) → ② r 선형.
#     로그공간 보간은 --yspace log 비교 옵션(제로빈 때문에 기본은 linear).
#   · hull 규율(④ v9 승계): 목표 질량은 "양쪽" 브래킷 폴더 질량 범위의
#     교집합에서만 생성 — 한쪽 노드 외삽(클램프) 금지. 보간 폴더의 최저
#     질량 = floor(r_lo) (r_lo < r_new < r_hi, floor는 r 감소 방향으로 상승).
#   · 네임스페이스 격리(§3-3 위생): 출력 dir = Spectra_Data_sfdm_{ch}_offint_r{r:.2f}
#     — production glob '*_off_r*'와 불일치(①③④⑥ 안전 확인). 폴더마다
#     _INTERP_MANIFEST.json 기록(합성/파생 데이터 영구 표식). 결손 18 재생성 아님.
#   · CSV 포맷 = production 동일(컬럼 Energy_GeV,dNdE; MM_mpsi{m}GeV_interp_photon.csv)
#     → MG5Interpolator(base_dir=...)가 그대로 소비 가능.
#
#   모드:
#     gen : 소수-r 폴더 생성. 기본 Δr=0.02, r∈(0.10,1.00) 노드 제외 36값.
#     val : leave-one-out — 내부 노드 r_v(0.2..0.9)를 숨기고 (r_v±0.1)로 재구성
#           → Spectra_Data_sfdm_{ch}_offintVAL_r{r_v:.2f} 생성 + production 대비
#           스펙트럼 상대 L1(전체-x / fit-창) 표 출력·CSV 저장.
#           (fit-층 σv*/TS 대조는 노트북 셀 cell_interp_r_validation_v1.py 담당.)
#
#   사용 예 (neutrino, conda env fermi — numpy/pandas만 사용):
#     cd /home/haebarg/MG5_aMC_v3_5_12
#     python interp_offshell_spectra.py val --channels 4b,4tau,2b2tau
#     python interp_offshell_spectra.py gen --channels 4b,4tau,2b2tau --dr 0.02
#   옵션: --rlist 0.45,0.47 (gen 특정 r만) / --force (기존 폴더 재생성)
#         / --dry (계획만 출력) / --yspace lin|log
# =============================================================================
import os, re, sys, glob, json, time, argparse
import numpy as np
import pandas as pd

MG5_BASE = os.environ.get('MG5_BASE', '/home/haebarg/MG5_aMC_v3_5_12')
SPECTRA_ROOT = os.environ.get('SPECTRA_ROOT', f'{MG5_BASE}/Spectra')

def _resolve_root():
    """통합 루트(Spectra/) 우선; 미구성이면 구 루트(MG5_BASE)로 fallback."""
    if glob.glob(f'{SPECTRA_ROOT}/Spectra_Data_sfdm_*_off_r*'):
        return SPECTRA_ROOT
    if glob.glob(f'{MG5_BASE}/Spectra_Data_sfdm_*_off_r*'):
        print(f'[root] {SPECTRA_ROOT} 미구성 — 구 루트(MG5_BASE) 사용 '
              f'(consolidate_spectra.sh 실행 전 상태)')
        return MG5_BASE
    return SPECTRA_ROOT

SCRIPT_VER = 'interp_offshell_spectra v2 (2026-07-10; SPECTRA_ROOT 통합 루트 대응 — 수학·로직 v1.1 동일)'
# fit 창(⑥ _EWIN 동일 취지): 17yr 14빈 [0.275, 51.9] GeV
EWIN = (0.275, 51.9)


ROOT = None  # CLI에서 _resolve_root()로 확정; import 사용 시 직접 지정


# ---------- production 폴더/CSV 읽기 (MG5Interpolator._read_csv 규약 미러) ------
def read_photon_csv(fpath):
    df = pd.read_csv(fpath)
    low = {c.lower().replace(' ', '').replace('/', ''): c for c in df.columns}
    e_key = next((low[k] for k in low
                  if k in ('energy_gev', 'energy', 'e_gev', 'e', 'egev')), None)
    dn_key = next((low[k] for k in low
                   if k in ('dnde', 'dn_de', 'dndegev', 'flux')), None)
    if e_key is not None and dn_key is not None:
        df = df.sort_values(e_key)
        return df[e_key].to_numpy(float), df[dn_key].to_numpy(float)
    d = df.to_numpy(float)
    return d[:, 0], d[:, 1]


def prod_r_dirs(channel):
    out = []
    for d in glob.glob(f'{ROOT}/Spectra_Data_sfdm_{channel}_off_r*'):
        if os.path.isdir(d):
            m = re.search(r'_r([0-9.]+)$', d.rstrip('/'))
            if m:
                out.append((float(m.group(1)), d))
    return sorted(out)


class NodeFolder:
    """production r-노드 1개: 공통 x-빈 위 dN/dx 행렬 + 질량 정렬 보간."""

    def __init__(self, r, path, xg_ref=None):
        self.r, self.path = float(r), path
        masses, rows, srcs = [], [], []
        self.xg = xg_ref
        n_size, n_shift = 0, 0                 # 불일치 유형: 빈 개수 상이 / 값 이동
        bstat = []                             # (m, n_bins, x_min, x_max)
        for f in sorted(glob.glob(os.path.join(path, 'MM_mpsi*GeV_*_photon.csv'))):
            base = os.path.basename(f)
            try:
                m = float(base.split('mpsi')[1].split('GeV')[0])
                e, y = read_photon_csv(f)
            except Exception:
                continue
            v = (e > 0) & np.isfinite(y) & (y >= 0)
            if v.sum() < 4:
                continue
            e, y = e[v], y[v]
            x = e / m                      # 공통 x-빈 전제(⑥ 동일) — 아래서 검증
            dndx = m * y                   # dN/dx = m · dN/dE
            bstat.append((m, x.size, float(x.min()), float(x.max())))
            if self.xg is None:
                self.xg = x.copy()
            # rtol 1e-3: 파일명 질량(소수 2자리) 반올림에 의한 그리드 시프트
            # (~5e-4, 빈폭 대비 무시 가능)는 동일 빈으로 간주; 진짜 빈-스킴
            # 불일치만 리샘플 fallback.
            if x.size == self.xg.size and np.allclose(x, self.xg, rtol=1e-3,
                                                      atol=0.0):
                row = dndx
            else:                          # 빈-스킴 불일치 → log-x 위 선형(값) 리샘플
                if x.size != self.xg.size:
                    n_size += 1
                else:
                    n_shift += 1
                lg = np.log10(self.xg)
                row = np.interp(lg, np.log10(x), dndx, left=0.0, right=0.0)
            masses.append(m); rows.append(row); srcs.append(base)
        if not masses:
            raise FileNotFoundError(f'no usable CSV in {path}')
        order = np.argsort(masses)
        self.masses = np.asarray(masses)[order]
        self.mat = np.asarray(rows)[order]          # (n_m, n_x) dN/dx
        self.srcs = [srcs[i] for i in order]
        self.n_resamp = n_size + n_shift
        nb = np.array([b[1] for b in bstat])
        xlo = np.array([b[2] for b in bstat]); xhi = np.array([b[3] for b in bstat])
        self.bin_report = (f'[bins] {os.path.basename(path)}: files {len(bstat)} | '
                           f'n_bins {nb.min()}/{int(np.median(nb))}/{nb.max()} | '
                           f'x_lo {xlo.min():.2e}..{xlo.max():.2e} | '
                           f'x_hi {xhi.min():.4f}..{xhi.max():.4f} | '
                           f'canonical n={self.xg.size}'
                           f'(x∈[{self.xg.min():.2e},{self.xg.max():.4f}]) | '
                           f'mismatch size {n_size} / shift {n_shift}')
        print(self.bin_report)

    def dndx_at(self, m):
        """폴더 안 질량 선형 보간(고정 x-빈). 범위 밖 = None (외삽 금지)."""
        ms = self.masses
        if m < ms[0] - 1e-9 or m > ms[-1] + 1e-9:
            return None
        j = int(np.searchsorted(ms, m))
        if j < ms.size and abs(ms[j] - m) < 1e-9 * max(m, 1.0):
            return self.mat[j]
        if j == 0:
            return self.mat[0]
        w = (m - ms[j - 1]) / (ms[j] - ms[j - 1])
        return (1 - w) * self.mat[j - 1] + w * self.mat[j]


_NODE_CACHE = {}


def get_node(channel, r, path):
    key = (channel, round(float(r), 6))
    if key not in _NODE_CACHE:
        ref = None
        for k, nf in _NODE_CACHE.items():           # 채널 공통 x-빈 기준 공유
            if k[0] == channel:
                ref = nf.xg; break
        _NODE_CACHE[key] = NodeFolder(r, path, xg_ref=ref)
    return _NODE_CACHE[key]


# ---------- r-보간 코어 -------------------------------------------------------
def interp_r(nlo, nhi, r_new, yspace='lin'):
    """두 노드 폴더 사이 r 선형 보간 → (target_masses, dndx_matrix, xg).
    질량 그리드 = 두 폴더 질량의 합집합 ∩ 교집합 도메인(hull; 외삽 금지)."""
    m_lo, m_hi = nlo.masses, nhi.masses
    dom = (max(m_lo.min(), m_hi.min()), min(m_lo.max(), m_hi.max()))
    tgt = np.unique(np.concatenate([m_lo, m_hi]))
    tgt = tgt[(tgt >= dom[0] - 1e-9) & (tgt <= dom[1] + 1e-9)]
    w = (r_new - nlo.r) / (nhi.r - nlo.r)
    out_m, out_rows = [], []
    for m in tgt:
        a, b = nlo.dndx_at(m), nhi.dndx_at(m)
        if a is None or b is None:
            continue
        if yspace == 'log':
            pos = (a > 0) & (b > 0)
            row = np.zeros_like(a)
            row[pos] = 10 ** ((1 - w) * np.log10(a[pos]) + w * np.log10(b[pos]))
        else:
            row = (1 - w) * a + w * b
        out_m.append(m); out_rows.append(row)
    return np.asarray(out_m), np.asarray(out_rows), nlo.xg, dom, w


def write_folder(out_dir, channel, r_new, masses, rows, xg, meta, force=False):
    man_p = os.path.join(out_dir, '_INTERP_MANIFEST.json')
    if os.path.isdir(out_dir) and os.path.exists(man_p) and not force:
        print(f'  [skip] {os.path.basename(out_dir)} 존재(manifest 有) — --force로 재생성')
        return 0
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, 'MM_mpsi*_photon.csv')):
        os.remove(f)
    n = 0
    for m, row in zip(masses, rows):
        E = xg * m
        dNdE = row / m
        keep = E > 0
        df = pd.DataFrame({'Energy_GeV': E[keep], 'dNdE': dNdE[keep]})
        fname = f'MM_mpsi{m:.2f}GeV_interp_photon.csv'
        df.to_csv(os.path.join(out_dir, fname), index=False,
                  float_format='%.8e')
        n += 1
    meta = dict(meta, script=SCRIPT_VER, channel=channel, r=round(r_new, 4),
                n_masses=int(n), mass_min=float(masses.min()),
                mass_max=float(masses.max()),
                created=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(man_p, 'w') as fp:
        json.dump(meta, fp, indent=1, ensure_ascii=False)
    return n


# ---------- gen 모드 ----------------------------------------------------------
def run_gen(channels, dr, rlist, yspace, force, dry):
    report = []
    for ch in channels:
        nodes = prod_r_dirs(ch)
        if len(nodes) < 2:
            print(f'[{ch}] production 노드 부족 — skip'); continue
        r_nodes = np.array([r for r, _ in nodes])
        if rlist:
            targets = [r for r in rlist
                       if r_nodes.min() < r < r_nodes.max()
                       and not np.isclose(r, r_nodes, atol=1e-6).any()]
        else:
            grid = np.round(np.arange(r_nodes.min() + dr, r_nodes.max(), dr), 6)
            targets = [float(r) for r in grid
                       if not np.isclose(r, r_nodes, atol=1e-6).any()]
        print(f'[{ch}] production {len(nodes)}노드 '
              f'r={r_nodes.tolist()} → 보간 목표 {len(targets)}개 '
              f'(Δr={dr if not rlist else "rlist"}, yspace={yspace})')
        if dry:
            print(f'   targets: {targets}'); continue
        for r_new in targets:
            jlo = int(np.searchsorted(r_nodes, r_new) - 1)
            nlo = get_node(ch, *nodes[jlo])
            nhi = get_node(ch, *nodes[jlo + 1])
            masses, rows, xg, dom, w = interp_r(nlo, nhi, r_new, yspace)
            if masses.size == 0:
                print(f'   r={r_new:.2f}: 유효 질량 없음 — skip'); continue
            out_dir = f'{ROOT}/Spectra_Data_sfdm_{ch}_offint_r{r_new:.2f}'
            meta = dict(mode='gen', yspace=yspace, w=round(float(w), 6),
                        bracket=[nlo.r, nhi.r],
                        bracket_dirs=[os.path.basename(nodes[jlo][1]),
                                      os.path.basename(nodes[jlo + 1][1])],
                        mass_domain=[float(dom[0]), float(dom[1])],
                        note='derived by r-interpolation of production spectra '
                             '(NOT new MC); hull-clipped, no extrapolation')
            n = write_folder(out_dir, ch, r_new, masses, rows, xg, meta, force)
            if n:
                print(f'   r={r_new:.2f}: {n}질량 (m∈[{masses.min():.1f},'
                      f'{masses.max():.1f}], 브래킷 {nlo.r:g}/{nhi.r:g})'
                      f' → {os.path.basename(out_dir)}')
                report.append((ch, r_new, n, masses.min(), masses.max()))
    if report and not dry:
        rp = f'{ROOT}/offint_generation_report.csv'
        pd.DataFrame(report, columns=['channel', 'r', 'n_masses',
                                      'm_min', 'm_max']).to_csv(rp, index=False)
        print(f'[gen] report → {rp}')


# ---------- val 모드 (leave-one-out, 스펙트럼 층) ------------------------------
def run_val(channels, yspace, force):
    rows_out, detail_rows = [], []
    for ch in channels:
        nodes = prod_r_dirs(ch)
        r_nodes = np.array([r for r, _ in nodes])
        print(f'\n[{ch}] leave-one-out — 내부 노드 재구성 vs production '
              f'(yspace={yspace})')
        print(f'   지표: rel L1 = Σ|y_int−y_prod| / Σ y_prod  '
              f'(dN/dx, 전체 x / fit-창 x∈[{EWIN[0]}/m, min({EWIN[1]}/m,1)])')
        for jv in range(1, len(nodes) - 1):
            r_v, path_v = nodes[jv]
            nlo = get_node(ch, *nodes[jv - 1])
            nhi = get_node(ch, *nodes[jv + 1])
            nprod = get_node(ch, r_v, path_v)
            masses, rows, xg, dom, w = interp_r(nlo, nhi, r_v, yspace)
            if masses.size == 0:
                print(f'   r_v={r_v:g}: 도메인 공백 — skip'); continue
            out_dir = (f'{ROOT}/Spectra_Data_sfdm_{ch}_offintVAL_r{r_v:.2f}')
            meta = dict(mode='val-holdout', yspace=yspace,
                        bracket=[nlo.r, nhi.r], holdout=r_v,
                        note='leave-one-out reconstruction of a production '
                             'node from ±0.1 neighbours (validation only)')
            write_folder(out_dir, ch, r_v, masses, rows, xg, meta, force=True)
            # --- production 공통 질량에서 형상 대조 (질량별 상세 수집) ---
            det = []                       # (m, L1_all, L1_fit, mh2_lo, mh2_v, mh2_hi)
            for m, yp in zip(nprod.masses, nprod.mat):
                if m < masses.min() - 1e-9 or m > masses.max() + 1e-9:
                    continue                          # hull 쐐기 밖 — 대조 불가
                k = int(np.argmin(np.abs(masses - m)))
                if abs(masses[k] - m) > 1e-6 * max(m, 1.0):
                    # 목표그리드에 없으면 브래킷에서 직접 재구성
                    a, b = nlo.dndx_at(m), nhi.dndx_at(m)
                    if a is None or b is None:
                        continue
                    ww = (r_v - nlo.r) / (nhi.r - nlo.r)
                    yi = (1 - ww) * a + ww * b
                else:
                    yi = rows[k]
                den = yp.sum()
                if den <= 0:
                    continue
                l1a_m = np.abs(yi - yp).sum() / den
                xw = (xg >= EWIN[0] / m) & (xg <= min(EWIN[1] / m, 1.0))
                l1f_m = (np.abs(yi[xw] - yp[xw]).sum() / yp[xw].sum()
                         if (xw.sum() >= 4 and yp[xw].sum() > 0) else np.nan)
                det.append((m, l1a_m, l1f_m,
                            nlo.r * m, r_v * m, nhi.r * m))
            if not det:
                print(f'   r_v={r_v:g}: 공통 질량 없음 — skip'); continue
            D = np.array(det)
            l1a, l1f = D[:, 1], D[:, 2][np.isfinite(D[:, 2])]
            print(f'   r_v={r_v:g} (브래킷 {nlo.r:g}/{nhi.r:g}, 공통질량 {len(det)}): '
                  f'L1_all med={np.median(l1a)*100:5.2f}% max={l1a.max()*100:5.2f}% | '
                  f'L1_fit med={np.median(l1f)*100:5.2f}% max={l1f.max()*100:5.2f}%')
            fin_f = np.isfinite(D[:, 2])
            for j in np.argsort(-np.where(fin_f, D[:, 2], -np.inf))[:3]:
                m, _, lf, mlo_h, mv_h, mhi_h = D[j]
                print(f'      worst: m={m:7.1f}  L1_fit={lf*100:6.2f}%  '
                      f'm_h2 브래킷 {mlo_h:.1f}→{mhi_h:.1f} (@r_v {mv_h:.1f})')
            for m, a_, f_, mlo_h, mv_h, mhi_h in det:
                detail_rows.append((ch, r_v, m, a_, f_, mlo_h, mv_h, mhi_h))
            rows_out.append((ch, r_v, len(det), np.median(l1a), l1a.max(),
                             np.median(l1f), l1f.max()))
    if rows_out:
        rp = f'{ROOT}/offint_validation_report.csv'
        pd.DataFrame(rows_out, columns=['channel', 'r_holdout', 'n_masses',
                                        'L1all_med', 'L1all_max',
                                        'L1fit_med', 'L1fit_max']
                     ).to_csv(rp, index=False)
        dp = f'{ROOT}/offint_validation_detail.csv'
        pd.DataFrame(detail_rows,
                     columns=['channel', 'r_holdout', 'mass', 'L1_all',
                              'L1_fit', 'mh2_lo', 'mh2_at_rv', 'mh2_hi']
                     ).to_csv(dp, index=False)
        print(f'\n[val] report → {rp}')
        print(f'[val] detail → {dp}   '
              f'(fit-층 σv*/TS 대조 = 노트북 cell_interp_r_validation_v1.py)')


# ---------- CLI ----------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=SCRIPT_VER)
    ap.add_argument('mode', choices=['gen', 'val'])
    ap.add_argument('--channels', default='4b,4tau,2b2tau')
    ap.add_argument('--dr', type=float, default=0.02)
    ap.add_argument('--rlist', default='',
                    help='gen 전용: 쉼표구분 r 목록(지정 시 --dr 무시)')
    ap.add_argument('--yspace', choices=['lin', 'log'], default='lin')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    globals()['ROOT'] = _resolve_root()
    print(f'[root] 데이터 루트 = {ROOT}')
    chs = [c.strip() for c in a.channels.split(',') if c.strip()]
    rl = [float(x) for x in a.rlist.split(',') if x.strip()] if a.rlist else None
    t0 = time.time()
    if a.mode == 'gen':
        run_gen(chs, a.dr, rl, a.yspace, a.force, a.dry)
    else:
        run_val(chs, a.yspace, a.force)
    print(f'[done] {time.time()-t0:.1f}s')
