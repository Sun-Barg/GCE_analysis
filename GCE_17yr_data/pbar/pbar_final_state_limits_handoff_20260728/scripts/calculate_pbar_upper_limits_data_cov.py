#!/usr/bin/env python3
"""Calculate pbar DM upper limits with AMS-02 data covariance only.

This is a first likelihood workflow for the USINE BIG-grid products. It uses
only the reconstructed AMS-02 antiproton experimental covariance, C_data. It
therefore does not reproduce the final bounds of arXiv:2202.03076 yet, because
those use C_data + C_model and TOA-consistent flux predictions.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from pbar_likelihood_scaffold import (
    DEFAULT_AMS02,
    DEFAULT_DATA_COVARIANCE,
    DEFAULT_DM_TSV,
    DEFAULT_L_HAT_KPC,
    DEFAULT_LOG_BASE,
    DEFAULT_SECONDARY_DIR,
    DEFAULT_SECONDARY_FLUX_BASENAME,
    DEFAULT_SECONDARY_MANIFEST,
    DEFAULT_SIGMA_LOG_L,
    DEFAULT_SIGMAV_REF,
    AmsPbarData,
    DataCovariance,
    PrimaryDmGrid,
    SecondaryGrid,
    interp_primary_dm,
    interp_secondary,
    log_l_penalty,
    read_ams02_pbar_data,
    read_primary_dm_grid,
    read_secondary_grid,
)


DEFAULT_OUTPUT_DIR = Path("Codex_files/generated_outputs/likelihood_data_cov_toa_phi0p6849_continuous_L")
# Calore et al. 2202.03076 use Delta chi2 = 3.84 for their fixed-mass
# 95% CL upper-limit construction.  Keep this as the default for direct
# reference-paper comparisons; alternative thresholds remain available via
# --lr-target.
LR_95_CL = 3.84


def solve_covariance(covariance: np.ndarray, vector: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(covariance, vector)
    except np.linalg.LinAlgError:
        jitter = 1.0e-12 * float(np.max(np.diag(covariance)))
        return np.linalg.solve(covariance + np.eye(covariance.shape[0]) * jitter, vector)


def n2ll_for_arrays(
    data_flux: np.ndarray,
    covariance: np.ndarray,
    secondary_flux: np.ndarray,
    primary_flux_at_ref: np.ndarray,
    l_kpc: float,
    sigmav: float,
    sigmav_ref: float,
    l_hat_kpc: float,
    sigma_log_l: float,
    log_base: float,
    alpha_sec: float = 1.0,
    alpha_sec_sigma: Optional[float] = 0.0,
) -> float:
    model = alpha_sec * secondary_flux + primary_flux_at_ref * (sigmav / sigmav_ref)
    residual = data_flux - model
    chi2 = float(residual @ solve_covariance(covariance, residual))
    if alpha_sec_sigma is not None and alpha_sec_sigma > 0.0:
        chi2 += ((alpha_sec - 1.0) / alpha_sec_sigma) ** 2
    return chi2 + log_l_penalty(l_kpc, l_hat_kpc, sigma_log_l, log_base)


def best_alpha_for_fixed_sigmav(
    data_flux: np.ndarray,
    covariance: np.ndarray,
    secondary_flux: np.ndarray,
    primary_flux_at_ref: np.ndarray,
    sigmav: float,
    sigmav_ref: float,
    alpha_sec_sigma: Optional[float],
) -> float:
    if alpha_sec_sigma == 0.0:
        return 1.0
    residual_without_secondary = data_flux - primary_flux_at_ref * (sigmav / sigmav_ref)
    denom = float(secondary_flux @ solve_covariance(covariance, secondary_flux))
    numer = float(secondary_flux @ solve_covariance(covariance, residual_without_secondary))
    if alpha_sec_sigma is not None:
        inv_sigma2 = 1.0 / (alpha_sec_sigma * alpha_sec_sigma)
        denom += inv_sigma2
        numer += inv_sigma2
    if denom <= 0.0:
        return 1.0
    return max(0.0, numer / denom)


def _best_sigmav_for_fixed_alpha(
    data_flux: np.ndarray,
    covariance: np.ndarray,
    secondary_flux: np.ndarray,
    primary_flux_at_ref: np.ndarray,
    sigmav_ref: float,
    alpha_sec: float,
) -> float:
    signal_per_unit = primary_flux_at_ref / sigmav_ref
    residual_no_dm = data_flux - alpha_sec * secondary_flux
    denom = float(signal_per_unit @ solve_covariance(covariance, signal_per_unit))
    if denom <= 0.0:
        return 0.0
    numer = float(signal_per_unit @ solve_covariance(covariance, residual_no_dm))
    return max(0.0, numer / denom)


def best_sigmav_alpha_for_l(
    data_flux: np.ndarray,
    covariance: np.ndarray,
    secondary_flux: np.ndarray,
    primary_flux_at_ref: np.ndarray,
    sigmav_ref: float,
    alpha_sec_sigma: Optional[float],
) -> Tuple[float, float]:
    if alpha_sec_sigma == 0.0:
        alpha = 1.0
        sig = _best_sigmav_for_fixed_alpha(
            data_flux, covariance, secondary_flux, primary_flux_at_ref, sigmav_ref, alpha
        )
        return sig, alpha

    signal_per_unit = primary_flux_at_ref / sigmav_ref
    cinv_sec = solve_covariance(covariance, secondary_flux)
    cinv_sig = solve_covariance(covariance, signal_per_unit)
    cinv_data = solve_covariance(covariance, data_flux)

    a00 = float(secondary_flux @ cinv_sec)
    a01 = float(secondary_flux @ cinv_sig)
    a11 = float(signal_per_unit @ cinv_sig)
    b0 = float(secondary_flux @ cinv_data)
    b1 = float(signal_per_unit @ cinv_data)
    if alpha_sec_sigma is not None:
        inv_sigma2 = 1.0 / (alpha_sec_sigma * alpha_sec_sigma)
        a00 += inv_sigma2
        b0 += inv_sigma2

    candidates: list[Tuple[float, float]] = []
    try:
        alpha, sig = np.linalg.solve(np.asarray([[a00, a01], [a01, a11]]), np.asarray([b0, b1]))
        candidates.append((float(sig), float(alpha)))
    except np.linalg.LinAlgError:
        pass

    alpha_sig0 = best_alpha_for_fixed_sigmav(
        data_flux, covariance, secondary_flux, primary_flux_at_ref, 0.0, sigmav_ref, alpha_sec_sigma
    )
    candidates.append((0.0, alpha_sig0))
    candidates.append((_best_sigmav_for_fixed_alpha(data_flux, covariance, secondary_flux, primary_flux_at_ref, sigmav_ref, 0.0), 0.0))
    candidates.append((_best_sigmav_for_fixed_alpha(data_flux, covariance, secondary_flux, primary_flux_at_ref, sigmav_ref, 1.0), 1.0))

    valid = [(max(0.0, sig), max(0.0, alpha)) for sig, alpha in candidates if np.isfinite(sig) and np.isfinite(alpha)]
    if not valid:
        return 0.0, 1.0

    def chi(pair: Tuple[float, float]) -> float:
        sig, alpha = pair
        model = alpha * secondary_flux + primary_flux_at_ref * (sig / sigmav_ref)
        residual = data_flux - model
        value = float(residual @ solve_covariance(covariance, residual))
        if alpha_sec_sigma is not None and alpha_sec_sigma > 0.0:
            value += ((alpha - 1.0) / alpha_sec_sigma) ** 2
        return value

    return min(valid, key=chi)


def flux_arrays_for_l(
    data: AmsPbarData,
    secondary: SecondaryGrid,
    primary: PrimaryDmGrid,
    l_kpc: float,
    mdm_gev: float,
) -> Tuple[np.ndarray, np.ndarray]:
    secondary_flux = interp_secondary(secondary, l_kpc, data.rigidity_gv)
    primary_flux = interp_primary_dm(primary, l_kpc, mdm_gev, data.rigidity_gv)
    return secondary_flux, primary_flux



def golden_section_minimize(func, lo: float, hi: float, tol: float = 1.0e-5, max_iter: int = 80) -> Tuple[float, float]:
    """Minimize a scalar function on [lo, hi] with golden-section search."""
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        return lo, float(func(lo))
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    inv_phi2 = (3.0 - math.sqrt(5.0)) / 2.0
    a = lo
    b = hi
    h = b - a
    c = a + inv_phi2 * h
    d = a + inv_phi * h
    yc = float(func(c))
    yd = float(func(d))
    for _ in range(max_iter):
        if abs(b - a) <= tol:
            break
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = inv_phi * h
            c = a + inv_phi2 * h
            yc = float(func(c))
        else:
            a = c
            c = d
            yc = yd
            h = inv_phi * h
            d = a + inv_phi * h
            yd = float(func(d))
    x = 0.5 * (a + b)
    y = float(func(x))
    # Also test endpoints.  Interpolated grids can have interval-edge minima.
    candidates = [(x, y), (lo, float(func(lo))), (hi, float(func(hi)))]
    return min(candidates, key=lambda item: item[1])


def minimize_over_log_l(func, l_values: Sequence[float], tol: float = 1.0e-5) -> Tuple[float, float]:
    """Minimize func(L) continuously over the log-L grid span.

    The flux model is piecewise log-interpolated between generated L grid points.
    Minimizing each adjacent interval avoids assuming the full function is globally
    unimodal over [L_min, L_max].
    """
    values = np.asarray(sorted(float(x) for x in l_values), dtype=float)
    if np.any(values <= 0.0):
        raise ValueError("L values must be positive for log-L profiling")
    best_log_l = math.log(values[0])
    best_val = float(func(values[0]))
    for lo_l, hi_l in zip(values[:-1], values[1:]):
        lo = math.log(float(lo_l))
        hi = math.log(float(hi_l))

        def wrapped(log_l: float) -> float:
            return float(func(math.exp(log_l)))

        opt_log_l, opt_val = golden_section_minimize(wrapped, lo, hi, tol=tol)
        if opt_val < best_val:
            best_log_l = opt_log_l
            best_val = opt_val
    last_val = float(func(values[-1]))
    if last_val < best_val:
        return float(values[-1]), last_val
    return math.exp(best_log_l), best_val


def profile_l_for_sigmav(
    data: AmsPbarData,
    covariance: np.ndarray,
    secondary: SecondaryGrid,
    primary: PrimaryDmGrid,
    mdm_gev: float,
    sigmav: float,
    l_values: Sequence[float],
    sigmav_ref: float,
    l_hat_kpc: float,
    sigma_log_l: float,
    log_base: float,
    alpha_sec_sigma: Optional[float],
) -> Tuple[float, float, float]:
    alpha_by_l: dict[float, float] = {}

    def objective(l_kpc: float) -> float:
        sec, dm = flux_arrays_for_l(data, secondary, primary, float(l_kpc), mdm_gev)
        alpha = best_alpha_for_fixed_sigmav(
            data.flux,
            covariance,
            sec,
            dm,
            sigmav,
            sigmav_ref,
            alpha_sec_sigma,
        )
        alpha_by_l[float(l_kpc)] = alpha
        return n2ll_for_arrays(
            data.flux,
            covariance,
            sec,
            dm,
            float(l_kpc),
            sigmav,
            sigmav_ref,
            l_hat_kpc,
            sigma_log_l,
            log_base,
            alpha_sec=alpha,
            alpha_sec_sigma=alpha_sec_sigma,
        )

    best_l, best_val = minimize_over_log_l(objective, l_values)
    sec, dm = flux_arrays_for_l(data, secondary, primary, best_l, mdm_gev)
    best_alpha = best_alpha_for_fixed_sigmav(
        data.flux, covariance, sec, dm, sigmav, sigmav_ref, alpha_sec_sigma
    )
    return best_l, best_val, best_alpha


def global_best_for_mass(
    data: AmsPbarData,
    covariance: np.ndarray,
    secondary: SecondaryGrid,
    primary: PrimaryDmGrid,
    mdm_gev: float,
    l_values: Sequence[float],
    sigmav_ref: float,
    l_hat_kpc: float,
    sigma_log_l: float,
    log_base: float,
    alpha_sec_sigma: Optional[float],
) -> Tuple[float, float, float, float]:
    sigmav_at_l: dict[float, float] = {}
    alpha_at_l: dict[float, float] = {}

    def objective(l_kpc: float) -> float:
        sec, dm = flux_arrays_for_l(data, secondary, primary, float(l_kpc), mdm_gev)
        sig, alpha = best_sigmav_alpha_for_l(data.flux, covariance, sec, dm, sigmav_ref, alpha_sec_sigma)
        sigmav_at_l[float(l_kpc)] = float(sig)
        alpha_at_l[float(l_kpc)] = float(alpha)
        return n2ll_for_arrays(
            data.flux,
            covariance,
            sec,
            dm,
            float(l_kpc),
            sig,
            sigmav_ref,
            l_hat_kpc,
            sigma_log_l,
            log_base,
            alpha_sec=alpha,
            alpha_sec_sigma=alpha_sec_sigma,
        )

    best_l, best_n2ll = minimize_over_log_l(objective, l_values)
    sec, dm = flux_arrays_for_l(data, secondary, primary, best_l, mdm_gev)
    best_sigmav, best_alpha = best_sigmav_alpha_for_l(
        data.flux, covariance, sec, dm, sigmav_ref, alpha_sec_sigma
    )
    best_n2ll = n2ll_for_arrays(
        data.flux,
        covariance,
        sec,
        dm,
        best_l,
        best_sigmav,
        sigmav_ref,
        l_hat_kpc,
        sigma_log_l,
        log_base,
        alpha_sec=best_alpha,
        alpha_sec_sigma=alpha_sec_sigma,
    )
    return best_l, float(best_sigmav), float(best_alpha), best_n2ll


def find_upper_limit(
    data: AmsPbarData,
    covariance: np.ndarray,
    secondary: SecondaryGrid,
    primary: PrimaryDmGrid,
    mdm_gev: float,
    l_values: Sequence[float],
    sigmav_ref: float,
    l_hat_kpc: float,
    sigma_log_l: float,
    log_base: float,
    lr_target: float,
    max_sigmav: float,
    alpha_sec_sigma: Optional[float],
) -> Tuple[float, float, float, float, float, float]:
    best_l, best_sigmav, best_alpha, best_n2ll = global_best_for_mass(
        data,
        covariance,
        secondary,
        primary,
        mdm_gev,
        l_values,
        sigmav_ref,
        l_hat_kpc,
        sigma_log_l,
        log_base,
        alpha_sec_sigma,
    )

    def lr_at(sig: float) -> float:
        _, prof_n2ll, _ = profile_l_for_sigmav(
            data,
            covariance,
            secondary,
            primary,
            mdm_gev,
            sig,
            l_values,
            sigmav_ref,
            l_hat_kpc,
            sigma_log_l,
            log_base,
            alpha_sec_sigma,
        )
        return prof_n2ll - best_n2ll

    lo = best_sigmav
    hi = max(sigmav_ref, lo * 2.0 if lo > 0.0 else sigmav_ref)
    lr_hi = lr_at(hi)
    while lr_hi < lr_target and hi < max_sigmav:
        hi *= 2.0
        lr_hi = lr_at(hi)

    if lr_hi < lr_target:
        return math.nan, best_l, best_sigmav, best_alpha, best_n2ll, lr_hi

    for _ in range(80):
        mid = math.sqrt(lo * hi) if lo > 0.0 else 0.5 * hi
        lr_mid = lr_at(mid)
        if lr_mid >= lr_target:
            hi = mid
        else:
            lo = mid
        if hi > 0.0 and abs(hi - lo) / hi < 1.0e-4:
            break
    return hi, best_l, best_sigmav, best_alpha, best_n2ll, lr_at(hi)


def write_table(path: Path, rows: Sequence[dict]) -> None:
    columns = [
        "mDM_GeV",
        "sigmav95_cm3_s",
        "best_sigmav_cm3_s",
        "best_alpha_sec",
        "best_L_kpc",
        "best_minus2loglike",
        "lr_at_limit",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(f"{row[col]:.10e}" for col in columns) + "\n")


def maybe_plot_limits(path: Path, rows: Sequence[dict]) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    masses = np.asarray([row["mDM_GeV"] for row in rows], dtype=float)
    limits = np.asarray([row["sigmav95_cm3_s"] for row in rows], dtype=float)
    ok = np.isfinite(limits) & (limits > 0.0)
    if not np.any(ok):
        return None
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.loglog(masses[ok], limits[ok], marker="o", lw=1.6)
    ax.set_xlabel(r"$m_\chi$ [GeV]")
    ax.set_ylabel(r"95% CL $\langle\sigma v\rangle$ [cm$^3$ s$^{-1}$]")
    ax.set_title("BIG pbar limit, data covariance only")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = path / "pbar_95cl_upper_limits_data_cov_only.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate 95% CL pbar DM upper limits with AMS-02 C_data only."
    )
    parser.add_argument("--ams02", type=Path, default=DEFAULT_AMS02)
    parser.add_argument("--data-covariance", type=Path, default=DEFAULT_DATA_COVARIANCE)
    parser.add_argument(
        "--covariance-blocks",
        nargs="*",
        default=None,
        help="Optional covariance block names to use, e.g. stat. Defaults to all C_data blocks.",
    )
    parser.add_argument("--secondary-dir", type=Path, default=DEFAULT_SECONDARY_DIR)
    parser.add_argument("--secondary-manifest", type=Path, default=DEFAULT_SECONDARY_MANIFEST)
    parser.add_argument("--secondary-flux-basename", default=DEFAULT_SECONDARY_FLUX_BASENAME)
    parser.add_argument("--dm-grid-tsv", type=Path, default=DEFAULT_DM_TSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sigmav-ref", type=float, default=DEFAULT_SIGMAV_REF)
    parser.add_argument(
        "--primary-scale",
        type=float,
        default=1.0,
        help="Diagnostic multiplier applied to the primary DM template fluxes.",
    )
    parser.add_argument("--l-hat-kpc", type=float, default=DEFAULT_L_HAT_KPC)
    parser.add_argument("--sigma-log-l", type=float, default=DEFAULT_SIGMA_LOG_L)
    parser.add_argument("--log-base", type=float, default=DEFAULT_LOG_BASE)
    parser.add_argument("--lr-target", type=float, default=LR_95_CL)
    parser.add_argument("--max-sigmav", type=float, default=1.0e-21)
    parser.add_argument(
        "--secondary-scale-mode",
        choices=("fixed", "free", "gaussian"),
        default="fixed",
        help="Profile a global secondary-pbar normalization nuisance.",
    )
    parser.add_argument(
        "--secondary-scale-sigma",
        type=float,
        default=0.20,
        help="Gaussian sigma for --secondary-scale-mode gaussian.",
    )
    parser.add_argument(
        "--masses",
        type=float,
        nargs="*",
        default=None,
        help="Optional DM masses in GeV. Defaults to all available grid masses.",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = read_ams02_pbar_data(args.ams02)
    secondary = read_secondary_grid(args.secondary_dir, args.secondary_manifest, args.secondary_flux_basename)
    primary = read_primary_dm_grid(args.dm_grid_tsv, drop_highest_mass=True)
    if args.primary_scale <= 0.0:
        raise ValueError("--primary-scale must be positive")
    if args.primary_scale != 1.0:
        primary = PrimaryDmGrid(
            l_values_kpc=primary.l_values_kpc,
            mdm_values_gev=primary.mdm_values_gev,
            curves={
                key: type(curve)(rigidity_gv=curve.rigidity_gv, flux=curve.flux * args.primary_scale)
                for key, curve in primary.curves.items()
            },
        )
    covariance_provider = DataCovariance(args.data_covariance, blocks=args.covariance_blocks)
    covariance = covariance_provider.covariance(data, np.zeros_like(data.flux), None)  # type: ignore[arg-type]
    if args.secondary_scale_mode == "fixed":
        alpha_sec_sigma: Optional[float] = 0.0
    elif args.secondary_scale_mode == "free":
        alpha_sec_sigma = None
    else:
        alpha_sec_sigma = float(args.secondary_scale_sigma)
        if alpha_sec_sigma <= 0.0:
            raise ValueError("--secondary-scale-sigma must be positive for gaussian mode")

    l_values = [float(x) for x in secondary.l_values_kpc]
    masses = np.asarray(args.masses if args.masses else primary.mdm_values_gev, dtype=float)
    rows = []

    print("Calculating data-covariance-only upper limits")
    print(f"  AMS-02 bins: {len(data.flux)}")
    print(f"  C_data blocks: {', '.join(covariance_provider.block_names)}")
    print(f"  L profiling: continuous log-L interpolation over {len(l_values)} grid values from {min(l_values):.4g} to {max(l_values):.4g} kpc")
    print(f"  Masses: {len(masses)} values from {masses[0]:.4g} to {masses[-1]:.4g} GeV")
    print(f"  Secondary scale mode: {args.secondary_scale_mode}")
    print(f"  Primary DM template scale: {args.primary_scale:.6g}")
    print("  Caveat: C_model is not included; flux inputs are the selected TOA grid.")

    for mdm in masses:
        limit, best_l, best_sig, best_alpha, best_n2ll, lr_limit = find_upper_limit(
            data=data,
            covariance=covariance,
            secondary=secondary,
            primary=primary,
            mdm_gev=float(mdm),
            l_values=l_values,
            sigmav_ref=args.sigmav_ref,
            l_hat_kpc=args.l_hat_kpc,
            sigma_log_l=args.sigma_log_l,
            log_base=args.log_base,
            lr_target=args.lr_target,
            max_sigmav=args.max_sigmav,
            alpha_sec_sigma=alpha_sec_sigma,
        )
        rows.append(
            {
                "mDM_GeV": float(mdm),
                "sigmav95_cm3_s": float(limit),
                "best_sigmav_cm3_s": float(best_sig),
                "best_alpha_sec": float(best_alpha),
                "best_L_kpc": float(best_l),
                "best_minus2loglike": float(best_n2ll),
                "lr_at_limit": float(lr_limit),
            }
        )
        print(
            f"  mDM={mdm:.6g} GeV: sigmav95={limit:.6e}, "
            f"best sigmav={best_sig:.3e}, best alpha={best_alpha:.4g}, best L={best_l:.4g} kpc"
        )

    table_path = args.output_dir / "pbar_95cl_upper_limits_data_cov_only.tsv"
    write_table(table_path, rows)
    print(f"Wrote {table_path}")
    if not args.no_plot:
        plot_path = maybe_plot_limits(args.output_dir, rows)
        if plot_path:
            print(f"Wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
