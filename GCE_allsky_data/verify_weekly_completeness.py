#!/usr/bin/env python3
"""
verify_weekly_completeness.py

Photon + Spacecraft weekly 파일들이
  (1) 연속 주 누락 없이 받아졌는지
  (2) FITS header가 정상인지
  (3) photon ↔ SC 짝이 맞는지
검증.

Usage:
    python verify_weekly_completeness.py \
        --photon-dir ./photon_files \
        --sc-dir ./sc_files \
        [--start 9] [--end 950]
"""
import argparse
import re
from pathlib import Path

try:
    from astropy.io import fits
    HAVE_ASTROPY = True
except ImportError:
    HAVE_ASTROPY = False


PHOTON_RE = re.compile(r"lat_photon_weekly_w(\d{3})_p305_v001\.fits$")
SC_RE     = re.compile(r"lat_spacecraft_weekly_w(\d{3})_p310_v001\.fits$")


def list_weeks(directory: Path, regex: re.Pattern) -> dict[int, Path]:
    """Return {week_number: path} dict of files matching regex."""
    out = {}
    if not directory.is_dir():
        return out
    for p in directory.iterdir():
        m = regex.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def find_gaps(weeks: set[int], start: int, end: int) -> list[int]:
    """Return list of missing week numbers in [start, end] range."""
    actual_max = max(weeks) if weeks else start
    upper = min(end, actual_max)  # don't flag future weeks
    return sorted(w for w in range(start, upper + 1) if w not in weeks)


def check_fits(path: Path) -> tuple[bool, str]:
    """Try to open FITS file and check it's valid."""
    if not HAVE_ASTROPY:
        return True, "(astropy not available, header check skipped)"
    try:
        with fits.open(path, memmap=True) as hdul:
            n_hdu = len(hdul)
            if n_hdu < 2:
                return False, f"only {n_hdu} HDU"
            # Try to read EVENTS or SC_DATA HDU
            try:
                events_hdu = hdul[1]
                _ = events_hdu.header
                nrows = events_hdu.header.get("NAXIS2", 0)
                return True, f"{n_hdu} HDUs, {nrows} rows"
            except Exception as e:
                return False, f"HDU[1] read fail: {e}"
    except Exception as e:
        return False, f"open fail: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photon-dir", type=Path, default=Path("./photon_files"))
    ap.add_argument("--sc-dir",     type=Path, default=Path("./sc_files"))
    ap.add_argument("--start", type=int, default=9)
    ap.add_argument("--end",   type=int, default=950)
    ap.add_argument("--check-fits", action="store_true",
                    help="Validate FITS header for each file (slow)")
    ap.add_argument("--sample-fits", type=int, default=5,
                    help="Number of randomly sampled files to FITS-check (default 5)")
    args = ap.parse_args()

    print("=" * 70)
    print("Photon weekly inventory")
    print("=" * 70)
    photon = list_weeks(args.photon_dir, PHOTON_RE)
    p_weeks = set(photon.keys())
    print(f"  dir       : {args.photon_dir}")
    print(f"  count     : {len(p_weeks)}")
    if p_weeks:
        print(f"  range     : w{min(p_weeks):03d} -- w{max(p_weeks):03d}")
    p_gaps = find_gaps(p_weeks, args.start, args.end)
    if p_gaps:
        print(f"  ⚠ GAPS    : {len(p_gaps)} missing week(s)")
        # group consecutive gaps for readability
        groups, cur = [], []
        for w in p_gaps:
            if not cur or w == cur[-1] + 1:
                cur.append(w)
            else:
                groups.append(cur); cur = [w]
        if cur: groups.append(cur)
        for g in groups[:20]:
            if len(g) == 1:
                print(f"      w{g[0]:03d}")
            else:
                print(f"      w{g[0]:03d} -- w{g[-1]:03d}  ({len(g)} weeks)")
        if len(groups) > 20:
            print(f"      ... and {len(groups)-20} more groups")
    else:
        print(f"  ✓ no gaps in w{args.start:03d}--w{max(p_weeks):03d}")

    print()
    print("=" * 70)
    print("Spacecraft weekly inventory")
    print("=" * 70)
    sc = list_weeks(args.sc_dir, SC_RE)
    s_weeks = set(sc.keys())
    print(f"  dir       : {args.sc_dir}")
    print(f"  count     : {len(s_weeks)}")
    if s_weeks:
        print(f"  range     : w{min(s_weeks):03d} -- w{max(s_weeks):03d}")
    s_gaps = find_gaps(s_weeks, args.start, args.end)
    if s_gaps:
        print(f"  ⚠ GAPS    : {len(s_gaps)} missing week(s)")
        for w in s_gaps[:20]:
            print(f"      w{w:03d}")
        if len(s_gaps) > 20:
            print(f"      ... and {len(s_gaps)-20} more")
    else:
        print(f"  ✓ no gaps in w{args.start:03d}--w{max(s_weeks):03d}")

    print()
    print("=" * 70)
    print("Pairing check (photon ↔ spacecraft)")
    print("=" * 70)
    only_photon = sorted(p_weeks - s_weeks)
    only_sc     = sorted(s_weeks - p_weeks)
    paired      = p_weeks & s_weeks
    print(f"  paired    : {len(paired)}")
    print(f"  only photon: {only_photon[:20]}{' ...' if len(only_photon) > 20 else ''}")
    print(f"  only SC    : {only_sc[:20]}{' ...' if len(only_sc) > 20 else ''}")

    if args.check_fits and HAVE_ASTROPY:
        import random
        print()
        print("=" * 70)
        print(f"FITS header sanity check (random sample of {args.sample_fits})")
        print("=" * 70)
        sample = random.sample(sorted(p_weeks), min(args.sample_fits, len(p_weeks)))
        for w in sample:
            ok, info = check_fits(photon[w])
            mark = "✓" if ok else "✗"
            print(f"  {mark} photon w{w:03d}: {info}")
        if s_weeks:
            sample = random.sample(sorted(s_weeks), min(args.sample_fits, len(s_weeks)))
            for w in sample:
                ok, info = check_fits(sc[w])
                mark = "✓" if ok else "✗"
                print(f"  {mark} SC     w{w:03d}: {info}")

    print()
    print("=" * 70)
    print("Coverage summary (assuming Aug 2008 = w9)")
    print("=" * 70)
    if p_weeks:
        n_w = max(p_weeks) - 9 + 1
        n_yr = n_w / 52.18
        print(f"  Approx. coverage: ~{n_yr:.2f} years (w009 -- w{max(p_weeks):03d})")
        print(f"  17.5 yr target  : ~ w922 (target ≈ {round(9 + 17.5*52.18)})")

    rc = 0
    if p_gaps or s_gaps or only_photon or only_sc:
        rc = 1
    print()
    print(f"Exit code: {rc}  ({'OK' if rc == 0 else 'INCOMPLETE — re-run download script'})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
