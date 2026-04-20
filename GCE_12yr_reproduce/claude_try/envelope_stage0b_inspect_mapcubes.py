"""
Envelope Stage 0b - Inspect MapCubes directory to understand the format.

We need to know before constructing XML + gtsrcmaps:
  1. MapCube energy axis (38-bin native? or 14-bin?)
  2. Pixel grid (600x600?)
  3. Units (GeV cm-2 s-1 sr-1  vs  flux form)
  4. Which files exist for each model (pi0 / bremss / ics / pion)

NOTE: 'pion' vs 'pi0' duplicate files - pion_mapcube_modelI.fits AND
pi0_mapcube_modelI.fits both exist. Need to determine which one gtsrcmaps
should consume.
"""
import os, sys
import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK_DIR, OUT_DIR

MAPCUBES_DIR = f'{WORK_DIR}/MapCubes'


def inspect_file(path, label):
    if not os.path.exists(path):
        print(f"  [{label}] MISSING")
        return
    with fits.open(path) as h:
        print(f"\n  === {label} ===  ({os.path.getsize(path)/1024/1024:.1f} MB)")
        print(f"    {os.path.basename(path)}")
        for i, hdu in enumerate(h):
            name = hdu.name or f'ext{i}'
            shape = hdu.data.shape if hdu.data is not None else None
            dtype = hdu.data.dtype if hdu.data is not None else None
            print(f"    HDU[{i}] name={name}  shape={shape}  dtype={dtype}")

        hdr = h[0].header
        for k in ['NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
                   'CTYPE1', 'CTYPE2', 'CTYPE3',
                   'CDELT1', 'CDELT2', 'CDELT3',
                   'CRVAL1', 'CRVAL2', 'CRVAL3',
                   'CRPIX1', 'CRPIX2', 'CRPIX3',
                   'BUNIT']:
            v = hdr.get(k, '(not set)')
            print(f"      {k:<10} = {v}")

        if 'ENERGIES' in [ext.name for ext in h]:
            en = h['ENERGIES'].data
            ens = en['Energy']
            print(f"      ENERGIES ext: {len(ens)} nodes, "
                  f"{ens[0]:.2f} - {ens[-1]:.2f} MeV")
            if len(ens) <= 20:
                print(f"         {ens}")

        data = h[0].data
        if data is not None:
            print(f"      data stats: min={data.min():.3e} "
                  f"max={data.max():.3e} mean={data.mean():.3e} "
                  f"sum={data.sum():.3e}")


def main():
    os.chdir(WORK_DIR)
    print("=" * 90)
    print("Envelope Stage 0b : MapCubes directory inspection")
    print("=" * 90)

    if not os.path.isdir(MAPCUBES_DIR):
        print(f"ERROR: {MAPCUBES_DIR} not found")
        sys.exit(1)

    all_files = sorted(os.listdir(MAPCUBES_DIR))
    print(f"\nTotal files in MapCubes: {len(all_files)}")
    prefixes = {}
    for f in all_files:
        prefix = f.split('_')[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    print(f"File prefix counts:")
    for p, n in sorted(prefixes.items(), key=lambda x: -x[1]):
        print(f"  {p}: {n}")

    print("\n--- Sample files for Model X ---")
    for fn in [
        'pi0_mapcube_modelX.fits',
        'pion_mapcube_modelX.fits',
        'bremss_mapcube_modelX.fits',
        'ics_mapcube_modelX.fits',
        'isotropic_cube.fits',
    ]:
        inspect_file(f'{MAPCUBES_DIR}/{fn}', fn)

    print("\n--- Check pi0 vs pion - are they duplicates? ---")
    pi0_path = f'{MAPCUBES_DIR}/pi0_mapcube_modelX.fits'
    pion_path = f'{MAPCUBES_DIR}/pion_mapcube_modelX.fits'
    if os.path.exists(pi0_path) and os.path.exists(pion_path):
        with fits.open(pi0_path) as h1, fits.open(pion_path) as h2:
            d1 = h1[0].data
            d2 = h2[0].data
            if d1.shape == d2.shape:
                diff = np.max(np.abs(d1 - d2))
                rel = diff / max(np.max(np.abs(d1)), 1e-30)
                print(f"  shape match: {d1.shape}")
                print(f"  max abs diff: {diff:.3e}  (rel {rel:.3e})")
                if rel < 1e-6:
                    print("  -> files are IDENTICAL (safe to use either)")
                else:
                    print("  -> files DIFFER")
            else:
                print(f"  shape differs: {d1.shape} vs {d2.shape}")


if __name__ == '__main__':
    main()
