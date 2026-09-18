#!/usr/bin/env python3
"""
verify_bubble_template.py — is the 17yr bubble spatial map the same template
Sanghwan used for the 16yr fit?

Compares two SpatialMap FITS:
  - Sanghwan : /home/sanghwan/FermiLAT/Sanghwan/FermiLAT/Fermi_bubble_template.fits
  - 17yr     : <GCE_17yr_reproduce>/Fermi_Bubbles_template.fits   (the legacy one)

Reports: md5, shape, dtype, WCS keys, exact data equality, and a
SCALE-INVARIANT shape comparison (each map normalized to unit sum) so a
pure amplitude/normalization difference is distinguished from a genuine
different spatial template.

Usage:
  python3 verify_bubble_template.py \
      [--sanghwan /home/sanghwan/FermiLAT/Sanghwan/FermiLAT/Fermi_bubble_template.fits] \
      [--legacy  ../GCE_17yr_reproduce/Fermi_Bubbles_template.fits]
"""
import argparse, hashlib, os, sys
import numpy as np
from astropy.io import fits

DEF_SW  = '/home/sanghwan/FermiLAT/Sanghwan/FermiLAT/Fermi_bubble_template.fits'
DEF_LEG = '../GCE_17yr_reproduce/Fermi_Bubbles_template.fits'

WCS_KEYS = ['NAXIS', 'NAXIS1', 'NAXIS2', 'CTYPE1', 'CTYPE2',
            'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
            'CDELT1', 'CDELT2', 'CD1_1', 'CD2_2']


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def first_image(path):
    """Return (data, header) of the first HDU that has 2D+ image data."""
    with fits.open(path) as hdul:
        for h in hdul:
            if getattr(h, 'data', None) is not None and np.ndim(h.data) >= 2:
                return np.asarray(h.data, dtype=float), h.header
    raise ValueError(f'no image HDU with data in {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sanghwan', default=DEF_SW)
    ap.add_argument('--legacy',   default=DEF_LEG)
    a = ap.parse_args()

    for label, p in (('Sanghwan', a.sanghwan), ('17yr-legacy', a.legacy)):
        if not os.path.isfile(p):
            print(f'[MISSING] {label}: {p}')
            if label == '17yr-legacy':
                print('  -> pass --legacy with the real path (look in '
                      'GCE_17yr_reproduce/ for Fermi_Bubbles_template.fits)')
            sys.exit(2)

    print('=== files ===')
    m1, m2 = md5(a.sanghwan), md5(a.legacy)
    print(f'Sanghwan    {a.sanghwan}\n            md5={m1}')
    print(f'17yr-legacy {a.legacy}\n            md5={m2}')
    print(f'byte-identical: {m1 == m2}')

    d1, h1 = first_image(a.sanghwan)
    d2, h2 = first_image(a.legacy)

    print('\n=== geometry / WCS ===')
    print(f'{"key":<8}{"Sanghwan":>22}{"17yr-legacy":>22}')
    for k in WCS_KEYS:
        v1, v2 = h1.get(k, '—'), h2.get(k, '—')
        flag = '' if str(v1) == str(v2) else '   <-- DIFF'
        print(f'{k:<8}{str(v1):>22}{str(v2):>22}{flag}')
    print(f'data shape  Sanghwan={d1.shape}  17yr={d2.shape}')

    same_shape = (d1.shape == d2.shape)
    print('\n=== data comparison ===')
    if not same_shape:
        print('shapes DIFFER -> genuinely different templates (or different '
              'grid). Repoint BUBBLE_TEMPLATE to the Sanghwan original.')
        verdict = 'DIFFERENT (shape)'
    else:
        f1 = np.nan_to_num(d1, nan=0.0)
        f2 = np.nan_to_num(d2, nan=0.0)
        exact = np.allclose(f1, f2, rtol=1e-6, atol=0.0, equal_nan=True)
        # scale-invariant: normalize each to unit sum, compare shape only
        s1, s2 = f1.sum(), f2.sum()
        if s1 != 0 and s2 != 0:
            n1, n2 = f1 / s1, f2 / s2
            shape_only = np.allclose(n1, n2, rtol=1e-4, atol=1e-12)
            ratio = (s2 / s1)
        else:
            shape_only, ratio = False, float('nan')
        max_abs = float(np.max(np.abs(f1 - f2)))
        print(f'exact equal (rtol1e-6)         : {exact}')
        print(f'same SHAPE up to a scale factor: {shape_only}'
              f'   (sum ratio 17yr/Sanghwan = {ratio:.6g})')
        print(f'max |Sanghwan - 17yr|          : {max_abs:.6g}')
        if exact:
            verdict = 'IDENTICAL (no confound; repoint optional, for traceability)'
        elif shape_only:
            verdict = ('SAME SHAPE, DIFFERENT SCALE -> not a spatial confound, '
                       'but amplitude differs; repoint to Sanghwan original to '
                       'be exact and watch the bubble coefficient prior bound')
        else:
            verdict = ('DIFFERENT TEMPLATE -> a real hidden confound; MUST '
                       'repoint BUBBLE_TEMPLATE to the Sanghwan original')

    print('\n=== VERDICT ===')
    print(verdict)
    print('\nRepoint (if needed), both files, idempotent:')
    print("  sed -i \"s#BUBBLE_TEMPLATE *= *'\\./Fermi_Bubbles_template.fits'#"
          "BUBBLE_TEMPLATE       = '%s'#\" prepare_common.py" % a.sanghwan)
    print("  sed -i \"s#BUBBLE_TEMPLATE *= *'\\./Fermi_Bubbles_template.fits'#"
          "BUBBLE_TEMPLATE      = '%s'#\" run_one_model.py" % a.sanghwan)
    print('  grep -n BUBBLE_TEMPLATE prepare_common.py run_one_model.py')


if __name__ == '__main__':
    main()
