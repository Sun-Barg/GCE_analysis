from astropy.io import fits
import numpy as np

# Path는 본인 환경에 맞춰 수정
PATHS = {
    "4FGL-DR2": "/home/haebarg/GCE-Chi-square-fitting/GCE_12yr_data/gll_psc_v23.fit",
    "FL16Y":    "/home/haebarg/GCE-Chi-square-fitting/GCE_17yr_data/gll_psc_v40.fit",
}

for name, path in PATHS.items():
    hdul = fits.open(path)

    # Point sources: extension 1 (LAT_Point_Source_Catalog)
    psc = hdul[1].data
    glon = psc["GLON"]   # Galactic longitude in degrees, 0-360
    glat = psc["GLAT"]   # Galactic latitude  in degrees, -90 to +90

    # Convert GLON to (-180, +180] for symmetric box cut
    glon_sym = np.where(glon > 180, glon - 360, glon)

    # Cholis 60x60 ROI:  |GLON| <= 30,  |GLAT| <= 30
    in_roi_60 = (np.abs(glon_sym) <= 30) & (np.abs(glat) <= 30)
    n_psc_60 = in_roi_60.sum()

    # Cholis 40x40 fitting ROI
    in_roi_40 = (np.abs(glon_sym) <= 20) & (np.abs(glat) <= 20)
    n_psc_40 = in_roi_40.sum()

    # Extended sources: extension 2 (ExtendedSources)
    try:
        ext = hdul[2].data
        e_glon = np.where(ext["GLON"] > 180, ext["GLON"] - 360, ext["GLON"])
        e_glat = ext["GLAT"]
        n_ext_60 = ((np.abs(e_glon) <= 30) & (np.abs(e_glat) <= 30)).sum()
        n_ext_40 = ((np.abs(e_glon) <= 20) & (np.abs(e_glat) <= 20)).sum()
    except (IndexError, KeyError):
        n_ext_60 = n_ext_40 = 0

    ts_threshold_satisfied = (psc["Signif_Avg"] >= 7.0)
    n_strong_60 = (in_roi_60 & ts_threshold_satisfied).sum()
    n_strong_40 = (in_roi_40 & ts_threshold_satisfied).sum()
    print(f"  TS≥49 in 60x60: {n_strong_60} PSC")
    print(f"  TS≥49 in 40x40: {n_strong_40} PSC")

    print(f"{name}:")
    print(f"  60x60 ROI:  {n_psc_60} PSC + {n_ext_60} extended = {n_psc_60+n_ext_60}")
    print(f"  40x40 ROI:  {n_psc_40} PSC + {n_ext_40} extended = {n_psc_40+n_ext_40}")
    hdul.close()
