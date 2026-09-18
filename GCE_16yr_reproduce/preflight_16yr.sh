#!/usr/bin/env bash
# preflight_16yr.sh — verify GCE_16yr_reproduce/ is ready BEFORE prepare_common.py.
# Read-only; reports MISSING items + the exact fix command. No changes made.
# Run from inside ~/GCE-Chi-square-fitting/GCE_16yr_reproduce/.

set -uo pipefail
SW=/home/sanghwan/FermiLAT/Sanghwan
ok=1
note() { printf '  %-5s %s\n' "$1" "$2"; }
chk()  { if [ -e "$2" ] && { [ ! -f "$2" ] || [ -r "$2" ]; }; then note OK "$3: $2";
         else note MISS "$3: $2"; ok=0; fi; }

echo "=== cwd ==="
case "$PWD" in
  */GCE_16yr_reproduce) note OK "cwd = $PWD";;
  *) note MISS "run from GCE_16yr_reproduce/ (now: $PWD)"; ok=0;;
esac

echo "=== cwd inputs ==="
chk f "photon_data_17yr.txt" "gtselect photon listfile"
if [ -e MapCubes ]; then
  s=$(ls MapCubes/pion_mapcube_model*.fits 2>/dev/null | wc -l)
  note OK "MapCubes present (pion mapcubes: $s)"
  [ "$s" -ge 1 ] || { note MISS "MapCubes has no pion_mapcube_model*.fits"; ok=0; }
else
  note MISS "MapCubes/ (GDE model templates) — needed by run_one_model.py"; ok=0
fi

echo "=== ../ shared inputs ==="
chk f "../gll_iem_v07.fits"                              "galactic diffuse"
chk f "../iso_P8R3_SOURCE_V3_v1.txt"                     "isotropic (SOURCE) — NEW dep"
chk f "../GCE_allsky_data/lat_spacecraft_merged_17yr.fits" "merged SC FT2 (raw, reused)"
pw=$(ls ../GCE_allsky_data/photon_files/lat_photon_weekly_w*.fits 2>/dev/null | wc -l)
sw=$(ls ../GCE_allsky_data/sc_files/lat_spacecraft_weekly_w*.fits 2>/dev/null | wc -l)
note "$([ "$pw" -gt 0 ] && echo OK || echo MISS)" "photon weeklies: $pw"
note "$([ "$sw" -gt 0 ] && echo OK || echo MISS)" "sc weeklies: $sw"
[ "$pw" -gt 0 ] && [ "$sw" -gt 0 ] || ok=0

echo "=== Sanghwan ground-truth inputs (read-only) ==="
chk f "$SW/GC_analysis/gll_psc_v32.xml"                       "DR4 SourceList catalog"
chk f "$SW/GC_analysis/gll_psc_v35.fit"                       "DR4 Signif catalog"
chk d "$SW/Extended_14years/Templates"                        "extended templates dir"
chk f "$SW/FermiLAT/wimp_map_CAR_wide_test.fits"              "GCE spatial map (_test)"
chk f "$SW/FermiLAT/Fermi_bubble_template.fits"               "bubble spatial map"
chk f "$SW/GC_14yr/data/Model/isotropic_spectrum.txt"        "iso FileFunction spectrum"
chk f "$SW/GC_14yr/data/Model/fermi_bubble_spectrum.txt"     "bubble FileFunction spectrum"
chk f "GC_analysis_DR4/Model/bubble_constraints.txt"         "bubble chi2 constraint (copied)"
chk f "GC_analysis_DR4/Model/iso_constraints_full_err.txt"   "iso chi2 constraint (copied)"

echo "=== clean workdir (single-pass; no stale intermediates) ==="
stale=$(find GC_analysis_DR4 -maxdepth 2 -name '*.fits' 2>/dev/null | head)
if [ -z "$stale" ]; then note OK "GC_analysis_DR4/ has no stale *.fits"
else note MISS "stale FITS present (delete before prepare):"; echo "$stale"; ok=0; fi

echo
echo "=== suggested fixes for any MISS above ==="
cat <<EOF
  photon listfile : cp -p ../GCE_17yr_reproduce/photon_data_17yr.txt .
                    # then check entries resolve from this cwd:
                    head -2 photon_data_17yr.txt
  MapCubes        : ln -s ../GCE_17yr_reproduce/MapCubes MapCubes
  iso SOURCE      : # if ../iso_P8R3_SOURCE_V3_v1.txt missing, get Sanghwan's:
                    cp -p $SW/iso_P8R3_SOURCE_V3_v1.txt ../  2>/dev/null || \\
                    ls -l $SW/../iso_P8R3_SOURCE_V3_v1.txt $SW/iso_P8R3_SOURCE_V3_v1.txt 2>/dev/null
  stale FITS      : rm GC_analysis_DR4/**/*.fits   (keep Model/*.txt)
EOF

echo
if [ "$ok" -eq 1 ]; then echo "RESULT: READY — prepare_common.py can run"; exit 0
else echo "RESULT: NOT READY — resolve MISS items above first"; exit 1; fi
