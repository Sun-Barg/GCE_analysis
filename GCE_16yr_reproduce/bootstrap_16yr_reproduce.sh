#!/usr/bin/env bash
# bootstrap_16yr_reproduce.sh
# Set up GCE_16yr_reproduce/ for porting-diagnostic overlay against Sanghwan 16yr.
# Non-destructive: reads /home/sanghwan only; writes only under DST.
# Idempotent: re-running refreshes the manifest, never deletes analysis outputs.
#
# What it does NOT do: it does NOT patch any CONFIG block. It DUMPS the real
# CONFIG-relevant lines of each copied script so the exact patch can be written
# against the actual source (not a guess).

set -uo pipefail

SRC="${HOME}/GCE-Chi-square-fitting/GCE_17yr_reproduce"
DST="${HOME}/GCE-Chi-square-fitting/GCE_16yr_reproduce"
MANIFEST="${DST}/RESOLVED_MANIFEST.txt"

say()  { printf '%s\n' "$*"; }
hr()   { printf -- '------------------------------------------------------------\n'; }
md5()  { [ -f "$1" ] && md5sum "$1" | awk '{print $1}' || echo "MISSING"; }

say "=== bootstrap_16yr_reproduce ==="
say "SRC = $SRC"
say "DST = $DST"
hr

# ---- 0. sanity: SRC must exist ----
if [ ! -d "$SRC" ]; then
  say "FATAL: 17yr workdir not found: $SRC"
  exit 1
fi

# ---- 1. create dirs ----
mkdir -p "$DST" "$DST/GC_analysis_DR4/Model" "$DST/results_16yr" "$DST/logs"
: > "$MANIFEST"
{
  echo "# RESOLVED_MANIFEST  (generated $(date -Is))"
  echo "# SRC=$SRC"
  echo "# DST=$DST"
  echo "# Locked config: DR4 gll_psc_v32, evtype=1, 14-bin Cholis-exact,"
  echo "#   mask_scale=1.0, tmin=239557417 (2008-08-04Z), tmax=755538221 (2024-12-10Z)"
  echo
} >> "$MANIFEST"

# ---- 2. copy main-fit pipeline files (overlay scope; cov files excluded) ----
say "[2] copying main-fit pipeline files"
declare -A COPY=(
  ["cholis_masking.py"]="cholis_masking.py"
  ["prepare_common.py"]="prepare_common.py"
  ["run_one_model.py"]="run_one_model.py"
  ["launch_all_models.py"]="launch_all_models.py"
  ["launcher_watchdog.sh"]="launcher_watchdog.sh"
  ["GCE_17yr_visualization.ipynb"]="GCE_16yr_overlay.ipynb"
)
echo "## copied files (src -> dst : md5)" >> "$MANIFEST"
for s in "${!COPY[@]}"; do
  d="${COPY[$s]}"
  if [ -f "$SRC/$s" ]; then
    cp -p "$SRC/$s" "$DST/$d"
    say "  OK   $s  ->  $d"
    echo "$s -> $d : $(md5 "$DST/$d")" >> "$MANIFEST"
  else
    say "  WARN missing in SRC: $s"
    echo "$s -> $d : SRC_MISSING" >> "$MANIFEST"
  fi
done
echo >> "$MANIFEST"

# ---- 3. resolve Sanghwan ORIGINAL template / catalog / constraint files ----
# Originals first; only if an original is gone do we record a fallback copy to md5-verify.
say "[3] resolving Sanghwan originals (read-only)"
SW="/home/sanghwan/FermiLAT/Sanghwan"
declare -A ORIG=(
  ["wimp_map"]="$SW/FermiLAT/wimp_map_CAR_wide.fits"
  ["iso_spectrum"]="$SW/GC_14yr/data/Model/isotropic_spectrum.txt"
  ["bubble_spectrum"]="$SW/GC_14yr/data/Model/fermi_bubble_spectrum.txt"
  ["extended_dir"]="$SW/Extended_14years/Templates"
)
echo "## Sanghwan originals" >> "$MANIFEST"
for k in "${!ORIG[@]}"; do
  p="${ORIG[$k]}"
  if [ -e "$p" ]; then
    if [ -d "$p" ]; then
      n=$(ls -1 "$p" | wc -l)
      say "  OK   $k : DIR $p ($n entries)"
      echo "$k : $p : DIR n=$n" >> "$MANIFEST"
    else
      say "  OK   $k : $p"
      echo "$k : $p : $(md5 "$p")" >> "$MANIFEST"
    fi
  else
    say "  WARN $k ORIGINAL MISSING: $p   (fallback search below)"
    echo "$k : ORIGINAL_MISSING $p" >> "$MANIFEST"
  fi
done

# constraint files: search SRC Model dir + 17yr data dir; record md5 of whatever exists
say "[3b] constraint files (bubble/iso)"
echo "## constraint files (candidates + md5)" >> "$MANIFEST"
for cf in bubble_constraints.txt iso_constraints_full_err.txt; do
  found=0
  for cand in \
      "$SRC/GC_analysis_FL16Y/Model/$cf" \
      "${HOME}/GCE-Chi-square-fitting/GCE_17yr_data/$cf" \
      "$SW/GC_14yr/data/Model/$cf" ; do
    if [ -f "$cand" ]; then
      say "  OK   $cf : $cand  ($(md5 "$cand"))"
      echo "$cf : $cand : $(md5 "$cand")" >> "$MANIFEST"
      found=1
    fi
  done
  [ "$found" -eq 0 ] && { say "  WARN $cf not found in candidates"; echo "$cf : NOT_FOUND" >> "$MANIFEST"; }
done

# DR4 catalog gll_psc_v32 : search likely data dirs
say "[3c] DR4 catalog gll_psc_v32"
echo "## DR4 catalog candidates" >> "$MANIFEST"
catfound=0
for d in \
    "${HOME}/GCE-Chi-square-fitting/GCE_17yr_data" \
    "${HOME}/GCE-Chi-square-fitting/GCE_16yr_data" \
    "$SRC/GC_analysis_FL16Y" \
    "$SW/GC_analysis" ; do
  for hit in "$d"/gll_psc_v32* ; do
    if [ -f "$hit" ]; then
      say "  OK   $hit  ($(md5 "$hit"))"
      echo "DR4_catalog : $hit : $(md5 "$hit")" >> "$MANIFEST"
      catfound=1
    fi
  done
done
[ "$catfound" -eq 0 ] && { say "  WARN gll_psc_v32* not found — locate DR4 catalog manually"; echo "DR4_catalog : NOT_FOUND" >> "$MANIFEST"; }
echo >> "$MANIFEST"

# ---- 4. 16yr input data status (raw weekly + merged SC are reused as-is) ----
say "[4] 16yr input data status (window enforced via gtselect tmin/tmax, NOT by subsetting weeklies)"
echo "## input data status" >> "$MANIFEST"
for cand in \
    "${HOME}/GCE-Chi-square-fitting/GCE_allsky_data" \
    "$SRC/GCE_allsky_data" ; do
  if [ -d "$cand" ]; then
    wk=$(ls -1 "$cand" 2>/dev/null | grep -ic 'photon' || true)
    say "  weekly dir : $cand  (photon-like files: $wk)"
    echo "weekly_dir : $cand : photon_files=$wk" >> "$MANIFEST"
  fi
done
for sc in "$SRC"/sc_files_17yr.txt "$SRC"/*spacecraft*merged* "$SRC"/photon_data*.txt ; do
  [ -e "$sc" ] && { say "  input listfile/SC present : $sc"; echo "input : $sc" >> "$MANIFEST"; }
done
say "  tmin=239557417  tmax=755538221  (set these in gtselect; keep full weekly listfile)"
echo "window : tmin=239557417 tmax=755538221" >> "$MANIFEST"
echo >> "$MANIFEST"

# ---- 5. DUMP real CONFIG-relevant lines of each copied script (NO patching) ----
say "[5] dumping real CONFIG lines from copied scripts (paste these back for exact patches)"
echo "## CONFIG line dump (verbatim from copied SRC files)" >> "$MANIFEST"
PAT='17yr|FL16Y|gll_psc|DR=|DR_NUMBER|catalog|extended_directory|extended_dir|evtype|EXTEND_ENERGY|bin_def|mask_scale|tmin|tmax|scfile|spacecraft|photon_data|sigma_to_free|free_radius|max_free_radius|extra_radius|norms_free_only|galactic_index_free|variable_free|gll_iem|iso_P8R3|wimp_map|GCE_template|isotropic_spectrum|fermi_bubble_spectrum|results_17yr|GC_analysis_FL16Y|WORKDIR|workdir'
for f in cholis_masking.py prepare_common.py run_one_model.py launch_all_models.py launcher_watchdog.sh; do
  if [ -f "$DST/$f" ]; then
    say "  ---- $f ----"
    {
      echo
      echo "### $f"
      grep -nE "$PAT" "$DST/$f" 2>/dev/null
    } >> "$MANIFEST"
  fi
done

hr
say "Done. Review and share: $MANIFEST"
say "Originals/constraints/catalog resolved + verbatim CONFIG lines are in the manifest."
say "Next: send the manifest's CONFIG dump so exact (non-guessed) patches can be written."
