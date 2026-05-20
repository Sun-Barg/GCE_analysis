#!/bin/bash
# mirrorfix_step1_clean_rebuild.sh
# 12yr mirror-flip stale build 수정 — HANDOFF_12yr_mirrorflip_2026-07-03.md PART 3 Step 1
#
# 동작:
#   Phase A1: bad 38 모델의 fit 산출물(3 config × {.dat,_fit.npz,_likelihood_value})을
#             results_12yr/_pre_mirrorfix_bad38/ 로 아카이브 (삭제 아님 — 포렌식 보존)
#   Phase A2: bad 38 모델의 intermediate 8개 삭제
#             (srcmap yes/no_convol ×2 + pion/bremss/ics × yes/no_convol ×6)
#             — 정확한 파일명만 지정, 글롭 없음 → 38bin/grp14/pregroup14 변형 비접촉
#   Phase B : RUN_PHASE=prepare 병렬 재빌드 (현재=올바른 방향의 공유 MapCube에서
#             gtsrcmaps ×2 → gtmodel ×6; XML은 기존 것 skip)
#
# 금지사항(핸드오프 PART 4): MapCubes/ 비접촉, 정상 42개 비접촉, XML 비접촉.
#
# 사용 (conda activate fermi 후):
#   DRY_RUN=1 bash mirrorfix_step1_clean_rebuild.sh     # 계획만 출력 (기본)
#   DRY_RUN=0 setsid nohup bash mirrorfix_step1_clean_rebuild.sh \
#       > logs/step1_mirrorfix_$(date +%m%d_%H%M).log 2>&1 &
set -u
cd "$HOME/GCE-Chi-square-fitting/GCE_12yr_reproduce" || exit 2

# 핸드오프 PART 2 확정 38개 (abf 24 + 859 일부 14)
BAD38="II III IV V VI VII VIII IX L LI LII LIV LV LVI LVII LVIII LIX LX LXI LXII LXIII LXIV LXV LXVI LXVII LXVIII LXIX LXX LXXI LXXII LXXIII LXXIV LXXV LXXVI LXXVII LXXVIII LXXIX LXXX"

DRY_RUN=${DRY_RUN:-1}
PREP_WORKERS=${PREP_WORKERS:-6}
D=GC_analysis_DR2
ARCH=results_12yr/_pre_mirrorfix_bad38
PLAN=mirrorfix_step1_plan.txt
SUFFIXES=("" "_hisbub_noConstr" "_hisbub_constr")
EXTS=(".dat" "_fit.npz" "_likelihood_value")

n38=$(echo $BAD38 | wc -w)
[ "$n38" -eq 38 ] || { echo "[FATAL] BAD38 count=$n38 != 38"; exit 2; }

# fermi env 확인 (Phase B에 GtApp 필요)
python -c "from GtApp import GtApp" 2>/dev/null \
  || { echo "[FATAL] fermi env 아님 — conda activate fermi 후 실행"; exit 2; }

echo "== TPL_SUFFIX 정의 (production 접미사가 '' 기본인지 눈으로 확인) =="
grep -n "TPL_SUFFIX" run_one_model.py | head -5
echo

mkdir -p "$ARCH" logs
: > "$PLAN"
n_arch=0; n_del=0; n_absent=0

for M in $BAD38; do
  # --- Phase A1: fit 산출물 아카이브 (results_12yr/ + cwd 잔존분) ---
  for S in "${SUFFIXES[@]}"; do
    for E in "${EXTS[@]}"; do
      for f in "results_12yr/GCE_model_${M}_front_12yr_cholis${S}${E}" \
               "./GCE_model_${M}_front_12yr_cholis${S}${E}"; do
        if [ -e "$f" ]; then
          n_arch=$((n_arch+1)); echo "[arch] $f" >> "$PLAN"
          [ "$DRY_RUN" = "0" ] && mv "$f" "$ARCH/"
        fi
      done
    done
  done
  # --- Phase A2: intermediate 8개 삭제 (정확한 파일명) ---
  for f in \
    "$D/GC_Extended_srcmap_12yr_front_clean_model_${M}.fits" \
    "$D/GC_Extended_srcmap_12yr_front_clean_model_${M}_no_convol.fits" \
    "$D/GC_pion_model${M}_12yr_front_clean.fits" \
    "$D/GC_pion_model${M}_12yr_front_clean_no_convol.fits" \
    "$D/GC_bremss_model${M}_12yr_front_clean.fits" \
    "$D/GC_bremss_model${M}_12yr_front_clean_no_convol.fits" \
    "$D/GC_ics_model${M}_12yr_front_clean.fits" \
    "$D/GC_ics_model${M}_12yr_front_clean_no_convol.fits" ; do
    if [ -e "$f" ]; then
      n_del=$((n_del+1)); echo "[del ] $f" >> "$PLAN"
      [ "$DRY_RUN" = "0" ] && rm "$f"
    else
      n_absent=$((n_absent+1)); echo "[absent] $f" >> "$PLAN"
    fi
  done
done

echo "== 계획 요약: archive $n_arch (기대 ≤342) / delete $n_del + absent $n_absent (합 304 기대) =="
echo "   상세 목록: $PLAN"
grep "\[absent\]" "$PLAN" || echo "   (absent 없음 — 8×38 전부 존재)"

if [ "$DRY_RUN" != "0" ]; then
  echo
  echo "DRY_RUN=1 — 위 계획 확인 후:"
  echo "  DRY_RUN=0 setsid nohup bash $0 > logs/step1_mirrorfix_\$(date +%m%d_%H%M).log 2>&1 &"
  exit 0
fi

echo
echo "== Phase B: prepare-only 재빌드 시작 (${PREP_WORKERS} workers, $(date)) =="
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset BUBBLE_OVERRIDE_DIR RANK_SUFFIX USE_CONSTRAINT || true

printf '%s\n' $BAD38 | xargs -P "$PREP_WORKERS" -n1 -I{} bash -c '
  M="$1"
  echo "[prep start] $M  $(date +%H:%M:%S)"
  RUN_PHASE=prepare python -u run_one_model.py "$M" > "logs/prep_mirrorfix_${M}.log" 2>&1
  rc=$?
  echo "[prep done ] $M  rc=$rc  $(date +%H:%M:%S)"
' _ {}

echo
echo "== Phase B 종료 $(date). 이상 검사 =="
grep -l "FATAL" logs/prep_mirrorfix_*.log 2>/dev/null \
  && echo "^^^ FATAL 로그 존재 — 해당 로그 확인 필요" \
  || echo "(FATAL 없음)"

n_new=0
for M in $BAD38; do
  ok=1
  for f in \
    "$D/GC_Extended_srcmap_12yr_front_clean_model_${M}.fits" \
    "$D/GC_Extended_srcmap_12yr_front_clean_model_${M}_no_convol.fits" \
    "$D/GC_pion_model${M}_12yr_front_clean.fits" \
    "$D/GC_pion_model${M}_12yr_front_clean_no_convol.fits" \
    "$D/GC_bremss_model${M}_12yr_front_clean.fits" \
    "$D/GC_bremss_model${M}_12yr_front_clean_no_convol.fits" \
    "$D/GC_ics_model${M}_12yr_front_clean.fits" \
    "$D/GC_ics_model${M}_12yr_front_clean_no_convol.fits" ; do
    [ -e "$f" ] || { ok=0; echo "[missing] $f"; }
  done
  n_new=$((n_new+ok))
done
echo "== 재빌드 완비 모델: $n_new/38 =="
echo "다음: python verify_mirrorfix_scope.py  (orientation 재검증 — FLIPPED 0건 확인 후 step2)"
