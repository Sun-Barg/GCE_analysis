#!/bin/bash
# mirrorfix_step2_refit.sh — HANDOFF PART 3 Step 2: 38개 모델 × 3 config 재fit (114 fit)
#
# 전제: step1 재빌드 완료 + verify_mirrorfix_scope.py 에서 FLIPPED 0건.
#       (게이트: verify_mirrorfix_scope.csv 에 FLIPPED 행이 있으면 즉시 중단)
#
# 순서: production(무접미사) → 셀a(_hisbub_noConstr) → 셀b(_hisbub_constr)
#       launcher .launcher.pid 단일 인스턴스 락 때문에 순차 실행.
#       각 config: 패치된 launcher_watchdog.sh 를 백그라운드로 띄우고
#       suffix별 .dat 38/38 완료를 10분 주기로 폴링. watchdog 사망 시 최대 3회 재기동.
#       정상 42개는 is_complete 로 skip → 38개만 실행됨 (핸드오프 Step 2).
#
# 사용 (conda activate fermi 후):
#   setsid nohup bash mirrorfix_step2_refit.sh \
#       > logs/refit_mirrorfix_$(date +%m%d_%H%M).log 2>&1 &
#
# worker 수: WD_WORKERS=4 기본 (핸드오프; 38×3/4w ≈ 38h).
#            MG5 트랙 종료 확인 시 WD_WORKERS=6 으로 단축 가능.
set -u
cd "$HOME/GCE-Chi-square-fitting/GCE_12yr_reproduce" || exit 2

BAD38="II III IV V VI VII VIII IX L LI LII LIV LV LVI LVII LVIII LIX LX LXI LXII LXIII LXIV LXV LXVI LXVII LXVIII LXIX LXX LXXI LXXII LXXIII LXXIV LXXV LXXVI LXXVII LXXVIII LXXIX LXXX"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WD_WORKERS=${WD_WORKERS:-4}
export WD_MAX_RUNTIME_HR=${WD_MAX_RUNTIME_HR:-48}
unset BUBBLE_OVERRIDE_DIR RANK_SUFFIX USE_CONSTRAINT || true
mkdir -p logs

# ---- 검증 게이트 ----
[ -f verify_mirrorfix_scope.csv ] \
  || { echo "[FATAL] verify_mirrorfix_scope.csv 없음 — 검증부터 실행"; exit 2; }
grep -q ',FLIPPED,' verify_mirrorfix_scope.csv \
  && { echo "[FATAL] verify CSV에 FLIPPED 잔존 — 재fit 금지"; exit 2; }
grep -q ',MISSING,' verify_mirrorfix_scope.csv \
  && { echo "[FATAL] verify CSV에 MISSING 잔존 — 재빌드 완비 후 재검증 필요"; exit 2; }

echo "== step2 시작 $(date)  WD_WORKERS=$WD_WORKERS =="
free -g | head -2

count_done () {  # $1 = suffix ('' | _hisbub_noConstr | _hisbub_constr)
  local n=0
  for M in $BAD38; do
    if [ -e "results_12yr/GCE_model_${M}_front_12yr_cholis$1.dat" ] \
    || [ -e "./GCE_model_${M}_front_12yr_cholis$1.dat" ]; then
      n=$((n+1))
    fi
  done
  echo $n
}

run_cfg () {  # $1=label  $2=suffix  $3...=config env (KEY=VAL ...)
  local label="$1" sfx="$2"; shift 2
  local restarts=0
  echo
  echo "==== config ${label} 시작 $(date) — 시작 시점 $(count_done "$sfx")/38 ===="
  setsid env "$@" bash launcher_watchdog.sh \
      > "logs/watchdog_${label}_$(date +%m%d_%H%M).log" 2>&1 &
  local wpid=$!
  while :; do
    sleep 600
    local n; n=$(count_done "$sfx")
    echo "[wait] ${label}: ${n}/38  $(date '+%m-%d %H:%M')"
    [ "$n" -eq 38 ] && break
    if ! kill -0 "$wpid" 2>/dev/null; then
      restarts=$((restarts+1))
      if [ "$restarts" -gt 3 ]; then
        echo "[FATAL] ${label}: watchdog 3회 초과 종료, ${n}/38 —"
        echo "        permanent_failed 가능성. logs/log_*.txt 및 watchdog 로그 확인."
        return 2
      fi
      echo "[warn] watchdog(${label}) 종료 감지(${n}/38) — 재기동 ${restarts}/3"
      setsid env "$@" bash launcher_watchdog.sh \
          >> "logs/watchdog_${label}_restart.log" 2>&1 &
      wpid=$!
    fi
  done
  # launcher 완전 종료 대기 (파일 이동/락 해제)
  while pgrep -f "launch_all_models.py" >/dev/null 2>&1; do sleep 60; done
  kill "$wpid" 2>/dev/null || true
  echo "==== config ${label} 완료 $(date) ===="
}

# 핸드오프 PART 5 env 그대로 (production = 토글 미설정)
run_cfg production      ""                 || exit 2
run_cfg hisbub_noConstr "_hisbub_noConstr" \
    BUBBLE_OVERRIDE_DIR=./his_v3_products_modelI USE_CONSTRAINT=0 \
    RANK_SUFFIX=_hisbub_noConstr           || exit 2
run_cfg hisbub_constr   "_hisbub_constr" \
    BUBBLE_OVERRIDE_DIR=./his_v3_products_modelI USE_CONSTRAINT=1 \
    RANK_SUFFIX=_hisbub_constr             || exit 2

echo
echo "== 3 config 전부 완료 $(date) =="
echo "다음(핸드오프 Step 3): GCE_12yr_visualization.ipynb 04d 5-pipe 셀 재실행"
echo "  기대: 12yr_P/12yr_a 양봉 소멸(70|10 형태), rho vs Cholis 회복,"
echo "        L/IV/VI 등 bad cluster 모델 정상 순위 복귀 (Cholis 2112.09706 p16-17 연속 랭킹 기준)"
