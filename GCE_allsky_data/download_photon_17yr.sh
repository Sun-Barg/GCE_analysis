#!/bin/bash
###############################################################################
# download_photon_17yr.sh
#
# Fermi-LAT weekly photon (P305_v001) 파일을 17.5년치까지 확장 다운로드.
# - 이미 받은 파일은 skip
# - 중간에 누락된 주 (예: w512) 자동 재시도
# - 서버에 아직 없는 주 (404)는 future로 분류
# - 다운로드 후 최소 크기 검증 (corrupt 방지)
#
# Usage:
#   ./download_photon_17yr.sh                  # 기본: ./photon_files 사용
#   ./download_photon_17yr.sh /path/to/dir     # 디렉토리 지정
#   ./download_photon_17yr.sh ./photon_files 920  # END_WEEK 명시
#
# 참고:
#   - 12.5yr ≈ w670, 17.5yr ≈ w920–w925 부근 (Aug 2008 시작 기준)
#   - 현재 시점(Apr 2026) 서버에 release된 최신 주는 대략 w920~w940
#   - 기본 END_WEEK=950으로 잡고 future는 자동으로 skip되게 둠
###############################################################################

set -u  # undefined var error
# NOTE: do NOT set -e — 404 (future weeks) should not abort the loop

# ---- Configuration ---------------------------------------------------------
TARGET_DIR="${1:-./photon_files}"
START_WEEK="${START_WEEK:-9}"
END_WEEK="${2:-${END_WEEK:-950}}"
BASE_URL="https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/weekly/photon"
PREFIX="lat_photon_weekly"
SUFFIX="p305_v001.fits"
LOG_FILE="${LOG_FILE:-photon_download_17yr.log}"
MAX_RETRIES=3
MIN_FILE_SIZE=10000   # bytes; FITS header alone is several KB

# ---- Setup -----------------------------------------------------------------
mkdir -p "$TARGET_DIR"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

log "============================================================"
log "Photon download started"
log "Target dir : $TARGET_DIR"
log "Week range : w$(printf '%03d' $START_WEEK) -- w$(printf '%03d' $END_WEEK)"
log "Base URL   : $BASE_URL"
log "============================================================"

# ---- Counters --------------------------------------------------------------
declare -i n_downloaded=0
declare -i n_skipped=0
declare -i n_failed=0
declare -i n_future=0
failed_weeks=()
future_weeks=()

# ---- Main loop -------------------------------------------------------------
for w in $(seq $START_WEEK $END_WEEK); do
    week=$(printf "%03d" "$w")
    fname="${PREFIX}_w${week}_${SUFFIX}"
    url="${BASE_URL}/${fname}"
    target="${TARGET_DIR}/${fname}"

    # Skip if already downloaded and large enough
    if [[ -s "$target" ]]; then
        fsize=$(stat -c%s "$target" 2>/dev/null || echo 0)
        if (( fsize > MIN_FILE_SIZE )); then
            n_skipped+=1
            continue
        else
            log "  ⚠ w${week} exists but suspiciously small (${fsize}B) — re-downloading"
            rm -f "$target"
        fi
    fi

    # Try to download with retries
    success=0
    last_http=""
    for attempt in $(seq 1 $MAX_RETRIES); do
        # curl with HTTP code capture
        http_code=$(curl -fsS \
                        --connect-timeout 30 \
                        --max-time 600 \
                        -o "$target" \
                        -w "%{http_code}" \
                        "$url" 2>/dev/null) || http_code="000"
        last_http="$http_code"

        if [[ "$http_code" == "200" && -s "$target" ]]; then
            fsize=$(stat -c%s "$target")
            if (( fsize > MIN_FILE_SIZE )); then
                log "  ✓ w${week} downloaded (${fsize} B, attempt ${attempt})"
                n_downloaded+=1
                success=1
                break
            else
                log "  ⚠ w${week} downloaded but too small (${fsize} B) — retry"
                rm -f "$target"
            fi
        elif [[ "$http_code" == "404" ]]; then
            # Not yet released — don't retry
            rm -f "$target"
            break
        else
            rm -f "$target"
            sleep 3
        fi
    done

    if (( success == 0 )); then
        if [[ "$last_http" == "404" ]]; then
            n_future+=1
            future_weeks+=("w${week}")
            # 연속 404가 5번 이상이면 종료 (의미 있는 future detection)
            if (( n_future > 0 && n_future % 1 == 0 )); then
                # 마지막 5개가 모두 미래면 종료
                if (( ${#future_weeks[@]} >= 5 )); then
                    last5="${future_weeks[@]: -5}"
                    last5_arr=($last5)
                    consecutive=1
                    prev_w=$(echo "${last5_arr[0]}" | tr -d 'w' | sed 's/^0*//')
                    for fw in "${last5_arr[@]:1}"; do
                        cur_w=$(echo "$fw" | tr -d 'w' | sed 's/^0*//')
                        if (( cur_w == prev_w + 1 )); then
                            consecutive=$((consecutive + 1))
                        else
                            consecutive=1
                        fi
                        prev_w=$cur_w
                    done
                    if (( consecutive >= 5 )); then
                        log "  ↻ ${consecutive}주 연속 404 감지 — 서버 최신 주 도달, 루프 종료"
                        break
                    fi
                fi
            fi
        else
            n_failed+=1
            failed_weeks+=("w${week}(HTTP:$last_http)")
            log "  ✗ w${week} FAILED after ${MAX_RETRIES} attempts (last HTTP: $last_http)"
        fi
    fi
done

# ---- Summary ---------------------------------------------------------------
log "============================================================"
log "Photon download finished"
log "  downloaded     : $n_downloaded"
log "  skipped (have) : $n_skipped"
log "  failed         : $n_failed"
log "  future (404)   : $n_future"
log "============================================================"

if (( n_failed > 0 )); then
    log "Failed weeks: ${failed_weeks[*]}"
    log "↻ 다시 실행하면 자동으로 retry 합니다."
fi

if (( n_future > 0 )); then
    log "Future weeks (서버에 아직 release 안됨): ${future_weeks[*]:0:10}$([ ${#future_weeks[@]} -gt 10 ] && echo " ...")"
fi

# Final inventory
total_files=$(ls "$TARGET_DIR"/${PREFIX}_w*_${SUFFIX} 2>/dev/null | wc -l)
log "Total photon files in $TARGET_DIR : $total_files"

exit $(( n_failed > 0 ? 1 : 0 ))
