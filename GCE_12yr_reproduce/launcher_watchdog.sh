#!/usr/bin/env bash
# launcher_watchdog.sh — Phase B launcher auto-restart + orphan recovery.
#
# Pattern observed 2026-05-15:
#   launch_all_models.py dies silently after 1-6 hr (cause unknown, dmesg
#   clean). Worker wrappers always survive and produce .dat correctly,
#   but .dat is left orphan in workdir and lockfile remains stale —
#   manual recovery is needed each time.
#
# This script automates that:
#   1. Every 60s, check launcher cmdline match.
#   2. If launcher dead AND we are not yet at 80/80:
#        a. mv all workdir orphan .dat (+ companions) into results_12yr/
#        b. rm stale .launcher.pid
#        c. respawn launcher with the canonical command
#   3. Auto-exit on:
#        a. 80/80 .dat in results_12yr/ (success)
#        b. MAX_RESTARTS exceeded (safety brake)
#        c. .watchdog.stop file appears (manual stop)
#
# Lock + manual control:
#   .watchdog.pid    — single-instance lock (this watchdog's own pid)
#   .watchdog.stop   — touch this file to ask watchdog to stop cleanly
#   watchdog.log     — append-only timeline of all watchdog actions
#
# Usage:
#   nohup ./launcher_watchdog.sh > watchdog_console.log 2>&1 &
#   disown
#
#   # stop the watchdog (but leave launcher alone):
#   touch .watchdog.stop
#
#   # kill the watchdog AND any running launcher:
#   kill $(cat .watchdog.pid)
#   kill $(cat .launcher.pid 2>/dev/null) 2>/dev/null
#
# Author: haebarg (2026-05-16)

set -uo pipefail   # not -e (we want to handle most failures inline)

# ============================================================
# CONFIG — edit here if launcher invocation changes
# ============================================================
WORKERS=${WD_WORKERS:-16}
MAX_RETRIES=3
MAX_RUNTIME_HR=${WD_MAX_RUNTIME_HR:-12}
RESULTS_DIR='results_12yr'
LAUNCHER_SCRIPT='launch_all_models.py'
WRAPPER_PATTERN='run_one_model_wrapper'   # for orphan inference / future hooks
LAUNCHER_PATTERN='launch_all_models'      # for liveness probe
LAUNCHER_LOCK='.launcher.pid'
TARGET_COUNT=80
# rank study: 셀 .dat suffix (production은 '' -> glob 동작 불변; launcher RANK_SUFFIX와 동일)
RANK_SUFFIX="${RANK_SUFFIX:-}"

POLL_SEC=60
MAX_RESTARTS=20

WATCHDOG_LOCK='.watchdog.pid'
WATCHDOG_STOP='.watchdog.stop'
WATCHDOG_LOG='watchdog.log'

# ============================================================
# Helpers
# ============================================================

_ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() {
    local msg="[$(_ts)] $*"
    echo "$msg"            # to stdout (nohup file)
    echo "$msg" >> "$WATCHDOG_LOG"
}

count_dat() {
    ls "$RESULTS_DIR"/GCE_model_*_front_12yr_cholis${RANK_SUFFIX}.dat 2>/dev/null | wc -l
}

launcher_alive() {
    # Match by full cmdline (pgrep -c uses comm, not cmdline — unreliable
    # for python scripts, see 2026-05-15 false-negative confusion).
    local n
    n=$(pgrep -af "$LAUNCHER_PATTERN" 2>/dev/null | wc -l)
    [ "$n" -ge 1 ]
}

worker_count() {
    pgrep -af "$WRAPPER_PATTERN" 2>/dev/null | wc -l
}

recover_orphans() {
    # Move workdir orphan .dat + companions into results dir.
    # Echo each rename; safe to call when none exist.
    local moved=0
    shopt -s nullglob
    for dat in GCE_model_*_front_12yr_cholis${RANK_SUFFIX}.dat; do
        local m=${dat#GCE_model_}
        m=${m%_front_12yr_cholis${RANK_SUFFIX}.dat}
        for ext in .dat _fit.npz _likelihood_value; do
            local src="GCE_model_${m}_front_12yr_cholis${RANK_SUFFIX}${ext}"
            local dst="${RESULTS_DIR}/${src}"
            if [ -f "$src" ] && [ ! -f "$dst" ]; then
                mv "$src" "$dst" && moved=$((moved+1))
            elif [ -f "$src" ] && [ -f "$dst" ]; then
                rm -f "$src"   # duplicate, drop the workdir copy
            fi
        done
    done
    shopt -u nullglob
    echo "$moved"
}

start_launcher() {
    # Run launcher in its own background process so this script can keep
    # polling. Use a date-stamped log so we don't clobber prior attempts.
    local stamp
    stamp=$(date +%m%d_%H%M)
    local logf="launcher_phaseB_${stamp}.log"

    # Make sure DIAG_SAVE_CHAIN is not inherited (production = chain off)
    nohup env -u DIAG_SAVE_CHAIN \
        python "$LAUNCHER_SCRIPT" \
            --workers "$WORKERS" \
            --max-retries "$MAX_RETRIES" \
            --max-runtime-hr "$MAX_RUNTIME_HR" \
        > "$logf" 2>&1 &
    local pid=$!
    disown 2>/dev/null || true
    log "spawned launcher pid=$pid  log=$logf"
}

acquire_lock() {
    if [ -e "$WATCHDOG_LOCK" ]; then
        local old
        old=$(cat "$WATCHDOG_LOCK" 2>/dev/null || true)
        if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
            echo "[FATAL] another watchdog (pid=$old) is already running."
            echo "        if it's actually dead, remove $WATCHDOG_LOCK and retry."
            exit 2
        else
            echo "[warn] stale watchdog lockfile (pid $old gone), reclaiming."
        fi
    fi
    echo "$$" > "$WATCHDOG_LOCK"
}

release_lock() {
    if [ -e "$WATCHDOG_LOCK" ]; then
        local own
        own=$(cat "$WATCHDOG_LOCK" 2>/dev/null || true)
        if [ "$own" = "$$" ]; then
            rm -f "$WATCHDOG_LOCK"
        fi
    fi
}

cleanup_on_exit() {
    release_lock
}
trap cleanup_on_exit EXIT
trap 'log "received signal — exiting"; exit 130' INT TERM

# ============================================================
# Main
# ============================================================
acquire_lock

if [ ! -f "$LAUNCHER_SCRIPT" ]; then
    log "[FATAL] $LAUNCHER_SCRIPT not found in cwd ($(pwd))"
    exit 2
fi
if [ ! -d "$RESULTS_DIR" ]; then
    log "[FATAL] $RESULTS_DIR/ not found in cwd"
    exit 2
fi
rm -f "$WATCHDOG_STOP"   # ensure clean state

restart_count=0
initial_dat=$(count_dat)

log "watchdog start  pid=$$  initial_dat=$initial_dat/$TARGET_COUNT"
log "  config: workers=$WORKERS  max_retries=$MAX_RETRIES  max_runtime_hr=$MAX_RUNTIME_HR"
log "  poll=${POLL_SEC}s  max_restarts=$MAX_RESTARTS"

# If launcher not already running at start, spawn one immediately.
if ! launcher_alive; then
    log "[init] no launcher detected — recovering orphans then spawning"
    n_orphan=$(recover_orphans)
    if [ "$n_orphan" -gt 0 ]; then
        log "[init] recovered $n_orphan orphan files into $RESULTS_DIR/"
    fi
    if [ -e "$LAUNCHER_LOCK" ]; then
        rm -f "$LAUNCHER_LOCK"
        log "[init] removed stale $LAUNCHER_LOCK"
    fi
    start_launcher
    restart_count=$((restart_count + 1))
else
    log "[init] launcher already running, watchdog only monitors"
fi

# Main polling loop
while true; do
    sleep "$POLL_SEC"

    # Manual stop request?
    if [ -e "$WATCHDOG_STOP" ]; then
        log "[stop] $WATCHDOG_STOP detected — exiting watchdog (launcher unaffected)"
        rm -f "$WATCHDOG_STOP"
        exit 0
    fi

    dat=$(count_dat)

    # Success?
    if [ "$dat" -ge "$TARGET_COUNT" ]; then
        log "[success] $dat/$TARGET_COUNT — final recovery + exit"
        # One final orphan sweep in case last batch finished after launcher died
        n=$(recover_orphans)
        [ "$n" -gt 0 ] && log "  final recovery moved $n files"
        log "watchdog done — total restarts: $restart_count"
        exit 0
    fi

    # Liveness
    if launcher_alive; then
        # Heartbeat every ~10 polls (10 min). Keeps log readable.
        if [ $((SECONDS / POLL_SEC % 10)) -eq 0 ]; then
            workers=$(worker_count)
            log "[ok] launcher alive  workers=$workers  dat=$dat/$TARGET_COUNT  restarts=$restart_count"
        fi
        continue
    fi

    # Launcher dead — recover
    log "[dead] launcher process gone  dat=$dat/$TARGET_COUNT"
    n_orphan=$(recover_orphans)
    if [ "$n_orphan" -gt 0 ]; then
        new_dat=$(count_dat)
        log "  recovered $n_orphan orphan files into $RESULTS_DIR/  new_dat=$new_dat"
    fi
    if [ -e "$LAUNCHER_LOCK" ]; then
        rm -f "$LAUNCHER_LOCK"
        log "  removed stale $LAUNCHER_LOCK"
    fi

    if [ "$restart_count" -ge "$MAX_RESTARTS" ]; then
        log "[brake] hit MAX_RESTARTS=$MAX_RESTARTS — exiting; manual intervention needed"
        exit 3
    fi

    start_launcher
    restart_count=$((restart_count + 1))
    log "  restart_count now $restart_count / $MAX_RESTARTS"
done
