#!/usr/bin/env python3
# =====================================================================
# monitor.py — 파이프라인 실시간 상태 모니터링
#
# [실행 방법]
#   python3 monitor.py          # 30초 간격 지속 모니터링
#   python3 monitor.py once     # 현재 상태 1회 출력 후 종료
#
# [감지 항목]
#   - Phase 1 srcmap 진행률
#   - Phase 2 피팅 진행률
#   - BinnedAnalysis 소스맵 재생성 여부 (v4 수정 검증)
#   - 워커 상태 (정상/행 걸림/죽음)
#   - 메모리 경고
# =====================================================================

import os
import sys
import time
import glob
import subprocess
import psutil

INTERVAL_SEC   = 600
# v4~v6 등 버전에 무관하게 감지
PIPELINE_NAMES = ['GCE_pipeline_final', 'GCE_analysis_pipeline', 'Gce_pipeline', 'phase1_srcmap', 'phase2_fitting']
LOG_FILE       = 'pipeline_run.log'
WARN_MEM_PCT   = 115.0

def read_tail(path, n=50):
    """파일 마지막 n줄을 읽어 반환합니다."""
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        return lines[-n:]
    except Exception:
        return []

def get_workers():
    """파이프라인 워커 프로세스 목록을 반환합니다."""
    workers = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_times',
                                   'status', 'memory_info']):
        try:
            cmd = ' '.join(p.info['cmdline'] or [])
            if any(name in cmd for name in PIPELINE_NAMES) and                p.info['status'] in ('running', 'sleeping'):
                workers.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return workers

def check_log_for_regen(log_lines):
    """
    Phase 2 진입 후 'Generating SourceMap' 이 나타나는지 감지합니다.
    Phase 2 이후에 이 메시지가 있으면 ptsrc 문제가 재발한 것입니다.
    """
    phase2_started = False
    regen_after_phase2 = []
    for line in log_lines:
        if 'Phase 2' in line or '피팅' in line:
            phase2_started = True
        if phase2_started and 'Generating SourceMap' in line:
            regen_after_phase2.append(line.strip())
    return regen_after_phase2

def check(verbose=True):
    sep  = "=" * 60
    now  = time.strftime('%Y-%m-%d %H:%M:%S')

    # ── 프로세스 상태 ──────────────────────────────────────────────
    workers = get_workers()
    main_proc = [w for w in workers
                 if 'Rl' not in (w.status() or '')]
    fit_workers = [w for w in workers
                   if w.cpu_percent(interval=0.1) > 50]

    # ── srcmap 진행 ────────────────────────────────────────────────
    n_fits   = len(glob.glob('Source_Maps_Smart/*.fits'))
    n_done   = len(glob.glob('Source_Maps_Smart/*.done'))
    n_xml    = len(glob.glob('Fitted_XML_models_Smart/*.xml'))
    n_errors = len(glob.glob('Error_Logs/error_fit_*.log'))

    # ── 메모리 ────────────────────────────────────────────────────
    vm       = psutil.virtual_memory()
    mem_warn = vm.percent >= WARN_MEM_PCT

    # ── 로그 분석 ─────────────────────────────────────────────────
    log_lines    = read_tail(LOG_FILE, 100)
    regen_issues = check_log_for_regen(log_lines)
    last_log     = log_lines[-1].strip() if log_lines else '(로그 없음)'

    # ── 출력 ──────────────────────────────────────────────────────
    if verbose:
        print(f"\n{sep}")
        print(f"파이프라인 상태 모니터  [{now}]")
        print(sep)

        # 프로세스
        print(f"\n  [프로세스]")
        print(f"  전체 워커 수 : {len(workers)}개")
        if len(workers) == 0:
            print("  ❌ 파이프라인이 실행 중이 아닙니다!")
        else:
            print(f"  ✅ 실행 중")

        # srcmap
        print(f"\n  [Phase 1] srcmap 생성")
        print(f"  완료 .done : {n_done:3d} / 80개  "
              f"({'완료' if n_done == 80 else f'{n_done/80*100:.0f}%'})")

        # 피팅
        print(f"\n  [Phase 2] likelihood 피팅")
        print(f"  완료 XML   : {n_xml:3d} / 80개  "
              f"({'완료' if n_xml == 80 else f'{n_xml/80*100:.0f}%'})")
        print(f"  에러 로그  : {n_errors:3d}개")

        # 메모리
        mem_mark = "⚠️ " if mem_warn else "✅"
        print(f"\n  [메모리]")
        print(f"  {mem_mark} {vm.used/1024**3:.1f} GB 사용 / "
              f"{vm.total/1024**3:.1f} GB 전체 "
              f"(가용 {vm.available/1024**3:.1f} GB, {vm.percent:.1f}%)")

        # 핵심 검증: BinnedAnalysis 재생성 여부
        print(f"\n  [핵심 검증] Phase 2 중 소스맵 재생성 감지")
        if regen_issues:
            print(f"  ❌ 경고! 피팅 중 소스맵 재생성이 감지됐습니다!")
            print(f"     → ptsrc 설정 문제가 재발했습니다. 즉시 종료 권장.")
            for line in regen_issues[:3]:
                print(f"     {line}")
        else:
            if n_done == 80 and n_xml == 0:
                print(f"  ⏳ Phase 2 진행 중 — 아직 재생성 없음 (정상)")
            elif n_xml > 0:
                print(f"  ✅ 재생성 없이 피팅 완료 모델 {n_xml}개")
            else:
                print(f"  ✅ 현재까지 소스맵 재생성 없음")

        # 마지막 로그
        print(f"\n  [마지막 로그]")
        print(f"  {last_log[:80]}")

        # 판정
        print(f"\n{'─'*60}")
        if len(workers) == 0:
            print("  ⛔ 파이프라인 중단됨 — 재실행 필요")
        elif regen_issues:
            print("  ⛔ 소스맵 재생성 감지 — 즉시 종료 후 원인 분석 필요")
        elif n_xml == 80:
            print("  🎉 파이프라인 완료!")
        elif n_done < 80:
            print(f"  ⏳ Phase 1 진행 중 ({n_done}/80)")
        elif n_done == 80 and n_xml == 0 and len(workers) > 0:
            print(f"  ⏳ Phase 2 시작 대기 중 또는 첫 피팅 진행 중")
        elif n_xml > 0:
            print(f"  ⏳ Phase 2 진행 중 ({n_xml}/80)")
        print(sep)

    return {
        'workers'      : len(workers),
        'n_done'       : n_done,
        'n_xml'        : n_xml,
        'n_errors'     : n_errors,
        'mem_warn'     : mem_warn,
        'regen_issues' : regen_issues,
    }


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'loop'

    if mode == 'once':
        check(verbose=True)
    else:
        print(f"파이프라인 모니터 시작 (갱신 간격: {INTERVAL_SEC}초)")
        print("종료: Ctrl+C\n")
        try:
            while True:
                state = check(verbose=True)
                # 완료 또는 이상 징후 시 강조
                if state['regen_issues']:
                    print("\n⛔ 소스맵 재생성 감지 — 즉시 확인하세요!\n")
                if state['n_xml'] == 80:
                    print("\n🎉 전체 완료!\n")
                    break
                time.sleep(INTERVAL_SEC)
        except KeyboardInterrupt:
            print("\n모니터 종료")
