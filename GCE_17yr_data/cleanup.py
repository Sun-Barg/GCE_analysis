#!/usr/bin/env python3
# =====================================================================
# cleanup.py — 재실행 전 파일 상태 점검 및 정리 스크립트
#
# [사용법]
#   python3 cleanup.py check       # 현재 상태만 확인 (아무것도 삭제 안 함)
#   python3 cleanup.py soft        # 피팅 결과만 초기화 (srcmap 보존)
#   python3 cleanup.py hard        # srcmap 포함 전체 초기화 (Phase 1부터 재실행)
#
# [권장 사용 시나리오]
#   - optimizer/tol 변경 후 재실행  → soft
#   - srcmap 자체가 의심될 때       → hard
#   - 실행 전 현재 상태 확인        → check
# =====================================================================

import os
import sys
import glob
import shutil

# =====================================================================
# 파일 분류 정의
# =====================================================================

# 절대 보존 — 원본 데이터 (삭제 시 전체 재분석 필요)
PROTECTED = [
    'GCE_17yr_ccube.fits',
    'GCE_17yr_ltcube.fits',
    'GCE_17yr_expcube_large.fits',
    'GCE_17yr_expcube.fits',
    'GCE_17yr_filtered.fits',
    'GCE_17yr_gti.fits',
    'GCE_17yr_Base_SourceMap.fits',
    'GCE_17yr_Base_SourceMap.fits.done',
    'GCE_17yr_Base_Model.xml',
    'XML_models',
    'LAT_extended_sources_16years',
]

# 재사용 가능 중간 산출물 (soft에서 보존, hard에서 삭제)
SRCMAP_FILES  = sorted(glob.glob('Source_Maps_Smart/*.fits'))
SRCMAP_DONE   = sorted(glob.glob('Source_Maps_Smart/*.done'))

# 재실행 시 반드시 초기화 (soft + hard 모두)
RESET_TARGETS = {
    'Likelihood_Results_Final.csv' : 'csv_header',   # 헤더만 남김
    'Error_Logs/error_fit_*.log'   : 'glob_delete',
    'Fitted_XML_models_Smart/*.xml': 'glob_delete',
}

# 임시 파일 (soft + hard 모두 삭제 가능)
TEMP_TARGETS = [
    'workdirs',
    'workdir_srcmap_*',
    'workdir_memtest_*',
    'Memory_Logs',
    'system_memory_monitor.log',
    'gtsrcmaps_run.log',
    'GCE_17yr_Base_SourceMap.fits_*.fits',
    '__pycache__',
]

CSV_HEADER = "Model,LogLikelihood,FitStatus,WorkerPeakGB\n"


# =====================================================================
# 유틸
# =====================================================================
def gb(path):
    """파일 또는 디렉토리 크기를 GB로 반환합니다."""
    try:
        if os.path.isfile(path):
            return os.path.getsize(path) / 1024**3
        elif os.path.isdir(path):
            total = 0
            for root, _, files in os.walk(path):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            return total / 1024**3
    except OSError:
        return 0.0

def count_glob(pattern):
    return len(glob.glob(pattern))

def size_glob(pattern):
    return sum(gb(f) for f in glob.glob(pattern))


# =====================================================================
# CHECK: 현재 상태 출력
# =====================================================================
def check():
    sep = "=" * 60
    print(sep)
    print("파일 상태 점검 결과")
    print(sep)

    # 원본 데이터
    print("\n[보존] 원본 데이터")
    for p in PROTECTED:
        exists = os.path.exists(p)
        size   = gb(p)
        mark   = "✅" if exists else "❌ 없음!"
        print(f"  {mark} {p:<45s} {size:.2f} GB" if exists else
              f"  {mark} {p}")

    # srcmap 상태
    print(f"\n[Phase 1] srcmap 산출물")
    print(f"  .fits 파일 : {len(SRCMAP_FILES):3d}개  "
          f"({size_glob('Source_Maps_Smart/*.fits'):.1f} GB)")
    print(f"  .done 파일 : {len(SRCMAP_DONE):3d}개  "
          f"← 이 수만큼 Phase 1 스킵됨")
    if len(SRCMAP_FILES) != len(SRCMAP_DONE):
        print(f"  ⚠️  .fits 와 .done 수가 다릅니다 — "
              f"불완전한 srcmap이 있을 수 있습니다.")
        fits_names = {os.path.basename(f).replace('.done','')
                      for f in SRCMAP_DONE}
        for f in SRCMAP_FILES:
            base = os.path.basename(f)
            if base + '.done' not in [os.path.basename(d) for d in SRCMAP_DONE]:
                print(f"    .done 없음: {base}")

    # 피팅 결과
    print(f"\n[Phase 2] 피팅 결과")
    n_fitted = count_glob('Fitted_XML_models_Smart/*.xml')
    n_errors = count_glob('Error_Logs/error_fit_*.log')
    print(f"  완료 XML   : {n_fitted:3d}개")
    print(f"  에러 로그  : {n_errors:3d}개")

    csv = 'Likelihood_Results_Final.csv'
    if os.path.exists(csv):
        with open(csv) as f:
            lines = f.readlines()
        n_data = len(lines) - 1   # 헤더 제외
        print(f"  결과 CSV   : {n_data:3d}행 기록됨")
        # 오염 검사: 이전 실패 상태가 섞여 있는지 확인
        error_rows = [l.strip() for l in lines[1:] if 'Error' in l]
        warning_rows = [l.strip() for l in lines[1:] if 'Warning' in l]
        if error_rows:
            print(f"  ⚠️  Error 행 {len(error_rows)}개 — soft reset 권장")
        if warning_rows:
            print(f"  ℹ️  Warning 행 {len(warning_rows)}개")
    else:
        print(f"  결과 CSV   : 없음")

    # 임시 파일
    print(f"\n[임시] 정리 가능한 파일")
    for pattern in TEMP_TARGETS:
        matches = glob.glob(pattern)
        if matches:
            size = sum(gb(m) for m in matches)
            print(f"  {pattern:<40s} {len(matches)}개  {size:.2f} GB")

    print(f"\n{'─'*60}")
    print("권장 실행 명령:")
    if n_fitted > 0 or n_errors > 0:
        print("  python3 cleanup.py soft   ← 피팅 결과 초기화 후 재실행")
    else:
        print("  python3 cleanup.py check  ← 현재 상태 이상 없음")
    print(sep)


# =====================================================================
# SOFT RESET: 피팅 결과만 초기화 (srcmap 보존)
# =====================================================================
def soft():
    print("=" * 60)
    print("Soft Reset — srcmap 보존, 피팅 결과 초기화")
    print("=" * 60)

    # 피팅 XML 삭제
    xmls = glob.glob('Fitted_XML_models_Smart/*.xml')
    if xmls:
        for f in xmls:
            os.remove(f)
        print(f"  🗑  Fitted XML 삭제       : {len(xmls)}개")

    # 에러 로그 삭제
    errs = glob.glob('Error_Logs/error_fit_*.log')
    if errs:
        for f in errs:
            os.remove(f)
        print(f"  🗑  Error 로그 삭제       : {len(errs)}개")

    # CSV 헤더만 남기기
    with open('Likelihood_Results_Final.csv', 'w') as f:
        f.write(CSV_HEADER)
    print(f"  🔄 CSV 초기화             : 헤더만 유지")

    # 임시 파일 삭제
    _clean_temp()

    print(f"\n  ✅ Soft Reset 완료")
    print(f"     Source_Maps_Smart/ ({len(SRCMAP_DONE)}개 .done) 보존됨")
    print(f"     → Phase 1 스킵하고 Phase 2부터 재실행됩니다")


# =====================================================================
# HARD RESET: srcmap 포함 전체 초기화
# =====================================================================
def hard():
    print("=" * 60)
    print("Hard Reset — 전체 초기화 (Phase 1부터 재실행)")
    print("=" * 60)
    print("⚠️  srcmap 파일 전체를 삭제합니다. 계속하시겠습니까? [y/N] ", end='')

    confirm = input().strip().lower()
    if confirm != 'y':
        print("취소됨.")
        return

    # srcmap 전체 삭제 (*.fits_*.fits 임시파일 포함)
    fits = (glob.glob('Source_Maps_Smart/*.fits') +
            glob.glob('Source_Maps_Smart/*.fits_*.fits'))
    done = glob.glob('Source_Maps_Smart/*.done')
    size = sum(gb(f) for f in fits)
    removed_fits = 0
    for f in fits + done:
        try:
            os.remove(f)
            removed_fits += 1
        except FileNotFoundError:
            pass
    print(f"  🗑  srcmap 파일 삭제      : {removed_fits}개  ({size:.1f} GB)")

    # 피팅 결과 삭제
    xmls = glob.glob('Fitted_XML_models_Smart/*.xml')
    errs = glob.glob('Error_Logs/error_fit_*.log')
    for f in xmls + errs:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    print(f"  🗑  Fitted XML 삭제       : {len(xmls)}개")
    print(f"  🗑  Error 로그 삭제       : {len(errs)}개")

    # CSV 초기화
    with open('Likelihood_Results_Final.csv', 'w') as f:
        f.write(CSV_HEADER)
    print(f"  🔄 CSV 초기화             : 헤더만 유지")

    # 임시 파일
    _clean_temp()

    print(f"\n  ✅ Hard Reset 완료")
    print(f"     → Phase 1 (srcmap 생성) 부터 재실행됩니다")
    print(f"     예상 추가 소요: srcmap 5코어 기준 ~8시간")


# =====================================================================
# 임시 파일 공통 정리
# =====================================================================
def _clean_temp():
    removed = 0
    size    = 0.0
    for pattern in TEMP_TARGETS:
        for match in glob.glob(pattern):
            s = gb(match)
            try:
                if os.path.isdir(match):
                    shutil.rmtree(match)
                else:
                    os.remove(match)
                size    += s
                removed += 1
            except OSError as e:
                print(f"  ⚠️  삭제 실패: {match} ({e})")
    if removed:
        print(f"  🗑  임시 파일/디렉토리    : {removed}개  ({size:.2f} GB)")
    # Memory_Logs 디렉토리는 srcmap 로그만 삭제, fit 로그는 유지
    src_logs = glob.glob('Memory_Logs/mem_srcmap_*.txt')
    if src_logs:
        for f in src_logs:
            os.remove(f)
        print(f"  🗑  srcmap 메모리 로그    : {len(src_logs)}개")


# =====================================================================
# 메인
# =====================================================================
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

    mode = sys.argv[1] if len(sys.argv) > 1 else 'check'

    if mode == 'check':
        check()
    elif mode == 'soft':
        soft()
    elif mode == 'hard':
        hard()
    else:
        print(f"알 수 없는 모드: {mode}")
        print("사용법: python3 cleanup.py [check|soft|hard]")
        sys.exit(1)
