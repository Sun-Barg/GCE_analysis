# =====================================================================
# phase1_srcmap.py — srcmap 생성 전용 (split 파일 병합 방식)
#
# [실행 방법]
#   python3 -u phase1_srcmap.py > srcmap_stdout.log 2> srcmap_stderr.log
#
# [방식]
#   Source_Maps/ 의 기존 split 파일(_0,_1,_2.fits)을 활용합니다.
#   Step 1: ptsrc=no 로 diffuse 전용 파일 생성 (~30분/모델)
#   Step 2: Source_Maps/ split 파일에서 FL16Y extension 병합
#   Step 3: diffuse extension 추가
#   Step 4: 단일 FITS 파일로 저장 → Source_Maps_Perfect/
#
# [결과]
#   FL16Y 1371개 + Diffuse 4개 포함
#   BinnedAnalysis 로드 시 재생성 최소화 (2개 미만)
# =====================================================================

import os
import time
import glob
import multiprocessing as mp
import traceback
import psutil

NUM_CORES          = 4
MEM_WARN_THRESHOLD = 88.0

BASE_DIR      = os.path.abspath('.')
SRC_MAPS_DIR  = os.path.join(BASE_DIR, 'Source_Maps')
OUT_DIR       = os.path.join(BASE_DIR, 'Source_Maps_Perfect')
DIFFUSE_KEYS  = ['Bremss_', 'ICS_', 'Pi0_', 'iso_']


def build_srcmap(roman_name):
    from astropy.io import fits
    import gt_apps as my_apps

    os.makedirs(OUT_DIR, exist_ok=True)

    base      = os.path.join(SRC_MAPS_DIR,
                f'GCE_17yr_srcmap_Model_{roman_name}.fits')
    out       = os.path.join(OUT_DIR,
                f'GCE_17yr_srcmap_Model_{roman_name}.fits')
    tmp       = os.path.join(OUT_DIR,
                f'GCE_17yr_srcmap_Model_{roman_name}_diffuse_tmp.fits')
    done_flag = out + '.done'
    xml_file  = os.path.join(BASE_DIR, 'XML_models',
                f'GCE_17yr_FL16Y_Model_{roman_name}.xml')
    error_dir = os.path.join(BASE_DIR, 'Error_Logs')
    os.makedirs(error_dir, exist_ok=True)

    if os.path.exists(done_flag):
        return (roman_name, 'Skip')

    split_files = sorted(glob.glob(f'{base}_*.fits'))
    if not split_files:
        return (roman_name, 'Error: Source_Maps/ split 파일 없음')

    try:
        for f in [out, tmp]:
            if os.path.exists(f):
                os.remove(f)

        # Step 1: diffuse 전용 파일 생성
        my_apps.srcMaps['expcube'] = os.path.join(BASE_DIR, 'GCE_17yr_ltcube.fits')
        my_apps.srcMaps['cmap']    = os.path.join(BASE_DIR, 'GCE_17yr_ccube.fits')
        my_apps.srcMaps['srcmdl']  = xml_file
        my_apps.srcMaps['bexpmap'] = os.path.join(BASE_DIR, 'GCE_17yr_expcube_large.fits')
        my_apps.srcMaps['outfile'] = tmp
        my_apps.srcMaps['irfs']    = 'P8R3_ULTRACLEANVETO_V3'
        my_apps.srcMaps['evtype']  = 3
        my_apps.srcMaps['ptsrc']   = 'no'
        my_apps.srcMaps['copyall'] = 'no'
        my_apps.srcMaps['chatter'] = 0
        my_apps.srcMaps.run()

        # Step 2: split 파일에서 FL16Y 병합
        with fits.open(base, memmap=False) as main:
            combined = fits.HDUList([
                main[0].copy(), main[1].copy(), main[2].copy()
            ])

        for sp_path in split_files:
            with fits.open(sp_path, memmap=False) as sp:
                for hdu in sp[1:]:
                    combined.append(hdu.copy())

        # Step 3: diffuse extension 추가 (중복 방지)
        with fits.open(tmp, memmap=False) as diff_f:
            seen = set()
            for hdu in diff_f[1:]:
                if any(k in hdu.name for k in DIFFUSE_KEYS):
                    if hdu.name not in seen:
                        combined.append(hdu.copy())
                        seen.add(hdu.name)

        # Step 4: 저장
        combined.writeto(out, overwrite=True)

        if os.path.exists(tmp):
            os.remove(tmp)

        # 검증
        with fits.open(out) as h:
            names = [hdu.name for hdu in h]
            fl16y = len([n for n in names if 'FL16Y' in n])
            diff  = len([n for n in names if
                         any(k in n for k in DIFFUSE_KEYS)])
            size  = os.path.getsize(out) / 1024**3

        if fl16y == 0:
            return (roman_name,
                    f'Error: FL16Y 0개 (Diffuse:{diff}, {size:.1f}GB)')

        open(done_flag, 'w').close()
        return (roman_name, f'OK: {size:.1f}GB FL16Y:{fl16y} Diffuse:{diff}')

    except Exception as e:
        with open(os.path.join(error_dir,
                  f'error_srcmap_{roman_name}.log'), 'w') as ef:
            ef.write(traceback.format_exc())
        for f in [tmp]:
            if os.path.exists(f):
                os.remove(f)
        return (roman_name, f'Error: {e}')


def memory_monitor(threshold_pct, stop_event):
    log_path = 'srcmap_memory.log'
    with open(log_path, 'w') as f:
        f.write("timestamp,used_pct,avail_gb\n")
    while not stop_event.is_set():
        vm    = psutil.virtual_memory()
        ts    = time.strftime('%H:%M:%S')
        avail = vm.available / 1024**3
        with open(log_path, 'a') as f:
            f.write(f"{ts},{vm.percent:.1f},{avail:.2f}\n")
        if vm.percent >= threshold_pct:
            print(f"\n  ⚠️  [메모리 경고 {ts}] {vm.percent:.1f}% "
                  f"(가용 {avail:.1f} GB)", flush=True)
        time.sleep(10)


if __name__ == '__main__':
    start = time.time()

    print("=" * 60)
    print(f"Phase 1: srcmap 생성 (split 병합) [코어: {NUM_CORES}]")
    print("=" * 60)

    vm = psutil.virtual_memory()
    print(f"  메모리: {vm.used/1024**3:.1f} GB / "
          f"{vm.total/1024**3:.1f} GB "
          f"(가용 {vm.available/1024**3:.1f} GB)")

    n_split = len(glob.glob(os.path.join(SRC_MAPS_DIR, '*.fits_0.fits')))
    print(f"  Source_Maps/ split 파일: {n_split}개 모델")
    if n_split == 0:
        print("  ❌ Source_Maps/ 에 split 파일이 없습니다.")
        import sys; sys.exit(1)
    print()

    models = []
    with open('NAMING_CONVENTION_OF_DIFFUSE_EMISSION_MODELS.dat') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                models.append(parts[0])

    models_to_run = [
        m for m in models
        if glob.glob(os.path.join(SRC_MAPS_DIR,
           f'GCE_17yr_srcmap_Model_{m}.fits_0.fits'))
    ]
    n_done = len(glob.glob(os.path.join(OUT_DIR, '*.done')))
    print(f"  처리 대상: {len(models_to_run)}개  "
          f"(완료: {n_done}개, 남은: {len(models_to_run)-n_done}개)\n")

    stop_event   = mp.Event()
    monitor_proc = mp.Process(target=memory_monitor,
                              args=(MEM_WARN_THRESHOLD, stop_event),
                              daemon=True)
    monitor_proc.start()

    summary = {'ok': 0, 'skip': 0, 'error': 0}

    with mp.Pool(processes=NUM_CORES, maxtasksperchild=1) as pool:
        for roman_name, status in pool.imap_unordered(
                build_srcmap, models_to_run, chunksize=1):
            mem = psutil.virtual_memory().percent
            if status == 'Skip':
                print(f"  ⏩ {roman_name:10s} (스킵)", flush=True)
                summary['skip'] += 1
            elif status.startswith('OK'):
                print(f"  ✅ {roman_name:10s} | {status} "
                      f"| 시스템 {mem:.1f}%", flush=True)
                summary['ok'] += 1
            else:
                print(f"  ❌ {roman_name:10s} | {status}", flush=True)
                summary['error'] += 1

    stop_event.set()
    monitor_proc.join(timeout=5)

    elapsed = (time.time() - start) / 60
    print(f"\n{'='*60}")
    print(f"Phase 1 완료 — {elapsed:.1f}분")
    print(f"  성공: {summary['ok']}  스킵: {summary['skip']}  "
          f"실패: {summary['error']}")
    print(f"  출력: Source_Maps_Perfect/")
    print(f"{'='*60}")
