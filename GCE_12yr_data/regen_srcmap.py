import os
import glob
import subprocess
import multiprocessing as mp
import time

# =====================================================================
# 1. 원흉(빈 껍데기 파일들) 일괄 삭제
# =====================================================================
bad_files = glob.glob('Source_Maps_12yr/*.fits')
for f in bad_files:
    os.remove(f)
print(f"🗑️ 기존의 불완전한 소스 맵 껍데기 파일 {len(bad_files)}개를 모두 삭제했습니다.\n")

# =====================================================================
# 2. 안전하고 확실한 Source Map 생성 함수
# =====================================================================
def generate_srcmap_safe(roman_name):
    xml_file = f'XML_models_12yr/GCE_12yr_4FGLDR2_Model_{roman_name}.xml'
    outfile = f'Source_Maps_12yr/GCE_12yr_srcmap_Model_{roman_name}.fits'

    # gt_apps 대신 터미널 명령어를 직접 구성합니다.
    cmd = [
        "gtsrcmaps",
        "expcube=GCE_12yr_ltcube.fits",
        "cmap=GCE_12yr_ccube.fits",
        f"srcmdl={xml_file}",
        "bexpmap=GCE_12yr_expcube_large.fits",
        f"outfile={outfile}",
        "irfs=P8R3_ULTRACLEANVETO_V3",
        "evtype=3",
        "ptsrc=yes",
        "chatter=0" # 콘솔 출력 최소화
    ]

    try:
        # subprocess를 이용해 C++ 프로그램이 진짜 끝날 때까지 대기
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
        # 파일이 생성되었는지, 그리고 용량이 충분히 큰지(최소 10MB 이상) 교차 검증!
        if os.path.exists(outfile) and os.path.getsize(outfile) > 10000000:
            return (roman_name, "✅ 생성 완료 (정상 용량 확인)")
        else:
            return (roman_name, "❌ 생성 실패 (파일이 없거나 너무 작음)")
            
    except subprocess.CalledProcessError as e:
        # 에러가 나면 숨기지 않고 확실하게 출력
        err_msg = e.stderr.decode('utf-8').strip()
        return (roman_name, f"❌ 에러 발생: {err_msg}")

# =====================================================================
# 3. 병렬 실행 블록
# =====================================================================
if __name__ == '__main__':
    models_info = []
    with open('NAMING_CONVENTION_OF_DIFFUSE_EMISSION_MODELS.dat', 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = line.split()
            if len(parts) >= 2: models_info.append(parts[0])

    NUM_CORES = 8 # 여유 메모리에 맞게 8코어 세팅
    print(f"▶️ 80개 모델의 Source Map을 [강제 검증 모드]로 다시 구웁니다. (코어: {NUM_CORES})")
    print("--------------------------------------------------")
    
    start_time = time.time()
    
    with mp.Pool(processes=NUM_CORES) as pool:
        for result in pool.imap_unordered(generate_srcmap_safe, models_info):
            print(f"Model {result[0]} : {result[1]}")
            
    print(f"\n🎉 진짜 소스 맵 재생성 완료! (소요 시간: {(time.time() - start_time)/3600:.2f} 시간)")
