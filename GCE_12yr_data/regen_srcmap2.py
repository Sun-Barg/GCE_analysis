import os
import subprocess
import multiprocessing as mp
import time

def generate_diffuse_srcmap(roman_name):
    xml_file = f'XML_models_12yr/GCE_12yr_4FGLDR2_Model_{roman_name}.xml'
    outfile = f'Source_Maps_12yr/GCE_12yr_srcmap_Model_{roman_name}.fits'

    cmd = [
        "gtsrcmaps",
        "expcube=GCE_12yr_ltcube.fits",
        "cmap=GCE_12yr_ccube.fits",
        f"srcmdl={xml_file}",
        "bexpmap=GCE_12yr_expcube_large.fits",
        f"outfile={outfile}",
        "irfs=P8R3_ULTRACLEANVETO_V3",
        "evtype=3",
        "ptsrc=no",   # ⭐ 오타 수정 완료 (ptsrcs -> ptsrc)
        "chatter=0"
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return (roman_name, "✅ 생성 완료 (Diffuse Only)")
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode('utf-8').strip()
        return (roman_name, f"❌ 에러 발생: {err_msg}")

if __name__ == '__main__':
    models_info = []
    with open('NAMING_CONVENTION_OF_DIFFUSE_EMISSION_MODELS.dat', 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = line.split()
            if len(parts) >= 2: models_info.append(parts[0])

    NUM_CORES = 8 # 코어 8개 사용
    print(f"▶️ [최적화 모드] 확산 배경(Diffuse)만 소스 맵으로 구워냅니다. (코어: {NUM_CORES})")
    
    start_time = time.time()
    with mp.Pool(processes=NUM_CORES) as pool:
        for result in pool.imap_unordered(generate_diffuse_srcmap, models_info):
            print(f"Model {result[0]} : {result[1]}")
            
    print(f"\n🎉 소스 맵 생성 완료! (소요 시간: {(time.time() - start_time)/60:.2f} 분)")
