# do not run in jupyter의 %%bash가 아닌, 순수 Python 코드로 실행합니다.
import os
import multiprocessing as mp
import time

def run_perfect_srcmaps(roman_name):
    import gt_apps as my_apps
    
    # 12년 치 전용 폴더 설정
    xml_dir = 'XML_models_12yr'
    srcmap_dir = 'Source_Maps_12yr' 
    
    if not os.path.exists(srcmap_dir):
        os.makedirs(srcmap_dir, exist_ok=True)
        
    xml_file = os.path.join(xml_dir, f'GCE_12yr_4FGLDR2_Model_{roman_name}.xml')
    outfile = os.path.join(srcmap_dir, f'GCE_12yr_srcmap_Model_{roman_name}.fits')
    
    # 이미 잘 구워진 파일은 건너뜀 (대략 30MB 이상)
    if os.path.exists(outfile) and os.path.getsize(outfile) > 30000000:
        return (roman_name, "이미 존재함 (건너뜀)")
        
    try:
        my_apps.srcMaps['expcube'] = 'GCE_12yr_ltcube.fits'          
        my_apps.srcMaps['cmap']    = 'GCE_12yr_ccube.fits'              
        my_apps.srcMaps['srcmdl']  = xml_file           
        my_apps.srcMaps['bexpmap'] = 'GCE_12yr_expcube_large.fits'         
        my_apps.srcMaps['outfile'] = outfile  
        my_apps.srcMaps['irfs']    = 'P8R3_ULTRACLEANVETO_V3'           
        my_apps.srcMaps['evtype']  = 3                                
        
        # ⭐ 점광원까지 모조리 FITS 파일 안에 구워 넣습니다!
        my_apps.srcMaps['ptsrc']   = 'yes'  
        my_apps.srcMaps['chatter'] = 0

        my_apps.srcMaps.run()
        return (roman_name, "✅ 생성 성공")
    except Exception as e:
        return (roman_name, f"❌ 에러 발생: {e}")

if __name__ == '__main__':
    start_time = time.time()
    
    naming_conv_file = '/home/haebarg/GCE-Chi-square-fitting/GCE_TEMPLATES_FILES_v3/NAMING_CONVENTION_OF_DIFFUSE_EMISSION_MODELS.dat'
    models_info = []
    with open(naming_conv_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                models_info.append(parts[0])

    NUM_CORES = 4
    print(f"▶️ 80개 12.5년 치 모델의 Source Maps 병렬 생성을 시작합니다. (코어: {NUM_CORES})")
    print("--------------------------------------------------")

    with mp.Pool(processes=NUM_CORES) as pool:
        for result in pool.imap_unordered(run_perfect_srcmaps, models_info):
            print(f"Model {result[0]} : {result[1]}")

    end_time = time.time()
    print(f"\n🎉 소스 맵 병렬 생성 완료! (소요 시간: {(end_time - start_time)/3600:.2f} 시간)")
