# do not run in jupyter의 %%bash가 아닌, 순수 Python 코드로 실행합니다.
import os
import multiprocessing as mp
import time

# =====================================================================
# 수정된 단일 모델 피팅 함수
# =====================================================================
def run_fitting_for_model(roman_name):
    import BinnedAnalysis as ba
    import os
    
    xml_dir = 'XML_models_12yr'
    srcmap_dir = 'Source_Maps_12yr'
    fit_xml_dir = 'Fitted_XML_models_12yr'
    
    xml_file = os.path.join(xml_dir, f'GCE_12yr_4FGLDR2_Model_{roman_name}.xml')
    srcmap_file = os.path.join(srcmap_dir, f'GCE_12yr_srcmap_Model_{roman_name}.fits')
    out_xml_file = os.path.join(fit_xml_dir, f'GCE_12yr_fitted_Model_{roman_name}.xml')
    
    if not os.path.exists(srcmap_file):
        return (roman_name, None, "Source map 파일 누락")
        
    try:
        # 🚨 wmap 파라미터 삭제! (srcmap_file 안에 들어있는 WEIGHTS 블록을 알아서 읽습니다)
        obs = ba.BinnedObs(srcMaps=srcmap_file,
                           expCube='GCE_12yr_ltcube.fits',
                           binnedExpMap='GCE_12yr_expcube_large.fits',
                           irfs='P8R3_ULTRACLEANVETO_V3')
                           
        # 피팅 객체 생성 및 NewMinuit 알고리즘 사용
        like = ba.BinnedAnalysis(obs, srcModel=xml_file, optimizer='NewMinuit')
        
        # 피팅 수행 (tol=1e-2 수준으로 정밀도 설정)
        likeObj = like.fit(covar=True, tol=1e-2, verbosity=0)
        
        # 피팅된 파라미터 저장 및 우도 값(-log L) 반환
        logL = like.logLike.value()
        like.logLike.writeXml(out_xml_file)
        
        return (roman_name, logL, "Success")
        
    except Exception as e:
        return (roman_name, None, f"Error: {e}")


# =====================================================================
# 2. 메인 실행 블록 (멀티프로세싱)
# =====================================================================
if __name__ == '__main__':
    start_time = time.time()
    
    fit_xml_dir = 'Fitted_XML_models_12yr'
    if not os.path.exists(fit_xml_dir):
        os.makedirs(fit_xml_dir)

    # 80개 모델 리스트 불러오기
    naming_conv_file = '/home/haebarg/GCE-Chi-square-fitting/GCE_TEMPLATES_FILES_v3/NAMING_CONVENTION_OF_DIFFUSE_EMISSION_MODELS.dat'
    models_info = []
    with open(naming_conv_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                models_info.append(parts[0])

    # 안정성을 위해 코어 수를 4개로 제한합니다. (필요시 조정 가능)
    NUM_CORES = 4  
    print(f"▶️ 12.5년 치 80개 모델의 병렬 우도 피팅을 시작합니다. (코어: {NUM_CORES}, 마스크 적용)")
    print("--------------------------------------------------")

    results_file = 'Likelihood_Results_12yr_Masked.csv'
    with open(results_file, 'w') as f:
        f.write("Model,LogLikelihood,Status\n")

    # 병렬 처리 시작
    with mp.Pool(processes=NUM_CORES) as pool:
        for result in pool.imap_unordered(run_fitting_for_model, models_info):
            roman_name, logL, status = result
            
            if status == "Success":
                print(f" ✅ [완료] Model {roman_name} | -log(L) = {logL:.2f}")
            else:
                print(f" ❌ [실패] Model {roman_name} | 이유: {status}")
                
            # 결과가 나오는 대로 CSV에 즉시 저장
            with open(results_file, 'a') as f:
                f.write(f"{roman_name},{logL},{status}\n")

    end_time = time.time()
    print(f"\n🎉 병렬 피팅 완료! (소요 시간: {(end_time - start_time)/3600:.2f} 시간)")
    print(f"📊 최종 결과가 {results_file} 에 저장되었습니다.")
