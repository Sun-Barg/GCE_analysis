import os
import subprocess
import multiprocessing as mp
import time

# =====================================================================
# 1. 단일 모델 피팅 함수 (gtlike CLI 직접 호출)
# =====================================================================
def run_gtlike_cli(roman_name):
    xml_file = f'XML_models_12yr_opt/GCE_12yr_4FGLDR2_Model_{roman_name}.xml'
    srcmap_file = f'Source_Maps_12yr/GCE_12yr_srcmap_Model_{roman_name}.fits'
    out_xml_file = f'Fitted_XML_models_12yr/GCE_12yr_fitted_Model_{roman_name}.xml'
    
    if not os.path.exists(srcmap_file):
        return (roman_name, None, "Source map 파일 누락")
        
    # 파이썬 래퍼 대신, 터미널 명령어를 직접 구성합니다.
    cmd = [
        "gtlike",
        "irfs=P8R3_ULTRACLEANVETO_V3",
        "expcube=GCE_12yr_ltcube.fits",
        "bexpmap=GCE_12yr_expcube_large.fits",
        f"cmap={srcmap_file}",       # Binned 모드에서는 cmap에 소스 맵 파일을 넣습니다.
        f"srcmdl={xml_file}",
        f"sfile={out_xml_file}",     # 피팅이 완료된 파라미터가 저장될 XML
        "optimizer=NewMinuit",
        "wmap=GCE_12yr_mask_b2.fits",# 은하 적도면 마스크 완벽 지원!
        "statistic=BINNED",          # Binned Likelihood 모드
        "chatter=2"                  # 결과를 파싱하기 위해 출력 레벨 설정
    ]
    
    try:
        # 프로세스 실행 및 터미널 출력(stdout) 캡처
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # 출력된 텍스트에서 "-log(Likelihood)" 값을 파싱하여 추출합니다.
        logL = None
        for line in result.stdout.split('\n'):
            if "Optimum -log(Likelihood):" in line:
                logL = float(line.split(':')[1].strip())
                break
                
        if logL is not None:
            return (roman_name, logL, "Success")
        else:
            return (roman_name, None, "우도 값 파싱 실패 (로그 확인 필요)")
            
    except subprocess.CalledProcessError as e:
        return (roman_name, None, f"Error: 프로세스 실패")

# =====================================================================
# 2. 메인 실행 블록 (멀티프로세싱)
# =====================================================================
if __name__ == '__main__':
    start_time = time.time()
    
    fit_xml_dir = 'Fitted_XML_models_12yr'
    if not os.path.exists(fit_xml_dir):
        os.makedirs(fit_xml_dir)

    models_info = []
    with open('NAMING_CONVENTION_OF_DIFFUSE_EMISSION_MODELS.dat', 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = line.split()
            if len(parts) >= 2: models_info.append(parts[0])

    # 메모리 상황에 맞춰 4~8코어 정도로 실행하세요.
    NUM_CORES = 4  
    print(f"▶️ [Native CLI 모드] 12.5년 치 80개 모델의 병렬 우도 피팅을 시작합니다. (코어: {NUM_CORES})")
    print("--------------------------------------------------")

    results_file = 'Likelihood_Results_12yr_Masked_CLI.csv'
    with open(results_file, 'w') as f:
        f.write("Model,LogLikelihood,Status\n")

    with mp.Pool(processes=NUM_CORES) as pool:
        for result in pool.imap_unordered(run_gtlike_cli, models_info):
            roman_name, logL, status = result
            
            if status == "Success":
                print(f" ✅ [완료] Model {roman_name} | -log(L) = {logL:.2f}")
            else:
                print(f" ❌ [실패] Model {roman_name} | 이유: {status}")
                
            with open(results_file, 'a') as f:
                f.write(f"{roman_name},{logL},{status}\n")

    end_time = time.time()
    print(f"\n🎉 병렬 피팅 완료! (소요 시간: {(end_time - start_time)/3600:.2f} 시간)")
