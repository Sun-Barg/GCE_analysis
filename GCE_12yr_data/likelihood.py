import os
from BinnedAnalysis import *

# =====================================================================
# 1. 폴더 및 파일 설정
# =====================================================================
xml_dir = 'XML_models_12yr'                 # 12.5년 치 XML 폴더
srcmap_dir = 'Source_Maps_12yr'             # 방금 정리한 소스 맵 폴더
fit_xml_dir = 'Fitted_XML_models_12yr'      # 피팅이 완료된 파라미터가 저장될 새 폴더

if not os.path.exists(fit_xml_dir):
    os.makedirs(fit_xml_dir)

naming_conv_file = '/home/haebarg/GCE-Chi-square-fitting/GCE_TEMPLATES_FILES_v3/NAMING_CONVENTION_OF_DIFFUSE_EMISSION_MODELS.dat'
models_info = []
with open(naming_conv_file, 'r') as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            models_info.append(parts[0])

# 최종 결과(우도 값)를 저장할 CSV 파일
results_file = 'Likelihood_Results_12yr.csv'
with open(results_file, 'w') as f:
    f.write("Model,LogLikelihood\n")

print("▶️ 80개 모델(12.5년 치)에 대한 Binned Likelihood Fitting을 시작합니다...\n")

# =====================================================================
# 2. 80개 모델 자동 피팅 루프
# =====================================================================
for roman_name in models_info:
    # 12.5년 치 + 4FGL-DR2 이름 규칙 적용
    xml_file = os.path.join(xml_dir, f'GCE_12yr_4FGLDR2_Model_{roman_name}.xml')
    srcmap_file = os.path.join(srcmap_dir, f'GCE_12yr_srcmap_Model_{roman_name}.fits')
    out_xml_file = os.path.join(fit_xml_dir, f'GCE_12yr_fitted_Model_{roman_name}.xml')

    if not os.path.exists(srcmap_file):
        print(f"⚠️ {srcmap_file} 이(가) 없어 건너뜁니다.")
        continue

    print(f"========================================")
    print(f"🚀 피팅 시작: Model {roman_name}")
    print(f"========================================")

    try:
        # 1. BinnedObs 객체 로드 (12.5년 치 데이터, 소스맵, 노출지도 묶기)
        obs = BinnedObs(srcMaps=srcmap_file,
                        expCube='GCE_12yr_ltcube.fits',
                        binnedExpMap='GCE_12yr_expcube_large.fits',
                        irfs='P8R3_ULTRACLEANVETO_V3')

        # 2. Likelihood 최적화 객체 생성 (NewMinuit 알고리즘 사용)
        like = BinnedAnalysis(obs, srcModel=xml_file, optimizer='NewMinuit')

        # 3. 우도 피팅 수행 (서버 성능에 따라 모델당 십여 분~수 시간 소요)
        likeObj = like.fit(covar=True, tol=1e-2, verbosity=0)

        # 4. 피팅된 최종 Log-Likelihood 값 추출
        logL = like.logLike.value()
        print(f"  ✅ 피팅 성공! -log(L) = {logL}")

        # 5. 최적화된 변수들이 기록된 새 XML 파일 저장
        like.logLike.writeXml(out_xml_file)

        # 6. CSV 파일에 결과 기록 (중간에 끊겨도 데이터가 남도록 바로바로 저장)
        with open(results_file, 'a') as f:
            f.write(f"{roman_name},{logL}\n")

    except Exception as e:
        print(f"  ❌ 에러 발생 (Model {roman_name}): {e}")

print("\n🎉 12.5년 치 데이터의 모든 우도 피팅이 완료되었습니다! CSV 결과를 확인하세요.")
