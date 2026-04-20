# Diagnostic experiments A + B — 해석 가이드

## 가설
5 모델이 두 그룹으로 갈림:
- **Group A** (X, XLVIII, LIII): 저에너지 ratio ≈ 0.46-0.49 (under-predicted)
- **Group B** (XV, XLIX): 저에너지 ratio ≈ 1.54-1.59 (over-predicted)

제안된 원인: **ICS-GCE 공간 degeneracy**. Group B의 ICS 맵이 저에너지에서 GCE NFW² 템플릿과 매우 유사한 형태를 가져, fit이 c_ics를 0에 가깝게 누르고 잉여 카운트를 c_gce에 몰아 넣음.

---

## Experiment B : 공간 correlation (진단)

**`diagnostic_B_spatial_corr.py`** — 각 모델의 각 energy bin에서 mask된 픽셀들에 대해:
```
ρ(GCE, ICS) = Σ (GCE_i - <GCE>)(ICS_i - <ICS>) / (σ_GCE σ_ICS)
```
또한 ρ(GCE, π⁰), ρ(GCE, bremss), ρ(GCE, bubble), ρ(ICS, π⁰) 도 비교용으로.

**예상**:
- 만약 가설이 맞다면: Group B의 `ρ(GCE, ICS)` at E<1 GeV가 Group A보다 **현저히 높음** (예: 0.95 vs 0.75)
- 1-10 GeV는 양 그룹 유사 (ratio도 양쪽 다 ~0.8-0.9)

**산출물**:
- `diagnostic_B_spatial_correlation.txt` — 전체 수치 표
- `diagnostic_B_spatial_correlation.png` — 5 모델 × 14 bin heatmap (5 pair)
- `diagnostic_B_spatial_correlation.npz` — raw 데이터

**러닝 타임**: ~수십 초 (fits.open만, MCMC 없음).

---

## Experiment A : c_ics 하한 prior re-fit (수술)

**`diagnostic_A_ics_floor_fit.py`** — 기존 5-param fit에 `c_ics ≥ 0.1` 조건을 추가하여 병렬 재실행.

**논리**:
- 가설이 맞으면: Group B의 c_ics가 원래 저에너지에서 ~0에 가까웠을 것. Floor 걸면 ICS가 실제 값을 가질 수 밖에 없고, 잉여는 다른 컴포넌트가 흡수 → **GCE 저에너지 ratio가 떨어짐 (1.5+ → ~0.8-0.9)**.
- 가설이 틀리면: Floor가 유의미한 영향 없음, 또는 -2lnL이 현저히 악화 (fit이 억지로 맞춰지는 증상).

**Floor 값 조정 방법**:
```bash
export C_ICS_FLOOR=0.2
export FIT_SUFFIX=icsfloor0p20
bash run_diagnostics.sh
```

**산출물**:
- `GCE_model_{M}_haebarg_v_claude_{SUFFIX}.pkl` — floor 적용 fit 결과
- `diagnostic_A_comparison_{SUFFIX}.txt` — orig vs floor 비교 표
- `diagnostic_A_comparison_{SUFFIX}.png` — 3-row plot (SED / ratio / c_ics per-bin)

**러닝 타임**: ~15분 (기존 병렬 인프라 재사용).

---

## 결과 해석 매트릭스

| 시나리오 | Experiment B | Experiment A | 결론 |
|---|---|---|---|
| **1. degeneracy 가설 확정** | ρ(GCE,ICS) Group B 저E에서 현저히 높음 | Group B 저E ratio 1.5 → 0.8-0.9로 하락, Group A는 거의 변화 없음 | 저에너지 튐은 ICS-GCE degeneracy. "1-10 GeV offset (~0.87)은 진짜 systematic, 저에너지 튐은 인공물". XV/XLIX를 북쪽 환경에서 쓸 때 ICS에 soft prior 필요. |
| **2. 반만 맞음** | Corr 유사 | Group B ratio 조금 떨어짐 but A도 영향 받음 | Degeneracy가 **부분적** 원인. Spectral prior 부재가 공범. Cholis의 spectral prior (BPL index) 도입 실험 필요. |
| **3. 가설 틀림** | Corr 양 그룹 유사 | Floor 걸어도 ratio 안 떨어짐 or -2lnL 크게 악화 | 저에너지 튐은 실제로 **Group B 물리 모델 자체의 저에너지 diffuse 차이** 반영. 이 경우 Group B는 "남쪽/전체"에 부적합, Group A가 더 "정직한" 결과. |

---

## 사용법

```bash
bash /home/claude/pipeline_claude_v0/run_diagnostics.sh
```

또는 단계별:
```bash
python diagnostic_B_spatial_corr.py                           # ~수십 초
python diagnostic_A_ics_floor_fit.py X XV XLVIII XLIX LIII    # ~15 분 (병렬)
python diagnostic_A_compare.py                                # <1 초
```

다른 floor 값 실험:
```bash
export C_ICS_FLOOR=0.3
export FIT_SUFFIX=icsfloor0p30
python diagnostic_A_ics_floor_fit.py X XV XLVIII XLIX LIII
python diagnostic_A_compare.py
```

---

## 추가 실험 (필요 시)

가설 확정 시 후속:
- **spectral prior 실험**: c_gce에 BPL-like shape prior (index=-1.42 around E<E_break, -2.63 above). Sanghwan과 저자 사이 차이의 정확한 지점 진단.
- **5-param → 6-param**: pion + bremss를 별도 coefficient로 풀어 c_gas coupling을 제거. Cholis Eq. 7 vs 개별 피팅 비교.

Experiment B 상관관계 계산은 빠르니 bubble-GCE, iso-GCE 같은 다른 pair도 같이 보여 degeneracy 전체 지형을 파악할 수 있음.
