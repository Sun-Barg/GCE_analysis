# Claude v0 Pipeline — Cholis+2022 (arXiv:2112.09706) 5-model 재현

## 목표
Cholis et al. 2022의 5개 best-fit 모델 (X, XV, XLVIII, XLIX, LIII) 의 GCE SED 재현 및
저자 결과 (`Figures_12_and_14_GCE_Spectra/GCE_Model{name}_flux_Inner40x40_masked_disk.dat`) 와의 직접 비교.

## 파일 구성
```
pipeline_claude_v0/
├── config.py                       # 모든 경로/상수 (서버 의존)
├── prepare_masks.py                # Cholis Zhong&Cholis 2024 mask + fliplr 보정 + disk mask 결합
├── stage1_maps.py                  # gtsrcmaps + gtmodel (한 모델씩 처리, argv 인자)
├── stage1_parallel.py              # ⭐ Stage 1 병렬 런처 (5 모델 subprocess 동시 실행)
├── stage2_fit.py                   # BinLikelihood 클래스 + serial 실행 버전
├── stage2_parallel.py              # ⭐ 70 task (model × bin) / 64 worker Pool 실행
├── compare_to_cholis.py            # 5-panel 비교 plot + summary table
├── run_pipeline.sh                 # 직렬 실행 (디버그용)
└── run_pipeline_parallel.sh        # ⭐ 병렬 실행 (권장, 64 core 활용)
```

## 병렬화 설계

### Stage 1 (gtsrcmaps + gtmodel)
- **이슈**: GtApp은 C++ 확장 기반이라 fork-unsafe (`multiprocessing.Pool(fork)` 사용 금지).
- **해결**: `subprocess.Popen`으로 5 모델을 **독립 프로세스**로 동시 실행.
- **리소스**: 각 gtsrcmaps 호출 ~0.5-2GB RAM + 1-2 core (올바른 `LD_LIBRARY_PATH` ordering 기준).
  - 5 동시 × ~2GB = ~10GB 최악, 5 × 2core = 10core.
- **속도**: 125분 직렬 → **~25분** 병렬 (5×).

### Stage 2 (emcee MCMC)
- **이슈**: 5 모델 × 14 bin = 70개 독립 task. 각 bin fit은 완전 독립적.
- **해결**: `multiprocessing.Pool(64, initializer=_worker_init)` + `imap_unordered(chunksize=1)`
  - 각 worker가 **한 번만** ccube/expcube/mask/external constraints 로드 (initializer).
  - 각 task는 (model, ebin) → 해당 component FITS만 열고 emcee serial 실행.
  - load balancing: chunksize=1 + imap_unordered (빠른 bin이 먼저 끝나면 다른 task 받음).
- **리소스**:
  - worker당 공유 데이터 ~100MB (ccube + expcube + mask).
  - 64 workers × ~150MB = ~10GB (fork의 copy-on-write로 실제는 더 작음).
  - fit 시 BinLikelihood 객체 ~20MB × 64 = ~1.3GB.
- **속도**: 직렬 500분 → **~10-15분** 병렬 (task당 ~7-10분, 70 tasks / 64 workers = 1.1 wave).

### 기타 최적화
- `scipy.special.gammaln` + `gammaln(obs+1)` 사전 계산 → hot loop는 순수 numpy 산술 (~88× 속도)
- `solid_angle_per_pixel` 벡터화 + `sr_per_pixel.npy` 캐시 (이미 존재함)
- BinLikelihood `__init__`에서 mask boolean 적용 미리 수행 (hot loop는 1D 배열 연산)

## 출력 (`$WORK_DIR/haebarg_v_claude/`)
- `GCE_model_{M}_haebarg_v_claude.dat`     — `[E, flux_best, std, lower, upper]` (Cholis 포맷)
- `GCE_model_{M}_haebarg_v_claude_logL.txt` — per-bin max log L (길이 14)
- `GCE_model_{M}_haebarg_v_claude.pkl`     — 전체 fit 결과 (coef, std, GCE_avg, total -2lnL)
- `cholis_full_mask_14bin_400x400.npy`     — fliplr 보정된 Cholis mask × disk mask
- `compare_5models_vs_cholis.png`          — 5-panel SED + ratio plot
- `compare_5models_summary.txt`            — 비교 표
- `log_stage{1,2}_parallel.txt`            — 실행 로그
- `log_stage1_{M}.txt`                     — 모델별 stage 1 로그 (진단용)

## 실행

**권장 방식 (병렬, 전체 ~50분 예상):**
```bash
bash /home/claude/pipeline_claude_v0/run_pipeline_parallel.sh
```

**단계별 실행:**
```bash
python prepare_masks.py
python stage1_parallel.py X XV XLVIII XLIX LIII    # ~25 min (5 subprocess 동시)
python stage2_parallel.py X XV XLVIII XLIX LIII    # ~15 min (70 task / 64 worker)
python compare_to_cholis.py
```

**직렬 디버그 실행:**
```bash
bash /home/claude/pipeline_claude_v0/run_pipeline.sh
```

## 실행 시간 요약

| 단계 | 직렬 | 병렬 (64 core) | 속도 향상 |
|---|---|---|---|
| Stage 1 (maps) | 125 min | ~25 min | 5× |
| Stage 2 (MCMC) | 500 min | ~15 min | ~30× |
| Compare | <1 min | <1 min | — |
| **Total** | **~10.4 hr** | **~45 min** | **~14×** |

## 이전 시도 대비 핵심 변경점
1. **Mask**: v9.7 v2 검증된 Cholis Zhong&Cholis 2024 mask + `np.flip(axis=2)` 보정 사용
2. **Asymmetric chi² 부호 수정**: `model > data → upper_err`, `model < data → lower_err`
3. **Error floor**: `min(σ) = 5% × flux` — c_iso, c_bub 발산 방지
4. **emcee 100×1500×500 burn-in** (Sanghwan 1000×400 대비 더 보수적)

## 정직한 예상
이전 3번 시도 (해밝 v8.x/v9.x, Sanghwan port) 모두 1–10 GeV ratio ≈ 0.80–0.87로 수렴.
이 파이프라인도 동일한 **~13% offset에 정착할 가능성이 높음** (방법론적 차이로 확인됨).

가치:
- (a) chi² 부호 수정 + error floor가 c_iso 수렴 패턴을 바꾸는지 확인
- (b) 5 모델 모두 동일하게 ~0.87 offset이면 → systematic이 **모델-독립적**임을 증명
- (c) XLIX (top-tier)가 가장 좋은 -2lnL을 보이는지 → 저자 주장 독립 검증
- (d) 병렬화 덕분에 **여러 설정 실험을 빠르게 반복**할 수 있음 (기존: 하루 1회, 신규: 하루 10회+)

## 병렬 실행 모니터링
```bash
# Stage 2 worker 실행 중 상태
ps -eo pid,pcpu,pmem,cmd --sort=-pcpu | grep stage2_parallel | head

# 진행 상황 실시간
tail -f $OUT_DIR/log_stage2_parallel.txt

# 메모리
free -g
```
