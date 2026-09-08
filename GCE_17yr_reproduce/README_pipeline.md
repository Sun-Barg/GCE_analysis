# 17yr GCE Pipeline — File Registry
 
**Last updated**: 2026-05-23
**Maintainer**: haebarg (Jeonbuk National University, Prof. Seodong Shin advised)
**Working directory**: `~/GCE-Chi-square-fitting/GCE_17yr_reproduce/`
 
This document is the authoritative index of all production code files in
the 17yr GCE analysis pipeline. It exists so that any new conversation,
hand-off, or audit can quickly understand what each file does, which
other files it depends on, and where the live state currently stands.
 
The pipeline reproduces the analysis of **Cholis et al. 2022
(arXiv:2112.09706)** using ~17.5 years of Fermi-LAT data and the FL16Y
point-source catalog (`gll_psc_v40.fit`, LAT 16-Year Source List,
arXiv:2602.22148, a 5FGL precursor). The analysis extends Sanghwan Kim's
2025 master's thesis (16-year analysis) to 17 years and from 4FGL-DR4 to
FL16Y. Sanghwan's code lives at `/home/sanghwan/FermiLAT/Sanghwan/` and
serves as **implementation reference only** (Sanghwan has left the
project; the methodological reference is Cholis 2022 alone).
 
---
 
## Project layout
 
```
~/GCE-Chi-square-fitting/GCE_17yr_reproduce/
├── cholis_masking.py                       <- shared module (Phase 2)
├── prepare_common.py                       <- main data prep (Phase 2)
├── prepare_one_roi_cov.py                  <- per-ROI cov prep worker (Phase 2)
├── launch_all_roi_prep.py                  <- 22-ROI prep launcher (Phase 2)
├── run_one_model.py                        <- per-model main-fit worker (v3)
├── launch_all_models.py                    <- 80-model launcher (v3.1)
├── run_one_roi_cov.py                      <- per-ROI cov MCMC worker (Pool removed, phase-split)
├── run_one_roi_cov_wrapper.py              <- cov phase-split wrapper (prepare/mcmc subprocess)
├── launch_all_rois.py                      <- 22-ROI cov MCMC launcher (RUNNER=wrapper)
├── make_gce_template.py                    <- MAIN GCE NFW² template builder (norm-bug fix, NEW 2026-05-19)
├── build_mapcubes.py                       <- raw GALPROP → fermitools MapCube reproducer (provenance, NEW 2026-05-23)
├── make_perroi_ccube.py                    <- per-ROI ccube builder (cov notebook cell 6 정식화)
├── make_wimp_map_per_roi.py                <- per-ROI wimp_map builder (NFW² translate)
├── build_cov_matrix.py                     <- cov matrix assembly
├── validate_cov_matrix.py                  <- cov matrix sanity checks
├── launcher_watchdog.sh                    <- main 80-model launcher watchdog
├── launcher_watchdog_cov.sh                <- cov 22-ROI launcher watchdog
├── GC_analysis-60x60-models_17yr_v13.ipynb <- legacy notebook (main fit)
├── GCE_covariance_marix_calculation_17yr_v13.ipynb  <- legacy (cov)
├── GCE_17yr_visualization.ipynb            <- result visualization
├── GC_analysis_FL16Y/                      <- workdir for analysis outputs
│   ├── *.fits                              <- CCUBE / LTCUBE / expcube / gtsrcmaps / gtmodel
│   └── Model/                              <- XMLs / masks / source_classification.npz
├── results_17yr/                           <- final per-model .dat outputs (80/80)
├── results_cov_17yr/                       <- per-ROI cov .dat + cov matrix npz (22/22)
└── logs/cov_prep/                          <- per-ROI prep logs
```
 
---
 
## Phase 2 + v3 / v3.1 pipeline files
 
Six files were redesigned in the recent refactoring cycle to eliminate
the **stale-file silent reuse vulnerability** (mid-run SIGKILL leaving
partial intermediate files, then the next launcher pass reusing them as
if they were healthy). Every step now ships with an integrity check; a
file that fails its check causes an explicit FATAL rather than silent
poisoning of downstream analysis.
 
### 1. `cholis_masking.py` — shared module
 
**Purpose**: shared utilities imported by every Phase 2 / v3 script.
 
**Contents**:
- `CHOLIS_TABLE_III` (Cholis 2022 Table III, 14 energy bins, 2 mask radii each)
- `masking(significance, locations, energy, image_file, mask_scale=1.0)`
  — per-energy-bin circular point-source mask (`mask_scale=1.0` is paper-strict)
- Coordinate transforms: `equatorial_to_galactic`, `galactic_to_equatorial`
- 9 integrity-check helpers, all returning `(ok: bool, msg: str)`:
  - `verify_fits(path)` — generic FITS sanity (opens + `verify('exception')`)
  - `verify_cube(path, expected_nebins=14, expected_xy=None)` — 3D FITS cube
  - `verify_event_file(path, min_events)` — gtselect / gtmktime output
  - `verify_sc_merged(path)` — merged SC FT2 (TIMESYS, TSTART/TSTOP, span ≈ 17yr)
  - `verify_ltcube(path)` — livetime cube
  - `verify_xml(path, min_sources=0)` — XML model file
  - `verify_mask_npy(path, expected_shape)` — mask .npy
  - `verify_bin_def(path, expected_nebins=14)` — bin_definitions.fits
  - `verify_dat(path, expected_nbins=14)` — final per-model .dat (added in v3)
**Verifier contract**: `(ok, msg)`. Caller decides — `ok=True` → safe to skip
rebuild; `ok=False` → abort with explicit message (no silent stale reuse).
 
### 2. `prepare_common.py` — main data prep (16 steps, ~1× per workdir)
 
**Purpose**: reproduces the main fit notebook cells 5–25 as a sequential
script with integrity-checked skip.
 
**Outputs** (in working dir + `GC_analysis_FL16Y/[Model/]`):
- Steps 1–4: photon listfile, SC merged FT2, bin_definitions.fits, gtselect / gtmktime
- Steps 5–9: gtbin (GC_ccube), gtltcube (Allsky_ltcube), gtexpcube2 ×2 (CENTER, EDGE)
- Step 10: SourceList → `GC_model_FL16Y.xml` (PSC + diffuse, main GC ROI)
- Step 11: iso/gal prune → `GC_psc_model_FL16Y.xml` (PSC only)
- Step 12: source classification → `source_classification.npz`
  (sig_ra_dec_values, not_sig_ra_dec_values from main psc XML)
- Step 13: main psc mask → `GC_mask_60x60_definitions_FL16Y.npy` (14, 600, 600)
- Step 14: disk mask → `GC_disk_mask_60x60_definitions.npy`
- Step 15: empty model XML → `empty_model.xml`
- Step 16: model-independent template XMLs → `GC_{GCE,isotropic,fermi_bubble}_model.xml`
**CLI**:
```bash
python3 prepare_common.py
python3 prepare_common.py --force-step 7
python3 prepare_common.py --force-step 7,8,9
python3 prepare_common.py --force-all
```
 
**Skip policy**: file exists + verifier OK → skip; exists + verifier FAIL → `sys.exit(2)`;
absent → build → re-verify. All XML / multi-output writes use atomic `.tmp + os.rename`.
 
**Status (2026-05-14)**: complete. All 16 outputs verified-OK on disk.
 
### 3. `prepare_one_roi_cov.py` — per-ROI cov prep worker
 
**Purpose**: reproduces cov notebook cells 10, 14 per-ROI portions only.
 
> ⚠️ **Pre-prerequisite (2026-05-19)**: `GCE_template_NFW2.fits` must be
> a VALID template (normalization integral 1.0). Run
> `python make_gce_template.py --check` first; if it reports BUGGY
> (integral 0.7259), run `python make_gce_template.py` BEFORE
> prepare_common.py / the 80-model fit. The main pipeline references
> this file but does not generate it.
 
**Prerequisites** (from `prepare_common.py`):
- `Allsky_ltcube_17yr_front_clean.fits` (Step 7)
- `GC_ccube_17yr_front_clean.fits` (Step 6)
- `bin_definitions.fits` (Step 3)
- `Model/GC_psc_model_FL16Y.xml` (Step 11)
- `Model/source_classification.npz` (Step 12)
**3 steps per ROI**:
1. gtexpcube2 (CENTER, xref=roi) → `GC_expcube_center_17yr_front_clean_l{roi}.fits`
2. SourceList + iso/gal prune → `Model/GC_model_FL16Y_l{roi}.xml` + `Model/GC_psc_model_FL16Y_l{roi}.xml`
3. Per-ROI PSC mask → `Model/GC_mask_60x60_definitions_FL16Y_l{roi}.npy`
**Legacy quirk preserved** (cov cell 14 L19–L20): the per-ROI mask uses
the MAIN source classification from `source_classification.npz` (not the
per-ROI XML). Consequence: per-ROI mask content is bit-identical to the
main psc mask. The per-ROI XML produced in Step 2 IS used downstream
by `run_one_roi_cov.py` for gtsrcmaps/gtmodel. See the GOTCHA block in
`build_psc_mask_roi()` for the fix point if this behavior is ever
revisited.
 
**CLI**:
```bash
python3 prepare_one_roi_cov.py 25
python3 prepare_one_roi_cov.py -70 --force-step 2
python3 prepare_one_roi_cov.py 25 --force-all
```
 
**Valid ROI**: `roi != 0`, `abs(roi) ∈ [20, 70]`, `roi % 5 == 0` → **exactly 22 ROIs** (Cholis 2022 L1637 `20°≤|ℓ|≤70°`; ±20 INCLUDED). `assert len(ALL_ROIS)==22`.
 
**Status (2026-05-14)**: 22 ROI all complete via `launch_all_roi_prep.py`.
 
### 4. `launch_all_roi_prep.py` — 22-ROI prep launcher
 
**Purpose**: spawns up to N concurrent `prepare_one_roi_cov.py <roi>`
subprocesses with continuous slot refill.
 
**Differences vs `launch_all_rois.py` (cov MCMC launcher)**:
- Default `--workers 4` (prep is light, ~1 GB/process) vs cov MCMC's 2
- No move-to-results-dir (prep outputs stay in workdir for cov MCMC)
- Prerequisite check at startup (5 prepare_common outputs)
- rc=2 → `permanent_failed` (stale-file FATAL is non-retryable)
- `--worker-args "--force-step 3"` passthrough to worker
- Logs in `logs/cov_prep/roi_l{ROI}.log` (auto-created)
**Status (2026-05-14)**: completed successfully, 22/22 ROIs.
 
### 5. `run_one_model.py` (v3) — per-model main-fit worker
 
**Purpose**: per-model subprocess for the main 80-model fit. Reproduces
the legacy notebook cell 30 logic for a single GDE model passed as argv.
 
**Steps**:
1. Build `GC_model{M}_test.xml` (PSC + 6 new sources)
2. Build `GC_Extended_model{M}_test.xml` (6 sources only)
3. gtsrcmaps × 2 (convol=yes / convol=no)
4. Per-component XMLs (pion/bremss/ics) + gtmodel × 6
5. Model-independent template XMLs (GCE/iso/bubble) + gtmodel × 6
6. Data load (CCUBE, exp, mask, components, observed counts)
7. External constraints (bubble_constraints.txt, iso_constraints_full_err.txt)
8. Likelihood class with Bug A + B + C patches
9. MCMC per bin (100 walkers, 1000 steps, 400 burn-in, **serial — Pool removed**, SIGKILL fix)
10. Save `.dat` / `_likelihood_value` / `_fit.npz`
**v3 changes** (Phase 2 integration, 2026-05-14):
- imports verifiers from `cholis_masking`
- 2 new helpers: `_check_or_abort(path, verifier_fn, label)` and
  `_verify_built_or_abort(...)` — same skip-or-FATAL policy as
  prepare_common.py Step.run()
- 8 skip sites converted: final `.dat`, 5 per-model XMLs, gtsrcmaps × 2,
  gtmodel × 12. Atomic `.tmp + rename` writes for all XMLs.
- Final `.dat` stale → auto-delete + rerun (since fresh rerun re-derives
  from clean intermediates). Other intermediates stale → FATAL.
**CLI**:
```bash
python run_one_model.py X       # one model, e.g. "X"
```
 
### 6. `launch_all_models.py` (v3.1) — 80-model launcher
 
**Purpose**: spawns up to N concurrent `run_one_model.py M` subprocesses
with continuous slot refill + auto-move completed `.dat` to `results_17yr/`.
 
**v3.1 features**:
- **`cleanup_stale_intermediates()` at startup** (auto-runs by default;
  `--no-cleanup` to skip): scans every per-model intermediate, verifies
  each, deletes ONLY stale ones (healthy ones preserved so workers can
  skip). Also handles 6 shared template gtmodel files (if any stale,
  deletes all 6 for consistency).
- **rc=2 (worker FATAL) → `permanent_failed`** (non-retryable).
- **rc=0 + is_complete check** (both required for "done") — tighter than
  v3.0 which only checked is_complete.
**CLI**:
```bash
nohup python launch_all_models.py > launcher_$(date +%m%d_%H%M).log 2>&1 &
disown
 
python launch_all_models.py --workers 6 --max-runtime-hr 24
python launch_all_models.py --models I,II,X --workers 4
python launch_all_models.py --no-cleanup           # rare, debug only
```
 
---
 
## Cov pipeline (2026-05-18 FINAL — phase-split wrapper)
 
> 상세: `REF_cov_pipeline_17yr_FINAL.md`. 구
> `REF_cov_subprocess_pipeline.md` (3-layer/tqdm 가설) 는 폐기됨.
 
### `run_one_roi_cov.py` + `run_one_roi_cov_wrapper.py` + `launch_all_rois.py`
 
Per-ROI cov MCMC. **SIGKILL fix 적용**: fermitools (gtsrcmaps/gtmodel)
후 같은 프로세스 emcee 진입 시 외부 SIGKILL (12yr lesson #10 일반화,
main Jobs 3-8 확정). Fix = (a) Pool 제거 → serial emcee, (b) prepare
(fermitools) 와 mcmc 를 별도 subprocess 로 분리 (`run_one_roi_cov_wrapper.py`
가 `RUN_PHASE=prepare/mcmc` env 로 2회 호출). `launch_all_rois.py`
`RUNNER_SCRIPT='run_one_roi_cov_wrapper.py'`.
 
**메모리 (17yr 실측, 2026-05-18 단위검증)**: cov gtsrcmaps RSS
**~1.5 GB/ROI** (peak sum_rss 1.5GB, max_proc 1.4GB). 기존 "30-50GB"
는 16yr/타 조건 추정치로 17yr 과 불일치. load 1.04 = ROI당 ~1코어.
→ `--workers 8` 안전 (메모리·CPU 제약 배제). per-ROI ~87분
(prepare 24 + mcmc 63). 출력 `results_cov_17yr/`.
 
Prerequisites: `prepare_one_roi_cov.py` 산출 + `make_perroi_ccube.py`
(per-ROI ccube 22) + `make_wimp_map_per_roi.py` (wimp_map 22).
 
### `launcher_watchdog_cov.sh`
 
`launcher_watchdog.sh` 에서 CONFIG+glob 만 cov 용 치환 (로직 동일,
main 14회 자동복구 검증 승계). launcher silent death (시작 ~2시간 후,
원인 미상·OOM 아님) 시 60s polling 으로 orphan 회수 + 재시작,
22/22 도달 시 자동 종료. 본실행 2026-05-18 에서 1회 사망·무손실 복구.
 
### `make_perroi_ccube.py` — per-ROI ccube (cov notebook cell 6 정식화)
 
cov 노트북에 per-ROI ccube 생성 cell 부재 (cell 0 markdown 만 언급).
기존 20개는 미첨부 코드로 생성·±20 누락 상태였음. 이 스크립트가
누락 cell 6 을 정식화: gtbin 파라미터를 기존 `l-25` 헤더에서 byte
역추출 (CCUBE, 600×600, binsz 0.1, GAL/CAR, **xref=roi**, yref=0,
ebinfile=bin_definitions.fits, 14 bins). idempotent. 음수 ROI 는
`--rois=-20,20` 등호 형식 필수 (argparse 옵션 오인 회피).
 
### `make_wimp_map_per_roi.py` — wimp_map per ROI (NFW² translate)
 
ROI별 NFW² LOS 재적분 (translate) — Cholis 2022 L1477 "GCE is the
only template translated". 정규화 합=1.0000 검증. (cov 노트북
cell 9 의 16yr "header 재해석" 방식 아님 — 그 방식은 Cholis 와
불일치, `REF_GCE_covariance_16yr_SUMMARY.md` 🔴 정정 블록 참조.)
fermitools 無 → SIGKILL 위험 0.
 
### `make_gce_template.py` — MAIN GCE spatial template (NEW 2026-05-19)
 
Generates `./GCE_template_NFW2.fits`, the GC-centered NFW² LOS template
that the **main 80-model pipeline** references as the GCE SpatialMap
(`prepare_common.py` L114 / `run_one_model.py` L103, `WIMP_MAP_PATH`).
 
**Why it exists**: the main pipeline *referenced* this file but **no
pipeline code generated it** (verified 2026-05-19: zero writeto/np.save
for it in any .py/.ipynb). It came from a now-lost port of the 12yr
`Wimp_map_creation.ipynb`, whose normalization line — copied verbatim
into every generating cell — is a **bug**:
 
```
norm = np.sum(counts_map[0]) * (np.pi/180)**2 * (0.01)**2   # BUG
```
- `counts_map[0]` is the 0th ROW (top edge, |b| large) where NFW²≈0;
  the intent was `np.sum(counts_map)` (whole map).
- `(0.01)**2` is the 0.01-deg highresol pixel; the delivered file is
  0.1 deg, so the pixel term must be `(0.1)**2`.
The delivered (buggy) file had normalization integral **0.725858**
(should be 1.0). Confirmed 2026-05-19. The **correct** normalization —
matching the cov pipeline's `make_wimp_map_per_roi.py` (cov-notebook
cell 9, Cholis-correct, integral 1.0):
 
```
norm = np.sum(raw) * (np.pi/180)**2 * (0.1)**2   # integral = 1.0
```
 
Cross-check: regenerated GC-centered template sum = 3.282807e+05,
**identical** to cov `wimp_map_l-20.fits` sum 3.282807e+05 — same NFW²
profile, differing only by translation (main stays GC-centered per
Cholis L1477 "GCE is the only template translated"; cov shifts to ROI
longitude). NFW² core is verbatim from `make_wimp_map_per_roi.py`
L64-93 (= cov-notebook cell 6): rho_s=0.2710150839697834, r_s=20,
γ=1.2, r_0=8.5, cutoff 120°, inner reg at angle(0.05,0.05).
 
**Safety**: idempotent (skip if integral≈1.0 unless `--force`); the
buggy/old file is moved to `GCE_template_NFW2.fits.buggy_bak_<ts>`
before overwrite (preserves old 80-model provenance); post-write verify
(shape, finite, integral≈1.0). Modes: `--check` (status only), default
(regenerate if missing/buggy), `--force`.
 
```bash
python make_gce_template.py --check   # report integral (buggy=0.7259)
python make_gce_template.py           # backup buggy → write fixed (~3 min)
```
 
> ⚠️ **SUPERSEDED (2026-06-08) — DEAD, do not act on**: this "BUGGY
> template → INVALID → Model X rank 1→27 regression → must rerun" note is
> PRE-FLIP. The full 80-model 17yr main fit HAS been rerun after the
> MapCube axis=2 flip (`data[:,:,::-1]` on pion/bremss/ics; memory
> #16/#23). The current ranking is VALID — our 17yr best = Model X (#1),
> Cholis-XLIX → #11, Spearman ρ = 0.97 vs Cholis. There is no "rank 1→27
> regression". Kept only so the stale phrasing is recognized as stale.
 
### `build_mapcubes.py` — raw GALPROP → fermitools MapCube reproducer (NEW 2026-05-23)
 
Builds `./MapCubes_v2/{bremss,pion,ics}_mapcube_model<ROMAN>.fits` from
the Cholis Zenodo raw GALPROP maps
(`../GCE_TEMPLATES_FILES_v3/GALACTIC_DIFFUSE_EMISSION_MAPS_0p25deg/{bremss,pi0,ICS}_<hash>_Map_flux_E_50-814008_MeV_InnerGalaxy_60x60.fits`)
for all 80 GDE models × 3 components.
 
**Why it exists**: the original 80-model `./MapCubes/` files were
produced by a now-lost port of Sanghwan's `Converting_into_mapcube_test.ipynb`
CELL 5. Reproducibility was 0 — no script in the working dir generated
them, and Sanghwan's notebook was a 41-cell debugging notebook in which
CELL 5 was just one cell. `build_mapcubes.py` formalizes the conversion
as a self-contained, provenanced, idempotent, verifiable single-file
script (no fermitools dependency).
 
**Conversion logic** (verified against existing 80 MapCubes,
per-pixel binary identical for sample models X / XIII / II, 2026-05-23):
 
```
dN/dE [photons cm^-2 s^-1 sr^-1 MeV^-1]
    = raw_data [GeV cm^-2 s^-1 sr^-1]  (= E^2 · dPhi/dE)
      * (1e-3 / E_GeV^2)
```
 
applied per-bin with `E_GeV[i] = sqrt(Emin[i] * Emax[i]) * 1e-3`. The
factor is **model-independent multiplicative**; no spatial flip; raw
WCS preserved (CDELT1=-0.25, CRPIX1/2=120.5, GLON-CAR / GLAT-CAR
projection — already paper-faithful: left edge = +30 deg l, right edge
= -30 deg l, standard galactic convention).
 
**Retraction (2026-05-23)**: the earlier
hypothesis that the MapCube conversion stage was the root cause of
(β) ranking inversion ("1.76× model-dependent jump from raw to
MapCube") is **falsified**. The "1.76×" was a 38-bin total-integral
ratio measured by `DIAG_phase3_raw_template_audit.py`; that measurement
itself is correct, but the **interpretation** as a conversion-logic
defect was wrong. Per-bin (where the spectrum-shape weighting cancels)
the stage A (raw) ↔ stage B (our MapCube) per-bin sum ratio is
**1.0000 exactly** for every model × bin × component pair tested
(38 bins, 3 components, 3 models, all `max_rel = 0` after float32 cast).
 
**Implication for the existing pipeline**: no 80-model rerun is needed
on the MapCube account. `./MapCubes/` files are paper-faithful (single
multiplicative factor 1e-3/E_GeV²) and fit results are uncompromised by
spatial orientation. **Corrected 2026-06-08:** the "mask convention V44
(`np.flip(mask, axis=2)`)" clause is STALE — the fit code has NO mask
flip; the psc mask is built in the CCUBE frame and applied as-is (un-flipped
mask covers 88% of source cores vs 58% if flipped — do NOT flip the mask;
memory #22). And "(β) ranking inversion" is DEAD, not an open unidentified
phenomenon: post-flip (`data[:,:,::-1]` on pion/bremss/ics) the
17yr ranking is valid (best = Model X, #1).
 
**Sanghwan CELL 5 caveat**: that notebook applies
`np.flip(input_model[0].data, axis=2)` to match an **external base
template** (`Galprop_file/bremss_mapcube_modelI.gz`) that happened to be
in flipped orientation. The flip is a base-template-matching
operation, not a property of the raw-to-MapCube conversion itself.
`build_mapcubes.py` correctly omits the flip; raw WCS is already
paper-faithful and our pipeline's mask/CCUBE all use that same
non-flipped convention.
 
**Spatial check on ICS** (2026-05-23): ICS raw is left-right symmetric
about the GC (smooth ISRF + CR electron distribution), so flip-vs-noflip
is undetectable from ICS alone. The flip orientation was established
via bremss / pion (H I, H₂ asymmetric tracers). All three components
verify after the no-flip convention is applied.
 
**Verification mode** (`--verify-against-existing`): for each newly
built MapCube, compares against `./MapCubes/<comp>_mapcube_model<M>.fits`
on (a) per-bin sum ratio (tolerance 1e-5 around 1.0) and (b) max
per-pixel relative diff (tolerance 1e-6). Sample run 2026-05-23:
9/9 verify OK (X, XIII, II × bremss, pion, ics; all median_ratio=1.000000,
all max_rel=0.00e+00).
 
**CLI**:
```bash
python build_mapcubes.py --models X,XIII,II --verify-against-existing
python build_mapcubes.py --verify-against-existing      # all 80
python build_mapcubes.py --check                        # validate ./MapCubes_v2/ only
python build_mapcubes.py --force                        # rebuild even if exists
```
 
**Output FITS structure**: HDU0 PrimaryHDU (float32, shape (38, 240, 240),
BUNIT='photon/cm2/s/MeV/sr', WCS verbatim from raw), HDU1 ENERGIES
BinTableHDU (38 E values, MeV, copied verbatim from raw). Provenance
written into HDU0 HISTORY: raw filename + md5, build date, conversion
formula, "Spatial: raw orientation preserved (no flip)".
 
> ✓ **Status**: `./MapCubes/` (production, 80×3 = 240 files) remains
> the active reference for all fits. `./MapCubes_v2/` (built by this
> script) is a verified reproducer for provenance / future re-analysis,
> not a swap-in replacement.
 
### `build_cov_matrix.py` + `validate_cov_matrix.py`
 
22 ROI fit.npz → 14×14 systematic cov. 순수 numpy. docstring
`(20,)→(22,)` 정정 완료 (2026-05-18). 검증 (2026-05-18): cond
2.35e5, sigma_sys peak 1.25e-6 @0.31GeV → 1.49e-7 @35GeV (Calore+
1409.0042 패턴 일치), symmetric, diag≥0.
 
---
 
## ✅ Resolved: SIGKILL pattern (2026-05-18)
 
### Root cause (confirmed)
 
`launch_all_models.py` / cov workers received external SIGKILL because
**fermitools (`GtApp`: gtsrcmaps/gtmodel) leaves fork-unsafe state
(GALPROP / fits mmap) in the Python process; entering emcee MCMC over
it — whether via Pool fork or same-process — triggers external SIGKILL.**
This is the generalization of 12yr lesson #10. Earlier "VS Code tqdm
PTY overflow" hypothesis (`REF_cov_subprocess_pipeline.md`) was wrong.
 
Evidence (main pipeline Jobs 3-8, 2026-05-14/15):
 
| Job | condition | result |
|---|---|---|
| 3 | Pool, fresh srcmap | SIGKILL (MCMC entry) |
| 4 | Pool removed, srcmap reused | ✅ 67.4 min |
| 5 | Pool removed, fresh srcmap, single proc | SIGKILL (pre-MCMC) |
| 6 | Pool removed, srcmap reused | ✅ 68.3 min |
| 8 | wrapper (prepare/mcmc split), fresh | ✅ 94.7 min |
 
PBS mem always <2GB/20GB (never OOM); dmesg clean. The 2026-04-29
dmesg OOM entries were unrelated historical.
 
### Fix (applied to both main and cov)
 
1. **emcee Pool removed** → serial `EnsembleSampler(...)`.
2. **prepare (fermitools) and mcmc (emcee) split into separate
   subprocesses** via `RUN_PHASE` env + a wrapper
   (`run_one_model_wrapper.py` / `run_one_roi_cov_wrapper.py`).
   prepare subprocess exits after Step 5; fermitools state dies with
   it; mcmc runs in a fresh process.
3. **`launcher_watchdog*.sh`** auto-recovers the (unrelated) launcher
   silent-death (still occurs ~2h in, cause unknown, NOT OOM/PBS) —
   60s polling, orphan recovery, restart, auto-exit at target count.
### Current results state (2026-05-18)
 
- `results_17yr/*.dat`: ✅ **VALID — rerun complete (post-flip)**.
  ~~INVALIDATED 2026-05-19~~ is SUPERSEDED (Corrected 2026-06-08). The
  buggy GCE template was regenerated (integral 1.0, cov-consistent) and
  the full 80-model main fit was rerun after the MapCube axis=2 flip.
  Current ranking is VALID: our 17yr best = Model X (#1), Cholis-XLIX →
  #11, Spearman ρ = 0.97. No further rerun is needed on the template
  account (see the 2026-06-08 correction above). (launcher_watchdog was validated: 14
  restarts, zero data loss.)
- `results_cov_17yr/*.dat`: **22 / 22** (cov MCMC complete;
  launcher_watchdog 1 restart 2026-05-18 03:44, zero data loss,
  exited `[success] 22/22` 08:57)
- `GCE_systematic_covariance_matrix_17yr.npz`: built & validated
  (cond 2.35e5, sigma_sys 1.25e-6→1.49e-7, Calore+ pattern ✓)
- 22 ROI prep: complete and verified
- Methodology confirmed against Cholis 2022 原文: 22 ROIs (L1637),
  GCE-only translate (L1477), per-ROI ccube xref=roi (L1462).
  `make_perroi_ccube.py` formalizes the missing cov-notebook cell 6.
### OPEN residual — c_iso → 0 at mid-E (NOT "not a bug", NOT accepted)
 
**Corrected 2026-06-08.** The old header "(not a bug)" and "the author accepted
V48 conclusion (proceed)" are RETIRED — the author never agreed; c_iso → 0 is an
OPEN, unreconciled residual.
 
**17yr determination** (Plot 6 = `06_coefficient_comparison.png`): c_iso
rails toward the [0,∞) floor at MID energies and RECOVERS above ~10 GeV.
Model X: ~1–6 GeV (14-bin mean c_iso 0.28; healthy at bin 0 = 0.44).
Model XLIX: ~0.3–7 GeV / whole sub-10-GeV (mean 0.11; the one
cross-checked vs gcepy). NOT universal — Model I stays smooth & nonzero
(mean 1.93). Do NOT conflate this mid-E floor with (a) the separate
bin 0–1 FL16Y mask-aggression effect, nor (b) the 12yr-V48 "7/9 models
near boundary at 1–10 GeV" count below (a different track / metric).
 
12yr V48 decomposition (kept for record): Cash Poisson ~3.4e5/bin vs
chi²_iso dynamic range ~10 → external constraint effectively dead vs
Poisson dominance; reproduces 12yr V48. c-coeffs are
systematic-uncertainty indicators, not physical GDE calibration; GCE
flux envelope robust to c-swap. Rejected: chi² weight increase /
informative prior (paper deviation) / nsteps increase. Cholis 2022 is
the only methodological reference; paper-exact is mandatory unless
explicitly approved by the author.
 
## File version conventions
 
When updating any file, the convention in this project is:
- Increment the `[vN]` marker in the docstring's "changes" list
- Add a one-line entry describing the change
- Preserve the previous behavior unless explicitly documented otherwise
- Atomic writes (`.tmp + os.rename`) for any persistent output
When changing CONFIG values (paths, IRFS, DR_NUMBER, etc), keep them
byte-identical across:
- `prepare_common.py` CONFIG block
- `prepare_one_roi_cov.py` CONFIG block
- `run_one_model.py` CONFIG block
- The legacy notebooks' cell 3 (for reference parity)
---
 
## Key analysis-method constants (Cholis 2022 paper-exact)
 
- **Energy binning**: 14 bins, 0.274698 – 51.9312 GeV (`_EXTEND_ENERGY_RANGE=False`)
- **Energy bin centers**: geometric mean of edges, in GeV
- **IRFS**: `P8R3_CLEAN_V3`, evclass=256, evtype=1 (FRONT only)
- **MCMC**: 100 walkers × 1000 steps × 400 burn-in, **serial emcee (Pool removed — SIGKILL fix)**
- **Mask scale**: 1.0 (Cholis Table III strict)
- **DR number**: 4 (fermitools max; FL16Y format compatible with DR=4 parser)
- **Spacecraft**: merged FT2 from 17yr photon weeklies (w009–w934),
  preserving full SC_DATA header for time-system keys
---
 
## References
 
- **Primary methodology**: Cholis et al. 2022 (arXiv:2112.09706)
- **FL16Y catalog**: arXiv:2602.22148v2
- **Calore+ systematic study**: arXiv:1409.0042
- **Pedagogical**: Profumo 2019, Lisanti TASI 2015, Lin TASI 2019, Slatyer TASI 2016
- **GCE discovery**: Goodenough & Hooper 2009
---
 
## Quick-start: typical workflow
 
```bash
# (0) GCE spatial template — MUST be valid (integral 1.0) before anything
python make_gce_template.py --check    # if BUGGY (0.7259):
python make_gce_template.py            # backup buggy → write fixed (~3 min)
 
# (1) Main data prep — once per workdir
python prepare_common.py
 
# (2) Per-ROI cov prep — 22 ROIs in parallel
python launch_all_roi_prep.py --workers 4
 
# (3) Main 80-model fit — phase-split wrapper + watchdog (SIGKILL resolved)
unset DIAG_SAVE_CHAIN
nohup ./launcher_watchdog.sh > watchdog_console.log 2>&1 &
disown
 
# (4) Per-ROI ccube + wimp_map (cov prerequisites)
python make_perroi_ccube.py                          # 22 per-ROI ccube (누락만)
python make_wimp_map_per_roi.py --workers 8          # 22 wimp_map (NFW² translate)
 
# (5) Cov MCMC — 22 ROIs, phase-split wrapper + watchdog
#     launcher_watchdog_cov.sh CONFIG: WORKERS=8 TARGET_COUNT=22
unset DIAG_SAVE_CHAIN
nohup ./launcher_watchdog_cov.sh > watchdog_cov_console.log 2>&1 &
disown
 
# (6) Cov matrix assembly + validation
python build_cov_matrix.py --cov-dir results_cov_17yr --plot
 
# (7) Visualization
jupyter notebook GCE_17yr_visualization.ipynb
```
 
---
 
*This README is maintained as the single-source-of-truth file index.
When new files are added to the pipeline, or when the SIGKILL issue is
resolved, update both this file and the relevant docstrings.*
