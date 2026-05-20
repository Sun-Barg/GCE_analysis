# BIG Baseline Reproducibility Dossier

Date written: 2026-06-03 KST

Baseline label: `tentative_baseline`

Alias note: for further analysis, `tentative_baseline` means this
`calore_cdata_best_branch_20260601` branch: BIG/NFW/`b bbar`, true local
`kCIRELLI19`, TOA `phi = 0.732 GV`, `RhoSol = 0.385 GeV/cm^3`, and
`C_data`-only likelihood with no `C_model`.

Detailed audit file:

- `Codex_files/generated_outputs/tentative_baseline/README.md`

This documents the cleaned baseline used for the final Calore `C_data`-only comparison. Non-baseline USINE input/config files were moved, not deleted, under:

- `Codex_files/.archived_usine_inputs_20260603/`
- moved manifest: `Codex_files/.archived_usine_inputs_20260603/moved_nonbaseline_inputs_manifest.txt`
- visible keep manifest: `Codex_files/.archived_usine_inputs_20260603/visible_big_baseline_keep_manifest.txt`

## Visible Baseline Families

The visible generated-config directory is intentionally restricted to four BIG baseline families:

- `Codex_files/configs/generated/BIG_1D_L_grid/`
- `Codex_files/configs/generated/BIG_1D_L_grid_parents_known_good_sourcefit/`
- `Codex_files/configs/generated/BIG_2D_DM_NFW_bbbar_grid_kCIRELLI19_20260601/`
- `Codex_files/configs/generated/BIG_2D_DM_NFW_bbbar_m10p5_kcirelli19_grid/`

The first two regenerate the secondary antiproton background through `usine_pbar`. The last two regenerate the true local `kCIRELLI19` 2D primary-DM antiproton grids: the full mass grid and the exact `mDM = 10.5 GeV` Table-3 diagnostic grid.

## Visible Input Tables

Kept USINE input tables are the ones referenced by the BIG baseline configs or by USINE core lookup:

- AMS data: `DOC_PBAR/AMS_DATA/AMS02_published_{1HBAR,H,He,C,N,O,Ne,Mg,Si,BC,LiC,BeC,BeB}_rigidity.USINE`
- AMS covariance copies: `DOC_PBAR/AMS_DATA/cov_AMS02_...` and `DOC_PBAR/AMS_DATA/Cov/cov_AMS02_...` for pbar/H/He/C/O/BC/p.
- Parent/nuclear data: `inputs/crdata_crdbHEAO3.dat`
- Core CR tables: `inputs/atomic_properties.dat`, `inputs/solarsystem_abundances2003.dat`, `inputs/crcharts_Zmax30_ghost84-97mixed.dat`, `inputs/crcharts_Zmax30_ghost84.dat`, `inputs/crcharts_Zmax30_ghost97.dat`
- Nuclear production/destruction XS: `inputs/XS_NUCLEI/sigProdWebber03+Coste12.dat`, `inputs/XS_NUCLEI/sigProdGALPROP17_OPT12.dat`, `inputs/XS_NUCLEI/sigInelTripathi99+Coste12.dat`, `DOC_PBAR/XS/paramIIB_bestfit_30Si.dat`
- Antiproton XS: `inputs/XS_ANTINUC/dSdEProd_pbar_1H4He+HHe_Donato01.dat`, `inputs/XS_ANTINUC/sigInelANN_pbar+HHe_Donato01.dat`, `inputs/XS_ANTINUC/sigInelNONANN_pbar+HHe_Donato01.dat`, `inputs/XS_ANTINUC/dSdENAR_pbar+HHe_Duperray05_Anderson.dat`
- Prompt DM antiproton yield: `inputs/PPPC4DMID-spectra/2019/AtProduction_antiprotons.dat`

## Secondary Antiproton Baseline

Executable:

```bash
./bin/usine_pbar
```

Archived command wrapper:

```bash
bash Codex_files/.archived_codex_codes_20260603/run_usine_pbar_big_l_grid_phi0p732_known_good_sourcefit_20260531.txt
```

Core command form:

```bash
USINE_PBAR_COMBOS_ETYPES_PHIFF="1H-BAR:kR:0.732" ./bin/usine_pbar \
  "$PBAR_INIT" \
  "$PARENT_INIT" \
  "$OUT_DIR" \
  output_1D_BIG_fit/fit_result.out \
  NONE \
  NONE \
  0 \
  1 \
  0. \
  0
```

Secondary manifests:

- pbar init grid: `Codex_files/configs/generated/BIG_1D_L_grid/manifest.txt`
- parent/source-fit init grid: `Codex_files/configs/generated/BIG_1D_L_grid_parents_known_good_sourcefit/manifest.txt`

Secondary output consumed by the likelihood:

- archived directory: `Codex_files/.archived_generated_outputs_20260603/calore_phi0p732_grid_20260531/BIG_L_grid_phi0p732_known_good_sourcefit/`
- flux basename: `local_fluxes_1HBAR_R_Model1DKisoVc_SolMod0DFF_phi0_732GV_1.out`

Secondary setup:

- propagation model: `Model1DKisoVc`
- source spatial template: `ASTRO_STD|GREEN15`
- propagated list includes `1H-BAR` and nuclei up to `30Si`
- pure secondaries: `1H-BAR, Li, Be, B`
- tertiary antiprotons: enabled for `1H-BAR`
- parent/source treatment: fixed BIG transport from L-scaling; parent fit adjusts source parameters/modulation, then `usine_pbar` transfers the parent solution to the pbar run
- covariance/XS sampling in this baseline command: `COV=NONE`, `XS=NONE`, `IS_FLUXERR=0`, `N_SAMPLES=1`
- secondary normalization in final likelihood: fixed, `alpha_sec = 1`

## Primary DM Baseline

Executable:

```bash
./bin/usine -l
```

Full-grid command pattern:

```bash
./bin/usine -l \
  "$INIT_FILE" \
  "$OUT_DIR" \
  "1H-BAR:kR:0.732" \
  0. \
  1 \
  1 \
  0
```

Kept full-grid manifest:

- `Codex_files/configs/generated/BIG_2D_DM_NFW_bbbar_grid_kCIRELLI19_20260601/manifest.txt`

Kept exact low-mass diagnostic manifest:

- `Codex_files/configs/generated/BIG_2D_DM_NFW_bbbar_m10p5_kcirelli19_grid/manifest.txt`

Primary output consumed by the likelihood:

- archived grid: `Codex_files/.archived_generated_outputs_20260603/cirelli19_true_usine_full_20260601/BIG_2D_DM_NFW_bbbar_grid_phi0p732_kCIRELLI19/plots_primary_curve/big_2d_dm_explicit_primary_curve_pbar_flux_grid.tsv`
- exact 10.5 GeV diagnostic grid: `Codex_files/.archived_generated_outputs_20260603/cirelli19_true_usine_full_20260601/BIG_2D_DM_NFW_bbbar_m10p5_phi0p732_kCIRELLI19/`

DM setup:

- propagation model: `Model2DKisoVc`
- DM profile: NFW, `NFW97_ANNIHIL_CYL`
- profile scale radius: `rs = 19.6 kpc`
- local density: `RhoSol = 0.385 GeV/cm^3`
- solar radius: `R_sun = 8.2 kpc`
- diffusion cylinder radius: `R = 20 kpc`
- thin disk half-height: `h = 0.1 kpc`
- annihilation/decay selector: `Type = kANNIHILATION`
- self-conjugacy/source factor: `Delta = 2`
- reference cross section: `Sigmav = 3.0e-26 cm^3/s`
- lifetime field present but irrelevant for annihilation: `Tau = 1.0e27`
- channel: pure `b bbar`; `BranchingRatios` vector has index 11 set to 1
- prompt table/model: `kCIRELLI19`, read from `inputs/PPPC4DMID-spectra/2019/AtProduction_antiprotons.dat`
- reader convention audited previously: `x = Ekin / mDM`, table values as `dN/dlog10x`, converted internally to `dN/dEkin`
- active species: `DM_STD|1H-BAR`
- secondary contribution in DM primary grid: off; exotic contribution on

## L Grid And BIG Transport Scaling

The common L grid is:

```text
1.0000000, 1.3179806, 1.7370729, 2.2894285, 3.0174224,
3.9769043, 5.2414828, 6.9081728, 9.1048379, 12.0000000 kpc
```

Baseline transport coefficients stored in the configs:

- `Va = 5.001 km/s`
- `Vc = 0.851 km/s`
- `K0 = 0.037 kpc^2/Myr`
- `delta = 0.515`
- `eta_t = 1`
- `Rhi = 247 GV`
- `Deltahigh = 0.18`
- `shi = 0.04`
- `Rlow = 4.651 GV`
- `deltalow = -0.803`
- `slow = 5.00e-02`

Active L-scaling formulae:

```text
Vc_eff(L) = 0.851 * (L/5)^0.600
Va_eff(L) = 5.001 * (L/5)^0.000
K0_eff(L) = 0.037 * (L/5)^0.907
delta_eff(L) = 0.515 * (L/5)^0.020
Rlow_eff(L) = 4.651 * (L/5)^0.015
deltalow_eff(L) = -0.803 * (L/5)^0.025
```

The active diffusion coefficient is:

```text
K(R) = (0.037*(L/5)^0.907) * beta^eta_t * (Rig/1.0)^(0.515*(L/5)^0.020)
       * (1 + (Rig/(4.651*(L/5)^0.015))^(((-0.803*(L/5)^0.025)
       - (0.515*(L/5)^0.020))/slow))^slow
       * (1 + (Rig/Rhi)^(Deltahigh/shi))^(-shi)
```

Reacceleration:

```text
Kpp = (4/3) * ((5.001*(L/5)^0.000) * 1.022712e-3 * beta * Etot)^2
      / ((0.515*(L/5)^0.020) * (4-(0.515*(L/5)^0.020)^2)
      * (4-(0.515*(L/5)^0.020)) * K00)
```

Effective values at the profiled best-region L used often by the likelihood:

- `L = 2.0084652687 kpc` is the null best L in the C_data-only likelihood output.
- The generated L grid brackets this value between `1.7370729` and `2.2894285 kpc`; the likelihood profiles continuously in log-L.

## Solar Modulation

USINE init model:

- solar modulation model: `SolMod0DFF`
- init-file `ParVals`: `phi = 0.5 GV`
- `Rig0 = 0.2 GV`

Actual baseline flux extraction requests TOA output at:

- `phi = 0.732 GV`
- output request string for secondary and primary: `1H-BAR:kR:0.732`

The final likelihood uses the generated TOA `phi0_732GV` flux products, not the init-file default `0.5 GV` value.

## Likelihood And Report Command

Visible end-to-end wrapper:

```bash
bash Codex_files/codex_codes/run_calore_cdata_best_branch_end_to_end_20260603.txt
```

Default output root:

- `Codex_files/generated_outputs/calore_cdata_best_branch_20260601/end_to_end_reproduction_20260603/`

This wrapper reruns the baseline secondary `usine_pbar` grid, the baseline 2D
primary-DM `usine -l` grid, primary-grid extraction with `--divide-by-r-power
2.8 --drop-highest-mass`, the `C_data`-only upper-limit calculation, and the
final ours-vs-Calore report plot. Existing products under the wrapper output
root are reused so interrupted runs can be resumed.

Visible final wrapper:

```bash
bash Codex_files/codex_codes/run_calore_cdata_best_branch_20260601.txt
```

This wrapper reads archived already-generated flux grids and writes:

- `Codex_files/generated_outputs/calore_cdata_best_branch_20260601/pbar_95cl_upper_limits_data_cov_only.tsv`
- `Codex_files/generated_outputs/calore_cdata_best_branch_20260601/report/`

Core likelihood command:

```bash
python3 Codex_files/codex_codes/calculate_pbar_upper_limits_data_cov.py \
  --secondary-dir Codex_files/.archived_generated_outputs_20260603/calore_phi0p732_grid_20260531/BIG_L_grid_phi0p732_known_good_sourcefit \
  --secondary-flux-basename local_fluxes_1HBAR_R_Model1DKisoVc_SolMod0DFF_phi0_732GV_1.out \
  --dm-grid-tsv Codex_files/.archived_generated_outputs_20260603/cirelli19_true_usine_full_20260601/BIG_2D_DM_NFW_bbbar_grid_phi0p732_kCIRELLI19/plots_primary_curve/big_2d_dm_explicit_primary_curve_pbar_flux_grid.tsv \
  --output-dir Codex_files/generated_outputs/calore_cdata_best_branch_20260601
```

Report command:

```bash
python3 Codex_files/codex_codes/report_calore_cdata_best_branch.py \
  --limits Codex_files/generated_outputs/calore_cdata_best_branch_20260601/pbar_95cl_upper_limits_data_cov_only.tsv \
  --reference Codex_files/.archived_generated_outputs_20260603/reference_extraction/2202_03076_official_source_fig5x1e4_totalx_20260601/bbbar_big_cdata_only_reference_comparison.tsv \
  --out-dir Codex_files/generated_outputs/calore_cdata_best_branch_20260601/report
```

Likelihood settings:

- data: AMS-02 antiproton flux from `DOC_PBAR/AMS_DATA/AMS02_published_1HBAR_rigidity.USINE`
- covariance: local AMS-02 `C_data` only, default file `DOC_PBAR/AMS_DATA/cov_AMS02_201105201505__1HBAR_R.dat`
- covariance blocks: all available blocks by default, including the local `x-sec` block
- no `C_model`
- secondary scale mode: fixed, `alpha_sec = 1`
- primary scale: `1`
- reference cross section for template scaling: `3.0e-26 cm^3/s`
- upper-limit criterion: `Delta chi2 = 3.84`
- L prior: `((log10 L - log10 4.96) / 0.197)^2`
- L profiling: continuous log-L profiling over the generated L grid span

## Current Best-Branch Result

From `report/README.md`:

- branch: BIG/NFW/bbar, true `kCIRELLI19`, explicit primary curve, TOA `phi = 0.732 GV`
- matched Calore masses: `21`
- ours/Calore ratio min/median/max: `0.52649 / 1.01342 / 1.36499`
- worst midmass residuals:
  - `mDM = 169.84993 GeV`: ratio `0.595044`
  - `mDM = 233.64801 GeV`: ratio `0.52649`
  - `mDM = 321.40958 GeV`: ratio `0.580353`
- Table-3 diagnostic at `mDM = 10.5 GeV`:
  - `null_best_L_kpc = 2.0084652687`
  - `best_sigmav_cm3_s = 2.5288556698e-26`
  - `best_L_kpc = 1.9723402513`
  - `sqrt_delta_chi2 = 2.4083919538`

## Archived But Relevant Evidence

The final root-cause gate is under:

- `Codex_files/generated_outputs/calore_cdata_best_branch_20260601/restart_primary_yield_gate_20260603/`

Current conclusion there:

- global normalization changes, `RhoSol` changes, no-L-prior semantics, available local/public yield substitutes, and local `kCIRELLI19` reader bugs were ruled out as full fixes.
- strongest unresolved candidate is the exact Calore/Cirelli `PPPC4DMIDbis` primary-yield artifact or generator configuration.
- no USINE rerun is approved without that exact artifact.
