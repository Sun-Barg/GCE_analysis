# Calore C_data-only Best Branch Report

## Inputs

- limits: `Codex_files/generated_outputs/calore_cdata_best_branch_20260601/pbar_95cl_upper_limits_data_cov_only.tsv`
- reference: `Codex_files/.archived_generated_outputs_20260603/reference_extraction/2202_03076_official_source_fig5x1e4_totalx_20260601/bbbar_big_cdata_only_reference_comparison.tsv`
- branch: BIG/NFW/bbar, true `kCIRELLI19`, explicit `__prim_`, TOA `phi = 0.732 GV`
- covariance: local AMS-02 `C_data` blocks, including `x-sec`
- likelihood: one-sided 95% CL with `Delta chi2 = 3.84`, continuous log-L profiling

## Metrics

- matched masses: `21`
- ratio min / median / max: `0.52649` / `1.01342` / `1.36499`
- within factor 2: `21/21`
- within factor 1.5: `18/21`
- max absolute log10 ratio: `0.27861`

## Worst Residuals

| mDM GeV | ours / Calore | best L kpc | ours sigmav95 | Calore sigmav95 |
|---:|---:|---:|---:|---:|
| 233.648 | 0.52649 | 2.00847 | 3.787233e-26 | 7.193368e-26 |
| 321.41 | 0.580353 | 2.00847 | 3.561288e-26 | 6.136416e-26 |
| 169.85 | 0.595044 | 2.00847 | 5.652554e-26 | 9.499395e-26 |
| 7798.77 | 1.36499 | 2.00847 | 3.225492e-24 | 2.363012e-24 |
| 1583.22 | 1.35237 | 2.00847 | 2.890796e-25 | 2.137572e-25 |
| 5669.3 | 1.32824 | 2.00847 | 1.904629e-24 | 1.433954e-24 |

## Table-3 Diagnostic

- `Table-3 diagnostic file was not found.`

## Interpretation

Use the metrics and worst-residual table above to judge this branch. For the current best branch, the remaining discrepancy is localized mainly around `mDM ~= 170-320 GeV`; diagnostic branches can shift this structure but should not be promoted unless the full curve and Table-3 analogue also improve.

Plot: `Codex_files/generated_outputs/calore_cdata_best_branch_20260601/report/calore_cdata_best_branch_comparison.png`
