# Antiproton Final-State Limit Handoff

Generated from the tentative baseline matching `calore_cdata_best_branch_20260601`.

## Contents

- `two_body/bb/`: standard `b bbar` baseline C_data-only antiproton limit and Calore comparison plot.
- `two_body/tautau/`: standard `tau+ tau-` C_data-only antiproton limit, prompt-overlay manifest, compact primary-flux grid, and sanity/literature comparison plots.
- `four_body/onshell_30/`: on-shell four-body limits for `4b`, `4tau`, and `2b2tau` at `r = 0.1 ... 1.0`.
- `prompt_spectra_inputs/`: PPPC two-body antiproton table and custom four-body antiproton prompt tables used for the overlays.
- `scripts/`: scripts used to prepare overlays, run limits, and make the comparison plots.

## Important Notes

- Two-body and four-body limits use the same fixed-secondary, C_data-only baseline likelihood setup.
- The standard `tau+ tau-` run uses a PPPC overlay: the baseline `b bbar` USINE init files are left unchanged, while the PPPC kCIRELLI19 `b` prompt column is replaced by the PPPC kCIRELLI19 `tau` column.
- Four-body folders include compact limit tables, manifests, and extracted primary/provenance TSVs, but not the bulky raw per-point USINE macro output directories.
- Four-body source prompt tables are included so the overlays can be rebuilt without searching the original analysis tree.
- The literature comparison bands for `tau+ tau-` are order-of-magnitude guides, not digitized curves.

## Main Files

- `two_body/bb/pbar_95cl_upper_limits_data_cov_only.tsv`
- `two_body/tautau/pbar_95cl_upper_limits_data_cov_only.tsv`
- `four_body/onshell_30/combined_onshell_pbar_limit_summary.tsv`
- `four_body/onshell_30/combined_onshell_pbar_limit_summary_finite_only.tsv`
- `four_body/onshell_30/onshell_30_pbar_limit_matrix.png`
- `two_body/tautau/report/tautau_vs_bb_vs_4tau_pbar_limits_zoom.png`
- `two_body/tautau/report/tautau_antiproton_literature_order_comparison.png`
