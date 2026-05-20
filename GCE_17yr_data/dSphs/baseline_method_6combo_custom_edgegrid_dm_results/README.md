# Baseline-method six-combination DM-result collection

This directory is a compact collection of the combined DM-limit results from the
baseline analysis method across the six duration/catalog cases.

Included result families:
- Standard annihilation channel: `bb`
- Effective four-body spectra: `effective_4b`, `effective_4tau`, `effective_2b2tau`
- On-shell four-body spectra: `onshell_4b`, `onshell_4tau`, `onshell_2b2tau` for r=0.1 through 1.0

Duration/catalog matrix:
- 14yr + 4FGL-DR3
- 14yr + 4FGL-DR4
- 14yr + FL16Y
- 17yr + 4FGL-DR3
- 17yr + 4FGL-DR4
- 17yr + FL16Y

J treatment: `Jprior`
Sample: `measured_30`
Custom-spectrum mass grid: `support_edges`

Top-level tables:
- `combined_limit_index.csv`: one row per spectrum and duration/catalog case.
- `all_limits_long_format.csv`: one row per mass point, spectrum, and duration/catalog case.
- `matrix_product_index.csv`: copied six-combination plot/CSV/PDF products.
- `bb_mauro_comparison_index.csv`: copied Mauro-comparison products for the standard bb channel.
- `rollup.json`: counts and completeness checks.

Directory layout:
- `matrix_products/`: PNG/PDF/CSV six-combination plots for each result family.
- `combined_limits/`: full combined limit tables, both original TXT and converted CSV.
- `combined_ts_arrays/`: compact combined TS arrays.
- `cross_case_by_r/`: on-shell plots comparing all six duration/catalog cases at fixed r.
- `fixed_combo_by_r_17yr_5fgl_Jprior_m3_to_1000GeV/`: on-shell 17.5 yr + FL16Y plots comparing 4b, 4tau, and 2b2tau at each r.
- `summaries/` and `manifests/`: copied JSON metadata.
- `bb_mauro_comparison/`: per-combination bb comparison files against Mauro reference products where present.

This gather step only copies and reformats existing combined products. It does
not rerun sourcefinding, SED fitting, TS-array generation, or limit extraction.
For the `support_edges` custom-spectrum grid mode, standard `bb` products are
included unchanged while custom spectra are gathered from the edge-aware TS-array
namespace.
