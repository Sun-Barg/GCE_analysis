# GCE_analysis — Dark Matter Interpretation of the Galactic Center Excess

Analysis pipeline for a dark-matter interpretation of the **Galactic Center
gamma-ray Excess (GCE)** using **~17.5 years of Fermi-LAT data** and the
**FL16Y** point-source catalog.

The analysis reproduces the methodology of **Cholis et al. 2022**
([arXiv:2112.09706](https://arxiv.org/abs/2112.09706)) and extends it in two
directions: a longer exposure (16 → 17.5 yr) and an updated point-source
catalog (4FGL-DR4 → FL16Y, `gll_psc_v40.fit`,
[arXiv:2602.22148](https://arxiv.org/abs/2602.22148)).

**Associated manuscript** *(in preparation, JCAP format)*
*Dark Matter interpretation of the Galactic Center Gamma-Ray Excess from the
17.5 years of Fermi-LAT data with updated point source catalog*
Hae Barg Kang, Sang Hwan Kim, Seodong Shin — Jeonbuk National University

---

## Start here

The most detailed document in this repository is
**[`GCE_17yr_reproduce/README_pipeline.md`](GCE_17yr_reproduce/README_pipeline.md)**.
It is the authoritative index of the production pipeline: for every file,
its purpose, inputs, outputs, prerequisites, CLI, and current state.

It is kept as a working record rather than a polished summary. Superseded
conclusions are marked as superseded and left in place, so that stale
phrasing encountered elsewhere can be recognised as stale. Four episodes are
documented end to end:

| Episode | Record |
|---|---|
| **SIGKILL, root cause** | Workers were killed externally when entering `emcee` after `fermitools` (`GtApp`) had run in the same process. Localised by a **six-job controlled experiment** varying pool on/off, fresh vs. reused source maps, and single vs. multi-process. The earlier "VS Code tqdm PTY overflow" hypothesis is documented **and rejected**, not deleted. Fix: serial sampler plus a `prepare` / `mcmc` subprocess split, so the fermitools state dies with the first process. |
| **GCE template normalisation** | The pipeline *referenced* a spatial template that no pipeline code generated. Its inherited normalisation summed only row 0 of the map and used the wrong pixel scale, giving integral **0.7259** instead of 1.0. Discovery → impact assessment → cross-validation against the covariance branch (identical map sum, `3.282807e+05`) → regeneration, with the old file preserved as a backup. |
| **MapCube axis orientation** | A separate defect found after the template fix: the diffuse component cubes required an axis-2 flip, and this — not the template — was the origin of an anomalous model ranking. The full 80-model fit was rerun afterwards. |
| **Mask orientation, settled quantitatively** | Whether the point-source mask needed the same flip was decided by measurement rather than by symmetry with the cube: the un-flipped mask covers **88 %** of source cores against **58 %** when flipped. The fit therefore applies the mask in the CCUBE frame, un-flipped, and the earlier note claiming a flip convention is marked stale. |

**Result state.** The 80-model main fit has been rerun after both fixes and
the current outputs are valid. The recovered model ranking is consistent
with the reference analysis; the rank correlation and per-model ordering are
reported in `README_pipeline.md`.

**Stale-file protection.** A mid-run kill could once leave partial
intermediates that the next launcher pass silently reused. The pipeline now
ships **10 integrity verifiers** on an `(ok, msg)` contract, atomic
`.tmp + os.rename` writes, and an explicit fail-on-stale policy — never
silent reuse. Unattended runs are held up by `launcher_watchdog*.sh`
(60 s polling, orphan reclamation, restart, auto-exit at target count):
**14 restarts, zero data loss** on the main run.

---

## Analysis at a glance

| | |
|---|---|
| Data | Fermi-LAT, ~17.5 yr (weeks w009–w934), `P8R3_CLEAN_V3`, evclass 256, evtype 1 (FRONT) |
| Energy binning | 14 bins, 0.274698 – 51.9312 GeV |
| Region | 60° × 60° around the Galactic Center, 600 × 600 × 14 counts cube |
| Forward model | `gtsrcmaps` (PSF convolution) + `gtexpcube2` + `gtmodel` — instrument response folded into the model, no deconvolution |
| Decomposition | 5-component Poisson template fit (π⁰, bremsstrahlung, ICS, isotropic, Fermi bubbles) plus a GCE NFW² template |
| Main fit | 80 Galactic-diffuse-emission models, MCMC 100 walkers × 1000 steps × 400 burn-in |
| Systematics | 22 ROIs at 20° ≤ \|ℓ\| ≤ 70°, step 5° → 14 × 14 covariance matrix (cond ≈ 2.4 × 10⁵) |
| Cross-checks | Parallel 4FGL-DR4 track for catalog comparison; synthetic-covariance generation with recovery verification; 16 yr reproduction compared against the predecessor analysis |

---

## Layout

```
GCE_17yr_reproduce/   17.5 yr production pipeline — workers, launchers,
                      watchdogs, integrity verifiers, covariance assembly,
                      synthetic-covariance validation, DR4 parallel track
                      + README_pipeline.md (start here)
GCE_16yr_reproduce/   16 yr reproduction and comparison against the
                      predecessor 16 yr analysis
GCE_12yr_reproduce/   12 yr reproduction (earliest validation stage)
GCE_12yr_data/        12 yr data-level modules (masks, likelihood,
                      source map generation)
GCE_17yr_data/        17 yr data-level modules (source map preparation,
                      fitting phases, catalog counting)
GCE_allsky_data/      photon / spacecraft acquisition and weekly-completeness
                      verification
Cov/                  16 yr covariance matrix products
GCE_16yr_data/        16 yr GDE fit results
Prompt_spectra/       DM annihilation prompt photon spectra — extraction and
                      interpolation code
PPPC4/                external reference spectra — PPPC4DMID (Cirelli et al.)
docs/                 miscellaneous notes
legacy/               superseded notebooks, kept for provenance
```

Large intermediate products (FITS cubes, livetime cubes, exposure maps,
per-model fit outputs) and bulk result tables are not tracked; they are
regenerated by the pipeline. Prompt spectrum data files are produced by the
companion MadGraph5 + Pythia8 pipeline (see *Related repositories*).

---

## Environment

[Fermitools](https://github.com/fermi-lat/Fermitools-conda) supplies
`gtselect`, `gtmktime`, `gtbin`, `gtltcube`, `gtexpcube2`, `gtsrcmaps` and
`gtmodel`, together with the Python modules the pipeline imports directly —
`GtApp`, `gt_apps`, `BinnedAnalysis`, `UnbinnedAnalysis` and `pyLikelihood`.
It is conda-only and is therefore not covered by `requirements.txt`:

```bash
conda create -n fermi -c conda-forge -c fermi fermitools
conda activate fermi
pip install -r requirements.txt
```

`requirements.txt` pins `numpy`, `scipy`, `astropy`, `pandas`, `matplotlib`
and `emcee` to the versions the current results were produced with, and
declares [`LATSourceModel`](https://github.com/physicsranger/make4FGLxml),
which builds the point-source XML model (see *Third-party components*).

One caveat applies to the 12-year comparison. It constructs `SourceList` with
`DR=2`, which reaches a branch where release 1.10.11 reads
`row.Pivot_energy` although the column is `Pivot_Energy`. A clean install
raises `AttributeError` there as soon as the region of interest contains a
`PLSuperExpCutoff2` source. The environment the 12-year results were produced
in carries that one-character correction in `LATSourceModel/SourceList.py`;
apply it before rerunning the 12-year path. The 16-year and 17.5-year paths
use `DR=4` and are unaffected.

A quick-start command sequence is at the end of `README_pipeline.md`.

---

## Contributions

| Period | Author | Scope |
|---|---|---|
| 2025-11 | Sang Hwan Kim | Repository initialisation; 16 yr GDE fit results and covariance matrices; directory organisation |
| 2026-04 – | Hae Barg Kang | 12 yr and 16 yr reproductions; design and implementation of the 17.5 yr pipeline (data preparation, per-model and per-ROI workers, launchers, watchdogs, integrity verifiers); covariance assembly, validation and synthetic-data recovery tests; template regeneration and orientation fixes |

Sang Hwan Kim's 16 yr code served as an implementation reference only; the
methodological reference for the 17.5 yr analysis is Cholis et al. 2022.

---

## Related repositories

- **`dm_spectra_pipeline`** — MadGraph5 + Pythia8 mass-scan orchestration,
  spectrum extraction, interpolation, and validation against PPPC4DMID. It
  produces the prompt spectra this analysis consumes. Released together with
  the manuscript.

---

## Third-party components

This repository contains only code written for this analysis. Work by others
is declared as a dependency or cited, not redistributed here. Each item below
must be obtained from its own source.

| Component | Author / reference | Terms | How to obtain |
|---|---|---|---|
| `LATSourceModel` — `SourceList` / `make4FGLxml`, builds the point-source XML model from the catalog | Tyrel Johnson. Originally distributed as a single file through the Fermi SSC user contributions, later restructured into a package | GPL-3.0 | `pip install LATSourceModel` · [source](https://github.com/physicsranger/make4FGLxml) |
| Cascade spectrum tables and loader examples | Elor, Rodd, Slatyer & Xue, [arXiv:1511.08787](https://arxiv.org/abs/1511.08787) | cite the paper | [CascadeSpectra](http://web.mit.edu/lns/research/CascadeSpectra.html), released by the authors |
| Masks for GCE analysis | Zhong & Cholis, [arXiv:2401.02481](https://arxiv.org/abs/2401.02481) | acknowledge the paper | [ymzhong/gce_mask](https://github.com/ymzhong/gce_mask) |
| Binned-likelihood and Python likelihood tutorials | Fermi SSC | — | [fermi-lat/AnalysisThreads](https://github.com/fermi-lat/AnalysisThreads) |

### How the table was built

The table above is the outcome of an audit, not of a record kept as the work
went along. Fifteen files — 39,089 lines — of other people's code had
accumulated in the tree over time, committed without license notices and
without any note of where they came from.

| Source | Files | Lines |
|---|---|---|
| `LATSourceModel` (Tyrel Johnson, GPL-3.0) | 9 | 5,541 |
| Elor, Rodd, Slatyer & Xue 2015, supplementary material | 2 | 25,136 |
| Zhong & Cholis 2024, mask index | 1 | 35 |
| Fermi SSC page saved as HTML | 1 | 2,449 |
| Fermi SSC tutorials | 2 | 5,928 |

The `LATSourceModel` copies were not one thing. Three generations sat side by
side, and separating them took an md5 comparison of every copy against the
published release on PyPI (1.10.11) — deliberately against the release rather
than against the locally installed package, which turned out not to be the
release either. Seven of the nine files matched the release byte for byte.
One, `GCE_12yr_data/SourceList.py`, differed from it by exactly one line. The
ninth, `GCE_12yr_data/make4FGLxml.py`, was an older single-file version
distributed through the SSC user contributions, from before the package was
restructured; the file of the same name under `GCE_17yr_data/` is the
release's own console-script module.

Two questions had to be answered before deleting any of it: whether the tree
depends on these copies, and whether the one modified line carries behaviour
the results rely on.

The first was settled by reading the imports. The vendored `SourceList.py`
opens with `from LATSourceModel.utilities import ...` and a matching import
of `LATSourceModel.model_components` — names that resolve against the
installed package, not against its siblings in the same directory. The copies
cannot run on their own, so their functional contribution is zero and
removing them changes nothing that executes.

The second was settled by the code path. The modified line sits in the
`PLSuperExpCutoff2` branch that `SourceList` reaches only when it is
constructed with a data-release number outside {3, 4}. Release 1.10.11 reads
`row.Pivot_energy` there, while the column is spelled `Pivot_Energy` where
the frame is built; on that branch the release raises `AttributeError`, and
the local change is the one-character correction. The 16-year and 17.5-year
pipelines construct `SourceList` with `DR=4` and never reach the branch, so
the results in this repository do not depend on the change. The 12-year
comparison uses `DR=2` and does.

So the files were removed from tracking and replaced by a declaration:
`LATSourceModel` pinned in `requirements.txt`, everything else cited in the
table above. Local working copies stay on disk and are ignored by git, so
nothing that ran before stops running.

| | Before | After |
|---|---|---|
| Third-party code | 15 files, 39,089 lines | 0 |
| Tracked files | 144 | 127 |

The notebook that had been using the modified copy,
`GCE_17yr_data/GCE_17yr_data_analysis.ipynb`, constructs `SourceList` with
`DR=4`; it never reached the branch and is unaffected either way. The place
where the change still matters is the 12-year comparison,
`GCE_12yr_data/GCE_12yr_data_compare.ipynb`, which uses `DR=2`. A comment in
that notebook and the note under *Environment* say what is required.

---

## Data provenance

- Fermi-LAT photon and spacecraft data: public, from the FSSC.
- FL16Y point-source catalog (`gll_psc_v40.fit`): LAT 16-Year Source List.
- `PPPC4/` contains reference spectra tables from **PPPC4DMID**
  (Cirelli et al., [arXiv:1012.4515](https://arxiv.org/abs/1012.4515),
  distributed at [marcocirelli.net](http://www.marcocirelli.net/PPPC4DMID.html)),
  included here so that the fit is reproducible. Please cite the original work.

---

## License

Code written for this analysis is released under the MIT License — see
[`LICENSE`](LICENSE). Third-party components are not included in this
repository; their own terms apply wherever they are used, as listed under
*Third-party components*.

---

## References

- Cholis, Zhong, McDermott, Surdutovich 2022 — [arXiv:2112.09706](https://arxiv.org/abs/2112.09706) *(primary methodology)*
- LAT 16-Year Source List — [arXiv:2602.22148](https://arxiv.org/abs/2602.22148)
- Calore, Cholis, Weniger 2015 — [arXiv:1409.0042](https://arxiv.org/abs/1409.0042) *(systematic covariance)*
- Goodenough & Hooper 2009 — [arXiv:0910.2998](https://arxiv.org/abs/0910.2998) *(GCE discovery)*

---

*Jeonbuk National University, Department of Physics — advised by Prof. Seodong Shin.*
