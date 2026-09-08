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
ships **9 integrity verifiers** on an `(ok, msg)` contract, atomic
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
GCE_12yr_data/        12 yr data-level modules (source lists, masks,
                      likelihood, model components)
GCE_17yr_data/        17 yr data-level modules (source map preparation,
                      fitting phases, catalog counting)
GCE_allsky_data/      photon / spacecraft acquisition and weekly-completeness
                      verification
Cov/                  16 yr covariance matrix products
GCE_16yr_data/        16 yr GDE fit results
Prompt_spectra/       DM annihilation prompt photon spectra — extraction and
                      interpolation code
CascadeSpectra/       cascade spectrum utilities
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

- Python 3, `numpy`, `scipy`, `astropy`, `emcee`, `matplotlib`
- [Fermitools](https://github.com/fermi-lat/Fermitools-conda) — `gtselect`,
  `gtmktime`, `gtbin`, `gtltcube`, `gtexpcube2`, `gtsrcmaps`, `gtmodel`

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

- **Prompt spectra pipeline** — MadGraph5 + Pythia8 mass-scan orchestration,
  spectrum extraction, interpolation, and validation against PPPC4DMID.
  *(link to be added)*

---

## Data provenance

- Fermi-LAT photon and spacecraft data: public, from the FSSC.
- FL16Y point-source catalog (`gll_psc_v40.fit`): LAT 16-Year Source List.
- `PPPC4/` contains reference spectra tables from **PPPC4DMID**
  (Cirelli et al., [arXiv:1012.4515](https://arxiv.org/abs/1012.4515)),
  redistributed for reproducibility. Please cite the original work.

---

## References

- Cholis, Zhong, McDermott, Surdutovich 2022 — [arXiv:2112.09706](https://arxiv.org/abs/2112.09706) *(primary methodology)*
- LAT 16-Year Source List — [arXiv:2602.22148](https://arxiv.org/abs/2602.22148)
- Calore, Cholis, Weniger 2015 — [arXiv:1409.0042](https://arxiv.org/abs/1409.0042) *(systematic covariance)*
- Goodenough & Hooper 2009 — GCE discovery

---

*Jeonbuk National University, Department of Physics — advised by Prof. Seodong Shin.*
