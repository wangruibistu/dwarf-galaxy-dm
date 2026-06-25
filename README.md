# diff-dwarf

A learned **diffusion prior** over dark-matter halo density profiles for the
kinematic inversion of dwarf spheroidal galaxies, with an exact, gradient-free
**amplitude-profiled importance posterior** against a spherical-Jeans likelihood.
The package also contains the parametric (gNFW / coreNFW / Burkert) Jeans fits
and the multi-population GMM split used to expose the density-profile-family
degeneracy.


## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Layout
```
src/dm_models/parametric_priors.py        gNFW/coreNFW/Burkert + Jeans likelihoods
src/dm_models/diffusion_prior/            DDPM + amplitude-profiled importance
                                          posterior + synthetic/realistic halo libraries
src/dynamical_modeling/                   Jeans (numpy + jax)
src/chemodynamical/                       multi-population GMM split
scripts/                                  reproduction scripts (below)
tests/                                    smoke tests (pytest)
```

## Reproducing the paper figures/tables
| Paper item | Script |
|---|---|
| Sculptor members (Walker+2009 × Gaia DR3) | `scripts/build_sculptor_members.py` |
| σ-profile, single-pop Jeans, PPC (Fig.) | `scripts/run_sculptor_jeans.py` |
| Multi-population split + single-vs-multi (Figs.) | `scripts/run_sculptor_multipop_jeans.py` |
| Family head-to-head + diffusion posterior (Table/Fig.) | `scripts/run_head_to_head_sculptor.py` |
| Mock injection–recovery (Table/Fig.) | `scripts/run_mock_injection_recovery.py` |
| Robustness: prior-sensitivity, ESS, estimator | `scripts/run_referee_tests_dwarf.py` |
| Library/loss/prior-predictive figures | `scripts/make_paper_figures_diffusion.py` |

## Data
Uses the public Walker et al. (2009) Magellan/MMFS Sculptor catalogue (VizieR
`J/AJ/137/3100`) cross-matched with Gaia DR3. `scripts/build_sculptor_members.py`
regenerates the processed member list.

## License
MIT (see `LICENSE`).
