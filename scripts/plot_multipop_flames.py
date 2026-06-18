#!/usr/bin/env python3
"""Figures for the FLAMES CaT multi-population test (replaces the old 148-star
Mg-proxy figures). Produces:
  results/figures/paper/fig_flames_gmm_split.pdf  -- [Fe/H] vs log R, MR/MP split
  results/figures/paper/fig_flames_single_vs_multi.pdf -- gamma(150pc) posteriors
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

d = pd.read_parquet(ROOT / "data/processed/sculptor_flames_members.parquet")
R = d["R_pc"].values; feh = d["feh"].values
npz = np.load(ROOT / "results/tables/multipop_flames.npz")
gS, gM = npz["g_single"], npz["g_hard"]
ReMR, ReMP = float(npz["ReHMR"]) * 1000, float(npz["ReHMP"]) * 1000

# Median-[Fe/H] hard split: the primary two-population analysis
# (the unsupervised GMM in [Fe/H] alone is not favoured by BIC and collapses
# to a symmetric, nearly co-spatial division -- see run_multipop_flames.py)
isMR = feh > np.median(feh)

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})

# --- Fig 1: GMM split in ([Fe/H], log R) ---
fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.scatter(feh[isMR], np.log10(R[isMR]), s=6, c="#c0392b", alpha=0.5, lw=0,
           label=f"MR ($N={isMR.sum()}$, $R_e\\approx{ReMR:.0f}$ pc)")
ax.scatter(feh[~isMR], np.log10(R[~isMR]), s=6, c="#2c6fbb", alpha=0.5, lw=0,
           label=f"MP ($N={(~isMR).sum()}$, $R_e\\approx{ReMP:.0f}$ pc)")
ax.set_xlabel("[Fe/H]"); ax.set_ylabel(r"$\log_{10}(R/\mathrm{pc})$")
ax.legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout()
out1 = ROOT / "results/figures/paper/fig_flames_gmm_split.pdf"
fig.savefig(out1); fig.savefig(out1.with_suffix(".png"), dpi=150)
print("wrote", out1)

# --- Fig 2: single vs two-pop gamma(150pc) posteriors ---
qS = np.percentile(gS, [16, 50, 84]); qM = np.percentile(gM, [16, 50, 84])
fig, ax = plt.subplots(figsize=(3.4, 3.0))
bins = np.linspace(0, 2, 60)
ax.hist(gS, bins=bins, density=True, color="0.45", alpha=0.65,
        label=f"single ({qS[1]:.2f}, w={qS[2]-qS[0]:.2f})")
ax.hist(gM, bins=bins, density=True, histtype="step", color="#c0392b", lw=1.6,
        label=f"two-pop, median split ({qM[1]:.2f}, w={qM[2]-qM[0]:.2f})")
ax.axvline(1.0, color="k", ls=":", lw=0.8)
ax.set_xlabel(r"$\gamma(150\,\mathrm{pc})$"); ax.set_ylabel("posterior density")
ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
out2 = ROOT / "results/figures/paper/fig_flames_single_vs_multi.pdf"
fig.savefig(out2); fig.savefig(out2.with_suffix(".png"), dpi=150)
print("wrote", out2)
print(f"single={qS[1]:.2f}[{qS[0]:.2f},{qS[2]:.2f}] two-pop={qM[1]:.2f}[{qM[0]:.2f},{qM[2]:.2f}]")
