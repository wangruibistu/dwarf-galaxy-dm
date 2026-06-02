#!/usr/bin/env python3
"""Sculptor real-data Jeans inversion.

Loads the cleaned member catalog from data/processed/sculptor_members_v0.parquet,
bins σ_los(R), fits a gNFW DM halo + Plummer tracer via spherical isotropic Jeans,
runs NUTS, and saves posteriors + figures.

Outputs:
  results/figures/sculptor_sigma_profile.png
  results/figures/sculptor_jeans_corner.png
  results/figures/sculptor_gamma150pc.png
  results/figures/sculptor_ppc.png
  results/tables/sculptor_jeans_posterior.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

from src.dynamical_modeling.jeans import sigma_los2_isotropic, gamma_at_radius

numpyro.set_host_device_count(4)


# ---------------------------------------------------------------------------
# 1. Load + clean
# ---------------------------------------------------------------------------
members_path = DATA / "processed" / "sculptor_members_v0.parquet"
scu = pd.read_parquet(members_path)
scu = scu[(scu["P_member"] > 0.5) & scu["v_los_kms"].notna()].reset_index(drop=True)
print(f"[load] {len(scu)} P>0.5 members with RV")

R = scu["R_pc"].values
v = scu["v_los_kms"].values
verr = scu["v_err_kms"].fillna(scu["v_err_kms"].median()).values

# Robust sigma-clipped systemic velocity
v_sys = np.median(v)
v_robust = v.copy()
for _ in range(3):
    mad = np.median(np.abs(v_robust - np.median(v_robust)))
    mask = np.abs(v_robust - np.median(v_robust)) < 4 * 1.4826 * mad
    v_robust = v_robust[mask]
v_sys = float(np.median(v_robust))
print(f"[stats] N={len(scu)}  v_sys={v_sys:.2f} km/s  "
      f"plain std={(v - v_sys).std():.2f}  "
      f"robust std={(v_robust - v_sys).std():.2f}")
dv = v - v_sys


# ---------------------------------------------------------------------------
# 2. Bin σ_los²(R)
# ---------------------------------------------------------------------------
log_R_edges = np.linspace(np.log10(50), np.log10(2000), 11)
binc = 0.5 * (log_R_edges[1:] + log_R_edges[:-1])
sig2 = np.zeros_like(binc)
sig2_err = np.zeros_like(binc)
n_bin = np.zeros_like(binc, dtype=int)
for i, (lo, hi) in enumerate(zip(log_R_edges[:-1], log_R_edges[1:])):
    sel = (np.log10(R) >= lo) & (np.log10(R) < hi)
    n_bin[i] = sel.sum()
    if n_bin[i] < 5:
        sig2[i] = np.nan; continue
    sig2[i] = dv[sel].var() - np.mean(verr[sel] ** 2)
    sig2_err[i] = max(sig2[i], 1.0) * np.sqrt(2 / n_bin[i])

ok = ~np.isnan(sig2) & (sig2 > 0)
print("\n[profile] σ_los(R):")
for i in np.where(ok)[0]:
    print(f"   R={10**binc[i]:6.0f} pc  N={n_bin[i]:3d}  "
          f"σ={np.sqrt(sig2[i]):5.2f} ± {sig2_err[i] / (2*np.sqrt(sig2[i])):.2f} km/s")

# Plot σ-profile
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(10 ** binc[ok], np.sqrt(sig2[ok]),
            yerr=sig2_err[ok] / (2 * np.sqrt(sig2[ok])),
            fmt='o', capsize=3)
ax.set_xscale("log")
ax.set_xlabel("R [pc]"); ax.set_ylabel(r"$\sigma_{\rm los}$ [km/s]")
ax.set_title(f"Sculptor σ_los profile (N={ok.sum()} bins, "
             f"P_member > 0.5, v_sys = {v_sys:.1f} km/s)")
fig.savefig(FIG / "sculptor_sigma_profile.png", dpi=200, bbox_inches="tight")
print(f"[fig] {FIG / 'sculptor_sigma_profile.png'}")


# ---------------------------------------------------------------------------
# 3. Jeans likelihood + NUTS
# ---------------------------------------------------------------------------
R_obs_kpc = jnp.array(10 ** binc[ok] / 1000.0)
sig2_obs = jnp.array(sig2[ok])
sig2_unc = jnp.array(sig2_err[ok] + 0.5)   # mild floor

def model(R, y, yerr):
    log_rho_s = numpyro.sample("log_rho_s", dist.Uniform(5.0, 10.0))
    log_r_s = numpyro.sample("log_r_s",   dist.Uniform(-2.0, 1.5))
    gamma = numpyro.sample("gamma",     dist.Uniform(0.0, 1.5))
    pred = sigma_los2_isotropic(R, (log_rho_s, log_r_s, gamma))
    numpyro.sample("obs", dist.Normal(pred, yerr), obs=y)


print("\n[mcmc] starting NUTS (1000 warmup + 2000 samples × 4 chains)")
kernel = NUTS(model, target_accept_prob=0.9)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000,
            num_chains=4, progress_bar=False)
mcmc.run(jax.random.PRNGKey(0), R_obs_kpc, sig2_obs, sig2_unc)
mcmc.print_summary(prob=0.9)


# ---------------------------------------------------------------------------
# 4. Save posterior + plots
# ---------------------------------------------------------------------------
samples = mcmc.get_samples()
arr = np.stack([np.asarray(samples[k]) for k in ["log_rho_s", "log_r_s", "gamma"]],
               axis=1)
np.savez(TAB / "sculptor_jeans_posterior.npz",
         log_rho_s=arr[:, 0], log_r_s=arr[:, 1], gamma=arr[:, 2],
         R_obs_kpc=np.asarray(R_obs_kpc), sig2_obs=np.asarray(sig2_obs))
print(f"[save] posterior → {TAB / 'sculptor_jeans_posterior.npz'}")

# Corner plot — DIY (no corner dep)
fig, axs = plt.subplots(3, 3, figsize=(8, 8))
labels = [r"$\log\rho_s$", r"$\log r_s$", r"$\gamma$"]
for i in range(3):
    for j in range(3):
        if i == j:
            axs[i, j].hist(arr[:, i], bins=40, color="C0", alpha=0.7)
            axs[i, j].set_xlabel(labels[i])
        elif i > j:
            axs[i, j].hexbin(arr[:, j], arr[:, i], gridsize=30, cmap="Blues", mincnt=1)
            if i == 2: axs[i, j].set_xlabel(labels[j])
            if j == 0: axs[i, j].set_ylabel(labels[i])
        else:
            axs[i, j].axis("off")
fig.suptitle(f"Sculptor gNFW Jeans posterior  (N={ok.sum()} bins)")
fig.tight_layout()
fig.savefig(FIG / "sculptor_jeans_corner.png", dpi=200, bbox_inches="tight")
print(f"[fig] {FIG / 'sculptor_jeans_corner.png'}")

# γ(150 pc) posterior
r_eval = 0.15
gl = np.asarray(jax.vmap(
    lambda a, b, g: gamma_at_radius(r_eval, a, b, g)
)(samples["log_rho_s"], samples["log_r_s"], samples["gamma"]))
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.hist(gl, bins=40, color="C0", alpha=0.7)
ax.axvline(np.median(gl), c="r", ls="--", lw=2,
           label=f"median {np.median(gl):.2f}")
ax.axvline(1.0, c="k", ls=":", label="NFW cusp γ=1")
ax.axvline(0.0, c="g", ls=":", label="core γ=0")
ax.set_xlabel(r"$\gamma(150\,{\rm pc})$"); ax.set_ylabel("posterior")
ax.set_title(f"median {np.median(gl):.2f}, "
             f"68%CI [{np.percentile(gl,16):.2f}, {np.percentile(gl,84):.2f}]")
ax.legend()
fig.savefig(FIG / "sculptor_gamma150pc.png", dpi=200, bbox_inches="tight")
print(f"[fig] {FIG / 'sculptor_gamma150pc.png'}")

# PPC
R_plot = jnp.linspace(R_obs_kpc.min() * 0.5, R_obs_kpc.max() * 2, 60)
idx = np.random.choice(arr.shape[0], 200, replace=False)
preds = np.stack([
    np.asarray(sigma_los2_isotropic(R_plot, tuple(arr[i]))) for i in idx
])
lo, mid, hi = np.percentile(preds, [16, 50, 84], 0)
fig, ax = plt.subplots(figsize=(6, 4))
ax.fill_between(np.asarray(R_plot), np.sqrt(np.clip(lo, 0, None)),
                np.sqrt(np.clip(hi, 0, None)),
                color="C0", alpha=0.3, label="model 68%")
ax.plot(np.asarray(R_plot), np.sqrt(mid), "C0-", label="model median")
ax.errorbar(np.asarray(R_obs_kpc), np.sqrt(np.asarray(sig2_obs)),
            yerr=np.asarray(sig2_unc) / (2 * np.sqrt(np.asarray(sig2_obs))),
            fmt="ko", capsize=3, label="data")
ax.set_xscale("log")
ax.set_xlabel("R [kpc]"); ax.set_ylabel(r"$\sigma_{\rm los}$ [km/s]")
ax.legend()
fig.savefig(FIG / "sculptor_ppc.pdf", bbox_inches="tight"); fig.savefig(FIG / "sculptor_ppc.png", dpi=200, bbox_inches="tight")
print(f"[fig] {FIG / 'sculptor_ppc.png'}")

print("\n=== HEADLINE NUMBERS ===")
print(f"  N members used      : {ok.sum()} bins, {len(scu)} stars")
print(f"  γ(150 pc) median    : {np.median(gl):.2f}")
print(f"  γ(150 pc) 68% CI    : [{np.percentile(gl,16):.2f}, "
      f"{np.percentile(gl,84):.2f}]")
print(f"  Walker+11 / Pascale+18 reference: γ ≲ 0.5 (cored preference)")
