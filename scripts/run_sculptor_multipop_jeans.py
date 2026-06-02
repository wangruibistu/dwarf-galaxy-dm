#!/usr/bin/env python3
"""Sculptor Phase 1 — multi-population chemodynamical Jeans.

Implementation of the Walker & Peñarrubia (2011) split-population idea:
  1. Take stars with measured Mg index (sigmg, [Fe/H] proxy)
  2. Fit a 2-component GMM in (sigmg, R) space — metal-rich (MR) compact pop
     vs metal-poor (MP) extended pop
  3. Compute the half-light radius R_e for each subpopulation
  4. Bin σ_los(R) for each subpopulation independently
  5. Joint NUTS Jeans: shared DM gNFW (log ρ_s, log r_s, γ),
     two tracer Plummer R_e (one per subpop)
  6. Compare γ posterior with the single-pop result from
     scripts/run_sculptor_jeans.py

Hypothesis: the multi-pop posterior should pull γ lower (closer to a core),
reproducing Walker & Peñarrubia 2011 finding for Sculptor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

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
# 1. Load with sigmg
# ---------------------------------------------------------------------------
scu = pd.read_parquet(DATA / "processed" / "sculptor_members_v0.parquet")
scu = scu[(scu["P_member"] > 0.5)
          & scu["v_los_kms"].notna()
          & scu["sigmg"].notna()].reset_index(drop=True)
print(f"[load] {len(scu)} P>0.5 members with finite sigmg")

R = scu["R_pc"].values
v = scu["v_los_kms"].values
verr = scu["v_err_kms"].fillna(2.0).values
sigmg = scu["sigmg"].values

v_sys = float(np.median(v))
dv = v - v_sys
print(f"[stats] v_sys={v_sys:.2f} km/s")


# ---------------------------------------------------------------------------
# 2. 2-component GMM in (sigmg, log R) space
# ---------------------------------------------------------------------------
# Standardise features
X = np.stack([sigmg, np.log10(R + 1)], axis=1)
X = (X - X.mean(0)) / X.std(0)
gmm = GaussianMixture(n_components=2, random_state=0,
                      covariance_type="full", n_init=5).fit(X)
post = gmm.predict_proba(X)

# Identify which component is metal-RICH (higher mean sigmg)
comp_sigmg_mean = [scu.loc[gmm.predict(X) == k, "sigmg"].mean() for k in range(2)]
i_MR = int(np.argmax(comp_sigmg_mean))
i_MP = 1 - i_MR
print(f"[gmm] MR sigmg-mean={comp_sigmg_mean[i_MR]:.3f}, "
      f"MP sigmg-mean={comp_sigmg_mean[i_MP]:.3f}")

scu["w_MR"] = post[:, i_MR]
scu["w_MP"] = post[:, i_MP]

# Effective half-light radius for each subpop (weighted median R)
def weighted_median(values, weights):
    idx = np.argsort(values)
    v_sorted = values[idx]
    w_sorted = weights[idx]
    cw = np.cumsum(w_sorted) / w_sorted.sum()
    return v_sorted[np.searchsorted(cw, 0.5)]

R_e_MR_pc = weighted_median(R, scu["w_MR"].values)
R_e_MP_pc = weighted_median(R, scu["w_MP"].values)
print(f"[gmm] MR R_e ≈ {R_e_MR_pc:.0f} pc, MP R_e ≈ {R_e_MP_pc:.0f} pc")
print(f"[gmm] MR effective N = {scu['w_MR'].sum():.1f}, "
      f"MP effective N = {scu['w_MP'].sum():.1f}")

# Diagnostic plot of GMM separation
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].scatter(sigmg, np.log10(R), c=scu["w_MR"], cmap="coolwarm",
               s=12, alpha=0.7, vmin=0, vmax=1)
axs[0].set_xlabel("sigmg [0.1 nm]")
axs[0].set_ylabel("log R [pc]")
axs[0].set_title(f"GMM weights (MR weight as colour)\n"
                 f"MR R_e≈{R_e_MR_pc:.0f}, MP R_e≈{R_e_MP_pc:.0f} pc")
axs[1].hist(sigmg[scu["w_MR"] > 0.5], bins=20, alpha=0.5, label="MR",
            color="C3")
axs[1].hist(sigmg[scu["w_MR"] <= 0.5], bins=20, alpha=0.5, label="MP",
            color="C0")
axs[1].set_xlabel("sigmg"); axs[1].legend()
fig.savefig(FIG / "sculptor_gmm_split.pdf", bbox_inches="tight"); fig.savefig(FIG / "sculptor_gmm_split.png", dpi=200, bbox_inches="tight")
print(f"[fig] {FIG / 'sculptor_gmm_split.png'}")


# ---------------------------------------------------------------------------
# 3. Weighted σ_los profiles per subpopulation
# ---------------------------------------------------------------------------
def weighted_sigma_profile(R, dv, verr, weights, n_bins=5):
    log_R_edges = np.percentile(R[weights > 0.1], np.linspace(0, 100, n_bins + 1))
    centers = 10 ** (0.5 * (np.log10(log_R_edges[1:]) + np.log10(log_R_edges[:-1])))
    sig2 = np.full(n_bins, np.nan)
    sig2_err = np.full(n_bins, np.nan)
    n_eff = np.zeros(n_bins)
    for i in range(n_bins):
        sel = (R >= log_R_edges[i]) & (R < log_R_edges[i+1])
        if sel.sum() < 5: continue
        w = weights[sel]
        wsum = w.sum()
        if wsum < 3: continue
        n_eff[i] = wsum
        mean = np.sum(w * dv[sel]) / wsum
        var = np.sum(w * (dv[sel] - mean) ** 2) / wsum
        sig2[i] = var - np.sum(w * verr[sel] ** 2) / wsum
        # Effective sample size for error
        sig2_err[i] = max(sig2[i], 1.0) * np.sqrt(2 / max(wsum, 3))
    return centers, sig2, sig2_err, n_eff

centers_MR, sig2_MR, err_MR, n_MR = weighted_sigma_profile(
    R, dv, verr, scu["w_MR"].values, n_bins=5)
centers_MP, sig2_MP, err_MP, n_MP = weighted_sigma_profile(
    R, dv, verr, scu["w_MP"].values, n_bins=5)
ok_MR = np.isfinite(sig2_MR) & (sig2_MR > 0)
ok_MP = np.isfinite(sig2_MP) & (sig2_MP > 0)
print("\n[profile MR]")
for i in np.where(ok_MR)[0]:
    print(f"   R={centers_MR[i]:5.0f} pc  Neff={n_MR[i]:5.1f}  σ={np.sqrt(sig2_MR[i]):5.2f} km/s")
print("[profile MP]")
for i in np.where(ok_MP)[0]:
    print(f"   R={centers_MP[i]:5.0f} pc  Neff={n_MP[i]:5.1f}  σ={np.sqrt(sig2_MP[i]):5.2f} km/s")

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(centers_MR[ok_MR], np.sqrt(sig2_MR[ok_MR]),
            yerr=err_MR[ok_MR] / (2*np.sqrt(sig2_MR[ok_MR])),
            fmt="rs", label=f"MR (R_e={R_e_MR_pc:.0f} pc)", capsize=3)
ax.errorbar(centers_MP[ok_MP], np.sqrt(sig2_MP[ok_MP]),
            yerr=err_MP[ok_MP] / (2*np.sqrt(sig2_MP[ok_MP])),
            fmt="bo", label=f"MP (R_e={R_e_MP_pc:.0f} pc)", capsize=3)
ax.set_xscale("log")
ax.set_xlabel("R [pc]"); ax.set_ylabel(r"$\sigma_{\rm los}$ [km/s]")
ax.legend()
ax.set_title("Sculptor multi-pop σ_los profiles")
fig.savefig(FIG / "sculptor_multipop_sigma.png", dpi=200, bbox_inches="tight")
print(f"[fig] {FIG / 'sculptor_multipop_sigma.png'}")


# ---------------------------------------------------------------------------
# 4. Joint Jeans NUTS — shared DM, two tracer Re
# ---------------------------------------------------------------------------
R_MR_kpc = jnp.array(centers_MR[ok_MR] / 1000.0)
sig2_obs_MR = jnp.array(sig2_MR[ok_MR])
sig2_unc_MR = jnp.array(err_MR[ok_MR] + 0.5)
R_MP_kpc = jnp.array(centers_MP[ok_MP] / 1000.0)
sig2_obs_MP = jnp.array(sig2_MP[ok_MP])
sig2_unc_MP = jnp.array(err_MP[ok_MP] + 0.5)

Re_MR_kpc = R_e_MR_pc / 1000.0
Re_MP_kpc = R_e_MP_pc / 1000.0


def model_multipop(R_MR, y_MR, e_MR, R_MP, y_MP, e_MP):
    log_rho_s = numpyro.sample("log_rho_s", dist.Uniform(5.0, 10.0))
    log_r_s = numpyro.sample("log_r_s",   dist.Uniform(-2.0, 1.5))
    gamma = numpyro.sample("gamma",     dist.Uniform(0.0, 1.5))
    pred_MR = sigma_los2_isotropic(R_MR, (log_rho_s, log_r_s, gamma),
                                     Re_kpc=Re_MR_kpc)
    pred_MP = sigma_los2_isotropic(R_MP, (log_rho_s, log_r_s, gamma),
                                     Re_kpc=Re_MP_kpc)
    numpyro.sample("obs_MR", dist.Normal(pred_MR, e_MR), obs=y_MR)
    numpyro.sample("obs_MP", dist.Normal(pred_MP, e_MP), obs=y_MP)


print("\n[mcmc] starting joint multi-pop NUTS (1000 warmup + 2000 samples × 4 chains)")
kernel = NUTS(model_multipop, target_accept_prob=0.92)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000,
            num_chains=4, progress_bar=False)
mcmc.run(jax.random.PRNGKey(0),
         R_MR_kpc, sig2_obs_MR, sig2_unc_MR,
         R_MP_kpc, sig2_obs_MP, sig2_unc_MP)
mcmc.print_summary(prob=0.9)

samples = mcmc.get_samples()
arr = np.stack([np.asarray(samples[k]) for k in ["log_rho_s", "log_r_s", "gamma"]],
               axis=1)
np.savez(TAB / "sculptor_multipop_jeans_posterior.npz",
         log_rho_s=arr[:, 0], log_r_s=arr[:, 1], gamma=arr[:, 2],
         R_e_MR=Re_MR_kpc, R_e_MP=Re_MP_kpc)
print(f"[save] {TAB / 'sculptor_multipop_jeans_posterior.npz'}")


# ---------------------------------------------------------------------------
# 5. γ(150 pc) and comparison vs single-pop
# ---------------------------------------------------------------------------
r_eval = 0.15
gl_multi = np.asarray(jax.vmap(
    lambda a, b, g: gamma_at_radius(r_eval, a, b, g)
)(samples["log_rho_s"], samples["log_r_s"], samples["gamma"]))

# load single-pop posterior for comparison
single_path = TAB / "sculptor_jeans_posterior.npz"
gl_single = None
if single_path.exists():
    z = np.load(single_path)
    g_single = z["gamma"]; lrs_single = z["log_r_s"]; lrh_single = z["log_rho_s"]
    gl_single = np.asarray(jax.vmap(
        lambda a, b, g: gamma_at_radius(r_eval, a, b, g)
    )(lrh_single, lrs_single, g_single))

fig, ax = plt.subplots(figsize=(6, 4))
if gl_single is not None:
    ax.hist(gl_single, bins=40, alpha=0.5, color="C3", density=True,
            label=f"single-pop  median {np.median(gl_single):.2f}")
ax.hist(gl_multi, bins=40, alpha=0.7, color="C0", density=True,
        label=f"multi-pop   median {np.median(gl_multi):.2f}")
ax.axvline(0.0, c="g", ls=":", label="core γ=0")
ax.axvline(1.0, c="k", ls=":", label="NFW cusp γ=1")
ax.set_xlabel(r"$\gamma(150\,{\rm pc})$"); ax.set_ylabel("posterior PDF")
ax.set_title(f"Sculptor γ(150 pc): single-pop vs multi-pop Jeans")
ax.legend(fontsize=8)
fig.savefig(FIG / "sculptor_gamma_single_vs_multi.pdf", bbox_inches="tight"); fig.savefig(FIG / "sculptor_gamma_single_vs_multi.png", dpi=200, bbox_inches="tight")
print(f"[fig] {FIG / 'sculptor_gamma_single_vs_multi.png'}")

print("\n=== HEADLINE NUMBERS ===")
print(f"  MR pop  N_eff={scu['w_MR'].sum():.0f}   R_e={R_e_MR_pc:.0f} pc")
print(f"  MP pop  N_eff={scu['w_MP'].sum():.0f}   R_e={R_e_MP_pc:.0f} pc")
print(f"  γ (asymp) multi-pop:  median {np.median(arr[:,2]):.2f}  "
      f"68%CI [{np.percentile(arr[:,2],16):.2f}, {np.percentile(arr[:,2],84):.2f}]")
print(f"  γ(150 pc) multi-pop:  median {np.median(gl_multi):.2f}  "
      f"68%CI [{np.percentile(gl_multi,16):.2f}, {np.percentile(gl_multi,84):.2f}]")
if gl_single is not None:
    delta = np.median(gl_single) - np.median(gl_multi)
    print(f"  γ(150 pc) single-pop: median {np.median(gl_single):.2f}")
    print(f"  Δγ (single − multi) = {delta:+.2f}")
    print(f"  → multi-pop {'PULLS γ LOWER (toward core)' if delta > 0.05 else 'no significant change'}")
print(f"  WP11 / Pascale+18 expectation: γ ≈ 0–0.4 (core)")
