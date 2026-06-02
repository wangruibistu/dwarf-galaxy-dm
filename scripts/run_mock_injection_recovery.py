#!/usr/bin/env python3
"""Mock injection-recovery test of the diffusion-prior inference scheme.

Referee-facing validation for the diffusion-prior methodology paper. The
head-to-head script demonstrates the method on real Sculptor data, but a
methods paper must also show that the proposed inference scheme recovers a
KNOWN inner slope when the truth is set by hand. This separates two things
the real-data posterior cannot: whether the data actually update the prior
(method works) versus whether the cored posterior is merely inherited from a
core-biased proof-of-concept prior (prior dominates).

We inject two Sculptor-like mock sigma_los^2(R) profiles -- one from a cuspy
gNFW halo and one from a cored Burkert halo -- at the real-data radial
coverage and noise level, then run the IDENTICAL amplitude-profiled
importance posterior used on the real data. We report, for each injection,
the injected truth gamma(150 pc), the no-data prior, and the data-updated
posterior.

Outputs:
    results/tables/mock_injection_recovery.npz
    results/figures/mock_injection_recovery.png/.pdf
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dm_models.parametric_priors import (
    gnfw_rho, burkert_rho, gnfw_menc, burkert_menc, sigma_los2_from_Menc,
)
from src.dm_models.diffusion_prior.realistic_halos import (
    make_realistic_dataset, R_GRID, N_RBINS,
)
from src.dm_models.diffusion_prior.diffusion import (
    make_schedule, ScoreMLP, train, sample, sample_posterior_importance,
)

DATA = ROOT / "data"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

R_EVAL = 0.15   # kpc, "150 pc"
Re_kpc = 0.28   # Sculptor half-light radius
_LOGR = np.log10(R_GRID)


# ---------------------------------------------------------------------------
# gamma(150 pc) estimator -- identical to the head-to-head pipeline so that
# the injected truth, prior, and posterior are summarised on the same footing.
# ---------------------------------------------------------------------------
def _gnfw_logrho(logr_, log_rho_s, log_r_s, gamma):
    x = (10 ** logr_) / (10 ** log_r_s)
    return log_rho_s - gamma * np.log10(x) - (3.0 - gamma) * np.log10(1.0 + x)


def gamma_gnfwfit(log_rho_arr, r_eval=R_EVAL):
    from scipy.optimize import curve_fit
    try:
        p, _ = curve_fit(_gnfw_logrho, _LOGR, log_rho_arr, p0=[8.0, 0.0, 0.8],
                         bounds=([2, -1.5, 0.0], [12, 1.5, 1.6]), maxfev=4000)
        x = r_eval / (10 ** p[1])
        return p[2] + (3.0 - p[2]) * x / (1.0 + x)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# 1. Real Sculptor radial coverage + noise level (so the mock is Sculptor-like)
# ---------------------------------------------------------------------------
members = pd.read_parquet(DATA / "processed" / "sculptor_members_v0.parquet")
members = members[(members["P_member"] > 0.5) & members["v_los_kms"].notna()].reset_index(drop=True)
R = members["R_pc"].values
v = members["v_los_kms"].values
verr = members["v_err_kms"].fillna(members["v_err_kms"].median()).values
v_robust = v.copy()
for _ in range(3):
    mad = np.median(np.abs(v_robust - np.median(v_robust)))
    v_robust = v_robust[np.abs(v_robust - np.median(v_robust)) < 4 * 1.4826 * mad]
dv = v - float(np.median(v_robust))
log_R_edges = np.linspace(np.log10(50), np.log10(2000), 11)
binc = 0.5 * (log_R_edges[1:] + log_R_edges[:-1])
sig2 = np.full_like(binc, np.nan); sig2_err = np.zeros_like(binc)
for i, (lo, hi) in enumerate(zip(log_R_edges[:-1], log_R_edges[1:])):
    sel = (np.log10(R) >= lo) & (np.log10(R) < hi)
    if sel.sum() < 5:
        continue
    sig2[i] = dv[sel].var() - np.mean(verr[sel] ** 2)
    sig2_err[i] = max(sig2[i], 1.0) * np.sqrt(2 / sel.sum())
ok = np.isfinite(sig2) & (sig2 > 0)
R_obs = (10 ** binc[ok]) / 1000.0
sig2_unc = sig2_err[ok] + 0.5
print(f"[grid] {ok.sum()} Sculptor-like radial bins, R = "
      f"{R_obs.min():.2f}-{R_obs.max():.2f} kpc")


# ---------------------------------------------------------------------------
# 2. Train the SAME diffusion prior used on the real data
# ---------------------------------------------------------------------------
print("[diff] building realistic halo library + training prior...")
X, C = make_realistic_dataset(n_samples=6000, seed=42)
mu = X.mean(0); sigma = X.std(0) + 1e-6
cm = C.mean(0); cs = C.std(0) + 1e-6
X_std = (X - mu) / sigma
C_std = (C - cm) / cs
sched = make_schedule(n_steps=200)
model = ScoreMLP(dim_x=N_RBINS, dim_c=4, hidden=256, seed=0)
t0 = time.perf_counter()
train(model, X_std, C_std, sched, n_epochs=100, batch=128, lr=2e-3, verbose=False)
train(model, X_std, C_std, sched, n_epochs=80, batch=128, lr=5e-4, verbose=False)
print(f"[diff] trained in {time.perf_counter()-t0:.1f}s")
cond_sculptor = (np.array([10.0, 6.5, 8.0, 0.5]) - cm) / cs


def rho_to_Menc(log_rho_arr):
    rho = 10 ** log_rho_arr
    Menc = np.cumsum(4 * np.pi * R_GRID ** 2 * rho * np.gradient(R_GRID))
    return lambda r: np.interp(np.atleast_1d(r), R_GRID, Menc)


def make_log_likelihood(sig2_obs):
    """Amplitude-profiled Jeans log-likelihood against a given mock profile."""
    inv_var = 1.0 / np.maximum(sig2_unc, 1.0) ** 2
    def loglike(x_std):
        x = x_std * sigma + mu
        out = np.zeros(x.shape[0])
        for i, log_rho in enumerate(x):
            try:
                s = sigma_los2_from_Menc(R_obs, rho_to_Menc(log_rho), Re_kpc=Re_kpc)
                if np.any(~np.isfinite(s)) or np.all(s <= 0):
                    out[i] = -1e6; continue
                A = np.sum(sig2_obs * s * inv_var) / np.sum(s * s * inv_var)
                if not np.isfinite(A) or A <= 0:
                    out[i] = -1e6; continue
                out[i] = -0.5 * np.sum((sig2_obs - A * s) ** 2 * inv_var)
            except Exception:
                out[i] = -1e6
        return out
    return loglike


# ---------------------------------------------------------------------------
# 3. Build the two injected truths (cuspy gNFW + cored Burkert)
# ---------------------------------------------------------------------------
def inject(name, log_rho_grid, menc_fn, seed):
    """Forward-model a clean mock, add Sculptor-level noise, return data+truth."""
    truth_gamma = gamma_gnfwfit(log_rho_grid)
    s = sigma_los2_from_Menc(R_obs, menc_fn, Re_kpc=Re_kpc)
    s = s * (np.median(sig2_obs_real) / np.median(s))  # scale to Sculptor amplitude
    rng = np.random.default_rng(seed)
    mock = s + rng.normal(0.0, sig2_unc)
    print(f"[inject] {name:5s}: truth gamma(150pc) = {truth_gamma:.2f}")
    return mock, truth_gamma


# Sculptor real sigma^2 scale, used only to set a realistic mock amplitude
sig2_obs_real = sig2[ok]

# Cuspy gNFW: asymptotic slope 1.2, r_s = 1.5 kpc
cusp_params = dict(log_rho_s=7.9, log_r_s=np.log10(1.5), gamma=1.2)
cusp_logrho = np.log10(gnfw_rho(R_GRID, **cusp_params))
cusp_menc = lambda r: gnfw_menc(r, **cusp_params)
mock_cusp, truth_cusp = inject("cusp", cusp_logrho, cusp_menc, seed=11)

# Strongly cored Burkert: large core radius r_s = 1.0 kpc gives a flat inner
# profile and a clearly cored gamma(150 pc), symmetric to the cuspy injection.
core_params = dict(log_rho_s=8.5, log_r_s=np.log10(1.0))
core_logrho = np.log10(burkert_rho(R_GRID, **core_params))
core_menc = lambda r: burkert_menc(r, **core_params)
mock_core, truth_core = inject("core", core_logrho, core_menc, seed=22)


# ---------------------------------------------------------------------------
# 4. Prior (no data) + posterior for each injection
# ---------------------------------------------------------------------------
print("[diff] prior-predictive (no data)...")
x_prior = sample(model, sched, cond=cond_sculptor, n=400, guidance=1.5)
gamma_prior = np.array([gamma_gnfwfit(lr) for lr in (x_prior * sigma + mu)])
gamma_prior = gamma_prior[np.isfinite(gamma_prior)]


def recover(name, mock):
    x_post, ess = sample_posterior_importance(
        model, sched, cond=cond_sculptor, n_out=400,
        log_likelihood_fn=make_log_likelihood(mock),
        n_prior=6000, guidance=1.5, return_diagnostics=True,
    )
    g = np.array([gamma_gnfwfit(lr) for lr in (x_post * sigma + mu)])
    g = g[np.isfinite(g)]
    print(f"[recover] {name:5s}: posterior gamma(150pc) = {np.median(g):.2f} "
          f"[{np.percentile(g,16):.2f}, {np.percentile(g,84):.2f}], ESS={ess:.0f}")
    return g, float(ess)


gamma_post_cusp, ess_cusp = recover("cusp", mock_cusp)
gamma_post_core, ess_core = recover("core", mock_core)


# ---------------------------------------------------------------------------
# 5. Summary + figure
# ---------------------------------------------------------------------------
def stat(g):
    return float(np.median(g)), float(np.percentile(g, 16)), float(np.percentile(g, 84))

print("\n=== INJECTION-RECOVERY SUMMARY ===")
print(f"{'case':6s} {'truth':>6s} {'prior med':>10s} {'post med':>9s} {'post 68% CI':>16s}")
for nm, truth, gp in [("cusp", truth_cusp, gamma_post_cusp),
                      ("core", truth_core, gamma_post_core)]:
    m, lo, hi = stat(gp)
    pm = np.median(gamma_prior)
    print(f"{nm:6s} {truth:6.2f} {pm:10.2f} {m:9.2f} {f'[{lo:.2f}, {hi:.2f}]':>16s}")

plt.rcParams.update({"font.family": "serif", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharex=True, sharey=True)
for ax, nm, truth, gp in [(axes[0], "cuspy injection", truth_cusp, gamma_post_cusp),
                          (axes[1], "cored injection", truth_core, gamma_post_core)]:
    ax.hist(gamma_prior, bins=20, range=(0, 1.6), density=True, color="0.8",
            label="prior (no data)")
    ax.hist(gp, bins=20, range=(0, 1.6), density=True, histtype="step",
            color="C3", lw=2, label="posterior")
    ax.axvline(truth, color="k", ls="--", lw=1.5, label=f"injected truth = {truth:.2f}")
    ax.axvline(np.median(gp), color="C3", ls=":", lw=1.2)
    ax.set_title(nm); ax.set_xlabel(r"$\gamma(150\,\mathrm{pc})$")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, ls=":")
axes[0].set_ylabel("density")
fig.tight_layout()
fig.savefig(FIG / "mock_injection_recovery.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "mock_injection_recovery.pdf", bbox_inches="tight")
print(f"[fig] {FIG / 'mock_injection_recovery.png'}")

np.savez(
    TAB / "mock_injection_recovery.npz",
    R_obs=R_obs, sig2_unc=sig2_unc,
    mock_cusp=mock_cusp, mock_core=mock_core,
    truth_cusp=truth_cusp, truth_core=truth_core,
    gamma_prior=gamma_prior,
    gamma_post_cusp=gamma_post_cusp, gamma_post_core=gamma_post_core,
    ess_cusp=ess_cusp, ess_core=ess_core,
)
print(f"[save] {TAB / 'mock_injection_recovery.npz'}")
