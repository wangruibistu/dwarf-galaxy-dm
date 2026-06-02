#!/usr/bin/env python3
"""Referee-response robustness tests for the diffusion-prior paper.

(A1) Prior-sensitivity of the family-choice swing Delta-gamma: re-fit the
     parametric families under alternative prior ranges and show the swing is
     stable.
(A2) gamma(150pc) estimator robustness: recompute the diffusion prior/posterior
     inner slope with an independent windowed log-derivative estimator and
     compare against the gNFW-fit estimator used in the paper.
(A3) Importance-sampling convergence: posterior ESS and gamma(150pc) quantiles
     as a function of N_prior.

Outputs: results/tables/referee_tests_dwarf.npz  (+ console table)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import emcee

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.dm_models.parametric_priors import (
    gnfw_gamma_local, corenfw_gamma_local, burkert_gamma_local,
    sigma_los2_from_Menc, gnfw_menc, corenfw_menc, burkert_menc,
)
from src.dm_models.diffusion_prior.realistic_halos import (
    make_realistic_dataset, R_GRID, N_RBINS,
)
from src.dm_models.diffusion_prior.diffusion import (
    make_schedule, ScoreMLP, train, sample, sample_posterior_importance,
)

DATA = ROOT / "data"; TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
R_EVAL, Re_kpc = 0.15, 0.28
_LOGR = np.log10(R_GRID)

# ---- load Sculptor sigma^2 (same pipeline as head-to-head) ----
m = pd.read_parquet(DATA / "processed" / "sculptor_members_v0.parquet")
m = m[(m["P_member"] > 0.5) & m["v_los_kms"].notna()].reset_index(drop=True)
R = m["R_pc"].values; v = m["v_los_kms"].values
verr = m["v_err_kms"].fillna(m["v_err_kms"].median()).values
vr = v.copy()
for _ in range(3):
    mad = np.median(np.abs(vr - np.median(vr))); vr = vr[np.abs(vr - np.median(vr)) < 4 * 1.4826 * mad]
dv = v - float(np.median(vr))
edges = np.linspace(np.log10(50), np.log10(2000), 11); binc = 0.5 * (edges[1:] + edges[:-1])
s2 = np.full_like(binc, np.nan); s2e = np.zeros_like(binc)
for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
    sel = (np.log10(R) >= lo) & (np.log10(R) < hi)
    if sel.sum() >= 5:
        s2[i] = dv[sel].var() - np.mean(verr[sel] ** 2); s2e[i] = max(s2[i], 1.0) * np.sqrt(2 / sel.sum())
ok = np.isfinite(s2) & (s2 > 0)
R_obs = (10 ** binc[ok]) / 1000.0; sig2_obs = s2[ok]; sig2_unc = s2e[ok] + 0.5


def run_emcee(loglike, dim, c0, sc, nw=32, ns=2500, burn=1000):
    rng = np.random.default_rng(0)
    p0 = rng.normal(c0, sc, size=(nw, dim))
    smp = emcee.EnsembleSampler(nw, dim, loglike); smp.run_mcmc(p0, ns, progress=False)
    return smp.get_chain(flat=True, discard=burn)


# ===================== A1: prior sensitivity =====================
print("=== A1: prior-sensitivity of Delta-gamma ===")

def gnfw_ll(gmax):
    def ll(t):
        lrs, lr, g = t
        if not (5 < lrs < 10 and -2.5 < lr < 1.5 and 0 < g < gmax): return -np.inf
        pred = sigma_los2_from_Menc(R_obs, lambda r: gnfw_menc(r, lrs, lr, g), Re_kpc=Re_kpc)
        if np.any(pred < 1e-3) or np.any(~np.isfinite(pred)): return -np.inf
        return -0.5 * np.sum(((sig2_obs - pred) / np.maximum(sig2_unc, 1.0)) ** 2)
    return ll

def corenfw_ll(lrc_min):
    def ll(t):
        lrs, lr, lrc, n = t
        if not (5 < lrs < 10 and -2.5 < lr < 1.5 and lrc_min < lrc < 1.0 and 0.1 < n < 2.5): return -np.inf
        pred = sigma_los2_from_Menc(R_obs, lambda r: corenfw_menc(r, lrs, lr, lrc, n), Re_kpc=Re_kpc)
        if np.any(pred < 1e-3) or np.any(~np.isfinite(pred)): return -np.inf
        return -0.5 * np.sum(((sig2_obs - pred) / np.maximum(sig2_unc, 1.0)) ** 2)
    return ll

def med_ci(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return np.median(x), np.percentile(x, 16), np.percentile(x, 84)

a1 = {}
for gmax in [1.2, 1.7, 2.0]:
    fl = run_emcee(gnfw_ll(gmax), 3, [8.0, -0.2, 0.5], [0.2, 0.1, 0.15])
    g = np.array([gnfw_gamma_local(R_EVAL, *t) for t in fl]).ravel()
    a1[f"gnfw_gmax{gmax}"] = med_ci(g); print(f"  gNFW gamma_max={gmax}: gamma150 = {med_ci(g)[0]:.2f}")
for lrc in [-2.5, -1.5]:
    fl = run_emcee(corenfw_ll(lrc), 4, [8.0, -0.2, -0.5, 1.0], [0.2, 0.1, 0.2, 0.2])
    g = np.array([corenfw_gamma_local(R_EVAL, *t) for t in fl]).ravel()
    a1[f"corenfw_lrcmin{lrc}"] = med_ci(g); print(f"  coreNFW logrc_min={lrc}: gamma150 = {med_ci(g)[0]:.2f}")
# Burkert fixed (2-param)
flb = run_emcee(lambda t: (-0.5*np.sum(((sig2_obs - sigma_los2_from_Menc(R_obs, lambda r: burkert_menc(r,t[0],t[1]), Re_kpc=Re_kpc))/np.maximum(sig2_unc,1.0))**2)) if (5<t[0]<10 and -2.5<t[1]<1.5) else -np.inf, 2, [8.0,-0.4],[0.2,0.1])
gb = np.array([burkert_gamma_local(R_EVAL, *t) for t in flb]).ravel(); a1["burkert"] = med_ci(gb)
print(f"  Burkert: gamma150 = {med_ci(gb)[0]:.2f}")
swing_base = a1["gnfw_gmax1.7"][0] - a1["burkert"][0]
swing_wide = a1["gnfw_gmax2.0"][0] - a1["burkert"][0]
print(f"  --> Delta-gamma (gNFW_1.7 - Burkert) = {swing_base:.2f}; (gNFW_2.0 - Burkert) = {swing_wide:.2f}")


# ===================== diffusion model (shared for A2,A3) =====================
print("\n[diff] training prior...")
X, C = make_realistic_dataset(n_samples=6000, seed=42)
mu, sigma = X.mean(0), X.std(0) + 1e-6; cm, cs = C.mean(0), C.std(0) + 1e-6
sched = make_schedule(n_steps=200); model = ScoreMLP(dim_x=N_RBINS, dim_c=4, hidden=256, seed=0)
train(model, (X - mu) / sigma, (C - cm) / cs, sched, n_epochs=100, batch=128, lr=2e-3, verbose=False)
train(model, (X - mu) / sigma, (C - cm) / cs, sched, n_epochs=80, batch=128, lr=5e-4, verbose=False)
cond = (np.array([10.0, 6.5, 8.0, 0.5]) - cm) / cs

def rho_to_Menc(lr):
    rho = 10 ** lr; Menc = np.cumsum(4 * np.pi * R_GRID ** 2 * rho * np.gradient(R_GRID))
    return lambda r: np.interp(np.atleast_1d(r), R_GRID, Menc)

def loglike(x_std):
    x = x_std * sigma + mu; out = np.zeros(x.shape[0]); iv = 1.0 / np.maximum(sig2_unc, 1.0) ** 2
    for i, lr in enumerate(x):
        try:
            s = sigma_los2_from_Menc(R_obs, rho_to_Menc(lr), Re_kpc=Re_kpc)
            if np.any(~np.isfinite(s)) or np.all(s <= 0): out[i] = -1e6; continue
            A = np.sum(sig2_obs * s * iv) / np.sum(s * s * iv)
            if not np.isfinite(A) or A <= 0: out[i] = -1e6; continue
            out[i] = -0.5 * np.sum((sig2_obs - A * s) ** 2 * iv)
        except Exception:
            out[i] = -1e6
    return out

from scipy.optimize import curve_fit
def gam_gnfwfit(lr, r=R_EVAL):
    def f(lx, a, b, g): x = (10**lx)/(10**b); return a - g*np.log10(x) - (3-g)*np.log10(1+x)
    try:
        p,_ = curve_fit(f, _LOGR, lr, p0=[8,0,0.8], bounds=([2,-1.5,0],[12,1.5,1.6]), maxfev=4000)
        x = r/(10**p[1]); return p[2] + (3-p[2])*x/(1+x)
    except Exception: return np.nan

def gam_windowed(lr, r=R_EVAL, halfwin=0.35):
    sel = np.abs(_LOGR - np.log10(r)) < halfwin
    return -np.polyfit(_LOGR[sel], lr[sel], 1)[0]  # raw local slope = -dlogrho/dlogr

from scipy.signal import savgol_filter
def gam_savgol(lr, r=R_EVAL, halfwin=0.35):
    """Smoothed, gNFW-shape-independent estimator: Savitzky-Golay smooth the
    profile, then local-linear slope at r. Avoids the gNFW functional form."""
    sm = savgol_filter(lr, window_length=9, polyorder=2)
    sel = np.abs(_LOGR - np.log10(r)) < halfwin
    return -np.polyfit(_LOGR[sel], sm[sel], 1)[0]


# ===================== A3: ESS convergence + A2 on the N=6000 run =====================
print("\n=== A3: ESS / gamma vs N_prior  (and A2 estimator comparison) ===")
a3 = {}
for Np in [2000, 6000, 18000]:
    xp, ess = sample_posterior_importance(model, sched, cond=cond, n_out=400,
                                          log_likelihood_fn=loglike, n_prior=Np,
                                          guidance=1.5, return_diagnostics=True)
    lrp = xp * sigma + mu
    g_fit = np.array([gam_gnfwfit(z) for z in lrp]); g_fit = g_fit[np.isfinite(g_fit)]
    g_win = np.array([gam_windowed(z) for z in lrp]); g_win = g_win[np.isfinite(g_win)]
    g_sav = np.array([gam_savgol(z) for z in lrp]); g_sav = g_sav[np.isfinite(g_sav)]
    a3[Np] = dict(ess=float(ess), fit=med_ci(g_fit), win=med_ci(g_win), sav=med_ci(g_sav))
    print(f"  N_prior={Np:6d}: ESS={ess:5.0f} | gNFW-fit={med_ci(g_fit)[0]:.2f}"
          f" [{med_ci(g_fit)[1]:.2f},{med_ci(g_fit)[2]:.2f}] | Savgol-smooth={med_ci(g_sav)[0]:.2f}"
          f" [{med_ci(g_sav)[1]:.2f},{med_ci(g_sav)[2]:.2f}] | raw-finite-diff={med_ci(g_win)[0]:.2f}")

np.savez(TAB / "referee_tests_dwarf.npz",
         a1=np.array(list(a1.items()), dtype=object),
         a3=np.array(list(a3.items()), dtype=object))
print(f"\n[save] {TAB / 'referee_tests_dwarf.npz'}")
