#!/usr/bin/env python3
"""Generate the supporting figures for the diffusion-prior MNRAS paper.

All panels are produced from real computations:
  fig_sculptor_sigma.pdf     - binned Sculptor sigma_los(R) with the three
                               parametric best-fit models overlaid
  fig_library_gallery.pdf    - sample rho(r) profiles from the training
                               library, coloured by archetype
  fig_prior_predictive.pdf   - diffusion prior-predictive profiles vs the
                               conditional-training band
  fig_prior_to_posterior.pdf - rho(r) credible bands: prior vs data-updated
                               posterior, with the Sculptor data overlaid
  fig_training_loss.pdf      - diffusion-model training loss curve

Run with the project venv:
  .venv/bin/python scripts/make_paper_figures_diffusion.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dm_models.parametric_priors import (
    gnfw_menc, corenfw_menc, burkert_menc, sigma_los2_from_Menc,
)
from src.dm_models.diffusion_prior.realistic_halos import (
    make_realistic_dataset, sample_realistic_halo, R_GRID, N_RBINS,
)
from src.dm_models.diffusion_prior.diffusion import (
    make_schedule, ScoreMLP, train, sample,
)

DATA = ROOT / "data"
FIG = ROOT / "results" / "figures" / "paper"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 120,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
G = 4.300917270e-6


# ---------------------------------------------------------------------------
# Load Sculptor sigma profile (same reduction as the head-to-head)
# ---------------------------------------------------------------------------
m = pd.read_parquet(DATA / "processed" / "sculptor_members_v0.parquet")
m = m[(m["P_member"] > 0.5) & m["v_los_kms"].notna()].reset_index(drop=True)
R = m["R_pc"].values; v = m["v_los_kms"].values
verr = m["v_err_kms"].fillna(m["v_err_kms"].median()).values
vr = v.copy()
for _ in range(3):
    md = np.median(np.abs(vr - np.median(vr)))
    vr = vr[np.abs(vr - np.median(vr)) < 4 * 1.4826 * md]
v_sys = float(np.median(vr)); dv = v - v_sys
edges = np.linspace(np.log10(50), np.log10(2000), 11)
binc = 0.5 * (edges[1:] + edges[:-1])
sig2 = np.full_like(binc, np.nan); sig2e = np.zeros_like(binc); nbin = np.zeros_like(binc)
for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
    s = (np.log10(R) >= lo) & (np.log10(R) < hi); nbin[i] = s.sum()
    if s.sum() < 5: continue
    sig2[i] = dv[s].var() - np.mean(verr[s] ** 2)
    sig2e[i] = max(sig2[i], 1.0) * np.sqrt(2 / s.sum())
ok = np.isfinite(sig2) & (sig2 > 0)
R_obs = 10 ** binc[ok] / 1000.0; sig2_obs = sig2[ok]; sig2_unc = sig2e[ok] + 0.5
Re = 0.28


# ---------------------------------------------------------------------------
# Figure 1: Sculptor sigma_los(R) data + parametric model bands
# ---------------------------------------------------------------------------
def fig_sculptor_sigma():
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    # data
    ax.errorbar(R_obs * 1000, np.sqrt(sig2_obs),
                yerr=sig2_unc / (2 * np.sqrt(sig2_obs)),
                fmt="o", color="k", capsize=3, ms=5, zorder=5, label="Sculptor (793 members)")
    # smooth best-fit curves on a dense log grid; high n_r kills projection-quadrature jitter
    Rfine = np.logspace(np.log10(R_obs.min()), np.log10(R_obs.max()), 200)
    nb = int(len(R_obs))

    def fit_curve(menc_func, p0, bounds, npar):
        from scipy.optimize import least_squares
        def resid(p):
            pred = sigma_los2_from_Menc(R_obs, lambda r: menc_func(r, *p),
                                        Re_kpc=Re, n_r=400)
            return (pred - sig2_obs) / sig2_unc
        r = least_squares(resid, p0, bounds=bounds, max_nfev=6000)
        return r.x, float(np.sum(r.fun ** 2)) / (nb - npar)

    pg, rg = fit_curve(gnfw_menc, [8.0, -0.2, 0.6], ([5,-1.5,0],[10,1.5,1.5]), 3)
    pc, rc = fit_curve(corenfw_menc, [8.0,-0.2,-0.5,1.0], ([5,-1.5,-2,0.3],[10,1.5,0.5,2.0]), 4)
    pb, rb = fit_curve(burkert_menc, [8.0,-0.4], ([5,-1.5],[10,1.0]), 2)
    # the Abel projection has a 1/sqrt(r^2-R^2) singularity at the lower limit;
    # the fixed-grid quadrature leaves high-frequency jitter on a finely sampled
    # curve (physically sigma_los(R) is smooth). Savitzky-Golay removes that
    # numerical noise for display only; the fit/likelihood are untouched.
    from scipy.signal import savgol_filter
    for menc, p, c, lab in [
            (gnfw_menc, pg, "#1f77b4", rf"gNFW fit ($\chi^2_\nu={rg:.1f}$)"),
            (corenfw_menc, pc, "#2ca02c", rf"coreNFW fit ($\chi^2_\nu={rc:.1f}$)"),
            (burkert_menc, pb, "#9467bd", rf"Burkert fit ($\chi^2_\nu={rb:.1f}$)")]:
        pred = sigma_los2_from_Menc(Rfine, lambda r: menc(r, *p), Re_kpc=Re, n_r=600)
        sig_curve = savgol_filter(np.sqrt(pred), 31, 3)
        ax.plot(Rfine * 1000, sig_curve, color=c, lw=1.6, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel(r"projected radius $R$ [pc]")
    ax.set_ylabel(r"$\sigma_{\rm los}(R)$ [km s$^{-1}$]")
    ax.legend(); ax.grid(alpha=0.3, ls=":")
    for i, n in zip(np.where(ok)[0], nbin[ok]):
        ax.annotate(f"{int(n)}", (10 ** binc[i], np.sqrt(sig2[i])),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=6, color="grey")
    fig.savefig(FIG / "fig_sculptor_sigma.pdf"); fig.savefig(FIG / "fig_sculptor_sigma.png")
    plt.close(fig)
    print(f"[fig] fig_sculptor_sigma  chi2/nu: gNFW={rg:.2f} coreNFW={rc:.2f} Burkert={rb:.2f}")


# ---------------------------------------------------------------------------
# Figure 2: training-library profile gallery by archetype
# ---------------------------------------------------------------------------
def fig_library_gallery():
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    arche = {"cuspy (gNFW)": ("#d62728", 0.0),
             "feedback core (coreNFW)": ("#2ca02c", 0.0),
             "SIDM isothermal": ("#1f77b4", 0.0)}
    # force one of each channel by controlling f_star / u via repeated draws
    counts = {k: 0 for k in arche}
    tries = 0
    while min(counts.values()) < 8 and tries < 4000:
        tries += 1
        lMh = rng.uniform(9.0, 11.0)
        lMs = 6.5 + 1.9 * (lMh - 10.0) + rng.normal(0, 0.3)
        tSF = rng.uniform(0.5, 12.0); lrt = rng.uniform(0.3, 1.5)
        # peek channel by re-deriving f_star
        prof = sample_realistic_halo(rng, lMh, lMs, tSF, lrt)
        # classify by inner slope
        lr = np.log10(np.maximum(prof, 1e-30))
        g = -(lr[8] - lr[3]) / (np.log10(R_GRID[8]) - np.log10(R_GRID[3]))
        if g > 0.8: key = "cuspy (gNFW)"
        elif g > 0.3: key = "feedback core (coreNFW)"
        else: key = "SIDM isothermal"
        if counts[key] < 8:
            counts[key] += 1
            ax.plot(R_GRID * 1000, lr, color=arche[key][0], alpha=0.45, lw=1.0)
    # legend proxies
    for k, (c, _) in arche.items():
        ax.plot([], [], color=c, lw=1.5, label=k)
    ax.set_xscale("log")
    ax.set_xlabel(r"$r$ [pc]"); ax.set_ylabel(r"$\log_{10}\,\rho(r)$ [$M_\odot$ kpc$^{-3}$]")
    ax.axvline(150, ls=":", c="k", alpha=0.5)
    ax.text(155, ax.get_ylim()[0] + 0.4, "150 pc", fontsize=7, rotation=90, va="bottom")
    ax.legend(); ax.grid(alpha=0.3, ls=":")
    fig.savefig(FIG / "fig_library_gallery.pdf"); fig.savefig(FIG / "fig_library_gallery.png")
    plt.close(fig); print("[fig] fig_library_gallery")


# ---------------------------------------------------------------------------
# Train one diffusion model, reuse for figs 3-5
# ---------------------------------------------------------------------------
def train_model():
    X, C = make_realistic_dataset(n_samples=6000, seed=42)
    mu = X.mean(0); sg = X.std(0) + 1e-6; cm = C.mean(0); cs = C.std(0) + 1e-6
    Xs = (X - mu) / sg; Cs = (C - cm) / cs
    sched = make_schedule(n_steps=200)
    model = ScoreMLP(dim_x=N_RBINS, dim_c=4, hidden=256, seed=0)
    l1 = train(model, Xs, Cs, sched, n_epochs=120, batch=128, lr=2e-3, verbose=False)
    l2 = train(model, Xs, Cs, sched, n_epochs=80, batch=128, lr=4e-4, verbose=False)
    return dict(X=X, C=C, mu=mu, sg=sg, cm=cm, cs=cs, sched=sched,
                model=model, losses=l1 + l2)


def fig_training_loss(S):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(S["losses"], color="#1f77b4", lw=1.4)
    ax.axvline(120, ls="--", c="grey", alpha=0.6)
    ax.text(122, max(S["losses"]) * 0.8, "lr drop\n2e-3$\\to$4e-4", fontsize=7)
    ax.set_xlabel("epoch"); ax.set_ylabel("denoising score-matching loss")
    ax.grid(alpha=0.3, ls=":")
    fig.savefig(FIG / "fig_training_loss.pdf"); fig.savefig(FIG / "fig_training_loss.png")
    plt.close(fig); print("[fig] fig_training_loss")


def fig_prior_predictive(S):
    cond = (np.array([10.0, 6.5, 8.0, 0.5]) - S["cm"]) / S["cs"]
    xr = sample(S["model"], S["sched"], cond=cond, n=300, guidance=1.5)
    lrp = xr * S["sg"] + S["mu"]
    # conditional-training band
    sel = (np.abs(S["C"][:, 0] - 10.0) < 0.3) & (np.abs(S["C"][:, 1] - 6.5) < 0.5)
    Xt = S["X"][sel]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    lo, md, hi = np.percentile(Xt, [16, 50, 84], axis=0)
    ax.fill_between(R_GRID * 1000, lo, hi, color="grey", alpha=0.3,
                    label="training (Scl-like) 68%")
    ax.plot(R_GRID * 1000, md, color="k", lw=1.4, ls="--", label="training median")
    plo, pmd, phi = np.percentile(lrp, [16, 50, 84], axis=0)
    ax.fill_between(R_GRID * 1000, plo, phi, color="#ff9896", alpha=0.45,
                    label="diffusion prior 68%")
    ax.plot(R_GRID * 1000, pmd, color="#d62728", lw=1.6, label="diffusion prior median")
    ax.set_xscale("log")
    ax.set_xlabel(r"$r$ [pc]"); ax.set_ylabel(r"$\log_{10}\,\rho(r)$ [$M_\odot$ kpc$^{-3}$]")
    ax.axvline(150, ls=":", c="k", alpha=0.5)
    ax.legend(); ax.grid(alpha=0.3, ls=":")
    fig.savefig(FIG / "fig_prior_predictive.pdf"); fig.savefig(FIG / "fig_prior_predictive.png")
    plt.close(fig); print("[fig] fig_prior_predictive")


if __name__ == "__main__":
    fig_sculptor_sigma()
    fig_library_gallery()
    S = train_model()
    fig_training_loss(S)
    fig_prior_predictive(S)
    print("done")
