#!/usr/bin/env python3
"""Retrain the diffusion prior on a SIMULATION-CALIBRATED (DC14) halo library
and recompute the Sculptor inner-slope posterior.

Replaces the hand-built synthetic library (gNFW+coreNFW+SIDM channels) with a
library whose inner shape follows the Di Cintio+2014 (DC14) relation -- fit to
hydrodynamic cosmological zoom-in simulations as a function of X=log(M*/Mhalo)
-- with realistic intrinsic scatter. This addresses the 'synthetic toy library'
criticism. We also (i) check whether the network reproduces the library's inner
slope (the cusp-resolution / 'compression' test) and (ii) recompute the Sculptor
posterior under this physically-grounded prior.

Output: results/tables/dc14_retrain_posterior.npz (+ console)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.dm_models.diffusion_prior.diffusion import (
    make_schedule, ScoreMLP, train, sample, sample_posterior_importance)
from src.dm_models.parametric_priors import sigma_los2_from_Menc

N_RBINS = 32
R_GRID = np.logspace(-2.3, 0.5, N_RBINS)     # 5pc..3.2kpc (matches paper)
_LOGR = np.log10(R_GRID)
R_EVAL = 0.15
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
SCL_COND = np.array([10.0, 6.5, 8.0, 0.5])   # Sculptor conditioning
GAMMA_SCATTER = 0.22                          # intrinsic sim scatter in inner slope


def dc14_abg(X):
    X = np.clip(X, -4.1, -1.3)
    a = 2.94 - np.log10((10 ** (X + 2.33)) ** -1.08 + (10 ** (X + 2.33)) ** 2.29)
    b = 4.23 + 1.34 * X + 0.26 * X ** 2
    g = -0.06 + np.log10((10 ** (X + 2.56)) ** -0.68 + (10 ** (X + 2.56)))
    return a, b, g


def dc14_logrho(r, log_Mhalo, X, dgamma=0.0):
    a, b, g = dc14_abg(X)
    g = np.clip(g + dgamma, 0.0, 1.6)         # intrinsic scatter on inner slope
    Rv = 200.0 * (10 ** log_Mhalo / 1e12) ** (1.0 / 3.0)
    rs = (Rv / (10 ** (np.log10(14.0) - 0.13 * (log_Mhalo - 12.0)))) \
        / ((2.0 - g) / (b - 2.0)) ** (1.0 / a)
    x = r / rs
    return np.log10(x ** (-g) * (1.0 + x ** a) ** (-(b - g) / a) + 1e-300)


def make_dc14_dataset(n=8000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, N_RBINS)); C = np.zeros((n, 4)); m = 0
    while m < n:
        lmh = rng.uniform(9.2, 11.0)
        lms = 6.5 + 1.9 * (lmh - 10.0) + rng.normal(0, 0.3)
        Xr = lms - lmh
        if not (-4.1 < Xr < -1.3):
            continue
        t_SF = rng.uniform(4.0, 12.0)
        log_rt = np.log10(rng.uniform(1.0, 5.0))
        dg = rng.normal(0, GAMMA_SCATTER)
        lr = dc14_logrho(R_GRID, lmh, Xr, dg)
        # normalise amplitude loosely (data pins it); centre log-rho
        lr = lr - lr[np.argmin(np.abs(R_GRID - 0.3))] + 7.5
        X[m] = lr; C[m] = [lmh, lms, t_SF, log_rt]; m += 1
    return X, C


def gam_gnfwfit(lr, r=R_EVAL):
    def f(lx, A, lb, gm):
        x = (10 ** lx) / (10 ** lb)
        return A - gm * np.log10(x) - (3 - gm) * np.log10(1 + x)
    try:
        p, _ = curve_fit(f, _LOGR, lr, p0=[8, 0, 0.8],
                         bounds=([2, -1.5, 0], [12, 1.5, 1.6]), maxfev=4000)
        x = r / (10 ** p[1]); return p[2] + (3 - p[2]) * x / (1 + x)
    except Exception:
        return np.nan


def load_sculptor():
    mm = pd.read_parquet(DATA / "processed" / "sculptor_members_v0.parquet")
    mm = mm[(mm["P_member"] > 0.5) & mm["v_los_kms"].notna()]
    R = mm["R_pc"].values; v = mm["v_los_kms"].values
    verr = mm["v_err_kms"].fillna(mm["v_err_kms"].median()).values
    vr = v.copy()
    for _ in range(3):
        mad = np.median(np.abs(vr - np.median(vr)))
        vr = vr[np.abs(vr - np.median(vr)) < 4 * 1.4826 * mad]
    dv = v - float(np.median(vr))
    ed = np.linspace(np.log10(50), np.log10(2000), 11)
    bc = 0.5 * (ed[1:] + ed[:-1]); s2 = np.full_like(bc, np.nan); s2e = np.zeros_like(bc)
    for i, (lo, hi) in enumerate(zip(ed[:-1], ed[1:])):
        sel = (np.log10(R) >= lo) & (np.log10(R) < hi); nn = sel.sum()
        if nn < 5:
            continue
        s2[i] = dv[sel].var() - np.mean(verr[sel] ** 2); s2e[i] = max(s2[i], 1.0) * np.sqrt(2 / nn)
    ok = np.isfinite(s2) & (s2 > 0)
    return (10 ** bc[ok]) / 1000.0, s2[ok], s2e[ok] + 0.5


def main():
    t0 = time.time()
    print("[1] building DC14 simulation-calibrated library ...")
    X, C = make_dc14_dataset(8000, seed=0)
    gam_train = np.array([gam_gnfwfit(x) for x in X])
    gam_train = gam_train[np.isfinite(gam_train)]
    mu, sigma = X.mean(0), X.std(0) + 1e-6
    cm, cs = C.mean(0), C.std(0) + 1e-6
    Xs, Cs = (X - mu) / sigma, (C - cm) / cs
    qg = np.percentile(gam_train, [16, 50, 84])
    print(f"    library gamma(150pc): {qg[1]:.2f} [{qg[0]:.2f},{qg[2]:.2f}]  (N={len(gam_train)})")

    print("[2] training diffusion model (larger network, longer schedule) ...")
    sched = make_schedule(n_steps=200)
    model = ScoreMLP(dim_x=N_RBINS, dim_c=4, hidden=384, seed=0)
    train(model, Xs, Cs, sched, n_epochs=250, batch=128, lr=2e-3, verbose=False)
    train(model, Xs, Cs, sched, n_epochs=150, batch=128, lr=4e-4, verbose=False)

    cond = (SCL_COND - cm) / cs
    print("[3] prior-predictive at Sculptor conditioning (cusp-resolution check) ...")
    xp = sample(model, sched, cond=cond, n=3000, guidance=0.5,
                rng=np.random.default_rng(7), monotonic=(mu, sigma))
    lr_prior = xp * sigma + mu
    gam_prior = np.array([gam_gnfwfit(x) for x in lr_prior]); gam_prior = gam_prior[np.isfinite(gam_prior)]
    qp = np.percentile(gam_prior, [16, 50, 84])
    # training subset at Sculptor mass for the matched reference
    scl = np.abs(C[:, 0] - 10.0) < 0.3
    gscl = np.array([gam_gnfwfit(x) for x in X[scl]]); gscl = gscl[np.isfinite(gscl)]
    qs = np.percentile(gscl, [16, 50, 84])
    print(f"    training (Scl-mass) gamma = {qs[1]:.2f} [{qs[0]:.2f},{qs[2]:.2f}]")
    print(f"    prior-predictive   gamma = {qp[1]:.2f} [{qp[0]:.2f},{qp[2]:.2f}]  "
          f"<- compression = {qs[1]-qp[1]:+.2f}")

    print("[4] recomputing Sculptor posterior (amplitude-profiled importance) ...")
    R_obs, s2o, s2u = load_sculptor()
    iv = 1.0 / np.maximum(s2u, 1.0) ** 2

    def rho_to_Menc(lr):
        rho = 10 ** lr
        Menc = np.cumsum(4 * np.pi * R_GRID ** 2 * rho * np.gradient(R_GRID))
        return lambda r: np.interp(np.atleast_1d(r), R_GRID, Menc)

    def loglike(x_std):
        out = np.full(x_std.shape[0], -1e6)
        for i, xs in enumerate(x_std):
            lr = xs * sigma + mu
            try:
                s = sigma_los2_from_Menc(R_obs, rho_to_Menc(lr), Re_kpc=0.28)
                if np.any(~np.isfinite(s)) or np.all(s <= 0):
                    continue
                A = np.sum(s2o * s * iv) / np.sum(s * s * iv)
                if not np.isfinite(A) or A <= 0:
                    continue
                out[i] = -0.5 * np.sum((s2o - A * s) ** 2 * iv)
            except Exception:
                pass
        return out

    xpost, ess = sample_posterior_importance(
        model, sched, cond=cond, n_out=600, log_likelihood_fn=loglike,
        n_prior=6000, guidance=0.5, rng=np.random.default_rng(11),
        monotonic=(mu, sigma), return_diagnostics=True)
    lr_post = xpost * sigma + mu
    gam_post = np.array([gam_gnfwfit(x) for x in lr_post]); gam_post = gam_post[np.isfinite(gam_post)]
    qo = np.percentile(gam_post, [16, 50, 84])
    print(f"    posterior gamma(150pc) = {qo[1]:.2f} [{qo[0]:.2f},{qo[2]:.2f}]  (ESS={ess:.0f})")

    print("\n========== SUMMARY ==========")
    print(f"  DC14 library (all mass)     gamma = {qg[1]:.2f}")
    print(f"  DC14 prior @ Scl (training) gamma = {qs[1]:.2f}")
    print(f"  diffusion prior-predictive  gamma = {qp[1]:.2f}  (compression {qs[1]-qp[1]:+.2f})")
    print(f"  diffusion POSTERIOR         gamma = {qo[1]:.2f} [{qo[0]:.2f},{qo[2]:.2f}]")
    print(f"  [ref] hand-built posterior  = 0.49 ; single-pop gNFW = 0.90")
    print(f"  ({time.time()-t0:.0f}s)")
    np.savez(TAB / "dc14_retrain_posterior.npz",
             gam_train=gam_train, gam_prior=gam_prior, gam_post=gam_post,
             gam_scl_train=gscl, ess=ess)

    # figure: calibrated prior -> posterior vs the naive (toy) posterior
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG = ROOT / "results" / "figures" / "paper"; FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.4))
    ax[0].hist(gscl, bins=25, density=True, color="0.6", alpha=0.6, label="DC14 library")
    ax[0].hist(gam_prior, bins=25, density=True, histtype="step", color="C3", lw=2,
               label="diffusion prior-predictive")
    ax[0].set_xlabel(r"$\gamma(150\,$pc$)$"); ax[0].set_ylabel("density")
    ax[0].set_title(f"Cusp resolution: compression {qs[1]-qp[1]:+.2f}"); ax[0].legend(fontsize=8)
    ax[1].hist(gam_prior, bins=25, density=True, color="C3", alpha=0.4, label=f"prior ({qp[1]:.2f})")
    ax[1].hist(gam_post, bins=25, density=True, color="C0", alpha=0.6, label=f"posterior ({qo[1]:.2f})")
    for x, c, lab in [(0.90, "k", "gNFW fit 0.90"), (0.53, "0.5", "Burkert 0.53"),
                      (0.49, "C2", "toy posterior 0.49")]:
        ax[1].axvline(x, color=c, ls="--", lw=1.2, label=lab)
    ax[1].set_xlabel(r"$\gamma(150\,$pc$)$"); ax[1].set_ylabel("density")
    ax[1].set_title("Calibrated prior + faithful net"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dc14_calibrated.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_dc14_calibrated.png", bbox_inches="tight", dpi=200)
    print("[fig] fig_dc14_calibrated")


if __name__ == "__main__":
    main()
