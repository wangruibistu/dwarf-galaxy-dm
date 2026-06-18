#!/usr/bin/env python3
"""EDGE reverse-direction control for the 'relocation' result.

The DC14 calibration (run_dc14_retrain_posterior.py) flips the hand-built cored
posterior (0.49) to a cusp (0.94). A referee can object that DC14, calibrated on
more massive systems, predicts a cusp at Sculptor's mass *almost by
construction*, so the reversal is close to a tautology and only demonstrates
relocation in the cusp direction.

This script runs the mirror-image experiment. We rebuild the calibrated library
from an EDGE-like core-formation prescription (Read+2016 coreNFW with an
EDGE-calibrated core radius rc = eta * R_half; e.g. Read, Agertz, Collins 2016;
Orkney+2021), which at Sculptor's mass predicts a CORE rather than a cusp. We
train the identical faithful (384-unit) network, and recompute the Sculptor
posterior under the *same* amplitude-profiled importance scheme and the *same*
real data. If the posterior now moves toward a core, the learned-prior result
tracks the prior's calibration in BOTH directions -- the relocation lesson, not
a one-way DC14 cusp artefact.

Output: results/tables/edge_reverse_posterior.npz (+ console)
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
R_GRID = np.logspace(-2.3, 0.5, N_RBINS)         # 5 pc .. 3.2 kpc (matches paper)
_LOGR = np.log10(R_GRID)
R_EVAL = 0.15
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
SCL_COND = np.array([10.0, 6.5, 8.0, 0.5])       # Sculptor conditioning
ETA_CORE = 3.0                                   # EDGE strong-core rc = eta * R_half
GAMMA_SCATTER = 0.22                             # matched to the DC14 run


def r_vir(log_Mhalo):
    return 200.0 * (10 ** log_Mhalo / 1e12) ** (1.0 / 3.0)          # kpc


def concentration(log_Mhalo):
    return 10 ** (np.log10(14.0) - 0.13 * (log_Mhalo - 12.0))


def r_half_of_mstar(log_Mstar):
    """Dwarf size-mass relation: R_half ~ 0.28 kpc at M*=10^6.5 (Sculptor),
    mild positive slope. Used to set the EDGE core radius rc = eta * R_half."""
    return 0.28 * 10 ** (0.25 * (log_Mstar - 6.5))


def edge_logrho(r, log_Mhalo, log_Mstar, n_core, dgamma=0.0):
    """log10 coreNFW density (Read, Agertz & Collins 2016) -- the EDGE-family
    core model. An NFW cusp is softened inside rc = eta * R_half by a tanh^n
    transfer. n_core in [0,1] sets the core strength (n->1 = full core, long
    star-formation duration); at Sculptor's mass rc >> 150 pc, so the inner
    slope is driven cored -- the deliberate mirror of the DC14 cusp library.
    `dgamma` adds intrinsic scatter by tilting the effective NFW inner slope."""
    c = concentration(log_Mhalo)
    rs = r_vir(log_Mhalo) / c
    g_in = float(np.clip(1.0 + dgamma, 0.3, 1.6))                   # NFW-ish cusp + scatter
    x = r / rs
    # generalised-NFW host cusp (gamma=g_in); enclosed mass via numeric integral
    rho_nfw = x ** (-g_in) * (1.0 + x) ** (-(3.0 - g_in))
    Mnfw = np.cumsum(4 * np.pi * r ** 2 * rho_nfw * np.gradient(r))
    rc = ETA_CORE * r_half_of_mstar(log_Mstar)
    f = np.tanh(r / rc)
    # coreNFW density: f^n rho_nfw + n f^{n-1}(1-f^2)/(4 pi r^2 rc) * Mnfw
    rho_core = f ** n_core * rho_nfw + (
        n_core * f ** (n_core - 1.0) * (1.0 - f ** 2) / (4 * np.pi * r ** 2 * rc) * Mnfw)
    rho_core = np.maximum(rho_core, 1e-300)
    return np.log10(rho_core)


def make_edge_dataset(n=8000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, N_RBINS)); C = np.zeros((n, 4)); m = 0
    iref = int(np.argmin(np.abs(R_GRID - 0.3)))
    while m < n:
        lmh = rng.uniform(9.2, 11.0)
        lms = 6.5 + 1.9 * (lmh - 10.0) + rng.normal(0, 0.3)
        Xr = lms - lmh
        if not (-4.1 < Xr < -1.3):
            continue
        # EDGE: core strength grows with star-formation duration / stellar mass
        n_core = float(np.clip(0.7 + 0.35 * (lms - 6.0) + rng.normal(0, 0.1), 0.4, 1.3))
        dg = rng.normal(0, GAMMA_SCATTER)
        t_SF = rng.uniform(4.0, 12.0)
        log_rt = np.log10(rng.uniform(1.0, 5.0))
        lr = edge_logrho(R_GRID, lmh, lms, n_core, dg)
        lr = lr - lr[iref] + 7.5                                    # loose amplitude
        X[m] = lr; C[m] = [lmh, lms, t_SF, log_rt]; m += 1
    return X, C


def gam_gnfwfit(lr, r=R_EVAL):
    def f(lx, A, lb, gm):
        x = (10 ** lx) / (10 ** lb)
        return A - gm * np.log10(x) - (3 - gm) * np.log10(1 + x)
    try:
        p, _ = curve_fit(f, _LOGR, lr, p0=[8, 0, 0.5],
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
    print("[1] building EDGE-like (coreNFW) core-forming library ...")
    X, C = make_edge_dataset(8000, seed=0)
    gam_train = np.array([gam_gnfwfit(x) for x in X]); gam_train = gam_train[np.isfinite(gam_train)]
    mu, sigma = X.mean(0), X.std(0) + 1e-6
    cm, cs = C.mean(0), C.std(0) + 1e-6
    Xs, Cs = (X - mu) / sigma, (C - cm) / cs
    qg = np.percentile(gam_train, [16, 50, 84])
    scl = np.abs(C[:, 0] - 10.0) < 0.3
    gscl = np.array([gam_gnfwfit(x) for x in X[scl]]); gscl = gscl[np.isfinite(gscl)]
    qs = np.percentile(gscl, [16, 50, 84])
    print(f"    library gamma(150pc) all      = {qg[1]:.2f} [{qg[0]:.2f},{qg[2]:.2f}]  (N={len(gam_train)})")
    print(f"    library gamma(150pc) Scl-mass = {qs[1]:.2f} [{qs[0]:.2f},{qs[2]:.2f}]  (N={scl.sum()})")
    if qs[1] > 0.6:
        print("    WARNING: EDGE library is not cored at Sculptor mass; check prescription")

    print("[2] training faithful diffusion model (384-unit, longer schedule) ...")
    sched = make_schedule(n_steps=200)
    model = ScoreMLP(dim_x=N_RBINS, dim_c=4, hidden=384, seed=0)
    train(model, Xs, Cs, sched, n_epochs=250, batch=128, lr=2e-3, verbose=False)
    train(model, Xs, Cs, sched, n_epochs=150, batch=128, lr=4e-4, verbose=False)

    cond = (SCL_COND - cm) / cs
    print("[3] prior-predictive at Sculptor conditioning ...")
    xp = sample(model, sched, cond=cond, n=3000, guidance=0.5,
                rng=np.random.default_rng(7), monotonic=(mu, sigma))
    lr_prior = xp * sigma + mu
    gam_prior = np.array([gam_gnfwfit(x) for x in lr_prior]); gam_prior = gam_prior[np.isfinite(gam_prior)]
    qp = np.percentile(gam_prior, [16, 50, 84])
    print(f"    prior-predictive gamma = {qp[1]:.2f} [{qp[0]:.2f},{qp[2]:.2f}]  "
          f"(library Scl {qs[1]:.2f}; fidelity gap {qs[1]-qp[1]:+.2f})")

    print("[4] recomputing Sculptor posterior (same data, same importance scheme) ...")
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

    print("\n========== EDGE reverse-direction control ==========")
    print(f"  EDGE library  @ Scl mass    gamma = {qs[1]:.2f}   (DC14 was 0.73)")
    print(f"  EDGE diffusion prior-pred   gamma = {qp[1]:.2f}   (DC14 was 0.66)")
    print(f"  EDGE POSTERIOR (real Scl)   gamma = {qo[1]:.2f} [{qo[0]:.2f},{qo[2]:.2f}]")
    print(f"  [ref] DC14 posterior = 0.94 ; hand-built toy = 0.49 ; gNFW fit = 0.90")
    print(f"  -> prior-driven posterior swing DC14->EDGE = {0.94 - qo[1]:+.2f}")
    print(f"  ({time.time()-t0:.0f}s)")
    np.savez(TAB / "edge_reverse_posterior.npz",
             gam_train=gam_train, gam_prior=gam_prior, gam_post=gam_post,
             gam_scl_train=gscl, qs=qs, qp=qp, qo=qo, ess=ess)
    print(f"[save] {TAB / 'edge_reverse_posterior.npz'}")


if __name__ == "__main__":
    main()
