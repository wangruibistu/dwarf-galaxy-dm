#!/usr/bin/env python3
"""Referee-response test: does the parametric family-choice swing in the Sculptor
inner slope survive non-zero velocity anisotropy?

The paper isolates the family-choice degeneracy at fixed isotropy (beta=0). A
referee will ask whether the Delta-gamma ~ 0.4 swing is an artefact of that
choice. Here we re-fit the same three density families to the same binned
sigma_los^2 profile under a constant-anisotropy (Binney-Mamon) spherical Jeans
projection, at beta = -0.3, 0, +0.3, and report the swing in each case.

Constant-beta anisotropic Jeans (Binney & Mamon 1982):
    nu sigma_r^2(r) = r^{-2 beta} \int_r^\infty s^{2 beta} nu(s) G M(s)/s^2 ds
    sigma_los^2(R) = (2/I(R)) \int_R^\infty (1 - beta R^2/r^2)
                       nu sigma_r^2 r / sqrt(r^2 - R^2) dr
beta=0 reduces to the isotropic projection used in the paper.

Output: results/tables/anisotropy_robustness.npz (+ console table)
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
    gnfw_menc, corenfw_menc, burkert_menc,
    gnfw_gamma_local, corenfw_gamma_local, burkert_gamma_local,
    plummer_density,
)

G_NEWT = 4.300917270e-6   # kpc (km/s)^2 / Msun
DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
R_EVAL = 0.15
BETAS = [-0.3, 0.0, 0.3]


def sigma_los2_aniso(R_obs_kpc, M_enc_func, beta, Re_kpc=0.28, n_r=300):
    """Constant-anisotropy line-of-sight dispersion projection."""
    R_obs_kpc = np.atleast_1d(R_obs_kpc).astype(float)
    r = np.logspace(np.log10(max(R_obs_kpc.min() * 0.05, 1e-3)), np.log10(50.0), n_r)
    nu = plummer_density(r, Re_kpc)
    M = M_enc_func(r)
    dr = np.gradient(r)
    # nu sigma_r^2 = r^{-2beta} \int_r^\infty s^{2beta} nu GM/s^2 ds
    integ = (r ** (2 * beta)) * nu * G_NEWT * M / r ** 2
    tail = np.flip(np.cumsum(np.flip(integ * dr)))      # \int_r^\infty ...
    nu_sig2 = (r ** (-2 * beta)) * tail
    out = np.zeros_like(R_obs_kpc)
    for i, Ri in enumerate(R_obs_kpc):
        m = r > Ri
        denom = np.sqrt(np.where(m, r ** 2 - Ri ** 2, 1.0))
        kern = (1.0 - beta * Ri ** 2 / r ** 2)
        num = 2.0 * np.sum(np.where(m, kern * nu_sig2 * r / denom, 0.0) * dr)
        den = 2.0 * np.sum(np.where(m, nu * r / denom, 0.0) * dr)
        out[i] = num / (den + 1e-30)
    return out


def make_ll(menc, bounds, beta, R_obs, s2, s2e, Re):
    def ll(theta):
        for v, (lo, hi) in zip(theta, bounds):
            if not (lo < v < hi):
                return -np.inf
        pred = sigma_los2_aniso(R_obs, lambda rr: menc(rr, *theta), beta, Re)
        if np.any(pred < 1e-3) or np.any(~np.isfinite(pred)):
            return -np.inf
        return -0.5 * np.sum(((s2 - pred) / np.maximum(s2e, 1.0)) ** 2)
    return ll


def fit(ll, c0, sc, nw=32, ns=2000, burn=800):
    rng = np.random.default_rng(0)
    p0 = rng.normal(c0, sc, size=(nw, len(c0)))
    smp = emcee.EnsembleSampler(nw, len(c0), ll)
    smp.run_mcmc(p0, ns, progress=False)
    return smp.get_chain(flat=True, discard=burn)


def main():
    members = pd.read_parquet(DATA / "processed" / "sculptor_members_v0.parquet")
    members = members[(members["P_member"] > 0.5) & members["v_los_kms"].notna()]
    R = members["R_pc"].values; v = members["v_los_kms"].values
    verr = members["v_err_kms"].fillna(members["v_err_kms"].median()).values
    vr = v.copy()
    for _ in range(3):
        mad = np.median(np.abs(vr - np.median(vr)))
        vr = vr[np.abs(vr - np.median(vr)) < 4 * 1.4826 * mad]
    dv = v - float(np.median(vr))
    edges = np.linspace(np.log10(50), np.log10(2000), 11)
    binc = 0.5 * (edges[1:] + edges[:-1]); s2 = np.full_like(binc, np.nan); s2e = np.zeros_like(binc)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (np.log10(R) >= lo) & (np.log10(R) < hi); n = sel.sum()
        if n < 5:
            continue
        s2[i] = dv[sel].var() - np.mean(verr[sel] ** 2); s2e[i] = max(s2[i], 1.0) * np.sqrt(2 / n)
    ok = np.isfinite(s2) & (s2 > 0)
    R_obs = (10 ** binc[ok]) / 1000.0; s2o = s2[ok]; s2u = s2e[ok] + 0.5
    Re = 0.28
    print(f"[data] {ok.sum()} sigma bins")

    fam = {
        "gNFW":    (gnfw_menc,    [(5,10),(-2.5,1.5),(0,1.7)],            [8.,-0.2,0.5],  [0.2,0.1,0.15], gnfw_gamma_local),
        "coreNFW": (corenfw_menc, [(5,10),(-2.5,1.5),(-2.5,1.0),(0.1,2.5)],[8.,-0.2,-0.5,1.],[0.2,0.1,0.2,0.2], corenfw_gamma_local),
        "Burkert": (burkert_menc, [(5,10),(-2.5,1.5)],                    [8.,-0.4],      [0.2,0.1],     burkert_gamma_local),
    }
    print(f"\n{'beta':>5s} {'gNFW':>14s} {'coreNFW':>14s} {'Burkert':>14s} {'swing':>7s}")
    rows = []
    for beta in BETAS:
        meds = {}
        for name, (menc, bnds, c0, sc, gam) in fam.items():
            t = time.time()
            flat = fit(make_ll(menc, bnds, beta, R_obs, s2o, s2u, Re), c0, sc)
            g = np.array([gam(R_EVAL, *th) for th in flat]).ravel()
            g = g[np.isfinite(g)]
            meds[name] = (np.median(g), np.percentile(g, 16), np.percentile(g, 84))
        swing = meds["gNFW"][0] - meds["Burkert"][0]
        print(f"{beta:+5.1f} "
              f"{meds['gNFW'][0]:6.2f}[{meds['gNFW'][1]:.2f},{meds['gNFW'][2]:.2f}] "
              f"{meds['coreNFW'][0]:5.2f}[{meds['coreNFW'][1]:.2f},{meds['coreNFW'][2]:.2f}] "
              f"{meds['Burkert'][0]:5.2f}[{meds['Burkert'][1]:.2f},{meds['Burkert'][2]:.2f}] "
              f"{swing:6.2f}")
        rows.append((beta, meds["gNFW"][0], meds["coreNFW"][0], meds["Burkert"][0], swing))
    rows = np.array(rows)
    np.savez(TAB / "anisotropy_robustness.npz", betas=rows[:, 0],
             gnfw=rows[:, 1], corenfw=rows[:, 2], burkert=rows[:, 3], swing=rows[:, 4])
    print(f"\n[result] family swing Delta-gamma across beta in [-0.3,0,0.3]: "
          f"{rows[:,4].min():.2f}-{rows[:,4].max():.2f}")


if __name__ == "__main__":
    main()
