#!/usr/bin/env python3
"""Multi-population inner-slope inference on the FLAMES CaT Sculptor sample
(1604 members) -- the data upgrade over the 148-star Mg subsample.

Split into metal-rich (MR, compact) and metal-poor (MP, extended) populations by
a 2-component GMM in ([Fe/H], log R), fit them jointly with the anisotropy-
marginalised two-population Jeans likelihood, and compare to a single-population
fit. Tests whether a clean, large CaT sample breaks the mass-anisotropy
degeneracy where the sparse Mg sample failed.

Output: results/tables/multipop_flames.npz (+ console)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import emcee
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.dynamical_modeling.multipop_jeans import (
    sigma_los2_aniso, multipop_loglike, gnfw_menc, gnfw_slope)

R_EVAL = 0.15
DATA = ROOT / "data" / "processed" / "sculptor_flames_members.parquet"


def wmedian(v, w):
    i = np.argsort(v); v, w = v[i], w[i]; c = np.cumsum(w) / w.sum()
    return v[np.searchsorted(c, 0.5)]


def wsigma_profile(R, dv, verr, w, nbin=6):
    edges = np.percentile(R[w > 0.1], np.linspace(0, 100, nbin + 1))
    cen = np.sqrt(edges[1:] * edges[:-1]); s2 = np.full(nbin, np.nan); e2 = np.full(nbin, np.nan)
    for i in range(nbin):
        m = (R >= edges[i]) & (R < edges[i + 1]); ww = w[m]; ws = ww.sum()
        if m.sum() < 8 or ws < 5:
            continue
        mu = np.sum(ww * dv[m]) / ws
        var = np.sum(ww * (dv[m] - mu) ** 2) / ws - np.sum(ww * verr[m] ** 2) / ws
        s2[i] = var; e2[i] = max(var, 1.0) * np.sqrt(2.0 / ws)
    ok = np.isfinite(s2) & (s2 > 0)
    return cen[ok] / 1000.0, s2[ok], e2[ok] + 0.5     # kpc


def fit(pops, seed, nstep=3000, burn=1200):
    nb = len(pops)
    lo = np.array([5.0, -1.5, 0.0] + [-0.5] * nb); hi = np.array([10.0, 1.2, 1.6] + [0.5] * nb)

    def lp(th):
        if np.any(th < lo) or np.any(th > hi):
            return -np.inf
        lrho, lrs, g = th[:3]; betas = th[3:]
        ll, _ = multipop_loglike(lambda r: gnfw_menc(r, lrho, lrs, g), list(betas), pops)
        return ll if np.isfinite(ll) else -np.inf

    nd = 3 + nb; nw = 6 * nd; rng = np.random.default_rng(seed)
    p0 = np.clip(rng.normal([8.0, -0.2, 0.6] + [0.0] * nb,
                            [0.3, 0.15, 0.2] + [0.1] * nb, (nw, nd)), lo + 1e-3, hi - 1e-3)
    s = emcee.EnsembleSampler(nw, nd, lp)
    try:                                  # emcee v3
        s.run_mcmc(p0, nstep, progress=False)
        f = s.get_chain(flat=True, discard=burn)
    except TypeError:                     # emcee v2
        s.run_mcmc(p0, nstep)
        f = s.chain[:, burn:, :].reshape(-1, nd)
    g150 = np.array([gnfw_slope(R_EVAL, lrs, gg) for lrs, gg in f[:, 1:3]])
    return g150, f


def main():
    t0 = time.time()
    d = pd.read_parquet(DATA)
    R = d["R_pc"].values; v = d["vlos"].values
    verr = d["evlos"].values; feh = d["feh"].values
    dv = v - np.median(v)
    print(f"[data] {len(d)} FLAMES CaT members; v_sys={np.median(v):.1f}")

    # 2-component GMM in [Fe/H] ONLY -- the population split must be chemical;
    # feeding radius into the clustering makes the "populations" inner/outer
    # radial shells, which over-constrains the mass profile and rails r_s.
    X = feh.reshape(-1, 1); X = (X - X.mean(0)) / X.std(0)
    gm = GaussianMixture(2, covariance_type="full", n_init=20, random_state=0).fit(X)
    post = gm.predict_proba(X)
    iMR = int(np.argmax([feh[gm.predict(X) == k].mean() for k in range(2)]))
    wMR = post[:, iMR]; wMP = post[:, 1 - iMR]
    ReMR = wmedian(R, wMR) / 1000.0; ReMP = wmedian(R, wMP) / 1000.0
    bic1 = GaussianMixture(1, n_init=5, random_state=0).fit(X).bic(X)
    bic2 = gm.bic(X)
    print(f"[gmm] MR N_eff={wMR.sum():.0f} Re={ReMR*1000:.0f}pc ; MP N_eff={wMP.sum():.0f} Re={ReMP*1000:.0f}pc")
    print(f"[gmm] BIC k=1: {bic1:.1f}  k=2: {bic2:.1f}  -> 1D [Fe/H] {'does NOT' if bic1 < bic2 else 'does'} favour 2 components")

    # hard median-[Fe/H] split: literature-motivated, stable against tail outliers
    med = np.median(feh)
    hMR = (feh > med).astype(float); hMP = 1.0 - hMR
    ReHMR = wmedian(R, hMR) / 1000.0; ReHMP = wmedian(R, hMP) / 1000.0
    print(f"[hard] MR N={int(hMR.sum())} Re={ReHMR*1000:.0f}pc ; MP N={int(hMP.sum())} Re={ReHMP*1000:.0f}pc")

    R1, s1, e1 = wsigma_profile(R, dv, verr, wMR)
    R2, s2, e2 = wsigma_profile(R, dv, verr, wMP)
    pMR = dict(R=R1, Re=ReMR, sig2_obs=s1, sig2_err=e1)
    pMP = dict(R=R2, Re=ReMP, sig2_obs=s2, sig2_err=e2)
    # single population: all members, Plummer scale = global weighted median
    Rs, ss, es = wsigma_profile(R, dv, verr, np.ones_like(R), nbin=8)
    pAll = dict(R=Rs, Re=wmedian(R, np.ones_like(R)) / 1000.0, sig2_obs=ss, sig2_err=es)

    Rh1, sh1, eh1 = wsigma_profile(R, dv, verr, hMR)
    Rh2, sh2, eh2 = wsigma_profile(R, dv, verr, hMP)
    pHMR = dict(R=Rh1, Re=ReHMR, sig2_obs=sh1, sig2_err=eh1)
    pHMP = dict(R=Rh2, Re=ReHMP, sig2_obs=sh2, sig2_err=eh2)

    print("[fit] single-population (free beta) ...")
    gS, fS = fit([pAll], seed=1)
    print("[fit] two-population, hard median-[Fe/H] split (free betas) ...")
    gH, fH = fit([pHMR, pHMP], seed=3)
    print("[fit] two-population, GMM soft split (free betas) ...")
    gM, fM = fit([pMR, pMP], seed=2)

    qS = np.percentile(gS, [16, 50, 84]); qM = np.percentile(gM, [16, 50, 84])
    qH = np.percentile(gH, [16, 50, 84])
    rsS = 10 ** np.percentile(fS[:, 1], 50); rsH = 10 ** np.percentile(fH[:, 1], 50)
    print("\n========== FLAMES CaT multi-population result ==========")
    print(f"  single-pop gamma(150pc) = {qS[1]:.2f} [{qS[0]:.2f},{qS[2]:.2f}]  width={qS[2]-qS[0]:.2f}  r_s={rsS:.2f} kpc")
    print(f"  two-pop (hard) gamma(150pc) = {qH[1]:.2f} [{qH[0]:.2f},{qH[2]:.2f}]  width={qH[2]-qH[0]:.2f}  r_s={rsH:.2f} kpc")
    print(f"  two-pop (GMM)  gamma(150pc) = {qM[1]:.2f} [{qM[0]:.2f},{qM[2]:.2f}]  width={qM[2]-qM[0]:.2f}")
    print(f"  tightening (hard) = {(qS[2]-qS[0])/(qH[2]-qH[0]):.2f}x")
    print(f"  [old Mg multipop was 1.30 [1.04,1.51] width 0.46 -- broadened vs single 0.27]")
    cored = qH[2] < 0.5; cusp = qH[0] > 0.8
    verdict = "CORED" if cored else ("CUSP" if cusp else "INTERMEDIATE/uncertain")
    print(f"  --> verdict: {verdict}   ({time.time()-t0:.0f}s)")

    np.savez(ROOT / "results" / "tables" / "multipop_flames.npz",
             g_single=gS, g_multi=gM, g_hard=gH, chain_single=fS, chain_hard=fH,
             ReMR=ReMR, ReMP=ReMP, ReHMR=ReHMR, ReHMP=ReHMP,
             NMR=wMR.sum(), NMP=wMP.sum(), NHMR=hMR.sum(), NHMP=hMP.sum(),
             bic1=bic1, bic2=bic2,
             R1=R1, s1=s1, e1=e1, R2=R2, s2=s2, e2=e2,
             Rh1=Rh1, sh1=sh1, eh1=eh1, Rh2=Rh2, sh2=sh2, eh2=eh2)


if __name__ == "__main__":
    main()
