#!/usr/bin/env python3
"""M3 acceptance: show the multi-population, anisotropy-marginalised Jeans
likelihood breaks the mass-anisotropy degeneracy.

We build a 2-population mock from a known gNFW DM halo with non-zero anisotropy,
then infer the inner slope marginalising over anisotropy with (A) a single
population and (B) both populations jointly. A single tracer with free beta
leaves the slope degenerate; the joint fit pins it and recovers the injection.
We test an injected cusp and an injected core.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import emcee

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.dynamical_modeling.multipop_jeans import (
    sigma_los2_aniso, multipop_loglike, gnfw_menc, gnfw_slope)

R_EVAL = 0.15
LRS_T, LRSCALE_T = 8.3, 0.0           # truth log_rho_s, log_r_s (r_s = 1 kpc)
RE1, RE2 = 0.15, 0.35                  # compact / extended tracer scales (kpc)
BETA1_T, BETA2_T = 0.20, -0.10         # injected anisotropies
NBIN = 8


def make_pop(Re, beta, gamma, rng, frac_err=0.12):
    R = np.logspace(np.log10(0.06), np.log10(1.2), NBIN)
    s2 = sigma_los2_aniso(R, lambda r: gnfw_menc(r, LRS_T, LRSCALE_T, gamma), beta, Re)
    err = frac_err * s2 + 1.0
    obs = s2 + rng.normal(0, err)
    return dict(R=R, Re=Re, sig2_obs=obs, sig2_err=err)


def run(pops, seed, nstep=3000, burn=1200):
    nb = len(pops)
    lo = np.array([6.0, -1.0, 0.0] + [-0.5] * nb)
    hi = np.array([10.0, 1.0, 1.6] + [0.5] * nb)

    def lp(th):
        if np.any(th < lo) or np.any(th > hi):
            return -np.inf
        lrs, lrsc, g = th[:3]; betas = th[3:]
        ll, _ = multipop_loglike(lambda r: gnfw_menc(r, lrs, lrsc, g),
                                 list(betas), pops)
        return ll if np.isfinite(ll) else -np.inf

    nd = 3 + nb; nw = 4 * nd
    rng = np.random.default_rng(seed)
    p0 = rng.normal([8.0, 0.0, 0.7] + [0.0] * nb, [0.2, 0.1, 0.2] + [0.1] * nb,
                    size=(nw, nd))
    p0 = np.clip(p0, lo + 1e-3, hi - 1e-3)
    s = emcee.EnsembleSampler(nw, nd, lp)
    s.run_mcmc(p0, nstep, progress=False)
    flat = s.get_chain(flat=True, discard=burn)
    g = np.array([gnfw_slope(R_EVAL, lrsc, gg) for lrsc, gg in flat[:, 1:3]])
    return g


def main():
    t0 = time.time()
    for label, gamma_t in [("cusp", 1.0), ("core", 0.25)]:
        rng = np.random.default_rng(0)
        p1 = make_pop(RE1, BETA1_T, gamma_t, rng)
        p2 = make_pop(RE2, BETA2_T, gamma_t, rng)
        inj = gnfw_slope(R_EVAL, LRSCALE_T, gamma_t)
        gA = run([p1], seed=1)                       # single pop, free beta
        gB = run([p1, p2], seed=2)                   # two pops, free betas
        qA = np.percentile(gA, [16, 50, 84]); qB = np.percentile(gB, [16, 50, 84])
        wA, wB = qA[2] - qA[0], qB[2] - qB[0]
        print(f"\n[{label}] injected gamma(150pc) = {inj:.2f}")
        print(f"  single-pop (free beta): {qA[1]:.2f} [{qA[0]:.2f},{qA[2]:.2f}]  width={wA:.2f}")
        print(f"  two-pop    (free betas): {qB[1]:.2f} [{qB[0]:.2f},{qB[2]:.2f}]  width={wB:.2f}")
        rec = abs(qB[1] - inj) < (qB[2] - qB[1] + qB[1] - qB[0])
        print(f"  --> joint recovers injection: {rec} ; tightening x{wA/wB:.1f}")
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
