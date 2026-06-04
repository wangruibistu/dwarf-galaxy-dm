#!/usr/bin/env python3
"""Referee-response test: what inner slope does a *simulation-calibrated* halo
library predict at Sculptor's mass?

Real FIRE-2/EDGE particle data are gated; we therefore use the Di Cintio et al.
(2014b, MNRAS 441, 2986; 'DC14') density profile, whose inner (alpha,beta,gamma)
shape parameters are fit to hydrodynamic cosmological zoom-in simulations as a
function of the stellar-to-halo mass ratio X = log10(M*/M_halo). DC14 is the
standard analytic encoding of the feedback core-cusp relation and is consistent
with the FIRE-2 result of Lazar et al. (2020).

We build a DC14 library over the same halo-mass range and abundance-matching
relation as the paper's hand-built library, measure the inner slope gamma(150pc)
of every profile with the identical gNFW-fit estimator, and report the
distribution -- in particular at Sculptor's conditioning (log M_halo ~ 10,
X ~ -3.5). This tests whether a simulation-grounded prior is cuspy, cored, or
intermediate at Sculptor's mass.

Output: results/tables/dc14_prior_predictive.npz (+ console)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

R_GRID = np.logspace(np.log10(0.005), np.log10(3.2), 32)   # kpc, 5pc..3.2kpc
_LOGR = np.log10(R_GRID)
R_EVAL = 0.15


# ---- DC14 (Di Cintio+2014b) shape parameters vs X = log10(M*/M_halo) ----
def dc14_abg(X):
    X = np.clip(X, -4.1, -1.3)
    a = 2.94 - np.log10((10 ** (X + 2.33)) ** -1.08 + (10 ** (X + 2.33)) ** 2.29)
    b = 4.23 + 1.34 * X + 0.26 * X ** 2
    g = -0.06 + np.log10((10 ** (X + 2.56)) ** -0.68 + (10 ** (X + 2.56)))
    return a, b, g


def dc14_logrho(r, log_Mhalo, X):
    """log10 DC14 density on radius array r [kpc]; amplitude arbitrary (slope only)."""
    a, b, g = dc14_abg(X)
    # r_{-2} from mass-concentration (paper's relation), then DC14 r_s
    logc = np.log10(14.0) - 0.13 * (log_Mhalo - 12.0)
    c = 10 ** logc
    Rv = 200.0 * (10 ** log_Mhalo / 1e12) ** (1.0 / 3.0)     # kpc
    r_2 = Rv / c
    rs = r_2 / ((2.0 - g) / (b - 2.0)) ** (1.0 / a)
    x = r / rs
    rho = x ** (-g) * (1.0 + x ** a) ** (-(b - g) / a)
    return np.log10(rho + 1e-300)


def dc14_local_slope(r, log_Mhalo, X):
    """Analytic DC14 local slope -dln rho/dln r at radius r."""
    a, b, g = dc14_abg(X)
    logc = np.log10(14.0) - 0.13 * (log_Mhalo - 12.0)
    Rv = 200.0 * (10 ** log_Mhalo / 1e12) ** (1.0 / 3.0)
    r_2 = Rv / 10 ** logc
    rs = r_2 / ((2.0 - g) / (b - 2.0)) ** (1.0 / a)
    xa = (r / rs) ** a
    return g + (b - g) * xa / (1.0 + xa)


def gam_gnfwfit(lr, r=R_EVAL):
    def f(lx, A, lb, gm):
        x = (10 ** lx) / (10 ** lb)
        return A - gm * np.log10(x) - (3 - gm) * np.log10(1 + x)
    try:
        p, _ = curve_fit(f, _LOGR, lr, p0=[8, 0, 0.8],
                         bounds=([2, -1.5, 0], [12, 1.5, 1.6]), maxfev=4000)
        x = r / (10 ** p[1])
        return p[2] + (3 - p[2]) * x / (1 + x)
    except Exception:
        return np.nan


def abundance_match(log_Mhalo, rng):
    return 6.5 + 1.9 * (log_Mhalo - 10.0) + rng.normal(0, 0.3)


def main():
    rng = np.random.default_rng(0)
    N = 4000
    gam_fit, gam_ana, Xs, lmh = [], [], [], []
    for _ in range(N):
        log_Mhalo = rng.uniform(9.2, 11.0)       # DC14-valid X along abund. matching
        log_Mstar = abundance_match(log_Mhalo, rng)
        X = log_Mstar - log_Mhalo
        if not (-4.1 < X < -1.3):
            continue
        lr = dc14_logrho(R_GRID, log_Mhalo, X)
        gam_fit.append(gam_gnfwfit(lr))
        gam_ana.append(float(dc14_local_slope(R_EVAL, log_Mhalo, X)))
        Xs.append(X); lmh.append(log_Mhalo)
    gam_fit = np.array(gam_fit); gam_ana = np.array(gam_ana)
    Xs = np.array(Xs); lmh = np.array(lmh)
    good = np.isfinite(gam_fit)

    # Sculptor conditioning subset: log M_halo ~ 10 (X ~ -3.5)
    scl = good & (np.abs(lmh - 10.0) < 0.3)

    def q(a):
        return np.percentile(a, [16, 50, 84])

    print(f"DC14 simulation-calibrated library ({good.sum()} profiles)")
    print(f"  full library  gamma(150pc)  gNFW-fit : {q(gam_fit[good])}")
    print(f"  full library  gamma(150pc)  analytic : {q(gam_ana[good])}")
    print(f"\nSculptor conditioning (logMhalo~10, X~-3.5; N={scl.sum()}):")
    qf = q(gam_fit[scl]); qa = q(gam_ana[scl])
    print(f"  gNFW-fit gamma(150pc) = {qf[1]:.2f} [{qf[0]:.2f}, {qf[2]:.2f}]")
    print(f"  analytic gamma(150pc) = {qa[1]:.2f} [{qa[0]:.2f}, {qa[2]:.2f}]")
    # DC14 asymptotic gamma at Sculptor's X
    _, _, gscl = dc14_abg(-3.5)
    print(f"  DC14 asymptotic gamma at X=-3.5 : {gscl:.2f}")
    print(f"\nCompare: paper hand-built library at Sculptor mass = 1.14 (cuspy);")
    print(f"         DMO NFW local slope at 150pc ~ {1.0 + 2*0.15/ (43/25.5):.2f} (cuspy);")
    print(f"         paper diffusion posterior = 0.49 (cored).")

    TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
    np.savez(TAB / "dc14_prior_predictive.npz",
             gam_fit=gam_fit[good], gam_ana=gam_ana[good], X=Xs[good], logMhalo=lmh[good],
             scl_gam_fit=gam_fit[scl], scl_gam_ana=gam_ana[scl])


if __name__ == "__main__":
    main()
