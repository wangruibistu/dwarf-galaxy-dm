"""M3 (Paper II): multi-population, anisotropy-marginalised spherical Jeans.

Two (or more) chemodynamically distinct tracer populations share the SAME
dark-matter potential M(<r) but have different tracer scale radii and, in
general, different velocity anisotropies. Fitting them jointly constrains the
enclosed mass at two radii and breaks the mass-anisotropy degeneracy that a
single population cannot. Anisotropy is treated as a free, marginalised
parameter per population (constant-beta Binney-Mamon projection; reduces to the
isotropic case at beta=0).

This module provides the forward model (predicted projected dispersion per
population) and a Gaussian binned log-likelihood, usable both directly and as
the simulator inside the SBI engine (M4).
"""
from __future__ import annotations
import numpy as np

G_NEWT = 4.300917270e-6     # kpc (km/s)^2 / Msun


def plummer_density(r, Re):
    return (3.0 / (4.0 * np.pi * Re ** 3)) * (1.0 + (r / Re) ** 2) ** (-2.5)


def sigma_los2_aniso(R_obs, M_enc_func, beta, Re, n_r=300):
    """Constant-anisotropy projected l.o.s. dispersion^2 for a Plummer tracer of
    scale Re in the potential with enclosed mass M_enc_func(r)."""
    R_obs = np.atleast_1d(R_obs).astype(float)
    r = np.logspace(np.log10(max(R_obs.min() * 0.05, 1e-3)), np.log10(50.0), n_r)
    nu = plummer_density(r, Re)
    M = M_enc_func(r)
    dr = np.gradient(r)
    integ = (r ** (2 * beta)) * nu * G_NEWT * M / r ** 2
    tail = np.flip(np.cumsum(np.flip(integ * dr)))
    nu_sig2 = (r ** (-2 * beta)) * tail
    out = np.zeros_like(R_obs)
    for i, Ri in enumerate(R_obs):
        m = r > Ri
        denom = np.sqrt(np.where(m, r ** 2 - Ri ** 2, 1.0))
        kern = 1.0 - beta * Ri ** 2 / r ** 2
        num = 2.0 * np.sum(np.where(m, kern * nu_sig2 * r / denom, 0.0) * dr)
        den = 2.0 * np.sum(np.where(m, nu * r / denom, 0.0) * dr)
        out[i] = num / (den + 1e-30)
    return out


def predict_multipop(M_enc_func, pops, betas):
    """pops: list of dicts {R, Re}; betas: list of constant-beta per pop.
    Returns list of predicted sigma_los^2 arrays."""
    return [sigma_los2_aniso(p["R"], M_enc_func, b, p["Re"]) for p, b in zip(pops, betas)]


def multipop_loglike(M_enc_func, betas, pops, amp=None):
    """Gaussian binned log-likelihood across populations. Each pop dict carries
    R, Re, sig2_obs, sig2_err. If amp is None the overall mass amplitude is
    profiled out analytically (shared across populations, since sigma^2 is linear
    in it)."""
    preds = predict_multipop(M_enc_func, pops, betas)
    if amp is None:
        num = den = 0.0
        for p, s in zip(pops, preds):
            iv = 1.0 / np.maximum(p["sig2_err"], 1.0) ** 2
            num += np.sum(p["sig2_obs"] * s * iv)
            den += np.sum(s * s * iv)
        amp = num / den if den > 0 else 1.0
        if not np.isfinite(amp) or amp <= 0:
            return -1e6, amp
    ll = 0.0
    for p, s in zip(pops, preds):
        iv = 1.0 / np.maximum(p["sig2_err"], 1.0) ** 2
        ll += -0.5 * np.sum((p["sig2_obs"] - amp * s) ** 2 * iv)
    return ll, amp


# ---- gNFW enclosed mass (reference DM model for mocks/tests) ----
def gnfw_menc(r, log_rho_s, log_r_s, gamma, n_grid=800):
    """Vectorised: integrate rho once on a fine grid, then interpolate M(<r)."""
    rs = 10 ** log_r_s; rho_s = 10 ** log_rho_s
    rr = np.atleast_1d(np.asarray(r, float))
    x = np.logspace(-3.5, np.log10(max(rr.max(), 1e-3) * 1.01), n_grid)
    rho = rho_s * (x / rs) ** (-gamma) * (1 + x / rs) ** (-(3 - gamma))
    integ = 4 * np.pi * x ** 2 * rho
    Mcum = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(x))])
    return np.interp(rr, x, Mcum)


def gnfw_slope(r, log_r_s, gamma):
    x = r / 10 ** log_r_s
    return gamma + (3 - gamma) * x / (1 + x)
