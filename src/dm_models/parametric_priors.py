"""Parametric DM density profiles for head-to-head comparison vs diffusion prior.

All profiles parameterised consistently so the same Jeans likelihood evaluates
on a common radial grid. We expose:

    rho(r)            : density at radius r [Msun / kpc^3]
    M_enc(r)          : enclosed mass M(<r) [Msun]
    gamma_local(r)    : local logarithmic slope - dlnrho/dlnr at r

Three families:
    gNFW       : (rho_s, r_s, gamma)   -- generalised NFW
    coreNFW    : (rho_s, r_s, r_c, n)  -- Read+ 2016 baryonic-feedback form
    Burkert    : (rho_s, r_s)          -- classical cored profile
"""

from __future__ import annotations
import numpy as np

R_GRID = np.logspace(-2.3, 0.5, 32)  # 5 pc to 3 kpc, matches diffusion-prior grid


# ---------------------------------------------------------------------------
# gNFW: rho = rho_s / [(r/r_s)^gamma (1 + r/r_s)^(3-gamma)]
# ---------------------------------------------------------------------------
def gnfw_rho(r, log_rho_s, log_r_s, gamma):
    rho_s = 10.0 ** log_rho_s
    r_s = 10.0 ** log_r_s
    x = r / r_s
    return rho_s / (x ** gamma * (1.0 + x) ** (3.0 - gamma))


def gnfw_menc(r, log_rho_s, log_r_s, gamma, n_grid=256):
    """Enclosed mass via cumulative spherical integration on a log grid."""
    rho_s = 10.0 ** log_rho_s
    r_s = 10.0 ** log_r_s
    rmax = max(np.atleast_1d(r).max(), 30 * r_s)
    grid = np.logspace(-3, np.log10(rmax), n_grid)
    integrand = 4.0 * np.pi * grid ** 2 * gnfw_rho(grid, log_rho_s, log_r_s, gamma)
    Menc = np.cumsum(integrand * np.gradient(grid))
    return np.interp(np.atleast_1d(r), grid, Menc)


def gnfw_gamma_local(r, log_rho_s, log_r_s, gamma):
    r_s = 10.0 ** log_r_s
    x = np.atleast_1d(r) / r_s
    return gamma + (3.0 - gamma) * x / (1.0 + x)


# ---------------------------------------------------------------------------
# coreNFW (Read+ 2016): rho_NFW * tanh(r/r_c)^n
# ---------------------------------------------------------------------------
def corenfw_rho(r, log_rho_s, log_r_s, log_r_c, n_core):
    r_c = 10.0 ** log_r_c
    nfw = gnfw_rho(r, log_rho_s, log_r_s, 1.0)
    f = np.tanh(r / r_c)
    return nfw * f ** n_core


def corenfw_menc(r, log_rho_s, log_r_s, log_r_c, n_core, n_grid=256):
    rho_s = 10.0 ** log_rho_s
    r_s = 10.0 ** log_r_s
    rmax = max(np.atleast_1d(r).max(), 30 * r_s)
    grid = np.logspace(-3, np.log10(rmax), n_grid)
    integrand = 4.0 * np.pi * grid ** 2 * corenfw_rho(
        grid, log_rho_s, log_r_s, log_r_c, n_core
    )
    Menc = np.cumsum(integrand * np.gradient(grid))
    return np.interp(np.atleast_1d(r), grid, Menc)


def corenfw_gamma_local(r, log_rho_s, log_r_s, log_r_c, n_core, eps=1e-3):
    r = np.atleast_1d(r)
    rho_plus = corenfw_rho(r * (1 + eps), log_rho_s, log_r_s, log_r_c, n_core)
    rho_minus = corenfw_rho(r * (1 - eps), log_rho_s, log_r_s, log_r_c, n_core)
    return -(np.log(rho_plus) - np.log(rho_minus)) / (2 * eps)


# ---------------------------------------------------------------------------
# Burkert: rho = rho_s / [(1 + x)(1 + x^2)], x = r/r_s
# ---------------------------------------------------------------------------
def burkert_rho(r, log_rho_s, log_r_s):
    rho_s = 10.0 ** log_rho_s
    r_s = 10.0 ** log_r_s
    x = r / r_s
    return rho_s / ((1 + x) * (1 + x ** 2))


def burkert_menc(r, log_rho_s, log_r_s, n_grid=256):
    rho_s = 10.0 ** log_rho_s
    r_s = 10.0 ** log_r_s
    rmax = max(np.atleast_1d(r).max(), 30 * r_s)
    grid = np.logspace(-3, np.log10(rmax), n_grid)
    integrand = 4.0 * np.pi * grid ** 2 * burkert_rho(grid, log_rho_s, log_r_s)
    Menc = np.cumsum(integrand * np.gradient(grid))
    return np.interp(np.atleast_1d(r), grid, Menc)


def burkert_gamma_local(r, log_rho_s, log_r_s, eps=1e-3):
    r = np.atleast_1d(r)
    rho_plus = burkert_rho(r * (1 + eps), log_rho_s, log_r_s)
    rho_minus = burkert_rho(r * (1 - eps), log_rho_s, log_r_s)
    return -(np.log(rho_plus) - np.log(rho_minus)) / (2 * eps)


# ---------------------------------------------------------------------------
# Shared spherical isotropic Jeans projection
#   sigma_los^2(R) = (2 / Sigma(R)) int_R^inf nu(r) sigma_r^2(r) r / sqrt(r^2 - R^2) dr
# with nu the Plummer tracer density and
#   sigma_r^2(r) = (1/nu) int_r^inf nu(r') G M(r') / r'^2 dr'
# ---------------------------------------------------------------------------
G_NEWT = 4.300917270e-6  # kpc (km/s)^2 / Msun


def plummer_density(r, Re):
    return (3.0 / (4.0 * np.pi * Re ** 3)) * (1.0 + (r / Re) ** 2) ** -2.5


def sigma_los2_from_Menc(R_obs_kpc, M_enc_func, Re_kpc=0.28, n_r=200):
    """Project sigma_los^2(R) given a generic M_enc(r) callable.

    M_enc_func(r_array) -> M_enc array, in Msun, on the same units as r [kpc]
    """
    R_obs_kpc = np.atleast_1d(R_obs_kpc).astype(float)
    r_grid = np.logspace(
        np.log10(max(R_obs_kpc.min() * 0.05, 1e-3)),
        np.log10(50.0),
        n_r,
    )
    nu = plummer_density(r_grid, Re_kpc)
    M = M_enc_func(r_grid)
    integrand = nu * G_NEWT * M / r_grid ** 2
    dr = np.gradient(r_grid)
    nu_sig2 = np.flip(np.cumsum(np.flip(integrand * dr)))
    sig2_r = nu_sig2 / (nu + 1e-30)

    out = np.zeros_like(R_obs_kpc)
    for i, Ri in enumerate(R_obs_kpc):
        mask = r_grid > Ri
        denom = np.sqrt(np.where(mask, r_grid ** 2 - Ri ** 2, 1.0))
        num = 2.0 * np.sum(np.where(mask, nu * sig2_r * r_grid / denom, 0.0) * dr)
        den = 2.0 * np.sum(np.where(mask, nu * r_grid / denom, 0.0) * dr)
        out[i] = num / (den + 1e-30)
    return out


# ---------------------------------------------------------------------------
# Sampler-agnostic log-likelihood given binned sigma_los profile
# ---------------------------------------------------------------------------
def jeans_loglike_gnfw(theta, R_obs, sig2_obs, sig2_err, Re_kpc=0.28):
    """Gaussian log-likelihood on binned sigma^2; clip large violations."""
    log_rho_s, log_r_s, gamma = theta
    if not (5 < log_rho_s < 10):
        return -np.inf
    if not (-2.5 < log_r_s < 1.5):
        return -np.inf
    if not (0 < gamma < 1.7):
        return -np.inf
    pred = sigma_los2_from_Menc(
        R_obs,
        lambda r: gnfw_menc(r, log_rho_s, log_r_s, gamma),
        Re_kpc=Re_kpc,
    )
    if np.any(pred < 1e-3) or np.any(~np.isfinite(pred)):
        return -np.inf
    chi2 = np.sum(((sig2_obs - pred) / np.maximum(sig2_err, 1.0)) ** 2)
    return -0.5 * chi2


def jeans_loglike_corenfw(theta, R_obs, sig2_obs, sig2_err, Re_kpc=0.28):
    log_rho_s, log_r_s, log_r_c, n_core = theta
    if not (5 < log_rho_s < 10):
        return -np.inf
    if not (-2.5 < log_r_s < 1.5):
        return -np.inf
    if not (-2.5 < log_r_c < 1.0):
        return -np.inf
    if not (0.1 < n_core < 2.5):
        return -np.inf
    pred = sigma_los2_from_Menc(
        R_obs,
        lambda r: corenfw_menc(r, log_rho_s, log_r_s, log_r_c, n_core),
        Re_kpc=Re_kpc,
    )
    if np.any(pred < 1e-3) or np.any(~np.isfinite(pred)):
        return -np.inf
    chi2 = np.sum(((sig2_obs - pred) / np.maximum(sig2_err, 1.0)) ** 2)
    return -0.5 * chi2


def jeans_loglike_burkert(theta, R_obs, sig2_obs, sig2_err, Re_kpc=0.28):
    log_rho_s, log_r_s = theta
    if not (5 < log_rho_s < 10):
        return -np.inf
    if not (-2.5 < log_r_s < 1.5):
        return -np.inf
    pred = sigma_los2_from_Menc(
        R_obs,
        lambda r: burkert_menc(r, log_rho_s, log_r_s),
        Re_kpc=Re_kpc,
    )
    if np.any(pred < 1e-3) or np.any(~np.isfinite(pred)):
        return -np.inf
    chi2 = np.sum(((sig2_obs - pred) / np.maximum(sig2_err, 1.0)) ** 2)
    return -0.5 * chi2
