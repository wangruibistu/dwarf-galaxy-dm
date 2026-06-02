"""Pure-numpy Jeans reference implementation (for smoke tests
without jax). The production version is `jeans.py` (jax)."""

from __future__ import annotations
import numpy as np

G = 4.300917270e-6  # kpc (km/s)^2 / Msun


def plummer_density(r, Re):
    return (3.0 / (4.0 * np.pi * Re**3)) * (1.0 + (r / Re) ** 2) ** -2.5


def gnfw_density(r, rho_s, r_s, gamma):
    x = r / r_s
    return rho_s / (x**gamma * (1.0 + x) ** (3.0 - gamma))


def gnfw_mass(r, log_rho_s, log_r_s, gamma, n_grid: int = 256):
    rho_s = 10.0**log_rho_s
    r_s   = 10.0**log_r_s
    rmax  = max(r.max(), 10.0 * r_s)
    grid  = np.logspace(-3, np.log10(rmax), n_grid)
    integrand = 4.0 * np.pi * grid**2 * gnfw_density(grid, rho_s, r_s, gamma)
    Menc = np.cumsum(integrand * np.gradient(grid))
    return np.interp(r, grid, Menc)


def sigma_los2_isotropic(R_kpc, params, Re_kpc: float = 0.28, n_r: int = 400):
    log_rho_s, log_r_s, gamma = params
    R_kpc = np.atleast_1d(R_kpc)
    r_grid = np.logspace(np.log10(max(R_kpc.min() * 0.05, 1e-3)),
                         np.log10(50.0), n_r)
    nu = plummer_density(r_grid, Re_kpc)
    M  = gnfw_mass(r_grid, log_rho_s, log_r_s, gamma)
    integrand = nu * G * M / r_grid**2
    dr = np.gradient(r_grid)
    nu_sig2 = np.flip(np.cumsum(np.flip(integrand * dr)))
    sig2_r  = nu_sig2 / (nu + 1e-30)

    out = np.zeros_like(R_kpc, dtype=float)
    for i, Ri in enumerate(R_kpc):
        mask = r_grid > Ri
        denom = np.sqrt(np.where(mask, r_grid**2 - Ri**2, 1.0))
        num = 2.0 * np.sum(np.where(mask, nu * sig2_r * r_grid / denom, 0.0) * dr)
        den = 2.0 * np.sum(np.where(mask, nu * r_grid / denom, 0.0) * dr)
        out[i] = num / (den + 1e-30)
    return out


def gamma_at_radius(r_kpc, log_rho_s, log_r_s, gamma):
    r_s = 10.0**log_r_s
    x = r_kpc / r_s
    return gamma + (3.0 - gamma) * x / (1.0 + x)
