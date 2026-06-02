"""Spherical Jeans inversion utilities (gNFW + Plummer tracer).

Reference: Walker & Peñarrubia 2011 (ApJ 742, 20); Read+ 2018 GravSphere paper.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp

G = 4.300917270e-6  # kpc (km/s)^2 / Msun


def plummer_density(r, Re):
    return (3.0 / (4.0 * jnp.pi * Re**3)) * (1.0 + (r / Re) ** 2) ** -2.5


def gnfw_density(r, rho_s, r_s, gamma):
    x = r / r_s
    return rho_s / (x**gamma * (1.0 + x) ** (3.0 - gamma))


def gnfw_mass(r, log_rho_s, log_r_s, gamma, n_grid: int = 256):
    """Enclosed gNFW mass via cumulative trapezoid on a log grid."""
    rho_s = 10.0**log_rho_s
    r_s = 10.0**log_r_s
    rmax = jnp.maximum(jnp.max(r), 10.0 * r_s)
    grid = jnp.logspace(-3, jnp.log10(rmax), n_grid)
    integrand = 4.0 * jnp.pi * grid**2 * gnfw_density(grid, rho_s, r_s, gamma)
    dr = jnp.gradient(grid)
    Menc = jnp.cumsum(integrand * dr)
    return jnp.interp(r, grid, Menc)


def sigma_los2_isotropic(R_kpc, params, Re_kpc: float = 0.28, n_r: int = 200):
    """Line-of-sight σ²(R) for isotropic β=0 Plummer tracer in gNFW DM halo.

    Parameters
    ----------
    R_kpc : projected radii (kpc)
    params : (log_rho_s, log_r_s, gamma)
    Re_kpc : tracer Plummer scale (kpc)
    """
    log_rho_s, log_r_s, gamma = params
    r_grid = jnp.logspace(jnp.log10(jnp.min(R_kpc) * 0.1), jnp.log10(50.0), n_r)
    nu = plummer_density(r_grid, Re_kpc)
    M = gnfw_mass(r_grid, log_rho_s, log_r_s, gamma)
    integrand = nu * G * M / r_grid**2
    dr = jnp.gradient(r_grid)
    # Integrate from infinity inward
    nu_sig2 = jnp.flip(jnp.cumsum(jnp.flip(integrand * dr)))
    sig2 = nu_sig2 / (nu + 1e-30)

    def project(Ri):
        mask = r_grid > Ri
        denom = jnp.sqrt(jnp.where(mask, r_grid**2 - Ri**2, 1.0))
        num = 2.0 * jnp.sum(jnp.where(mask, nu * sig2 * r_grid / denom, 0.0) * dr)
        den = 2.0 * jnp.sum(jnp.where(mask, nu * r_grid / denom, 0.0) * dr)
        return num / (den + 1e-30)

    return jax.vmap(project)(R_kpc)


def gamma_at_radius(r_kpc, log_rho_s, log_r_s, gamma):
    """Local logarithmic slope -d ln ρ / d ln r of gNFW at radius r."""
    r_s = 10.0**log_r_s
    x = r_kpc / r_s
    return gamma + (3.0 - gamma) * x / (1.0 + x)
