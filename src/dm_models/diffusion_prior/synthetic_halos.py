"""Synthetic halo profile dataset generator.

Stands in for the FIRE-2 / EDGE / Symphony halo library in the PoC. Produces
a population of dwarf-scale halo ρ(r) profiles drawn from a mixture of cusp,
core, and SIDM-like archetypes, conditioned on (log M_halo, log M_star).

The point is to give the diffusion model a non-trivial, multimodal distribution
of profile shapes that it must learn — exactly what FIRE/EDGE actually produce.
"""

from __future__ import annotations
import numpy as np

# Radius grid (log-spaced, kpc) shared by all profiles
N_RBINS = 32
R_GRID = np.logspace(-2.3, 0.5, N_RBINS)   # 5 pc to 3 kpc


def gnfw_profile(rho_s, r_s, gamma, r=R_GRID):
    x = r / r_s
    return rho_s / (x ** gamma * (1.0 + x) ** (3.0 - gamma))


def core_nfw_profile(rho_s, r_s, r_c, r=R_GRID):
    """coreNFW = NFW × f^n with f = tanh(r/r_c) (Read+ 2016)."""
    base = gnfw_profile(rho_s, r_s, 1.0, r)
    f = np.tanh(r / r_c)
    return base * f


def sidm_profile(rho_s, r_s, r_iso, sigma_iso, r=R_GRID):
    """SIDM isothermal core + NFW outer (Kaplinghat+ 2014)."""
    # Inside r_iso: isothermal core; outside: NFW
    nfw = gnfw_profile(rho_s, r_s, 1.0, r)
    iso = (sigma_iso ** 2 / (2 * np.pi * 4.3e-6 * r_iso ** 2)) * np.exp(
        -((r - 0) ** 2) / (2 * r_iso ** 2)
    )
    # Blend smoothly
    w = 1.0 / (1.0 + np.exp((r - r_iso) / (0.1 * r_iso)))
    return w * iso + (1 - w) * nfw


def sample_halo(rng, log_M_halo: float, log_M_star: float, archetype: str):
    """Sample one ρ(r) profile from the requested archetype.

    Returns ρ(r) at R_GRID in Msun/kpc³.
    """
    M_h = 10 ** log_M_halo
    # Concentration-mass relation (very approximate Wechsler+ 2002)
    c = 14.0 * (M_h / 1e12) ** -0.13 * np.exp(rng.normal(0, 0.1))
    r_vir = 200.0 * (M_h / 1e12) ** (1.0 / 3.0)  # kpc, crude
    r_s = r_vir / c
    # rho_s normalisation s.t. M(r_vir) = M_h (for γ=1 NFW)
    f_c = np.log(1 + c) - c / (1 + c)
    rho_s = M_h / (4 * np.pi * r_s ** 3 * f_c)

    if archetype == "cusp":
        gamma = float(np.clip(rng.normal(1.0, 0.1), 0.8, 1.3))
        return gnfw_profile(rho_s, r_s, gamma)

    if archetype == "core_feedback":
        # Strong feedback if M_star / M_halo > 1e-4
        ratio = 10 ** (log_M_star - log_M_halo)
        r_c = r_s * np.clip(0.5 * ratio / 1e-3, 0.05, 1.5)
        return core_nfw_profile(rho_s, r_s, r_c)

    if archetype == "sidm":
        sigma_iso = 10 * np.sqrt(M_h / 1e10) * np.exp(rng.normal(0, 0.15))
        r_iso = r_s * rng.uniform(0.3, 1.0)
        return sidm_profile(rho_s, r_s, r_iso, sigma_iso).clip(min=1e-2)

    raise ValueError(f"unknown archetype {archetype}")


def make_dataset(n_samples: int = 5000, seed: int = 0):
    """Generate the full synthetic dataset.

    Returns
    -------
    log_rho : (n, n_rbins) log10 ρ(r) — model inputs
    cond    : (n, 2) (log_M_halo, log_M_star) — conditioning vector
    archetype : (n,) string array — for diagnostics only
    """
    rng = np.random.default_rng(seed)
    archetypes = ["cusp", "core_feedback", "sidm"]
    out_rho = np.zeros((n_samples, N_RBINS))
    out_cond = np.zeros((n_samples, 2))
    out_arch = np.empty(n_samples, dtype=object)

    for i in range(n_samples):
        log_Mh = rng.uniform(9.0, 11.0)
        log_Ms = log_Mh + rng.uniform(-4.5, -2.5)
        arch = rng.choice(archetypes, p=[0.35, 0.4, 0.25])
        rho = sample_halo(rng, log_Mh, log_Ms, arch)
        rho = rho.clip(min=1e-2)
        out_rho[i] = np.log10(rho)
        out_cond[i] = [log_Mh, log_Ms]
        out_arch[i] = arch

    return out_rho, out_cond, out_arch
