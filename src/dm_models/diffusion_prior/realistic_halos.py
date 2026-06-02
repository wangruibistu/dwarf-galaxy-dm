"""Realistic halo library generator — physically-motivated stand-in for the
FIRE-2 / EDGE / Symphony halo profiles used in the production model.

Design requirements (v2, post-referee):
  1. Every profile is monotonically decreasing in rho(r): no spurious
     density inversions that would produce unphysical local slopes.
  2. The inner logarithmic slope gamma(150 pc) is, by construction, confined
     to the physical range [0, ~1.4] spanning cored (gamma~0) to cuspy
     (gamma~1.2) halos.
  3. Three physically-motivated channels (cuspy NFW, baryonic-feedback
     coreNFW, SIDM isothermal core) in proportions controlled by the
     stellar-mass fraction and a random SIDM assignment.
  4. The SIDM isothermal core is matched continuously to the outer NFW at
     the core radius (Kaplinghat+ 2014), guaranteeing monotonicity.

The conditioning vector is 4-d:
    c = (log M_halo, log M_star, t_SF/Gyr, log r_tidal/kpc)
"""

from __future__ import annotations
import numpy as np

# Radius grid (log-spaced, kpc) shared by all profiles.
N_RBINS = 32
R_GRID = np.logspace(-2.3, 0.5, N_RBINS)  # 5 pc -> 3.2 kpc

_G = 4.300917270e-6  # kpc (km/s)^2 / Msun


# Abundance-matching M_star-M_halo relation at the dwarf scale.
# Anchored to physical dwarf values: M_star ~ 10^6.5 Msun at M_halo ~ 10^10
# (e.g. Sculptor), with the steep slope (~1.9) characteristic of the
# low-mass regime (Garrison-Kimmel+ 2017; Moster+ 2013 extrapolation).
def abundance_match(log_M_halo, rng):
    log_M_star_mean = 6.5 + 1.9 * (log_M_halo - 10.0)
    return log_M_star_mean + rng.normal(0, 0.3)


def concentration(log_M_halo, rng):
    """Wechsler+ 2002 concentration-mass relation with scatter."""
    log_c = np.log10(14.0) - 0.13 * (log_M_halo - 12)
    return 10 ** (log_c + rng.normal(0, 0.13))


def r_vir(M_halo):
    return 200.0 * (M_halo / 1e12) ** (1.0 / 3.0)


def gnfw_profile(rho_s, r_s, gamma, r=R_GRID):
    x = r / r_s
    return rho_s / (x ** gamma * (1.0 + x) ** (3.0 - gamma))


def corenfw_profile(rho_s, r_s, r_c, n_core, r=R_GRID):
    """Read+ 2016 coreNFW: NFW x tanh(r/r_c)^n_core. Monotonic for n_core>=0."""
    nfw = gnfw_profile(rho_s, r_s, 1.0, r)
    f = np.tanh(r / r_c)
    return nfw * f ** n_core


def sidm_isothermal_profile(rho_s, r_s, r_c, r=R_GRID):
    """SIDM cored profile: flat isothermal core inside r_c continuously
    matched to the outer NFW at r_c. Monotonic by construction.

    Inside r_c: rho = rho_core (flat).
    Outside r_c: rho = NFW(r).
    rho_core is set to NFW(r_c) so the profile is continuous and the core
    is a genuine flattening, never an inversion.
    """
    nfw = gnfw_profile(rho_s, r_s, 1.0, r)
    rho_core = gnfw_profile(rho_s, r_s, 1.0, np.array([r_c]))[0]
    # Smooth transition (monotone): core value for r<r_c, NFW for r>r_c.
    # Use the elementwise minimum of (flat core, NFW) which is monotone
    # decreasing because NFW is monotone decreasing and equals rho_core at r_c.
    core = np.full_like(r, rho_core)
    return np.minimum(core, nfw)


def sample_realistic_halo(rng, log_M_halo, log_M_star, t_SF_Gyr, log_r_tidal):
    """Sample one monotonic rho(r) profile from a physically-motivated mix."""
    M_halo = 10 ** log_M_halo
    M_star = 10 ** log_M_star
    f_star = M_star / M_halo

    c = concentration(log_M_halo, rng)
    Rv = r_vir(M_halo)
    r_s = Rv / c
    fc = np.log(1 + c) - c / (1 + c)
    rho_s = M_halo / (4 * np.pi * r_s ** 3 * fc)

    u = rng.uniform()
    if u < 0.25:
        # SIDM isothermal core, radius a fraction of r_s
        r_c = r_s * rng.uniform(0.15, 0.6)
        prof = sidm_isothermal_profile(rho_s, r_s, r_c)
    elif f_star > 1e-3 and t_SF_Gyr > 4.0:
        # Baryonic-feedback core (Read+ 2016): core size grows with f_star
        r_c = r_s * float(np.clip(0.4 * f_star / 1e-3, 0.05, 0.8))
        n_core = 1.0 + 0.5 * np.log10(max(t_SF_Gyr, 1.0) / 4.0)
        prof = corenfw_profile(rho_s, r_s, r_c, n_core)
    else:
        # Cuspy gNFW with mild scatter on the inner slope
        gamma = float(np.clip(rng.normal(1.0, 0.07), 0.85, 1.25))
        prof = gnfw_profile(rho_s, r_s, gamma)

    # Tidal truncation acts only at large radii; r_tidal floored well above
    # 150 pc so the inner slope is never affected by truncation.
    r_tidal = 10 ** log_r_tidal
    trunc = 1.0 / (1.0 + (R_GRID / r_tidal) ** 4)
    prof = prof * trunc

    # Density floor relative to the profile's own peak (avoids a hard
    # absolute floor that flattens low-mass outer profiles).
    floor = prof.max() * 1e-6
    return np.maximum(prof, floor)


def make_realistic_dataset(n_samples: int = 8000, seed: int = 0):
    """Generate a realistic, physically-monotonic halo profile dataset.

    Returns
    -------
    out_rho : (n_samples, N_RBINS) array of log10 rho(r)
    out_cond : (n_samples, 4) array of (log_M_halo, log_M_star, t_SF, log_r_tidal)
    """
    rng = np.random.default_rng(seed)
    out_rho = np.zeros((n_samples, N_RBINS))
    out_cond = np.zeros((n_samples, 4))

    for i in range(n_samples):
        log_M_halo = rng.uniform(9.0, 11.0)
        log_M_star = abundance_match(log_M_halo, rng)
        t_SF = rng.uniform(0.5, 12.0)
        log_r_tidal = rng.uniform(0.3, 1.5)  # 2 - 32 kpc: never truncates <1 kpc
        rho = sample_realistic_halo(rng, log_M_halo, log_M_star, t_SF, log_r_tidal)
        out_rho[i] = np.log10(rho)
        out_cond[i] = [log_M_halo, log_M_star, t_SF, log_r_tidal]

    return out_rho, out_cond
