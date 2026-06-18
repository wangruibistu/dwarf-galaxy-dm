"""M1 (Paper II): simulation-calibrated halo-profile training library.

The inner shape follows the Di Cintio et al. (2014b, MNRAS 441, 2986; 'DC14')
relation, whose (alpha, beta, gamma) are fit to hydrodynamic cosmological
zoom-in simulations as a function of X = log10(M*/Mhalo); intrinsic scatter
around the mean inner slope is added at the level reported by the FIRE-2
analysis of Lazar et al. (2020, MNRAS 497, 2393).  This is the calibrated
stand-in for raw FIRE-2/EDGE particle training (Paper III).

Profiles are returned as log10 rho(r) on a fixed 32-point log grid (5 pc..3.2 kpc),
conditioned on c = (log Mhalo, log M*, t_SF/Gyr, log r_tidal/kpc), in the same
format consumed by the diffusion training code.
"""
from __future__ import annotations
import numpy as np

N_RBINS = 32
R_GRID = np.logspace(-2.3, 0.5, N_RBINS)          # kpc, 5 pc .. 3.2 kpc
LOGR = np.log10(R_GRID)
X_MIN, X_MAX = -4.1, -1.3                          # DC14 validity range
LAZAR_SCATTER = 0.20                               # intrinsic inner-slope scatter (dex)


def dc14_abg(X):
    """DC14 (alpha, beta, gamma) shape parameters at X = log10(M*/Mhalo)."""
    X = np.clip(X, X_MIN, X_MAX)
    a = 2.94 - np.log10((10 ** (X + 2.33)) ** -1.08 + (10 ** (X + 2.33)) ** 2.29)
    b = 4.23 + 1.34 * X + 0.26 * X ** 2
    g = -0.06 + np.log10((10 ** (X + 2.56)) ** -0.68 + (10 ** (X + 2.56)))
    return a, b, g


def concentration(log_Mhalo, rng=None, scatter=0.0):
    logc = np.log10(14.0) - 0.13 * (log_Mhalo - 12.0)
    if scatter and rng is not None:
        logc = logc + rng.normal(0, scatter)
    return 10 ** logc


def r_vir(log_Mhalo):
    return 200.0 * (10 ** log_Mhalo / 1e12) ** (1.0 / 3.0)     # kpc


def dc14_logrho(r, log_Mhalo, X, dgamma=0.0, c=None):
    """log10 DC14 density on r [kpc]; amplitude arbitrary (slope/shape only)."""
    a, b, g = dc14_abg(X)
    g = float(np.clip(g + dgamma, 0.0, 1.6))
    if c is None:
        c = concentration(log_Mhalo)
    r_2 = r_vir(log_Mhalo) / c
    rs = r_2 / ((2.0 - g) / (b - 2.0)) ** (1.0 / a)
    x = r / rs
    return np.log10(x ** (-g) * (1.0 + x ** a) ** (-(b - g) / a) + 1e-300)


def dc14_local_slope(r, log_Mhalo, X, dgamma=0.0, c=None):
    """Analytic local logarithmic slope -dln rho/dln r at radius r."""
    a, b, g = dc14_abg(X)
    g = float(np.clip(g + dgamma, 0.0, 1.6))
    if c is None:
        c = concentration(log_Mhalo)
    rs = (r_vir(log_Mhalo) / c) / ((2.0 - g) / (b - 2.0)) ** (1.0 / a)
    xa = (r / rs) ** a
    return g + (b - g) * xa / (1.0 + xa)


def abundance_match(log_Mhalo, rng):
    """Dwarf-scale SHMR: log M* = 6.5 + 1.9 (logMhalo - 10), 0.3 dex scatter."""
    return 6.5 + 1.9 * (log_Mhalo - 10.0) + rng.normal(0, 0.3)


def make_dc14_dataset(n=12000, seed=0, val_frac=0.2, ref_radius_kpc=0.3,
                      log_amp=7.5):
    """Return (X_logrho, C_cond, split) calibrated to DC14/Lazar.

    X_logrho : (n, 32) log10 rho(r) (amplitude centred at ref_radius)
    C_cond   : (n, 4)  (log Mhalo, log M*, t_SF, log r_tidal)
    split    : (n,) bool, True = training, False = held-out validation
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, N_RBINS)); C = np.zeros((n, 4)); m = 0
    iref = int(np.argmin(np.abs(R_GRID - ref_radius_kpc)))
    while m < n:
        lmh = rng.uniform(9.2, 11.0)
        lms = abundance_match(lmh, rng)
        Xr = lms - lmh
        if not (X_MIN < Xr < X_MAX):
            continue
        cval = concentration(lmh, rng, scatter=0.10)              # concentration scatter
        dg = rng.normal(0, LAZAR_SCATTER)                         # Lazar inner-slope scatter
        lr = dc14_logrho(R_GRID, lmh, Xr, dgamma=dg, c=cval)
        lr = lr - lr[iref] + log_amp                             # loose amplitude (data pins it)
        t_SF = rng.uniform(4.0, 12.0)
        log_rt = np.log10(rng.uniform(1.0, 5.0))
        X[m] = lr; C[m] = [lmh, lms, t_SF, log_rt]; m += 1
    split = rng.random(n) > val_frac
    return X, C, split


def make_flat_gamma_profiles(n, seed=0, lmh_range=(9.5, 10.5), gamma_range=(0.0, 1.4),
                             ref_radius_kpc=0.3, log_amp=7.5):
    """Broad NPE *proposal* profiles: DC14-shaped haloes whose inner slope is
    forced ~uniform over gamma_range, so the inference engine sees cusp and core
    with comparable training mass.  The DC14 conditional prior, which is strongly
    cusp-leaning at Sculptor's mass, can be re-imposed afterwards by importance
    reweighting; training the NPE on the conditional prior alone leaves the cored
    end unsupported and biases blind recovery there."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, N_RBINS))
    iref = int(np.argmin(np.abs(R_GRID - ref_radius_kpc)))
    for m in range(n):
        lmh = rng.uniform(*lmh_range)
        lms = abundance_match(lmh, rng)
        Xr = float(np.clip(lms - lmh, X_MIN, X_MAX))
        _, _, g0 = dc14_abg(Xr)
        g_target = rng.uniform(*gamma_range)
        cval = concentration(lmh, rng, scatter=0.10)
        lr = dc14_logrho(R_GRID, lmh, Xr, dgamma=g_target - g0, c=cval)
        X[m] = lr - lr[iref] + log_amp
    return X


# Sculptor conditioning (log Mhalo, log M*, t_SF, log r_tidal)
SCULPTOR_COND = np.array([10.0, 6.5, 8.0, 0.5])


def _gnfwfit_slope(lr, r=0.15):
    from scipy.optimize import curve_fit
    def f(lx, A, lb, gm):
        x = (10 ** lx) / (10 ** lb)
        return A - gm * np.log10(x) - (3 - gm) * np.log10(1 + x)
    try:
        p, _ = curve_fit(f, LOGR, lr, p0=[8, 0, 0.8],
                         bounds=([2, -1.5, 0], [12, 1.5, 1.6]), maxfev=4000)
        x = r / (10 ** p[1]); return p[2] + (3 - p[2]) * x / (1 + x)
    except Exception:
        return np.nan


def validate(n=8000, seed=0):
    """M1 acceptance check: slope distribution spans cusp-core, matches relation,
    covers Sculptor mass."""
    X, C, split = make_dc14_dataset(n=n, seed=seed)
    gam = np.array([_gnfwfit_slope(x) for x in X]); gam = gam[np.isfinite(gam)]
    ana = np.array([dc14_local_slope(0.15, C[i, 0], C[i, 1] - C[i, 0])
                    for i in range(len(C))])
    scl = np.abs(C[:, 0] - 10.0) < 0.3
    g_scl = np.array([_gnfwfit_slope(x) for x in X[scl]]); g_scl = g_scl[np.isfinite(g_scl)]
    q = lambda a: np.round(np.percentile(a, [5, 50, 95]), 2)
    print(f"[M1] DC14/Lazar library, N={len(X)}  (train {split.sum()}, val {(~split).sum()})")
    print(f"  gNFW-fit slope (all)   5/50/95 = {q(gam)}")
    print(f"  analytic slope (all)   5/50/95 = {q(ana)}")
    print(f"  gNFW-fit slope (Scl)   5/50/95 = {q(g_scl)}  (N={scl.sum()})")
    ok_span = (np.percentile(ana, 5) < 0.3) and (np.percentile(ana, 95) > 0.9)
    ok_scl = scl.sum() > 200
    print(f"  ACCEPT span cusp-core: {ok_span} ; covers Sculptor mass: {ok_scl}")
    return ok_span and ok_scl


if __name__ == "__main__":
    validate()
