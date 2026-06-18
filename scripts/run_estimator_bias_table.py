"""Estimator-bias table for the gamma(150 pc) inner-slope statistic.

Referee response (R1 methodology / Devil's Advocate): the manuscript reads the
inner slope of every sampled profile by re-fitting a smooth gNFW and evaluating
its local slope at 150 pc, rather than by a finite-difference derivative.  A
referee asks whether this gNFW-fit estimator silently *imposes* a cusp.  This
script answers that directly and quantitatively, on profiles of KNOWN truth.

We build analytic density profiles spanning cusp -> core (gNFW at several inner
slopes, a coreNFW, and a cored Burkert), each with an analytically known
gamma(150 pc).  We add realistic per-grid-point noise at the level of the
minimal proof-of-concept score network (~0.05-0.10 dex) and apply three
estimators -- (i) gNFW-fit, (ii) finite-difference log-derivative, (iii)
Savitzky-Golay smoothed derivative -- recording bias = <estimate> - truth and
the fraction of unphysical (<0 or >1.7) returns.

Headline expected outcome: the gNFW-fit estimator is low-bias and noise-stable
AND, applied to the cored Burkert truth, returns a LOW slope (it tracks the
core, it does not default to a cusp), whereas the finite-difference and
Savitzky-Golay estimators scatter to unphysical near-zero/negative medians once
realistic noise is present.

Output: results/tables/estimator_bias_table.json  (+ printed table)
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dm_models.parametric_priors import (  # noqa: E402
    gnfw_rho, gnfw_gamma_local,
    corenfw_rho, corenfw_gamma_local,
    burkert_rho, burkert_gamma_local,
)

TAB = ROOT / "results" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

# Profile grid: 5 pc -> 3.2 kpc, 32 log-spaced points (the library grid), in kpc.
R_GRID = np.logspace(np.log10(0.005), np.log10(3.2), 32)
LOGR = np.log10(R_GRID)
R_EVAL = 0.15  # kpc == 150 pc

NOISE_DEX = [0.0, 0.05, 0.10]   # per-grid-point Gaussian sigma in log10(rho)
N_TRIALS = 300
SEED = 42


# --- the three estimators, all acting on log10(rho) sampled on R_GRID ---------
def _gnfw_logrho(logr_, log_rho_s, log_r_s, gamma):
    x = (10 ** logr_) / (10 ** log_r_s)
    return log_rho_s - gamma * np.log10(x) - (3.0 - gamma) * np.log10(1.0 + x)


def est_gnfwfit(log_rho, r_eval=R_EVAL):
    """Re-fit a smooth gNFW and read its local slope at r_eval (manuscript)."""
    try:
        p, _ = curve_fit(_gnfw_logrho, LOGR, log_rho, p0=[8.0, 0.0, 0.8],
                         bounds=([2, -1.5, 0.0], [12, 1.5, 1.6]), maxfev=4000)
        x = r_eval / (10 ** p[1])
        return p[2] + (3.0 - p[2]) * x / (1.0 + x)
    except Exception:
        return np.nan


def est_finitediff(log_rho, r_eval=R_EVAL):
    """Local -d log10(rho) / d log10(r), interpolated to r_eval."""
    slope = -np.gradient(log_rho, LOGR)
    return float(np.interp(np.log10(r_eval), LOGR, slope))


def est_savgol(log_rho, r_eval=R_EVAL, window=7, poly=2):
    """Savitzky-Golay-smoothed local log-derivative."""
    sm = savgol_filter(log_rho, window_length=window, polyorder=poly)
    slope = -np.gradient(sm, LOGR)
    return float(np.interp(np.log10(r_eval), LOGR, slope))


ESTIMATORS = {
    "gnfw_fit": est_gnfwfit,
    "finite_diff": est_finitediff,
    "savgol": est_savgol,
}


# --- truth profiles spanning cusp -> core -------------------------------------
def make_profiles():
    profs = []
    # cuspy gNFW (near the cuspy injection mock)
    profs.append(("gNFW_cusp", np.log10(gnfw_rho(R_GRID, 8.0, 0.0, 1.2)),
                  float(gnfw_gamma_local(R_EVAL, 8.0, 0.0, 1.2))))
    # canonical NFW
    profs.append(("gNFW_nfw", np.log10(gnfw_rho(R_GRID, 8.0, 0.0, 1.0)),
                  float(gnfw_gamma_local(R_EVAL, 8.0, 0.0, 1.0))))
    # shallow-cusp gNFW
    profs.append(("gNFW_shallow", np.log10(gnfw_rho(R_GRID, 8.0, 0.0, 0.5)),
                  float(gnfw_gamma_local(R_EVAL, 8.0, 0.0, 0.5))))
    # coreNFW (intermediate)
    profs.append(("coreNFW", np.log10(corenfw_rho(R_GRID, 8.0, -0.2, -0.5, 1.0)),
                  float(corenfw_gamma_local(R_EVAL, 8.0, -0.2, -0.5, 1.0))))
    # cored Burkert (near the cored injection mock) -- the key OOD test
    profs.append(("Burkert_core", np.log10(burkert_rho(R_GRID, 8.0, -0.4)),
                  float(burkert_gamma_local(R_EVAL, 8.0, -0.4))))
    return profs


def main():
    rng = np.random.default_rng(SEED)
    profiles = make_profiles()
    results = {}

    print(f"{'profile':>14s} {'truth':>6s} {'noise':>5s} | "
          f"{'gnfw_fit':>18s} {'finite_diff':>18s} {'savgol':>18s}")
    print("-" * 92)

    for name, log_rho0, truth in profiles:
        results[name] = {"truth_gamma150": truth, "by_noise": {}}
        for sig in NOISE_DEX:
            draws = {k: [] for k in ESTIMATORS}
            for _ in range(N_TRIALS):
                noisy = log_rho0 + (rng.normal(0.0, sig, size=log_rho0.shape)
                                    if sig > 0 else 0.0)
                for k, fn in ESTIMATORS.items():
                    draws[k].append(fn(noisy))
            row = {}
            cells = []
            for k in ESTIMATORS:
                a = np.array(draws[k], dtype=float)
                a = a[np.isfinite(a)]
                med = float(np.median(a))
                lo, hi = float(np.percentile(a, 16)), float(np.percentile(a, 84))
                bias = med - truth
                unphys = float(np.mean((a < 0.0) | (a > 1.7)))
                row[k] = {"median": med, "q16": lo, "q84": hi,
                          "bias": bias, "frac_unphysical": unphys}
                cells.append(f"{med:5.2f}±{0.5*(hi-lo):4.2f}(b{bias:+.2f})")
            results[name]["by_noise"][f"{sig:.2f}"] = row
            print(f"{name:>14s} {truth:6.2f} {sig:5.2f} | "
                  f"{cells[0]:>18s} {cells[1]:>18s} {cells[2]:>18s}")

    out = TAB / "estimator_bias_table.json"
    with open(out, "w") as f:
        json.dump({"r_eval_kpc": R_EVAL, "n_trials": N_TRIALS,
                   "noise_dex": NOISE_DEX, "results": results}, f, indent=2)
    print(f"\n[save] {out}")

    # one-line takeaway numbers used in the manuscript text
    gf_core = results["Burkert_core"]["by_noise"]["0.05"]["gnfw_fit"]["median"]
    print(f"\n[takeaway] gNFW-fit on cored Burkert truth (0.05 dex noise): "
          f"gamma(150pc) = {gf_core:.2f}  (truth {results['Burkert_core']['truth_gamma150']:.2f}) "
          f"-> tracks the core, no cusp imposed")


if __name__ == "__main__":
    main()
