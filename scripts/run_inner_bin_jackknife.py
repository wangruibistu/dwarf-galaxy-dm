"""Inner-bin jackknife of the single-population gNFW Sculptor fit.

Referee response (R1 methodology): is the cusp-leaning single-population gNFW
inner slope driven by the sparse innermost sigma_los^2 bin (R~60 pc, N~22)?
We re-run the identical isotropic gNFW Jeans fit (same data, binning, priors and
sampler as run_head_to_head_sculptor.py) on (a) the full bin set and (b) the
bin set with the innermost valid bin removed, and compare gamma(150 pc).

A robust result is one where dropping the innermost bin shifts the median
gamma(150 pc) by much less than its 68 per cent width.

Output: results/tables/inner_bin_jackknife.json  (+ printed summary)
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import emcee

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dm_models.parametric_priors import (  # noqa: E402
    gnfw_gamma_local, jeans_loglike_gnfw,
)

DATA = ROOT / "data"
TAB = ROOT / "results" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

Re_kpc = 0.28
R_EVAL = 0.15
N_WALKERS, N_STEPS, BURN, SEED = 32, 2500, 1000, 0


def load_binned_profile():
    """Replicates the binning of run_head_to_head_sculptor.py exactly."""
    members = pd.read_parquet(DATA / "processed" / "sculptor_members_v0.parquet")
    members = members[(members["P_member"] > 0.5)
                      & members["v_los_kms"].notna()].reset_index(drop=True)
    R = members["R_pc"].values
    v = members["v_los_kms"].values
    verr = members["v_err_kms"].fillna(members["v_err_kms"].median()).values

    v_robust = v.copy()
    for _ in range(3):
        mad = np.median(np.abs(v_robust - np.median(v_robust)))
        mask = np.abs(v_robust - np.median(v_robust)) < 4 * 1.4826 * mad
        v_robust = v_robust[mask]
    v_sys = float(np.median(v_robust))
    dv = v - v_sys

    log_R_edges = np.linspace(np.log10(50), np.log10(2000), 11)
    binc = 0.5 * (log_R_edges[1:] + log_R_edges[:-1])
    sig2 = np.full_like(binc, np.nan)
    sig2_err = np.zeros_like(binc)
    nstar = np.zeros_like(binc, dtype=int)
    for i, (lo, hi) in enumerate(zip(log_R_edges[:-1], log_R_edges[1:])):
        sel = (np.log10(R) >= lo) & (np.log10(R) < hi)
        n = int(sel.sum())
        nstar[i] = n
        if n < 5:
            continue
        sig2[i] = dv[sel].var() - np.mean(verr[sel] ** 2)
        sig2_err[i] = max(sig2[i], 1.0) * np.sqrt(2 / n)

    ok = np.isfinite(sig2) & (sig2 > 0)
    R_obs = (10 ** binc[ok]) / 1000.0
    sig2_obs = sig2[ok]
    sig2_unc = sig2_err[ok] + 0.5
    nstar = nstar[ok]
    return R_obs, sig2_obs, sig2_unc, nstar, v_sys


def fit_gnfw(R_obs, sig2_obs, sig2_unc, seed=SEED):
    rng = np.random.default_rng(seed)
    p0 = rng.normal([8.0, -0.2, 0.5], [0.2, 0.1, 0.15], size=(N_WALKERS, 3))
    sampler = emcee.EnsembleSampler(
        N_WALKERS, 3,
        lambda t: jeans_loglike_gnfw(t, R_obs, sig2_obs, sig2_unc, Re_kpc),
    )
    sampler.run_mcmc(p0, N_STEPS)
    try:                                   # emcee >= 3
        flat = sampler.get_chain(flat=True, discard=BURN)
    except AttributeError:                 # emcee 2.x
        flat = sampler.chain[:, BURN:, :].reshape((-1, 3))
    gamma = np.array([gnfw_gamma_local(R_EVAL, *t) for t in flat]).flatten()
    gamma = gamma[np.isfinite(gamma)]
    return {
        "median": float(np.median(gamma)),
        "q16": float(np.percentile(gamma, 16)),
        "q84": float(np.percentile(gamma, 84)),
        "width68": float(np.percentile(gamma, 84) - np.percentile(gamma, 16)),
    }


def main():
    R_obs, sig2_obs, sig2_unc, nstar, v_sys = load_binned_profile()
    print(f"[load] {len(R_obs)} valid bins (v_sys={v_sys:.1f} km/s); "
          f"inner bin R={R_obs[0]*1000:.0f} pc, N={nstar[0]}")

    full = fit_gnfw(R_obs, sig2_obs, sig2_unc)
    drop = fit_gnfw(R_obs[1:], sig2_obs[1:], sig2_unc[1:])

    shift = drop["median"] - full["median"]
    robust = abs(shift) < full["width68"]
    print("\n=== gamma(150 pc): inner-bin jackknife ===")
    print(f"  full  ({len(R_obs)} bins): {full['median']:.2f} "
          f"[{full['q16']:.2f}, {full['q84']:.2f}]  width {full['width68']:.2f}")
    print(f"  drop inner ({len(R_obs)-1} bins): {drop['median']:.2f} "
          f"[{drop['q16']:.2f}, {drop['q84']:.2f}]  width {drop['width68']:.2f}")
    print(f"  median shift = {shift:+.2f}  "
          f"({'robust: shift < 68% width' if robust else 'WARNING: shift >= 68% width'})")

    out = TAB / "inner_bin_jackknife.json"
    with open(out, "w") as f:
        json.dump({
            "inner_bin_pc": float(R_obs[0] * 1000), "inner_bin_N": int(nstar[0]),
            "n_bins_full": int(len(R_obs)), "v_sys_kms": v_sys,
            "full": full, "drop_inner": drop,
            "median_shift": shift, "robust": bool(robust),
        }, f, indent=2)
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
