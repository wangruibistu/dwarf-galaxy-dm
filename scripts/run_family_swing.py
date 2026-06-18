#!/usr/bin/env python3
"""Model-independent family-swing replication on a second dSph.

Repeats the three-family head-to-head parametric Jeans fit (gNFW / coreNFW /
Burkert) under an identical isotropic projection, to test whether the
Delta-gamma family-choice swing isolated on Sculptor is generic. This is the
parametric, diffusion-free core of ``run_head_to_head_sculptor.py``,
parameterised by galaxy.

Usage:
    python scripts/run_family_swing.py --galaxy Fornax --Re 0.71

Outputs:
    results/tables/family_swing_<galaxy>.json
    results/figures/paper/fig_<galaxy>_family_swing.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import emcee

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.dm_models.parametric_priors import (
    gnfw_gamma_local, corenfw_gamma_local, burkert_gamma_local,
    jeans_loglike_gnfw, jeans_loglike_corenfw, jeans_loglike_burkert,
)

DATA = ROOT / "data" / "processed"
FIG = ROOT / "results" / "figures" / "paper"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

N_WALKERS, N_STEPS, BURN, R_EVAL = 32, 2500, 1000, 0.15


def bin_sigma(R_pc, dv, verr, rmin, rmax, nbin):
    edges = np.linspace(np.log10(rmin), np.log10(rmax), nbin + 1)
    binc = 0.5 * (edges[1:] + edges[:-1])
    sig2 = np.full(nbin, np.nan); sig2_err = np.zeros(nbin); counts = np.zeros(nbin, int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (np.log10(R_pc) >= lo) & (np.log10(R_pc) < hi)
        n = int(sel.sum()); counts[i] = n
        if n < 5:
            continue
        sig2[i] = dv[sel].var() - np.mean(verr[sel] ** 2)
        sig2_err[i] = max(sig2[i], 1.0) * np.sqrt(2 / n)
    return binc, sig2, sig2_err, counts


def run_emcee(name, log_post, dim, p0c, p0s):
    rng = np.random.default_rng(0)
    p0 = rng.normal(p0c, p0s, size=(N_WALKERS, dim))
    t0 = time.perf_counter()
    sampler = emcee.EnsembleSampler(N_WALKERS, dim, log_post)
    sampler.run_mcmc(p0, N_STEPS, progress=False)
    flat = sampler.get_chain(flat=True, discard=BURN)
    try:
        ess = (N_STEPS - BURN) * N_WALKERS / np.max(sampler.get_autocorr_time(quiet=True))
    except Exception:
        ess = float("nan")
    print(f"[mcmc] {name:8s} wall={time.perf_counter()-t0:5.1f}s ESS_min={ess:6.0f}")
    return flat


def pctl(s):
    s = np.asarray(s); s = s[np.isfinite(s)]
    return float(np.median(s)), float(np.percentile(s, 16)), float(np.percentile(s, 84))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--galaxy", default="Fornax")
    ap.add_argument("--Re", type=float, default=0.71, help="Plummer Re [kpc]")
    ap.add_argument("--rmin", type=float, default=80.0)
    ap.add_argument("--rmax", type=float, default=3000.0)
    ap.add_argument("--nbin", type=int, default=10)
    ap.add_argument("--pmember", type=float, default=0.5)
    args = ap.parse_args()
    tag = args.galaxy.lower()

    m = pd.read_parquet(DATA / f"{tag}_members_v0.parquet")
    m = m[(m["P_member"] > args.pmember) & m["v_los_kms"].notna()].reset_index(drop=True)
    R = m["R_pc"].values
    v = m["v_los_kms"].values
    verr = m["v_err_kms"].fillna(m["v_err_kms"].median()).values

    v_robust = v.copy()
    for _ in range(3):
        mad = np.median(np.abs(v_robust - np.median(v_robust)))
        v_robust = v_robust[np.abs(v_robust - np.median(v_robust)) < 4 * 1.4826 * mad]
    v_sys = float(np.median(v_robust))
    dv = v - v_sys
    sig_global = float(np.sqrt(np.maximum(dv.var() - np.mean(verr ** 2), 0.0)))
    print(f"[load] {len(m)} {args.galaxy} members  v_sys={v_sys:.1f} km/s  "
          f"sigma_global={sig_global:.1f} km/s")

    binc, sig2, sig2_err, counts = bin_sigma(R, dv, verr, args.rmin, args.rmax, args.nbin)
    ok = np.isfinite(sig2) & (sig2 > 0)
    R_obs = (10 ** binc[ok]) / 1000.0
    sig2_obs = sig2[ok]
    sig2_unc = sig2_err[ok] + 0.5
    print(f"[bin] {ok.sum()} valid bins; sigma_los={np.sqrt(sig2_obs).round(1)} km/s")

    Re = args.Re
    flat_g = run_emcee("gNFW", lambda t: jeans_loglike_gnfw(t, R_obs, sig2_obs, sig2_unc, Re),
                       3, [8.0, -0.2, 0.5], [0.2, 0.1, 0.15])
    g_gnfw = np.array([gnfw_gamma_local(R_EVAL, *t) for t in flat_g]).ravel()
    flat_c = run_emcee("coreNFW", lambda t: jeans_loglike_corenfw(t, R_obs, sig2_obs, sig2_unc, Re),
                       4, [8.0, -0.2, -0.5, 1.0], [0.2, 0.1, 0.2, 0.2])
    g_core = np.array([corenfw_gamma_local(R_EVAL, *t) for t in flat_c]).ravel()
    flat_b = run_emcee("Burkert", lambda t: jeans_loglike_burkert(t, R_obs, sig2_obs, sig2_unc, Re),
                       2, [8.0, -0.4], [0.2, 0.1])
    g_burk = np.array([burkert_gamma_local(R_EVAL, *t) for t in flat_b]).ravel()

    rows = {n: pctl(s) for n, s in
            [("gNFW", g_gnfw), ("coreNFW", g_core), ("Burkert", g_burk)]}
    dgamma = rows["gNFW"][0] - rows["Burkert"][0]
    print(f"\n=== {args.galaxy} gamma(150 pc) ===")
    for n, (md, lo, hi) in rows.items():
        print(f"  {n:8s} {md:.2f} [{lo:.2f},{hi:.2f}]")
    print(f"  Delta_gamma (gNFW-Burkert) = {dgamma:.2f}")

    out = dict(galaxy=args.galaxy, n_members=int(len(m)), v_sys=v_sys,
               sigma_global=sig_global, Re_kpc=Re, R_EVAL=R_EVAL,
               R_obs_kpc=R_obs.tolist(), sigma_los_kms=np.sqrt(sig2_obs).tolist(),
               bin_counts=counts[ok].tolist(),
               gamma=rows, delta_gamma_gnfw_burkert=dgamma)
    (TAB / f"family_swing_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[save] {TAB / f'family_swing_{tag}.json'}")

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 120})
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    samples = [g_gnfw, g_core, g_burk]
    labels = ["gNFW\n(3 par)", "coreNFW\n(4 par)", "Burkert\n(2 par)"]
    parts = ax.violinplot([s[np.isfinite(s)] for s in samples], positions=[0, 1, 2],
                          widths=0.75, showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], ["#1f77b4", "#2ca02c", "#9467bd"]):
        pc.set_facecolor(col); pc.set_alpha(0.75)
    ax.axhline(1.0, ls=":", c="k", alpha=0.5, label=r"NFW cusp ($\gamma=1$)")
    ax.axhline(0.0, ls=":", c="green", alpha=0.5, label=r"cored ($\gamma=0$)")
    ax.annotate("", xy=(0, rows["gNFW"][0]), xytext=(2, rows["Burkert"][0]),
                arrowprops=dict(arrowstyle="<->", color="grey", lw=1.2, alpha=0.7))
    ax.text(1.0, max(rows["gNFW"][0], 1.0) + 0.18,
            rf"$\Delta\gamma\approx{dgamma:.2f}$ from family choice",
            ha="center", fontsize=8, color="grey")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$\gamma(150\,\mathrm{pc})$")
    ax.set_title(f"{args.galaxy}: family-choice swing (N={len(m)})", fontsize=10)
    ax.legend(loc="upper right", fontsize=8); ax.set_ylim(-0.3, 1.9); ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(FIG / f"fig_{tag}_family_swing.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"fig_{tag}_family_swing.png", dpi=200, bbox_inches="tight")
    print(f"[fig] {FIG / f'fig_{tag}_family_swing.pdf'}")


if __name__ == "__main__":
    main()
