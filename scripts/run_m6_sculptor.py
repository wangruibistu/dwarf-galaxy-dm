#!/usr/bin/env python3
"""M6: amortised NPE posterior for Sculptor's inner slope on the real FLAMES data.

Pipeline: hard median-[Fe/H] split of the 1598-star FLAMES CaT sample ->
binned two-population sigma_los^2 profiles -> amortised MDN-NPE trained on the
mixture proposal (diffusion prior at Sculptor's conditioning + flat-gamma
DC14-shape profiles), with the simulator geometry matched to the data
(measured Re per population, real bin radii, observed fractional errors).

Reports the posterior under (a) the broad mixture proposal (likelihood-
dominated) and (b) the DC14 conditional prior re-imposed by importance
reweighting.  Pre-registered decision rule, fixed before looking at the
posterior:
  CUSP confirmed   if P(gamma < 0.5) < 0.05 under BOTH priors
  CORE confirmed   if P(gamma > 0.8) < 0.05 under BOTH priors
  otherwise        INTERMEDIATE -> add Fornax contrast object (Paper II plan)

Output: results/tables/m6_sculptor.npz, results/figures/paper/fig_m6_sculptor.pdf
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import jax
from scripts.paper2_pipeline import (build_pipeline, posterior_gamma,
                                     dc14_prior_weights, weighted_quantile,
                                     calibrate_width)

DATA = ROOT / "data" / "processed" / "sculptor_flames_members.parquet"
NBIN = 8


def binned_sigma2(R_kpc, dv, verr, nbin=NBIN):
    """Equal-count log bins; error-deconvolved sigma_los^2 per bin (km^2/s^2)."""
    edges = np.percentile(R_kpc, np.linspace(0, 100, nbin + 1))
    cen = np.sqrt(edges[1:] * edges[:-1])
    s2 = np.full(nbin, np.nan); e2 = np.full(nbin, np.nan)
    for i in range(nbin):
        m = (R_kpc >= edges[i]) & (R_kpc < edges[i + 1] + (i == nbin - 1) * 1e-9)
        n = m.sum()
        if n < 8:
            continue
        mu = dv[m].mean()
        var = np.mean((dv[m] - mu) ** 2) - np.mean(verr[m] ** 2)
        s2[i] = var; e2[i] = max(var, 1.0) * np.sqrt(2.0 / n)
    ok = np.isfinite(s2) & (s2 > 0)
    return cen[ok], s2[ok], e2[ok]


def main():
    t0 = time.time()
    d = pd.read_parquet(DATA)
    R = d["R_pc"].values / 1000.0          # kpc
    v = d["vlos"].values; verr = d["evlos"].values
    feh = d["feh"].values
    dv = v - np.median(v)
    mr = feh > np.median(feh)

    r1, s1, e1 = binned_sigma2(R[mr], dv[mr], verr[mr])
    r2, s2, e2 = binned_sigma2(R[~mr], dv[~mr], verr[~mr])
    re1 = np.median(R[mr]); re2 = np.median(R[~mr])
    frac = np.median(np.concatenate([e1 / s1, e2 / s2]))
    print(f"[M6] MR: N={mr.sum()} Re={re1*1000:.0f}pc bins={len(r1)} ; "
          f"MP: N={(~mr).sum()} Re={re2*1000:.0f}pc bins={len(r2)} ; "
          f"median frac err={frac:.3f}")

    print("[M6] building amortised pipeline matched to data geometry ...")
    P = build_pipeline(seed=0, n_sim=14000, npe_epochs=600, flat_frac=0.5,
                       re1=re1, re2=re2, r1=r1, r2=r2, frac_err=frac)
    cal = np.arange(P["ntr"], len(P["theta"]))
    P["widen"] = calibrate_width(P, cal)
    print(f"[M6] posterior width calibration factor = {P['widen']:.2f}")
    x_obs = np.concatenate([s1, s2])

    # observation-in-support check
    Xtr = P["X"]
    pct = np.array([(Xtr[:, j] < x_obs[j]).mean() for j in range(len(x_obs))])
    print(f"[M6] obs percentile within training X: min={pct.min():.3f} "
          f"max={pct.max():.3f}  {'OK' if (pct.min()>0.01 and pct.max()<0.99) else 'WARNING: edge of support'}")

    ps = posterior_gamma(P, x_obs, n=20000, key=jax.random.PRNGKey(42))
    g = ps[:, 0]
    qb = np.percentile(g, [16, 50, 84])
    pb_core = (g < 0.5).mean(); pb_cusp = (g > 0.8).mean()

    w = dc14_prior_weights(P, g)
    qd = weighted_quantile(g, [0.16, 0.50, 0.84], w)
    pd_core = w[g < 0.5].sum(); pd_cusp = w[g > 0.8].sum()

    print("\n========== M6: Sculptor inner slope (FLAMES, calibrated NPE) ==========")
    print(f"  broad proposal : gamma(150pc) = {qb[1]:.2f} [{qb[0]:.2f},{qb[2]:.2f}] "
          f"width={qb[2]-qb[0]:.2f}  P(g<0.5)={pb_core:.3f}  P(g>0.8)={pb_cusp:.3f}")
    print(f"  DC14 reweighted: gamma(150pc) = {qd[1]:.2f} [{qd[0]:.2f},{qd[2]:.2f}] "
          f"width={qd[2]-qd[0]:.2f}  P(g<0.5)={pd_core:.3f}  P(g>0.8)={pd_cusp:.3f}")

    cusp_conf = (pb_core < 0.05) and (pd_core < 0.05)
    core_conf = (pb_cusp < 0.05) and (pd_cusp < 0.05)
    if cusp_conf:
        verdict = "CUSP confirmed (gamma<0.5 excluded at 95% under both priors)"
        fornax = "Fornax optional -- decisive single-object statement available"
    elif core_conf:
        verdict = "CORE confirmed"
        fornax = "Fornax optional"
    else:
        verdict = "INTERMEDIATE / prior-sensitive"
        fornax = ("ADD FORNAX: single-object posterior not decisive; "
                  "the amortised pipeline applies at near-zero marginal cost")
    print(f"  --> verdict: {verdict}")
    print(f"  --> Fornax decision: {fornax}")

    TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
    FIG = ROOT / "results" / "figures" / "paper"; FIG.mkdir(parents=True, exist_ok=True)
    np.savez(TAB / "m6_sculptor.npz", g=g, w=w, qb=qb, qd=qd,
             pb_core=pb_core, pd_core=pd_core,
             r1=r1, s1=s1, e1=e1, r2=r2, s2=s2, e2=e2,
             re1=re1, re2=re2, frac=frac, obs_pct=pct)

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for (rr, ss, ee, lab, c) in [(r1, s1, e1, "MR", "#c0392b"), (r2, s2, e2, "MP", "#2c6fbb")]:
        ax[0].errorbar(rr * 1000, np.sqrt(ss), yerr=0.5 * ee / np.sqrt(ss),
                       fmt="o", ms=4, capsize=2, color=c, label=lab)
    ax[0].set_xscale("log"); ax[0].set_xlabel("R [pc]")
    ax[0].set_ylabel(r"$\sigma_{\rm los}$ [km/s]"); ax[0].legend(frameon=False)
    ax[0].set_title("FLAMES two-population dispersion")
    bins = np.linspace(0, 1.6, 64)
    ax[1].hist(g, bins=bins, density=True, color="0.6", alpha=0.6,
               label=f"broad proposal ({qb[1]:.2f})")
    ax[1].hist(g, bins=bins, density=True, weights=w * len(g), histtype="step",
               color="#c0392b", lw=1.6, label=f"DC14 prior ({qd[1]:.2f})")
    ax[1].axvline(0.5, color="k", ls=":", lw=0.8); ax[1].axvline(1.0, color="k", ls="--", lw=0.8)
    ax[1].set_xlabel(r"$\gamma(150\,$pc$)$"); ax[1].set_ylabel("posterior density")
    ax[1].legend(frameon=False, fontsize=8); ax[1].set_title("Sculptor inner slope (NPE)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_m6_sculptor.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_m6_sculptor.png", bbox_inches="tight", dpi=200)
    print(f"  [fig] fig_m6_sculptor   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
