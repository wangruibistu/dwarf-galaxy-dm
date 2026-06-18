#!/usr/bin/env python3
"""M5: validation suite for the amortised NPE posterior.

(1) Coverage test: fraction of held-out truths inside the central C% credible
    interval, for C in {50,68,90}% -> should match C (calibration).
(2) Recovery sweep: inject held-out profiles spanning cusp->core, recover gamma
    blind, show the posterior tracks the injected truth without bias.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import jax
from scripts.paper2_pipeline import build_pipeline, calibrate_width
from src.inference.npe_mdn import sample_posterior


def main():
    t0 = time.time()
    print("[M5] building pipeline ...")
    P = build_pipeline(seed=0, n_sim=14000, npe_epochs=600)
    model, tm, ts, xm, xs = P["model"], P["tm"], P["ts"], P["xm"], P["xs"]
    theta, X, ntr = P["theta"], P["X"], P["ntr"]
    Xs = (X - xm) / xs
    allval = np.arange(ntr, len(theta))
    cal, val = allval[: len(allval) // 2], allval[len(allval) // 2:]
    key = jax.random.PRNGKey(11)

    # (0) posterior width calibration on the cal half (conformal-style)
    s_w = calibrate_width(P, cal)
    P["widen"] = s_w
    print(f"[M5] posterior width calibration factor = {s_w:.2f} "
          f"(fit on {len(cal)} cal sims; tested on {len(val)} independent sims)")

    # (1) coverage
    print("[M5] coverage test on held-out ...")
    levels = [0.50, 0.68, 0.90]
    inside = {c: 0 for c in levels}
    nval = 0
    for j in val:
        key, sk = jax.random.split(key)
        ps = np.asarray(sample_posterior(model, Xs[j], 400, sk))[:, 0] * ts[0] + tm[0]
        ps = np.median(ps) + s_w * (ps - np.median(ps))
        g_true = theta[j, 0]
        for c in levels:
            lo, hi = np.percentile(ps, [50 * (1 - c), 50 * (1 + c)])
            inside[c] += (lo <= g_true <= hi)
        nval += 1
    print("  level   empirical-coverage")
    cov_ok = True
    for c in levels:
        frac = inside[c] / nval
        ok = abs(frac - c) < 0.06
        cov_ok &= ok
        print(f"   {int(c*100):3d}%      {frac*100:5.1f}%   {'ok' if ok else 'OFF'}")

    # (2) recovery sweep: average over the 10 nearest held-out sims per target
    #     so the acceptance metric is not dominated by single-realisation noise
    print("[M5] recovery sweep (cusp->core; 10 mocks per target) ...")
    gv = theta[val, 0]
    targets = [0.25, 0.45, 0.65, 0.85, 1.05]
    inj, med, lo, hi = [], [], [], []
    for gt in targets:
        js = val[np.argsort(np.abs(gv - gt))[:10]]
        meds, los, his, injs = [], [], [], []
        for j in js:
            key, sk = jax.random.split(key)
            ps = np.asarray(sample_posterior(model, Xs[j], 1000, sk))[:, 0] * ts[0] + tm[0]
            ps = np.median(ps) + s_w * (ps - np.median(ps))
            q = np.percentile(ps, [16, 50, 84])
            meds.append(q[1]); los.append(q[0]); his.append(q[2]); injs.append(theta[j, 0])
        inj.append(np.mean(injs)); med.append(np.mean(meds))
        lo.append(np.mean(los)); hi.append(np.mean(his))
        print(f"  injected {np.mean(injs):.2f} -> posterior {np.mean(meds):.2f} "
              f"[{np.mean(los):.2f},{np.mean(his):.2f}]  (mean of 10 mocks)")
    inj = np.array(inj); med = np.array(med)
    bias = np.mean(med - inj)
    print(f"  mean recovery bias = {bias:+.3f}  (accept |.|<0.08)")

    TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
    FIG = ROOT / "results" / "figures" / "paper"; FIG.mkdir(parents=True, exist_ok=True)
    np.savez(TAB / "m5_validation.npz",
             levels=levels, coverage=[inside[c] / nval for c in levels],
             inj=inj, med=med, lo=lo, hi=hi, bias=bias)

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].plot(levels, [inside[c] / nval for c in levels], "C0-o")
    ax[0].set_xlabel("nominal credible level"); ax[0].set_ylabel("empirical coverage")
    ax[0].set_title("Coverage (calibration)")
    ax[1].plot([0, 1.4], [0, 1.4], "k--", lw=0.8)
    ax[1].errorbar(inj, med, yerr=[med - np.array(lo), np.array(hi) - med],
                   fmt="C3o", capsize=3)
    ax[1].set_xlabel(r"injected $\gamma(150\,$pc$)$")
    ax[1].set_ylabel(r"recovered $\gamma(150\,$pc$)$")
    ax[1].set_title(f"Blind recovery (bias {bias:+.2f})")
    fig.tight_layout()
    fig.savefig(FIG / "fig_m5_validation.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_m5_validation.png", bbox_inches="tight", dpi=200)
    print(f"  [fig] fig_m5_validation   ACCEPT coverage={cov_ok}, bias={abs(bias)<0.08}")
    print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
