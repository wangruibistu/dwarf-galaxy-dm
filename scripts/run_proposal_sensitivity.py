"""Proposal-sensitivity of the real-data Sculptor inner slope.

The amortised-NPE real-data number (gamma=0.72) is produced under a *mixture
proposal*: a fraction `flat_frac` of the training simulations have their inner
slope drawn ~uniform over cusp->core, the rest from the DC14 conditional prior
(paper2_pipeline.build_pipeline). A referee can reasonably ask how much the 0.72
depends on this proposal engineering rather than on the data.

We re-run the entire pipeline on the *same* FLAMES two-population data for a grid
of flat_frac, holding everything else fixed (same n_sim, npe_epochs, geometry,
width calibration), and report the broad-proposal posterior median and width, the
DC14-reweighted median, and the pre-registered cusp/core tail probabilities. A
stable median across flat_frac shows 0.72 is data- and reweighting-driven, not an
artefact of the 1:1 mixing choice.

Output: results/tables/proposal_sensitivity.json (+ console)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import jax
from scripts.paper2_pipeline import (build_pipeline, posterior_gamma,
                                     dc14_prior_weights, weighted_quantile,
                                     calibrate_width)

DATA = ROOT / "data" / "processed" / "sculptor_flames_members.parquet"
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
NBIN = 8
N_SIM = 14000          # matched to the production m6 run for direct comparability
NPE_EPOCHS = 600       # matched to m6; ff=0.5 reproduces the headline gamma=0.72
FLAT_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]


def binned_sigma2(R_kpc, dv, verr, nbin=NBIN):
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
    R = d["R_pc"].values / 1000.0
    v = d["vlos"].values; verr = d["evlos"].values; feh = d["feh"].values
    dv = v - np.median(v); mr = feh > np.median(feh)
    r1, s1, e1 = binned_sigma2(R[mr], dv[mr], verr[mr])
    r2, s2, e2 = binned_sigma2(R[~mr], dv[~mr], verr[~mr])
    re1, re2 = np.median(R[mr]), np.median(R[~mr])
    frac = np.median(np.concatenate([e1 / s1, e2 / s2]))
    x_obs = np.concatenate([s1, s2])
    print(f"[data] MR N={mr.sum()} Re={re1*1000:.0f}pc ; MP N={(~mr).sum()} "
          f"Re={re2*1000:.0f}pc ; frac_err={frac:.3f}")

    rows = []
    for ff in FLAT_FRACS:
        ts = time.time()
        P = build_pipeline(seed=0, n_sim=N_SIM, npe_epochs=NPE_EPOCHS, flat_frac=ff,
                           re1=re1, re2=re2, r1=r1, r2=r2, frac_err=frac)
        cal = np.arange(P["ntr"], len(P["theta"]))
        P["widen"] = calibrate_width(P, cal)
        ps = posterior_gamma(P, x_obs, n=20000, key=jax.random.PRNGKey(42))
        g = ps[:, 0]
        qb = np.percentile(g, [16, 50, 84])
        pb_core = float((g < 0.5).mean()); pb_cusp = float((g > 0.8).mean())
        # DC14 reweighted (skip when flat_frac=1.0: no DC14 component in proposal)
        if ff < 1.0:
            w = dc14_prior_weights(P, g)
            qd = weighted_quantile(g, [0.16, 0.50, 0.84], w)
            pd_core = float(w[g < 0.5].sum())
        else:
            qd = [np.nan, np.nan, np.nan]; pd_core = np.nan
        rows.append(dict(flat_frac=ff, widen=float(P["widen"]),
                         broad_med=float(qb[1]), broad_lo=float(qb[0]),
                         broad_hi=float(qb[2]), broad_width=float(qb[2] - qb[0]),
                         dc14_med=float(qd[1]), pb_core=pb_core, pb_cusp=pb_cusp,
                         pd_core=pd_core))
        print(f"[ff={ff:.2f}] broad gamma={qb[1]:.2f} [{qb[0]:.2f},{qb[2]:.2f}] "
              f"width={qb[2]-qb[0]:.2f}  DC14={qd[1]:.2f}  "
              f"P(g<0.5)={pb_core:.2f}  P(g>0.8)={pb_cusp:.2f}  ({time.time()-ts:.0f}s)")

    meds = np.array([r["broad_med"] for r in rows])
    dc14_meds = np.array([r["dc14_med"] for r in rows if np.isfinite(r["dc14_med"])])
    print("\n========== proposal sensitivity ==========")
    print(f"  broad-proposal median range  : {meds.min():.2f} - {meds.max():.2f} "
          f"(span {meds.max()-meds.min():.2f}) over flat_frac in {FLAT_FRACS}")
    print(f"  DC14-reweighted median range : {dc14_meds.min():.2f} - {dc14_meds.max():.2f} "
          f"(span {dc14_meds.max()-dc14_meds.min():.2f})")
    print(f"  paper value (m6, ff=0.5, n_sim=14000) = 0.72")
    out = dict(n_sim=N_SIM, npe_epochs=NPE_EPOCHS, nbin=NBIN, frac_err=float(frac),
               flat_fracs=FLAT_FRACS, rows=rows,
               broad_med_span=float(meds.max() - meds.min()),
               dc14_med_span=float(dc14_meds.max() - dc14_meds.min()))
    (TAB / "proposal_sensitivity.json").write_text(json.dumps(out, indent=2))
    print(f"[save] {TAB / 'proposal_sensitivity.json'}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
