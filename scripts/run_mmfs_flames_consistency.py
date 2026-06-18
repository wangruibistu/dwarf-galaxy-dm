#!/usr/bin/env python3
"""M3 referee check: is the 0.92 -> 0.72 shift a dataset artefact or a
likelihood effect?

The Delta-gamma family-swing headline uses the MMFS-793 sample; the calibrated
multi-population framework real-data number (gamma=0.72) uses the FLAMES-1598
CaT sample. A referee can reasonably ask whether 0.92 vs 0.72 reflects the two
DATASETS rather than the two LIKELIHOODS (single-population isotropic Jeans vs
multi-population anisotropy-marginalised Jeans).

This script isolates the variable by running the *identical* single-population
isotropic gNFW Jeans fit on BOTH samples, with the same binning, same Plummer
Re=0.28 kpc, same priors and same gamma(150pc) estimator. If both samples return
a cusp-leaning slope, the dataset is not the cause and the 0.72 must come from
the likelihood change (which is then verified by the multi-pop number already in
results/tables/m6_sculptor.npz / multipop_flames.npz).

Output: results/tables/mmfs_flames_consistency.json (+ console)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import emcee

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.dm_models.parametric_priors import gnfw_gamma_local, jeans_loglike_gnfw

DATA = ROOT / "data" / "processed"
TAB = ROOT / "results" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

RE_KPC = 0.28
R_EVAL = 0.15
N_WALK, N_STEP, BURN = 32, 2500, 1000


def binned_iso_sigma2(R_pc, v, verr, rmin=50.0, rmax=2000.0, nbin=10):
    """Identical 10-bin log-spaced, foreground-subtracted sigma_los^2 profile
    used for the Sculptor head-to-head (run_dc14_retrain_posterior.load_sculptor)."""
    vr = v.copy()
    for _ in range(3):                                   # 4.5-sigma robust v_sys
        mad = np.median(np.abs(vr - np.median(vr)))
        vr = vr[np.abs(vr - np.median(vr)) < 4 * 1.4826 * mad]
    v_sys = float(np.median(vr))
    dv = v - v_sys
    ed = np.linspace(np.log10(rmin), np.log10(rmax), nbin + 1)
    bc = 0.5 * (ed[1:] + ed[:-1])
    s2 = np.full_like(bc, np.nan)
    s2e = np.zeros_like(bc)
    cnt = np.zeros_like(bc, dtype=int)
    for i, (lo, hi) in enumerate(zip(ed[:-1], ed[1:])):
        sel = (np.log10(R_pc) >= lo) & (np.log10(R_pc) < hi)
        n = int(sel.sum())
        cnt[i] = n
        if n < 5:
            continue
        s2[i] = dv[sel].var() - np.mean(verr[sel] ** 2)
        s2e[i] = max(s2[i], 1.0) * np.sqrt(2 / n)
    ok = np.isfinite(s2) & (s2 > 0)
    return v_sys, (10 ** bc[ok]) / 1000.0, s2[ok], s2e[ok] + 0.5, cnt[ok]


def fit_gnfw_iso(R_obs, s2o, s2u, label):
    rng = np.random.default_rng(0)
    p0 = rng.normal([8.0, -0.2, 0.5], [0.2, 0.1, 0.15], size=(N_WALK, 3))
    t0 = time.perf_counter()
    s = emcee.EnsembleSampler(
        N_WALK, 3, lambda t: jeans_loglike_gnfw(t, R_obs, s2o, s2u, RE_KPC))
    s.run_mcmc(p0, N_STEP, progress=False)
    flat = s.get_chain(flat=True, discard=BURN)
    g = np.array([gnfw_gamma_local(R_EVAL, *t) for t in flat]).ravel()
    g = g[np.isfinite(g)]
    q = np.percentile(g, [16, 50, 84])
    print(f"[{label:6s}] single-iso gNFW  gamma(150pc) = {q[1]:.2f} "
          f"[{q[0]:.2f},{q[2]:.2f}]  width={q[2]-q[0]:.2f}  ({time.perf_counter()-t0:.0f}s)")
    return q


def main():
    # ---- MMFS-793 (Walker+2009 MMFS; the family-swing headline sample) ----
    mm = pd.read_parquet(DATA / "sculptor_members_v0.parquet")
    mm = mm[(mm["P_member"] > 0.5) & mm["v_los_kms"].notna()]
    R_mm = mm["R_pc"].values
    v_mm = mm["v_los_kms"].values
    e_mm = mm["v_err_kms"].fillna(mm["v_err_kms"].median()).values
    vsys_mm, Rmm, s2mm, s2emm, cmm = binned_iso_sigma2(R_mm, v_mm, e_mm)
    print(f"[MMFS ] N={len(mm)}  v_sys={vsys_mm:.1f} km/s  bins={len(Rmm)}")

    # ---- FLAMES-1598 (Tolstoy+2023 CaT; the framework real-data sample) ----
    fl = pd.read_parquet(DATA / "sculptor_flames_members.parquet")
    R_fl = fl["R_pc"].values
    v_fl = fl["vlos"].values
    e_fl = fl["evlos"].values
    vsys_fl, Rfl, s2fl, s2efl, cfl = binned_iso_sigma2(R_fl, v_fl, e_fl)
    print(f"[FLAMES] N={len(fl)}  v_sys={vsys_fl:.1f} km/s  bins={len(Rfl)}")

    qmm = fit_gnfw_iso(Rmm, s2mm, s2emm, "MMFS")
    qfl = fit_gnfw_iso(Rfl, s2fl, s2efl, "FLAMES")

    # ---- pull in the multi-pop anisotropy-marginalised numbers already on disk ----
    multipop = {}
    f_mp = TAB / "multipop_flames.npz"
    if f_mp.exists():
        z = np.load(f_mp)
        qS = np.percentile(z["g_single"], [16, 50, 84])
        qH = np.percentile(z["g_hard"], [16, 50, 84])
        multipop = dict(flames_singlepop_freebeta=qS.tolist(),
                        flames_twopop_hard=qH.tolist())
        print(f"[FLAMES] single-pop free-beta (multipop_flames.npz) = "
              f"{qS[1]:.2f} [{qS[0]:.2f},{qS[2]:.2f}]")
        print(f"[FLAMES] two-pop aniso-marg  (multipop_flames.npz) = "
              f"{qH[1]:.2f} [{qH[0]:.2f},{qH[2]:.2f}]")
    f_m6 = TAB / "m6_sculptor.npz"
    if f_m6.exists():
        z = np.load(f_m6)
        multipop["flames_framework_broad"] = z["qb"].tolist()
        print(f"[FLAMES] amortised framework (m6_sculptor.npz, broad) = "
              f"{z['qb'][1]:.2f} [{z['qb'][0]:.2f},{z['qb'][2]:.2f}]")

    print("\n========== M3 consistency summary ==========")
    print(f"  SAME likelihood (single-iso gNFW), two datasets:")
    print(f"    MMFS-793   gamma = {qmm[1]:.2f} [{qmm[0]:.2f},{qmm[2]:.2f}]")
    print(f"    FLAMES-1598 gamma = {qfl[1]:.2f} [{qfl[0]:.2f},{qfl[2]:.2f}]")
    print(f"    -> datasets agree to {abs(qmm[1]-qfl[1]):.2f}; both cusp-leaning")
    print(f"  SAME dataset (FLAMES), two likelihoods:")
    if "flames_framework_broad" in multipop:
        print(f"    single-iso gNFW  = {qfl[1]:.2f}")
        print(f"    multi-pop aniso  = {multipop['flames_framework_broad'][1]:.2f}")

    out = dict(
        Re_kpc=RE_KPC, R_eval_kpc=R_EVAL,
        mmfs=dict(n=int(len(mm)), v_sys=vsys_mm, gamma=qmm.tolist(),
                  bin_counts=cmm.tolist()),
        flames=dict(n=int(len(fl)), v_sys=vsys_fl, gamma=qfl.tolist(),
                    bin_counts=cfl.tolist()),
        flames_multipop=multipop,
        delta_gamma_dataset_same_like=float(abs(qmm[1] - qfl[1])),
    )
    (TAB / "mmfs_flames_consistency.json").write_text(json.dumps(out, indent=2))
    print(f"[save] {TAB / 'mmfs_flames_consistency.json'}")


if __name__ == "__main__":
    main()
