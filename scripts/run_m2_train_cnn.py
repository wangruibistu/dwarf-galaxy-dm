#!/usr/bin/env python3
"""M2 acceptance: train the 1D-CNN score network on the DC14/Lazar library and
check the prior-predictive reproduces the library inner-slope distribution
(compression < 0.05 dex; KS test), at Sculptor's conditioning.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.dm_models.diffusion_prior.dc14_library import (
    make_dc14_dataset, SCULPTOR_COND, _gnfwfit_slope)
from src.dm_models.diffusion_prior.score_cnn import (
    make_schedule, ScoreCNN, train_cnn, sample_cnn)
import jax


def main():
    t0 = time.time()
    print("[M2] building DC14 library ...")
    X, C, split = make_dc14_dataset(n=9000, seed=0)
    Xtr, Ctr = X[split], C[split]
    mu, sig = X.mean(0), X.std(0) + 1e-6
    cm, cs = C.mean(0), C.std(0) + 1e-6
    Xs = (Xtr - mu) / sig
    Cs = (Ctr - cm) / cs

    print("[M2] training 1D-CNN score network ...")
    sched = make_schedule(T=200)
    model = ScoreCNN(dim_c=4, ch=64, k=5, n_blocks=4, emb=128,
                     key=jax.random.PRNGKey(0))
    model, losses = train_cnn(model, Xs, Cs, sched, n_epochs=300, batch=128,
                              lr=2e-3, seed=1, verbose=True)
    print(f"    final loss {losses[-1]:.4f}")

    cond = (SCULPTOR_COND - cm) / cs
    print("[M2] prior-predictive at Sculptor conditioning ...")
    xp = sample_cnn(model, sched, cond=cond, n=2000, guidance=0.5, seed=7,
                    monotone=(mu, sig))
    lr_prior = np.asarray(xp) * sig + mu
    gam_prior = np.array([_gnfwfit_slope(r) for r in lr_prior])
    gam_prior = gam_prior[np.isfinite(gam_prior)]

    # library reference at Sculptor mass
    scl = np.abs(C[:, 0] - 10.0) < 0.3
    gam_lib = np.array([_gnfwfit_slope(r) for r in X[scl]])
    gam_lib = gam_lib[np.isfinite(gam_lib)]

    qp = np.percentile(gam_prior, [16, 50, 84])
    ql = np.percentile(gam_lib, [16, 50, 84])
    compression = ql[1] - qp[1]
    ks = stats.ks_2samp(gam_prior, gam_lib)
    print("\n========== M2 RESULT ==========")
    print(f"  library  (Scl) gamma = {ql[1]:.3f} [{ql[0]:.2f},{ql[2]:.2f}]")
    print(f"  CNN prior-pred gamma = {qp[1]:.3f} [{qp[0]:.2f},{qp[2]:.2f}]")
    print(f"  compression = {compression:+.3f} dex   (accept |.|<0.05)")
    print(f"  KS(prior, library) D={ks.statistic:.3f} p={ks.pvalue:.3f}")
    accept = abs(compression) < 0.05
    print(f"  ACCEPT: {accept}   ({time.time()-t0:.0f}s)")

    TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
    np.savez(TAB / "m2_cnn_prior.npz", gam_prior=gam_prior, gam_lib=gam_lib,
             compression=compression, ks_D=ks.statistic, losses=np.array(losses))
    return accept


if __name__ == "__main__":
    main()
