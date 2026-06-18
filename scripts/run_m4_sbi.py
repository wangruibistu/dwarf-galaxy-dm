#!/usr/bin/env python3
"""M4: amortised NPE posterior for the Sculptor inner slope.

Simulator: draw a profile from the (fast MLP) diffusion prior conditioned on
Sculptor's mass -> measure gamma(150pc) -> sample anisotropies (beta1,beta2) and
a mass amplitude (log10 A) -> M3 multi-population anisotropic Jeans forward ->
two-population binned sigma_los^2 with realistic noise = observation x.
theta = (gamma150, beta1, beta2, log10 A). Train an MDN q(theta|x); validate with
SBC rank-uniformity and cored/cuspy mock recovery.

Output: results/tables/m4_sbi.npz (+ console)  and a trained MDN checkpoint.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import jax
from src.dm_models.diffusion_prior.dc14_library import (
    make_dc14_dataset, SCULPTOR_COND, _gnfwfit_slope, R_GRID)
from src.dm_models.diffusion_prior.diffusion import (
    make_schedule, ScoreMLP, train as train_diff, sample as sample_diff)
from src.dynamical_modeling.multipop_jeans import sigma_los2_aniso
from src.inference.npe_mdn import MDN, train_npe, sample_posterior
import jax.numpy as jnp

RE1, RE2 = 0.15, 0.35
R1 = np.logspace(np.log10(0.05), np.log10(0.8), 8)
R2 = np.logspace(np.log10(0.10), np.log10(1.5), 8)
N_SIM = 7000
FRAC_ERR = 0.12


def menc_func(log_rho):
    rho = 10 ** log_rho
    M = np.cumsum(4 * np.pi * R_GRID ** 2 * rho * np.gradient(R_GRID))
    return lambda r: np.interp(np.atleast_1d(r), R_GRID, M)


def build_sims(profiles, rng):
    """profiles: (N,32) log10 rho. Return theta (N,4), x (N,16), gam (N)."""
    th = np.zeros((len(profiles), 4)); X = np.zeros((len(profiles), 16))
    for i, lr in enumerate(profiles):
        g = _gnfwfit_slope(lr)
        b1 = rng.uniform(-0.4, 0.4); b2 = rng.uniform(-0.4, 0.4)
        logA = rng.uniform(-0.5, 0.7)
        Mf = menc_func(lr)
        s1 = (10 ** logA) * sigma_los2_aniso(R1, Mf, b1, RE1, n_r=200)
        s2 = (10 ** logA) * sigma_los2_aniso(R2, Mf, b2, RE2, n_r=200)
        e1 = FRAC_ERR * s1 + 1.0; e2 = FRAC_ERR * s2 + 1.0
        x = np.concatenate([s1 + rng.normal(0, e1), s2 + rng.normal(0, e2)])
        th[i] = [g, b1, b2, logA]; X[i] = x
    ok = np.all(np.isfinite(X), 1) & np.isfinite(th[:, 0])
    return th[ok], X[ok]


def main():
    t0 = time.time()
    print("[M4] training fast MLP diffusion prior on DC14 library ...")
    Xlib, Clib, split = make_dc14_dataset(n=9000, seed=0)
    mu, sig = Xlib.mean(0), Xlib.std(0) + 1e-6
    cm, cs = Clib.mean(0), Clib.std(0) + 1e-6
    sched = make_schedule(n_steps=200)
    dmodel = ScoreMLP(dim_x=32, dim_c=4, hidden=256, seed=0)
    train_diff(dmodel, (Xlib[split] - mu) / sig, (Clib[split] - cm) / cs, sched,
               n_epochs=220, batch=128, lr=2e-3, verbose=False)
    train_diff(dmodel, (Xlib[split] - mu) / sig, (Clib[split] - cm) / cs, sched,
               n_epochs=120, batch=128, lr=4e-4, verbose=False)

    print(f"[M4] sampling {N_SIM} prior profiles + building simulations ...")
    cond = (SCULPTOR_COND - cm) / cs
    prof = sample_diff(dmodel, sched, cond=cond, n=N_SIM, guidance=0.5,
                       rng=np.random.default_rng(1), monotonic=(mu, sig))
    prof = np.asarray(prof) * sig + mu
    rng = np.random.default_rng(2)
    theta, X = build_sims(prof, rng)
    print(f"    {len(theta)} valid simulations; gamma range "
          f"{np.percentile(theta[:,0],[5,50,95]).round(2)}")

    # standardise
    tm, ts = theta.mean(0), theta.std(0) + 1e-9
    xm, xs = X.mean(0), X.std(0) + 1e-9
    Ts, Xs = (theta - tm) / ts, (X - xm) / xs
    ntr = int(0.9 * len(Ts))
    print("[M4] training MDN NPE ...")
    model = MDN(dim_x=16, dim_theta=4, K=8, width=128, depth=3,
                key=jax.random.PRNGKey(0))
    model, losses = train_npe(model, Xs[:ntr], Ts[:ntr], n_epochs=400,
                              batch=256, lr=1e-3, verbose=True)

    # ---- validation: SBC rank uniformity on held-out (gamma dim) ----
    print("[M4] SBC + mock recovery ...")
    key = jax.random.PRNGKey(7)
    ranks = []
    val = range(ntr, len(Ts))
    for j in val:
        key, sk = jax.random.split(key)
        ps = np.asarray(sample_posterior(model, Xs[j], 200, sk))
        ranks.append((ps[:, 0] < Ts[j, 0]).mean())
    ranks = np.array(ranks)
    ks = stats.kstest(ranks, "uniform")
    print(f"  SBC(gamma) rank-uniformity KS D={ks.statistic:.3f} p={ks.pvalue:.3f}  (n={len(ranks)})")

    # mock recovery: a cored and a cuspy held-out sim
    gv = theta[ntr:, 0]
    idx_core = ntr + int(np.argmin(np.abs(gv - 0.35)))
    idx_cusp = ntr + int(np.argmin(np.abs(gv - 1.05)))
    for name, j in [("core", idx_core), ("cusp", idx_cusp)]:
        key, sk = jax.random.split(key)
        ps = np.asarray(sample_posterior(model, Xs[j], 3000, sk)) * ts + tm
        q = np.percentile(ps[:, 0], [16, 50, 84])
        inj = theta[j, 0]
        rec = q[0] <= inj <= q[2]
        print(f"  {name}: injected gamma={inj:.2f}  posterior={q[1]:.2f} "
              f"[{q[0]:.2f},{q[2]:.2f}]  recovered={rec}")

    TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
    np.savez(TAB / "m4_sbi.npz", ranks=ranks, ks_D=ks.statistic, ks_p=ks.pvalue,
             theta=theta, losses=np.array(losses))
    print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
