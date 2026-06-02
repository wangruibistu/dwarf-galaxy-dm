"""Verify the realistic halo dataset has the expected statistical structure
and works as a drop-in replacement for the synthetic dataset in the
diffusion-prior pipeline."""

from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dm_models.diffusion_prior.realistic_halos import (
    make_realistic_dataset, R_GRID, N_RBINS,
)
from src.dm_models.diffusion_prior.diffusion import (
    make_schedule, ScoreMLP, train, sample, sample_posterior,
)


def test_dataset_realism():
    print("  generating realistic dataset (n=4000)...")
    X, C = make_realistic_dataset(n_samples=4000, seed=0)
    print(f"  shape: X={X.shape}, C={C.shape}")
    # Check conditioning ranges
    print(f"  log_M_halo: {C[:,0].min():.1f} - {C[:,0].max():.1f}")
    print(f"  log_M_star: {C[:,1].min():.1f} - {C[:,1].max():.1f}")
    print(f"  t_SF      : {C[:,2].min():.1f} - {C[:,2].max():.1f} Gyr")
    print(f"  log r_tid : {C[:,3].min():.1f} - {C[:,3].max():.1f}")

    # Inner-bin density should span ~3+ dex (cusps vs cores)
    inner_std = X[:, 2].std()
    outer_std = X[:, -3].std()
    print(f"  inner-bin log10 ρ std = {inner_std:.2f}, outer = {outer_std:.2f}")
    assert inner_std > 0.8
    print("  PASS")
    return X, C


def test_diffusion_on_realistic(X, C):
    print("  training diffusion on realistic dataset...")
    sched = make_schedule(n_steps=100)
    mu, sigma = X.mean(0), X.std(0) + 1e-6
    cm, cs = C.mean(0), C.std(0) + 1e-6
    X_std = (X - mu) / sigma
    C_std = (C - cm) / cs
    model = ScoreMLP(dim_x=N_RBINS, dim_c=4, hidden=192, seed=0)
    t0 = time.time()
    losses = train(model, X_std, C_std, sched,
                   n_epochs=20, batch=128, lr=2e-3, verbose=False)
    dt = time.time() - t0
    init = float(np.mean(losses[:3])); fin = float(np.mean(losses[-3:]))
    print(f"  trained {len(X)} halos × {len(losses)} epochs in {dt:.1f} s")
    print(f"  loss: {init:.3f} -> {fin:.3f}  (reduction {(init-fin)/init*100:.0f}%)")
    assert fin < init * 0.9
    print("  PASS")
    return model, sched, mu, sigma, (cm, cs)


def test_dps_posterior_realistic(model, sched, mu, sigma, C_norm):
    """Posterior recovery on a mock observation, conditioning to
    Sculptor-like values."""
    print("  setting up mock observation (Sculptor-like)...")
    target_logM_300 = np.log10(4e7)   # Sculptor scale
    sig_obs = 0.3

    def log_likelihood(x_std):
        x = x_std * sigma + mu
        i = int(np.argmin(np.abs(R_GRID - 0.3)))
        M = np.array([
            np.trapezoid(4 * np.pi * R_GRID[:i+1]**2 * 10**xi[:i+1], R_GRID[:i+1])
            for xi in x
        ])
        return -0.5 * ((np.log10(M.clip(min=1)) - target_logM_300) / sig_obs) ** 2

    # Sculptor-like conditioning
    cond = (np.array([10.0, 6.5, 8.0, 0.5]) - C_norm[0]) / C_norm[1]
    print("  running DPS sampler...")
    x_std = sample_posterior(model, sched, cond, n=200,
                             log_likelihood_fn=log_likelihood,
                             guidance=0.5, dps_scale=0.03)
    x = x_std * sigma + mu
    i = int(np.argmin(np.abs(R_GRID - 0.3)))
    M = np.array([np.trapezoid(4 * np.pi * R_GRID[:i+1]**2 * 10**xi[:i+1],
                            R_GRID[:i+1]) for xi in x])
    logM = np.log10(M.clip(min=1))
    print(f"  target log M(<300 pc) = {target_logM_300:.2f}")
    print(f"  posterior median = {np.median(logM):.2f}")
    print(f"  posterior 68% CI = [{np.percentile(logM,16):.2f}, "
          f"{np.percentile(logM,84):.2f}]")
    bias = abs(np.median(logM) - target_logM_300)
    assert bias < 1.0, f"bias {bias:.2f} dex too large"
    print("  PASS")


if __name__ == "__main__":
    print("=== 1. realistic dataset stats ===")
    X, C = test_dataset_realism()
    print("\n=== 2. training on realistic halos ===")
    model, sched, mu, sigma, C_norm = test_diffusion_on_realistic(X, C)
    print("\n=== 3. DPS posterior on Sculptor-like mock ===")
    test_dps_posterior_realistic(model, sched, mu, sigma, C_norm)
    print("\nAll realistic-halo diffusion-prior tests passed.")
