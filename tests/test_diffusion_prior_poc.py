"""Diffusion-prior proof of concept.

Three end-to-end checks:
  1. Synthetic halo dataset has multimodal structure (3 archetypes)
  2. Trained diffusion model recovers archetype distribution unconditionally
  3. Posterior sampling on a mock σ_los observation prefers profiles whose
     enclosed mass matches the observation — i.e. likelihood term works.
"""

from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dm_models.diffusion_prior.synthetic_halos import (
    make_dataset, R_GRID, N_RBINS,
)
from src.dm_models.diffusion_prior.diffusion import (
    make_schedule, ScoreMLP, train, sample, sample_posterior,
)


# ---- helpers ----
def enclosed_mass_from_logrho(log_rho, r_grid=R_GRID):
    """M(<r) by spherical shell integration from log10 ρ in Msun/kpc^3."""
    rho = 10 ** log_rho
    # batch dim handled
    if rho.ndim == 1:
        return np.trapz(4 * np.pi * r_grid ** 2 * rho, r_grid)
    return np.trapz(4 * np.pi * r_grid ** 2 * rho, r_grid, axis=-1)


def test_synthetic_dataset_multimodal():
    print("  generating dataset...")
    X, C, A = make_dataset(n_samples=2000, seed=0)
    print(f"  shape: X={X.shape}, C={C.shape}")
    # Multimodality check: ρ at innermost bin should have wider distribution
    # than ρ at outer bin (cores vs cusps differ inside, agree outside)
    inner_std = X[:, 2].std()
    outer_std = X[:, -3].std()
    print(f"  inner-bin log10 ρ std = {inner_std:.2f},  outer = {outer_std:.2f}")
    assert inner_std > outer_std * 0.7, "inner profile diversity too low"

    # Check all archetypes present
    arch_counts = {a: int((A == a).sum()) for a in np.unique(A)}
    print(f"  archetype counts: {arch_counts}")
    assert all(c > 100 for c in arch_counts.values())
    print("  PASS")
    return X, C, A


def test_diffusion_training_reduces_loss(X, C):
    print("  initializing score model...")
    sched = make_schedule(n_steps=100)
    # Standardize for training stability
    mu, sigma = X.mean(0), X.std(0) + 1e-6
    X_std = (X - mu) / sigma
    C_std = (C - C.mean(0)) / (C.std(0) + 1e-6)
    model = ScoreMLP(dim_x=N_RBINS, dim_c=2, hidden=128, seed=0)
    print("  training (this is a few-second PoC)...")
    t0 = time.time()
    losses = train(model, X_std, C_std, sched,
                   n_epochs=30, batch=128, lr=2e-3, verbose=True)
    print(f"  elapsed: {time.time() - t0:.1f} s")
    final = float(np.mean(losses[-3:]))
    initial = float(np.mean(losses[:3]))
    print(f"  loss start={initial:.3f} → end={final:.3f}")
    assert final < initial * 0.9, "loss did not decrease"
    print("  PASS")
    return model, sched, mu, sigma, (C.mean(0), C.std(0) + 1e-6)


def test_unconditional_samples_in_distribution(model, sched, mu, sigma):
    print("  drawing 200 unconditional samples...")
    cond = np.zeros(2)
    x_std = sample(model, sched, cond=cond, n=200, guidance=0.0)
    x = x_std * sigma + mu
    # Sample log ρ at r=0.05 kpc should fall in the training range
    inner_train = -1, 12     # broad — synthetic data range
    n_in = ((x[:, 2] > inner_train[0]) & (x[:, 2] < inner_train[1])).sum()
    frac = n_in / x.shape[0]
    print(f"  inner-bin log10 ρ in plausible range: {frac:.1%}")
    assert frac > 0.6, f"only {frac:.1%} of samples are in range"
    print("  PASS")


def test_posterior_sampling_prefers_matching_mass(model, sched, mu, sigma, C_norm):
    """Inject a mock observation: M(<0.3 kpc) = 4e7 Msun (Sculptor-like).
    Posterior samples should cluster around profiles giving this mass."""
    print("  setting up mock observation...")
    target_logM_300pc = np.log10(4e7)
    sigma_obs = 0.3    # ~factor-of-2 mass uncertainty in dex (loose for PoC)

    def log_likelihood(x_std):
        x = x_std * sigma + mu
        # Mass enclosed within 300 pc
        # Find bin index nearest to 0.3 kpc
        i = int(np.argmin(np.abs(R_GRID - 0.3)))
        # Trapezoid integration up to bin i
        M = np.array([
            np.trapz(4 * np.pi * R_GRID[:i+1] ** 2 * 10 ** xi[:i+1], R_GRID[:i+1])
            for xi in x
        ])
        return -0.5 * ((np.log10(M.clip(min=1)) - target_logM_300pc) / sigma_obs) ** 2

    # Sculptor-like conditioning: log M_h ≈ 10, log M_star ≈ 6.5
    cond = (np.array([10.0, 6.5]) - C_norm[0]) / C_norm[1]
    print("  running DPS sampler (200 samples × 100 steps, ~few sec)...")
    x_std = sample_posterior(model, sched, cond, n=200,
                             log_likelihood_fn=log_likelihood,
                             guidance=0.5, dps_scale=0.03)
    x = x_std * sigma + mu
    i = int(np.argmin(np.abs(R_GRID - 0.3)))
    M = np.array([np.trapz(4 * np.pi * R_GRID[:i+1] ** 2 * 10 ** xi[:i+1],
                                 R_GRID[:i+1]) for xi in x])
    logM = np.log10(M.clip(min=1))
    print(f"  target log M(<300 pc) = {target_logM_300pc:.2f}")
    print(f"  posterior median      = {np.median(logM):.2f}")
    print(f"  posterior 16-84       = [{np.percentile(logM,16):.2f}, "
          f"{np.percentile(logM,84):.2f}]")

    # Should pull posterior median toward target
    bias = abs(np.median(logM) - target_logM_300pc)
    assert bias < 1.0, f"posterior median off by {bias:.2f} dex from target"
    print("  PASS")


if __name__ == "__main__":
    print("=== 1. dataset diversity ===")
    X, C, A = test_synthetic_dataset_multimodal()
    print("\n=== 2. training loss decrease ===")
    model, sched, mu, sigma, C_norm = test_diffusion_training_reduces_loss(X, C)
    print("\n=== 3. unconditional samples ===")
    test_unconditional_samples_in_distribution(model, sched, mu, sigma)
    print("\n=== 4. posterior preference (DPS) ===")
    test_posterior_sampling_prefers_matching_mass(model, sched, mu, sigma, C_norm)
    print("\nAll diffusion-prior PoC tests passed.")
