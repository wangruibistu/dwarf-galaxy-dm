"""Minimal denoising diffusion model on log ρ(r) profiles.

Pure numpy implementation. This is a *proof of concept* of the algorithm, not
a production trainer. Production version belongs in a torch/jax implementation
with proper U-Net (see roadmap in src/dm_models/diffusion_prior/README.md).

Algorithm:
  - VP-SDE (Ho+ 2020 DDPM): x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
  - Score model:  s_θ(x_t, t, c) ≈ -ε / σ_t
  - Loss:  E ‖s_θ(x_t, t, c) σ_t + ε‖²  (denoising-score-matching)
  - Sampler: DDPM ancestral sampling
  - Posterior: classifier-free guidance + Tweedie-formula likelihood injection
    via DPS (Chung+ 2023) for inverse problems
"""

from __future__ import annotations
import numpy as np
from scipy.special import expit


# ---------------------------------------------------------------------------
# Noise schedule (linear β; cosine would be better but linear suffices for PoC)
# ---------------------------------------------------------------------------
def make_schedule(n_steps: int = 200, beta_min: float = 1e-4, beta_max: float = 0.02):
    betas = np.linspace(beta_min, beta_max, n_steps)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return dict(
        betas=betas, alphas=alphas, alpha_bar=alpha_bar,
        sqrt_ab=np.sqrt(alpha_bar),
        sqrt_one_minus_ab=np.sqrt(1.0 - alpha_bar),
    )


# ---------------------------------------------------------------------------
# Score-function MLP (single hidden layer + residual; minimal but expressive)
# ---------------------------------------------------------------------------
class ScoreMLP:
    """Predicts noise ε given (x_t, t_embed, cond).

    Architecture: [input || t || c]  →  Dense(H)→ReLU  →  Dense(H)→ReLU  →  Dense(D)
    """

    def __init__(self, dim_x: int, dim_c: int, hidden: int = 256,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dim_x = dim_x
        self.dim_c = dim_c
        self.hidden = hidden
        d_in = dim_x + 16 + dim_c            # 16-dim sinusoidal t-embedding
        # He init
        self.W1 = rng.normal(0, np.sqrt(2 / d_in), (d_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, np.sqrt(2 / hidden), (hidden, dim_x))
        self.b3 = np.zeros(dim_x)

    def _t_embed(self, t):
        """Sinusoidal positional embedding for diffusion timestep."""
        t = np.atleast_1d(t).astype(float)
        freqs = np.exp(np.linspace(0, -np.log(1000.0), 8))
        ang = t[:, None] * freqs[None, :]
        return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)

    def forward(self, x_t, t, c):
        n = x_t.shape[0]
        te = self._t_embed(t)
        if te.shape[0] == 1: te = np.repeat(te, n, 0)
        if c.ndim == 1: c = np.repeat(c[None, :], n, 0)
        h0 = np.concatenate([x_t, te, c], axis=-1)
        h1 = np.maximum(h0 @ self.W1 + self.b1, 0.0)
        h2 = np.maximum(h1 @ self.W2 + self.b2, 0.0)
        out = h2 @ self.W3 + self.b3
        # Cache for backward
        self._cache = (x_t, te, c, h0, h1, h2)
        return out

    def backward(self, grad_out):
        """Backprop, return gradients dict."""
        x_t, te, c, h0, h1, h2 = self._cache
        dW3 = h2.T @ grad_out
        db3 = grad_out.sum(0)
        dh2 = grad_out @ self.W3.T
        dh2 *= (h2 > 0)
        dW2 = h1.T @ dh2
        db2 = dh2.sum(0)
        dh1 = dh2 @ self.W2.T
        dh1 *= (h1 > 0)
        dW1 = h0.T @ dh1
        db1 = dh1.sum(0)
        return dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3)

    def step(self, grads, lr=1e-3):
        for k, g in grads.items():
            setattr(self, k, getattr(self, k) - lr * g)

    def params(self):
        return {k: getattr(self, k).copy() for k in ["W1","b1","W2","b2","W3","b3"]}

    def load(self, params):
        for k, v in params.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(model: ScoreMLP, X, C, sched, n_epochs: int = 50,
          batch: int = 128, lr: float = 1e-3, p_drop_cond: float = 0.1,
          rng=None, verbose: bool = True):
    rng = rng or np.random.default_rng(1)
    n = X.shape[0]
    T = len(sched["betas"])
    losses = []
    for ep in range(n_epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        nb = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            x0 = X[idx]
            c  = C[idx].copy()
            # classifier-free guidance dropout
            mask = rng.random(c.shape[0]) < p_drop_cond
            c[mask] = 0.0
            t  = rng.integers(0, T, size=x0.shape[0])
            eps = rng.standard_normal(x0.shape)
            sqab = sched["sqrt_ab"][t][:, None]
            sqomab = sched["sqrt_one_minus_ab"][t][:, None]
            x_t = sqab * x0 + sqomab * eps
            pred_eps = model.forward(x_t, t, c)
            err = pred_eps - eps
            loss = (err ** 2).mean()
            epoch_loss += float(loss); nb += 1
            grad_out = 2 * err / err.size
            grads = model.backward(grad_out)
            model.step(grads, lr=lr)
        losses.append(epoch_loss / nb)
        if verbose and (ep % max(1, n_epochs // 10) == 0 or ep == n_epochs - 1):
            print(f"  epoch {ep:3d} | loss = {losses[-1]:.4f}")
    return losses


# ---------------------------------------------------------------------------
# Unconditional / conditional sampling
# ---------------------------------------------------------------------------
def project_monotonic(log_rho_std, mu, sigma):
    """Project standardised samples to physically monotone (non-increasing)
    log10 rho(r) profiles by least-squares isotonic regression, then
    re-standardise.

    rho(r) for a bound DM halo must be non-increasing in r. The minimal MLP
    score model occasionally generates small density inversions. We remove
    them with isotonic (pool-adjacent-violators) regression, which is the
    least-squares monotone fit -- unlike a cumulative minimum it does not
    collapse an inner bump down to a dip, so it does not bias the inner
    slope toward zero. This is a physical hard constraint, not a tuning knob.
    """
    from scipy.optimize import isotonic_regression
    x_phys = log_rho_std * sigma + mu
    out = np.empty_like(x_phys)
    for i in range(x_phys.shape[0]):
        # isotonic_regression fits non-decreasing; fit to the reversed array
        # (decreasing r->increasing index reversed) to obtain non-increasing.
        res = isotonic_regression(x_phys[i][::-1], increasing=True)
        out[i] = res.x[::-1]
    return (out - mu) / sigma


def sample(model: ScoreMLP, sched, cond, n: int, guidance: float = 2.0,
           rng=None, monotonic=None):
    """Conditional DDPM ancestral sampling.

    If ``monotonic`` is provided as a tuple ``(mu, sigma)`` the standardiser
    statistics, generated profiles are projected to be physically monotone
    (non-increasing in rho) before being returned.
    """
    rng = rng or np.random.default_rng(2)
    T = len(sched["betas"])
    x = rng.standard_normal((n, model.dim_x))
    c = np.broadcast_to(cond, (n, model.dim_c)).copy()
    c0 = np.zeros_like(c)
    for t in range(T - 1, -1, -1):
        z = rng.standard_normal(x.shape) if t > 0 else np.zeros_like(x)
        eps_c = model.forward(x, np.array([t]), c)
        eps_u = model.forward(x, np.array([t]), c0)
        eps = (1 + guidance) * eps_c - guidance * eps_u
        ab = sched["alpha_bar"][t]
        a  = sched["alphas"][t]
        b  = sched["betas"][t]
        # DDPM mean
        mu = (x - b / np.sqrt(1 - ab) * eps) / np.sqrt(a)
        x = mu + np.sqrt(b) * z
    if monotonic is not None:
        x = project_monotonic(x, monotonic[0], monotonic[1])
    return x


# ---------------------------------------------------------------------------
# Importance-reweighting posterior (exact, gradient-free, no tuning knobs)
# ---------------------------------------------------------------------------
def sample_posterior_importance(model, sched, cond, n_out, log_likelihood_fn,
                                n_prior=None, guidance=0.5, rng=None,
                                monotonic=None, return_diagnostics=False):
    """Posterior samples by likelihood-reweighting of the conditional prior.

    Draw ``n_prior`` profiles from the conditional diffusion prior, weight
    each by its Jeans likelihood w_i = exp(logL_i - max logL), and resample
    ``n_out`` profiles with replacement in proportion to the weights. This
    is an exact (self-normalised importance) posterior estimator, free of the
    finite-difference gradient approximation and step-size tuning of DPS,
    and is the method used for the published results.

    Returns the resampled posterior profiles (standardised). If
    ``return_diagnostics`` is True, also returns the effective sample size.
    """
    rng = rng or np.random.default_rng(3)
    if n_prior is None:
        n_prior = max(2000, 10 * n_out)
    x_prior = sample(model, sched, cond=cond, n=n_prior, guidance=guidance,
                     rng=rng, monotonic=monotonic)
    logL = np.asarray(log_likelihood_fn(x_prior))
    logL = np.where(np.isfinite(logL), logL, -np.inf)
    w = np.exp(logL - np.max(logL))
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        # likelihood uninformative; posterior = prior
        idx = rng.integers(0, n_prior, size=n_out)
    else:
        p = w / w_sum
        idx = rng.choice(n_prior, size=n_out, replace=True, p=p)
    x_post = x_prior[idx]
    if return_diagnostics:
        ess = (w_sum ** 2) / np.sum(w ** 2) if w_sum > 0 else 0.0
        return x_post, float(ess)
    return x_post


# ---------------------------------------------------------------------------
# Score-based posterior sampling (DPS, Chung+ 2023)
# ---------------------------------------------------------------------------
def sample_posterior(model, sched, cond, n: int, log_likelihood_fn,
                     guidance: float = 1.5, dps_scale: float = 0.5,
                     rng=None):
    """Sample from p(x | obs, cond) ∝ p(x | cond) × L(obs | x).

    log_likelihood_fn(x) -> (n,) array of log L for the n batch elements.
    Gradients are approximated by finite differences (fine for low-dim PoC).
    """
    rng = rng or np.random.default_rng(3)
    T = len(sched["betas"])
    x = rng.standard_normal((n, model.dim_x))
    c = np.broadcast_to(cond, (n, model.dim_c)).copy()
    c0 = np.zeros_like(c)

    def grad_logL(x):
        eps = 1e-3
        g = np.zeros_like(x)
        L0 = log_likelihood_fn(x)
        for d in range(x.shape[1]):
            xp = x.copy(); xp[:, d] += eps
            g[:, d] = (log_likelihood_fn(xp) - L0) / eps
        # Clip per-sample gradient norm — keeps DPS stable on log-likelihoods
        g_norm = np.linalg.norm(g, axis=-1, keepdims=True) + 1e-8
        max_norm = 5.0
        scale = np.minimum(max_norm / g_norm, 1.0)
        return g * scale

    for t in range(T - 1, -1, -1):
        z = rng.standard_normal(x.shape) if t > 0 else np.zeros_like(x)
        eps_c = model.forward(x, np.array([t]), c)
        eps_u = model.forward(x, np.array([t]), c0)
        eps   = (1 + guidance) * eps_c - guidance * eps_u
        ab = sched["alpha_bar"][t]; a = sched["alphas"][t]; b = sched["betas"][t]
        x_pred_0 = (x - np.sqrt(1 - ab) * eps) / np.sqrt(ab)         # Tweedie x̂₀(x_t)
        # likelihood gradient on x̂₀
        g = grad_logL(x_pred_0)
        mu = (x - b / np.sqrt(1 - ab) * eps) / np.sqrt(a)
        x  = mu + np.sqrt(b) * z + dps_scale * g
    return x
