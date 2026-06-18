"""M2 (Paper II): production 1D-CNN score network for the halo-profile diffusion
prior, in JAX/equinox.

A FiLM-conditioned 1D convolutional residual network predicts the DDPM noise
epsilon given a noised log10 rho(r) profile, the diffusion time, and the
conditioning vector. Trained with denoising score matching + classifier-free
guidance; sampled by ancestral DDPM with CFG and a monotone (non-increasing)
projection. Local convolutions preserve the sharp inner cusp that the minimal
MLP of Paper I compressed.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------
# VP-DDPM schedule
# --------------------------------------------------------------------------
def make_schedule(T=200, beta_min=1e-4, beta_max=2e-2):
    betas = jnp.linspace(beta_min, beta_max, T)
    alphas = 1.0 - betas
    ab = jnp.cumprod(alphas)
    return dict(betas=betas, alphas=alphas, alpha_bar=ab,
                sqrt_ab=jnp.sqrt(ab), sqrt_omab=jnp.sqrt(1 - ab), T=T)


def _t_embed(t, dim=16):
    t = jnp.atleast_1d(t).astype(jnp.float64)
    freqs = jnp.exp(jnp.linspace(0, -np.log(1000.0), dim // 2))
    ang = t[..., None] * freqs
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1).reshape(-1)


# --------------------------------------------------------------------------
# FiLM-conditioned residual 1D-CNN
# --------------------------------------------------------------------------
class ResBlock(eqx.Module):
    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d
    film: eqx.nn.Linear

    def __init__(self, ch, k, emb, key):
        k1, k2, k3 = jax.random.split(key, 3)
        p = k // 2
        self.conv1 = eqx.nn.Conv1d(ch, ch, k, padding=p, key=k1)
        self.conv2 = eqx.nn.Conv1d(ch, ch, k, padding=p, key=k2)
        self.film = eqx.nn.Linear(emb, 2 * ch, key=k3)

    def __call__(self, h, emb):
        hh = jax.nn.silu(self.conv1(h))
        sc, sh = jnp.split(self.film(emb), 2)
        hh = jax.nn.silu(sc[:, None] * hh + sh[:, None])
        hh = self.conv2(hh)
        return h + hh


class ScoreCNN(eqx.Module):
    in_conv: eqx.nn.Conv1d
    out_conv: eqx.nn.Conv1d
    blocks: list
    cond1: eqx.nn.Linear
    cond2: eqx.nn.Linear
    te_dim: int = eqx.field(static=True)
    dim_c: int = eqx.field(static=True)

    def __init__(self, dim_c=4, ch=64, k=5, n_blocks=4, emb=128, te_dim=16, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)
        ks = jax.random.split(key, n_blocks + 4)
        self.te_dim = te_dim
        self.dim_c = dim_c
        self.in_conv = eqx.nn.Conv1d(1, ch, k, padding=k // 2, key=ks[0])
        self.out_conv = eqx.nn.Conv1d(ch, 1, k, padding=k // 2, key=ks[1])
        self.cond1 = eqx.nn.Linear(te_dim + dim_c, emb, key=ks[2])
        self.cond2 = eqx.nn.Linear(emb, emb, key=ks[3])
        self.blocks = [ResBlock(ch, k, emb, ks[4 + i]) for i in range(n_blocks)]

    def __call__(self, x, t, c):
        emb = jnp.concatenate([_t_embed(t, self.te_dim), c])
        emb = jax.nn.silu(self.cond1(emb))
        emb = jax.nn.silu(self.cond2(emb))
        h = self.in_conv(x[None, :])
        for b in self.blocks:
            h = b(h, emb)
        return self.out_conv(h)[0]


# --------------------------------------------------------------------------
# Training (hand-rolled Adam; no optax)
# --------------------------------------------------------------------------
def train_cnn(model, X, C, sched, n_epochs=300, batch=128, lr=2e-3,
              p_drop=0.1, seed=1, verbose=True):
    Xj, Cj = jnp.asarray(X), jnp.asarray(C)
    n, D = Xj.shape
    T = sched["T"]
    params, static = eqx.partition(model, eqx.is_inexact_array)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def step(params, m, v, gstep, xb, tb, cb, noise):
        def loss_fn(p):
            mdl = eqx.combine(p, static)
            sab = sched["sqrt_ab"][tb][:, None]
            som = sched["sqrt_omab"][tb][:, None]
            xt = sab * xb + som * noise
            pred = jax.vmap(mdl)(xt, tb.astype(jnp.float64), cb)
            return jnp.mean((pred - noise) ** 2)
        loss, g = jax.value_and_grad(loss_fn)(params)
        m = jax.tree_util.tree_map(lambda mm, gg: b1 * mm + (1 - b1) * gg, m, g)
        v = jax.tree_util.tree_map(lambda vv, gg: b2 * vv + (1 - b2) * gg * gg, v, g)
        mh = jax.tree_util.tree_map(lambda mm: mm / (1 - b1 ** gstep), m)
        vh = jax.tree_util.tree_map(lambda vv: vv / (1 - b2 ** gstep), v)
        params = jax.tree_util.tree_map(
            lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps), params, mh, vh)
        return params, m, v, loss

    rng = np.random.default_rng(seed)
    gstep = 0; losses = []
    for ep in range(n_epochs):
        perm = rng.permutation(n); el = 0.0; nb = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            cb = Cj[idx]
            drop = rng.random(len(idx)) < p_drop
            cb = cb.at[jnp.asarray(np.where(drop)[0])].set(0.0)
            tb = jnp.asarray(rng.integers(0, T, len(idx)))
            noise = jnp.asarray(rng.standard_normal((len(idx), D)))
            gstep += 1
            params, m, v, loss = step(params, m, v, gstep, Xj[idx], tb, cb, noise)
            el += float(loss); nb += 1
        losses.append(el / nb)
        if verbose and (ep % max(1, n_epochs // 10) == 0 or ep == n_epochs - 1):
            print(f"  epoch {ep:3d} | loss {losses[-1]:.4f}")
    return eqx.combine(params, static), losses


# --------------------------------------------------------------------------
# Sampling (ancestral DDPM + CFG) and monotone projection
# --------------------------------------------------------------------------
def _pav_nonincreasing(y):
    """Pool-adjacent-violators: nearest non-increasing sequence."""
    y = np.asarray(y, float).copy()
    w = np.ones_like(y); n = len(y)
    val = list(y); wt = list(w); idx = [[i] for i in range(n)]
    i = 0
    while i < len(val) - 1:
        if val[i] < val[i + 1]:                     # violation (must be non-increasing)
            nv = (val[i] * wt[i] + val[i + 1] * wt[i + 1]) / (wt[i] + wt[i + 1])
            val[i] = nv; wt[i] += wt[i + 1]; idx[i] += idx[i + 1]
            del val[i + 1]; del wt[i + 1]; del idx[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = np.empty(n)
    for v_, ix in zip(val, idx):
        out[ix] = v_
    return out


def sample_cnn(model, sched, cond, n, guidance=0.5, seed=2, monotone=None):
    """Conditional ancestral DDPM sampling with classifier-free guidance.
    monotone=(mu,sigma): project to non-increasing log-rho after de-standardising."""
    key = jax.random.PRNGKey(seed)
    D = model.in_conv.weight.shape[0] if False else sched.get("D", None)
    # infer D from a dry run
    c = jnp.asarray(cond)
    c0 = jnp.zeros_like(c)
    T = sched["T"]
    key, sk = jax.random.split(key)
    # need D: sample dim from out_conv via a probe
    D = 32
    x = jax.random.normal(sk, (n, D))

    @jax.jit
    def denoise(x, t):
        tv = jnp.full((n,), t, dtype=jnp.float64)
        ec = jax.vmap(lambda xi, ti: model(xi, ti, c))(x, tv)
        eu = jax.vmap(lambda xi, ti: model(xi, ti, c0))(x, tv)
        return (1 + guidance) * ec - guidance * eu

    for t in range(T - 1, -1, -1):
        key, zk = jax.random.split(key)
        z = jax.random.normal(zk, x.shape) if t > 0 else jnp.zeros_like(x)
        eps = denoise(x, t)
        ab = sched["alpha_bar"][t]; a = sched["alphas"][t]; b = sched["betas"][t]
        mu = (x - b / jnp.sqrt(1 - ab) * eps) / jnp.sqrt(a)
        x = mu + jnp.sqrt(b) * z
    x = np.asarray(x)
    if monotone is not None:
        mu_s, sig_s = monotone
        lr = x * sig_s + mu_s
        lr = np.array([_pav_nonincreasing(row) for row in lr])
        x = (lr - mu_s) / sig_s
    return x
