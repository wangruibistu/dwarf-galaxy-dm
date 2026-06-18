"""M4 (Paper II): amortised Neural Posterior Estimation via a Mixture Density
Network (NPE-A style), in JAX/equinox.

q(theta | x) is a K-component diagonal-Gaussian mixture whose parameters are
output by an MLP of the observation x. Trained by maximising the conditional
log-likelihood of the simulated (theta, x) pairs. Avoids a torch+sbi dependency;
consistent with the JAX stack used for the diffusion prior.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)


class MDN(eqx.Module):
    trunk: eqx.nn.MLP
    K: int = eqx.field(static=True)
    dt: int = eqx.field(static=True)

    def __init__(self, dim_x, dim_theta, K=8, width=128, depth=3, key=None):
        key = key if key is not None else jax.random.PRNGKey(0)
        self.K = K; self.dt = dim_theta
        out = K * (1 + 2 * dim_theta)
        self.trunk = eqx.nn.MLP(dim_x, out, width, depth, activation=jax.nn.silu, key=key)

    def __call__(self, x):
        o = self.trunk(x)
        logit = o[:self.K]
        mean = o[self.K:self.K + self.K * self.dt].reshape(self.K, self.dt)
        lstd = o[self.K + self.K * self.dt:].reshape(self.K, self.dt)
        lstd = jnp.clip(lstd, -6.0, 3.0)
        return logit, mean, lstd


def _logprob_one(model, x, theta):
    logit, mean, lstd = model(x)
    logw = jax.nn.log_softmax(logit)
    var = jnp.exp(2 * lstd)
    comp = -0.5 * (((theta[None, :] - mean) ** 2) / var + 2 * lstd
                   + jnp.log(2 * jnp.pi)).sum(axis=1)
    return jax.scipy.special.logsumexp(logw + comp)


def train_npe(model, X, Theta, n_epochs=400, batch=256, lr=1e-3, seed=0, verbose=True):
    Xj, Tj = jnp.asarray(X), jnp.asarray(Theta)
    n = Xj.shape[0]
    params, static = eqx.partition(model, eqx.is_inexact_array)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    b1, b2, eps = 0.9, 0.999, 1e-8

    @jax.jit
    def step(params, m, v, g, xb, tb):
        def loss_fn(p):
            mdl = eqx.combine(p, static)
            lp = jax.vmap(lambda xi, ti: _logprob_one(mdl, xi, ti))(xb, tb)
            return -jnp.mean(lp)
        loss, grad = jax.value_and_grad(loss_fn)(params)
        m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, grad)
        v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v, grad)
        mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** g), m)
        vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** g), v)
        params = jax.tree_util.tree_map(
            lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps), params, mh, vh)
        return params, m, v, loss

    rng = np.random.default_rng(seed); g = 0; losses = []
    for ep in range(n_epochs):
        perm = rng.permutation(n); el = 0.0; nb = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]; g += 1
            params, m, v, loss = step(params, m, v, g, Xj[idx], Tj[idx])
            el += float(loss); nb += 1
        losses.append(el / nb)
        if verbose and (ep % max(1, n_epochs // 8) == 0 or ep == n_epochs - 1):
            print(f"  epoch {ep:3d} | -logq {losses[-1]:.3f}")
    return eqx.combine(params, static), losses


def sample_posterior(model, x, n, key):
    """Draw n samples from q(theta | x)."""
    logit, mean, lstd = model(jnp.asarray(x))
    k1, k2 = jax.random.split(key)
    comp = jax.random.categorical(k1, logit, shape=(n,))
    eps = jax.random.normal(k2, (n, model.dt))
    return mean[comp] + eps * jnp.exp(lstd[comp])
