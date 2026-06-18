"""Shared Paper II pipeline: simulation-calibrated diffusion prior + multi-pop
anisotropic Jeans simulator + amortised MDN-NPE. Imported by M4/M5/M6.

The NPE is trained on a *mixture proposal*: a fraction `flat_frac` of the
training profiles have their inner slope drawn ~uniform over cusp->core
(make_flat_gamma_profiles), the rest from the diffusion prior conditioned at
Sculptor's mass.  Training on the conditional DC14 prior alone leaves the cored
end unsupported (it is strongly cusp-leaning at Sculptor's mass) and biases
blind recovery of cored mocks; the DC14 prior is re-imposed afterwards by
importance reweighting where wanted.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import jax
from src.dm_models.diffusion_prior.dc14_library import (
    make_dc14_dataset, make_flat_gamma_profiles, SCULPTOR_COND,
    _gnfwfit_slope, R_GRID)
from src.dm_models.diffusion_prior.diffusion import (
    make_schedule, ScoreMLP, train as train_diff, sample as sample_diff)
from src.dynamical_modeling.multipop_jeans import sigma_los2_aniso
from src.inference.npe_mdn import MDN, train_npe, sample_posterior

RE1, RE2 = 0.15, 0.35
R1 = np.logspace(np.log10(0.05), np.log10(0.8), 8)
R2 = np.logspace(np.log10(0.10), np.log10(1.5), 8)
FRAC_ERR = 0.12


def menc_func(log_rho):
    rho = 10 ** log_rho
    M = np.cumsum(4 * np.pi * R_GRID ** 2 * rho * np.gradient(R_GRID))
    return lambda r: np.interp(np.atleast_1d(r), R_GRID, M)


def observe(log_rho, b1, b2, logA, rng=None, r1=None, r2=None,
            re1=RE1, re2=RE2, frac_err=FRAC_ERR):
    """Profile + (beta1,beta2,log10 A) -> noiseless or noisy 2-pop sigma^2 vector."""
    r1 = R1 if r1 is None else r1
    r2 = R2 if r2 is None else r2
    Mf = menc_func(log_rho)
    s1 = (10 ** logA) * sigma_los2_aniso(r1, Mf, b1, re1, n_r=200)
    s2 = (10 ** logA) * sigma_los2_aniso(r2, Mf, b2, re2, n_r=200)
    e1 = frac_err * s1 + 1.0; e2 = frac_err * s2 + 1.0
    if rng is not None:
        s1 = s1 + rng.normal(0, e1); s2 = s2 + rng.normal(0, e2)
    return np.concatenate([s1, s2]), np.concatenate([e1, e2])


def build_pipeline(seed=0, n_sim=7000, npe_epochs=400, flat_frac=0.5,
                   re1=RE1, re2=RE2, r1=None, r2=None, frac_err=FRAC_ERR,
                   loga_range=(-0.5, 0.7), verbose=False):
    r1 = R1 if r1 is None else np.asarray(r1, float)
    r2 = R2 if r2 is None else np.asarray(r2, float)
    Xlib, Clib, split = make_dc14_dataset(n=9000, seed=0)
    mu, sig = Xlib.mean(0), Xlib.std(0) + 1e-6
    cm, cs = Clib.mean(0), Clib.std(0) + 1e-6
    sched = make_schedule(n_steps=200)
    dmodel = ScoreMLP(dim_x=32, dim_c=4, hidden=256, seed=0)
    train_diff(dmodel, (Xlib[split] - mu) / sig, (Clib[split] - cm) / cs, sched,
               n_epochs=220, batch=128, lr=2e-3, verbose=False)
    train_diff(dmodel, (Xlib[split] - mu) / sig, (Clib[split] - cm) / cs, sched,
               n_epochs=120, batch=128, lr=4e-4, verbose=False)

    cond = (SCULPTOR_COND - cm) / cs
    n_flat = int(n_sim * flat_frac); n_diff = n_sim - n_flat
    prof_d = sample_diff(dmodel, sched, cond=cond, n=n_diff, guidance=0.5,
                         rng=np.random.default_rng(seed + 1), monotonic=(mu, sig))
    prof_d = np.asarray(prof_d) * sig + mu
    prof_f = make_flat_gamma_profiles(n_flat, seed=seed + 3)
    prof = np.vstack([prof_d, prof_f])
    is_flat = np.concatenate([np.zeros(n_diff, bool), np.ones(n_flat, bool)])

    rng = np.random.default_rng(seed + 2)
    nx = len(r1) + len(r2)
    th = np.zeros((n_sim, 4)); X = np.zeros((n_sim, nx))
    for i, lr in enumerate(prof):
        g = _gnfwfit_slope(lr)
        b1 = rng.uniform(-0.4, 0.4); b2 = rng.uniform(-0.4, 0.4); logA = rng.uniform(*loga_range)
        x, _ = observe(lr, b1, b2, logA, rng, r1=r1, r2=r2, re1=re1, re2=re2,
                       frac_err=frac_err)
        th[i] = [g, b1, b2, logA]; X[i] = x
    ok = np.all(np.isfinite(X), 1) & np.isfinite(th[:, 0])
    theta, X, is_flat = th[ok], X[ok], is_flat[ok]
    # shuffle BEFORE the train/held-out split: the raw order is diffusion-prior
    # first, flat-proposal last, so an ordered split would put only flat sims
    # in the held-out set and break cal/test exchangeability
    per = np.random.default_rng(seed + 7).permutation(len(theta))
    theta, X, is_flat = theta[per], X[per], is_flat[per]

    tm, ts = theta.mean(0), theta.std(0) + 1e-9
    xm, xs = X.mean(0), X.std(0) + 1e-9
    Ts, Xs = (theta - tm) / ts, (X - xm) / xs
    ntr = int(0.9 * len(Ts))
    model = MDN(dim_x=nx, dim_theta=4, K=8, width=128, depth=3, key=jax.random.PRNGKey(0))
    model, _ = train_npe(model, Xs[:ntr], Ts[:ntr], n_epochs=npe_epochs,
                         batch=256, lr=1e-3, verbose=verbose)
    return dict(model=model, tm=tm, ts=ts, xm=xm, xs=xs, theta=theta, X=X, ntr=ntr,
                is_flat=is_flat, dmodel=dmodel, sched=sched, mu=mu, sig=sig,
                cm=cm, cs=cs, cond=cond, prof=prof,
                r1=r1, r2=r2, re1=re1, re2=re2, frac_err=frac_err)


def posterior_gamma(P, x_raw, n=4000, key=None):
    """Posterior samples of theta given a RAW observation vector x_raw.
    Applies the global width-calibration factor P['widen'] (if set) around the
    per-column median -- conformal-style recalibration fit on held-out sims."""
    key = key if key is not None else jax.random.PRNGKey(123)
    xs = (np.asarray(x_raw) - P["xm"]) / P["xs"]
    ps = np.asarray(sample_posterior(P["model"], xs, n, key)) * P["ts"] + P["tm"]
    s = P.get("widen", 1.0)
    if s != 1.0:
        med = np.median(ps, axis=0)
        ps = med + s * (ps - med)
    return ps  # columns: gamma150, beta1, beta2, log10A


def calibrate_width(P, idx, level=0.68, n_draw=400, key=None):
    """Width factor by grid search using the SAME percentile-interval criterion
    as the coverage evaluation (median +/- s*(percentile - median)); a
    symmetric-interval conformal quantile does not transfer when the MDN
    posterior is skewed."""
    key = key if key is not None else jax.random.PRNGKey(99)
    Xs = (P["X"] - P["xm"]) / P["xs"]
    samples, truth = [], []
    for j in idx:
        key, sk = jax.random.split(key)
        samples.append(np.asarray(sample_posterior(P["model"], Xs[j], n_draw, sk))[:, 0])
        truth.append((P["theta"][j, 0] - P["tm"][0]) / P["ts"][0])
    qlo, qhi = 50 * (1 - level), 50 * (1 + level)

    def cov(s):
        c = 0
        for ps, t in zip(samples, truth):
            m = np.median(ps)
            lo, hi = np.percentile(m + s * (ps - m), [qlo, qhi])
            c += (lo <= t <= hi)
        return c / len(truth)

    ss = np.linspace(0.8, 1.8, 41)
    cs = np.array([cov(s) for s in ss])
    return float(ss[int(np.argmin(np.abs(cs - level)))])


def dc14_prior_weights(P, g_samples, bw=0.08):
    """Importance weights re-imposing the DC14 conditional prior on posterior
    gamma samples drawn under the mixture proposal."""
    from scipy.stats import gaussian_kde
    g_prop = P["theta"][:, 0]
    g_dc14 = P["theta"][~P["is_flat"], 0]
    k_prop = gaussian_kde(g_prop, bw_method=bw)
    k_dc14 = gaussian_kde(g_dc14, bw_method=bw)
    w = k_dc14(g_samples) / np.maximum(k_prop(g_samples), 1e-12)
    return w / w.sum()


def weighted_quantile(x, q, w):
    i = np.argsort(x); x, w = x[i], w[i]
    c = np.cumsum(w) / w.sum()
    return np.interp(np.asarray(q), c, x)
