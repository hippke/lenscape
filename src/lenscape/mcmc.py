"""Affine-invariant ensemble sampler (Goodman & Weare 2010 stretch move, red/blue half-ensemble updates as in emcee),
with optional fork-based parallel evaluation of the log-probability, integrated autocorrelation time (Sokal window)
and posterior summaries. No dependency beyond numpy.

    lp = lambda p: -0.5 * model.chi2(backend, p, data)            # flat priors inside the physical box
    res = run(lp, p0, scale, nwalkers=32, nsteps=3000, workers=8)
    summary(res["chain"][:, burn:], names)
"""
import multiprocessing as mp
import numpy as np

_LP = None                                     # log-probability for forked workers (set before the pool starts)


def _call(p):
    return _LP(p)


def _evaluate(pool, lp, ps):
    if pool is None:
        return np.array([lp(p) for p in ps])
    return np.array(pool.map(_call, list(ps), chunksize=max(1, len(ps) // (4 * pool._processes))))


def run(lp, p0, scale, nwalkers=32, nsteps=2000, a=2.0, seed=0, workers=1, init=None, progress=None):
    """Sample exp(lp). Walkers start in a Gaussian ball p0 + scale * N(0,1) (or at `init`, shape (nwalkers, ndim)).
    Returns dict(chain (nwalkers, nsteps, ndim), lnp (nwalkers, nsteps), acc (acceptance fraction per walker))."""
    global _LP
    rng = np.random.default_rng(seed)
    p0, scale = np.asarray(p0, float), np.asarray(scale, float)
    ndim = len(p0)
    if nwalkers < 2 * ndim or nwalkers % 2:
        raise ValueError("nwalkers must be even and >= 2 ndim")
    X = np.array(init, float) if init is not None else p0 + scale * rng.standard_normal((nwalkers, ndim))
    _LP = lp
    pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
    try:
        L = _evaluate(pool, lp, X)
        bad = ~np.isfinite(L)
        if bad.all():
            raise ValueError("no walker has a finite log-probability")
        chain = np.empty((nwalkers, nsteps, ndim))
        lnp = np.empty((nwalkers, nsteps))
        nacc = np.zeros(nwalkers)
        half = nwalkers // 2
        for it in range(nsteps):
            for s in (0, 1):
                act = np.arange(s * half, (s + 1) * half)
                oth = np.arange((1 - s) * half, (2 - s) * half)
                z = ((a - 1.0) * rng.random(half) + 1.0) ** 2 / a          # g(z) ~ 1/sqrt(z) on [1/a, a]
                Y = X[oth[rng.integers(0, half, half)]]
                prop = Y + z[:, None] * (X[act] - Y)
                Lp = _evaluate(pool, lp, prop)
                lr = (ndim - 1) * np.log(z) + Lp - L[act]
                ok = np.isfinite(Lp) & (np.log(rng.random(half)) < lr)
                X[act[ok]], L[act[ok]] = prop[ok], Lp[ok]
                nacc[act[ok]] += 1
            chain[:, it], lnp[:, it] = X, L
            if progress and (it + 1) % progress == 0:
                print(f"  mcmc step {it + 1}/{nsteps}: max lnp {L.max():.2f}, acc {nacc.mean() / (it + 1):.2f}",
                      flush=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()
    return dict(chain=chain, lnp=lnp, acc=nacc / nsteps)


def run_ensembles(lp, p0, scale, nwalkers=16, nsteps=2000, a=2.0, seed=0, workers=4):
    """Independent ensembles, one per process (no synchronisation between them), merged along the walker axis.
    Each ensemble needs its own burn-in. Returns the same dict as run(); acc per walker of all ensembles."""
    global _LP
    _LP = lp
    args = [(None, p0, scale, nwalkers, nsteps, a, seed + 1000 * k) for k in range(workers)]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_ens_child, args, chunksize=1)
    return dict(chain=np.concatenate([r["chain"] for r in res]), lnp=np.concatenate([r["lnp"] for r in res]),
                acc=np.concatenate([r["acc"] for r in res]), nens=workers)


def _ens_child(args):
    return run(_LP, *args[1:])


def autocorr_time(chain, c=5.0):
    """Integrated autocorrelation time per parameter (walker-averaged autocorrelation function, Sokal window
    M >= c tau). chain: (nwalkers, nsteps, ndim)."""
    nw, n, nd = chain.shape
    m = 1 << int(np.ceil(np.log2(2 * n)))
    tau = np.empty(nd)
    for d in range(nd):
        x = chain[:, :, d] - chain[:, :, d].mean(axis=1, keepdims=True)
        f = np.fft.rfft(x, m, axis=1)
        acf = np.fft.irfft(f * np.conj(f), m, axis=1)[:, :n].mean(axis=0)
        if acf[0] <= 0:
            tau[d] = np.nan; continue
        acf /= acf[0]
        t = 2.0 * np.cumsum(acf) - 1.0
        w = np.arange(n) >= c * t
        tau[d] = t[np.argmax(w)] if w.any() else t[-1]
    return tau


def summary(chain, names=None, q=(15.865, 50.0, 84.135)):
    """Median and 1-sigma equivalent interval per parameter; returns list of (name, median, -err, +err)."""
    flat = chain.reshape(-1, chain.shape[-1])
    names = names or [f"p{k}" for k in range(flat.shape[1])]
    out = []
    for k, nm in enumerate(names):
        lo, md, hi = np.percentile(flat[:, k], q)
        out.append((nm, md, md - lo, hi - md))
    return out
