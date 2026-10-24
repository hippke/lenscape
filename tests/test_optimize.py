"""lenscape.optimize (Nelder-Mead copied from SciPy) gives bit-identical results to SciPy; bisection root finder."""
import numpy as np
import pytest
from lenscape import optimize as O


def rosen(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


OPTIONS = [dict(), dict(adaptive=True), dict(maxfev=200, adaptive=True), dict(maxfev=1500, adaptive=True, fatol=1e-2),
           dict(maxfev=6000, xatol=1e-8, fatol=1e-3, adaptive=True), dict(maxiter=50)]


@pytest.mark.parametrize("opts", OPTIONS)
def test_identical_to_scipy(opts):
    so = pytest.importorskip("scipy.optimize")
    for x0 in ([1.3, 0.7, 0.8, 1.9, 1.2], [0.0, -1.0, 2.0], [3, 1]):
        a = O.minimize(rosen, x0, method="Nelder-Mead", options=opts)
        b = so.minimize(rosen, x0, method="Nelder-Mead", options=opts)
        assert np.array_equal(a.x, b.x) and a.fun == b.fun and a.nfev == b.nfev and a.nit == b.nit
        assert a.status == b.status and a.success == b.success


def test_identical_with_initial_simplex():
    so = pytest.importorskip("scipy.optimize")
    x0 = np.array([0.5, 0.5, 0.5])
    sim = np.vstack([x0] + [x0 + np.eye(3)[k] * 0.1 for k in range(3)])
    o = dict(maxfev=1000, adaptive=True, initial_simplex=sim, fatol=1e-2)
    a, b = O.minimize(rosen, x0, options=o), so.minimize(rosen, x0, method="Nelder-Mead", options=o)
    assert np.array_equal(a.x, b.x) and a.nfev == b.nfev


def test_lenscape_fit_identical_to_scipy(monkeypatch):
    """A complete lenscape fit (angle screen, explore, refine) is bit-identical with SciPy's Nelder-Mead."""
    so = pytest.importorskip("scipy.optimize")
    from lenscape import fit as F, config as C, data as D
    be = F.make_backend(C.REFINE_BACKEND)
    p = np.array([0.0, 0.08, 25.0, np.log10(1.1), np.log10(5e-3), 4.2, -2.8])
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(-40, 40, 200))
    d = D.Data(t, 800 * be.lc(p, t) + 100 + rng.normal(0, 20, 200), np.full(200, 20.0), np.zeros(200, int), ["T"])
    cfg = dict(C.DEFAULT, seed=dict(t0=0.2, u0=0.07, tE=27.0, logrho=-2.7))
    a = F.fit_point(p[3], p[4], d, cfg)
    monkeypatch.setattr(F, "minimize", so.minimize)
    b = F.fit_point(p[3], p[4], d, cfg)
    assert a[0] == b[0] and np.array_equal(a[1], b[1])


def test_bisect():
    assert abs(O.bisect(lambda x: x ** 3 - 2, 0, 2) - 2 ** (1 / 3)) < 1e-10
    with pytest.raises(ValueError):
        O.bisect(lambda x: x * x + 1, -1, 1)
