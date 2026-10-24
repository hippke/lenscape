"""Parallax, orbital motion (against VBBinaryLensing when installed), source radius chain, MCMC sampler."""
import numpy as np
import pytest
from lenscape import lightcurve as L, source as S, mcmc, contour

LN10 = np.log(10.0)
RA, DEC = "17:48:38.01", "-23:59:55.68"                     # KMT-2025-BLG-0121
P0 = np.array([756.463, 0.0311, 29.97, np.log10(1.0912), np.log10(0.0121), 3.449, np.log10(1.13e-3)])


def _vbbl():
    VB = pytest.importorskip("VBBinaryLensing")
    v = VB.VBBinaryLensing()
    v.Tol = v.RelTol = 1e-4
    v.parallaxsystem = 1                                    # (pi_EN, pi_EE)
    v.SetObjectCoordinates(f"{RA} {DEC}")
    return v


def _vb_par(p):
    t0, u0, tE, ls, lq, al, lr = p
    return [ls * LN10, lq * LN10, u0, al, lr * LN10, np.log(tE), t0 + 1e4]


def test_static_trajectory_equals_lc_full():
    t = np.linspace(740, 770, 400)
    ho = L.HighOrder(L.Trajectory(t, 2460000.0, RA, DEC, P0[0]), 1e-3, 1e-4, 0.01)
    A, _ = contour.lc_full(*P0[[3, 4, 1, 5, 6, 2, 0]], t, 1e-3, 1e-4, 0.01)
    assert np.allclose(ho.mag(np.concatenate([P0, [0, 0, 0, 0]])), A, rtol=1e-12)


def test_parallax_vs_vbbl():
    v = _vbbl()
    t = np.linspace(650, 860, 300)
    ho = L.HighOrder(L.Trajectory(t, 2460000.0, RA, DEC, P0[0]), 1e-3, 1e-4, 0.01)
    for pi in ((0.3, 0.0), (0.0, 0.3), (-0.2, 0.15)):
        Av = np.array(v.BinaryLightCurveParallax(_vb_par(P0) + list(pi), t + 1e4)[0])
        assert np.max(np.abs(ho.mag(np.concatenate([P0, pi])) / Av - 1)) < 3e-4


def test_orbital_motion_first_order_vs_vbbl():
    v = _vbbl()
    t = np.linspace(745, 768, 300)
    ho = L.HighOrder(L.Trajectory(t, 2460000.0, RA, DEC, P0[0]), 1e-3, 1e-4, 0.01)
    s = 10 ** P0[3]
    for dsdt, dadt in ((0.5, 0.0), (0.0, 0.5), (-0.004, 0.008)):     # VBBL circular orbit == linear to 1st order
        w = [dsdt / s / 365.25, dadt / 365.25, 1e-7]
        Av = np.array(v.BinaryLightCurveOrbital(_vb_par(P0) + [0.0, 0.0] + w, t + 1e4)[0])
        assert np.max(np.abs(ho.mag(np.concatenate([P0, [0, 0, dsdt, dadt]])) / Av - 1)) < 3e-4


def test_parallax_offsets_vanish_at_t0par():
    jd = 2460756.0 + np.array([-1e-3, 0.0, 1e-3])
    ds = L.parallax_offsets(jd, 2460756.0, *L.parse_coords(RA, DEC))
    assert np.abs(ds).max() < 1e-9                          # value and first derivative removed


def test_source_chain_reproduces_han2026_table5():
    # (V-I, I)_s, (V-I, I)_RGC, I_RGC,0, rho, tE -> theta_* [uas], theta_E [mas], mu [mas/yr] (Han et al. 2026)
    rows = [(3.184, 20.985, 3.492, 17.179, 14.326, 7.36e-4, 34.5, 0.780, 1.060, 11.22),
            (3.599, 20.424, 3.983, 17.646, 14.339, 3.13e-3, 19.31, 1.139, 0.364, 6.89),
            (1.549, 21.484, 1.843, 16.034, 14.538, 1.20e-3, 43.43, 0.337, 0.281, 2.36),
            (2.634, 21.772, 2.718, 16.430, 14.389, 1.13e-3, 29.97, 0.499, 0.442, 5.39)]
    for vi, i, vr, ir, ir0, rho, tE, th, thE, mu in rows:
        r = S.source_chain(vi, i, vr, ir, ir0, rho, tE)
        assert abs(r["theta_star"] / th - 1) < 0.03        # V-I -> V-K table differs from Bessell & Brett (1988)
        assert abs(r["theta_E"] / thE - 1) < 0.03 and abs(r["mu"] / mu - 1) < 0.03


def test_mcmc_gaussian():
    C = np.array([[1.0, 0.9], [0.9, 1.0]])
    Ci = np.linalg.inv(C)
    r = mcmc.run(lambda p: -0.5 * p @ Ci @ p, np.zeros(2), np.full(2, 0.1), nwalkers=16, nsteps=1500, seed=3)
    f = r["chain"][:, 300:].reshape(-1, 2)
    assert np.allclose(np.cov(f.T), C, atol=0.12)
    assert 0.3 < r["acc"].mean() < 0.9
    assert np.all(mcmc.autocorr_time(r["chain"][:, 300:]) < 100)


def test_mcmc_independent_ensembles():
    lp = lambda p: -0.5 * float(p @ p)
    r = mcmc.run_ensembles(lp, np.zeros(2), np.full(2, 0.1), nwalkers=8, nsteps=600, seed=1, workers=2)
    assert r["chain"].shape == (16, 600, 2)
    assert not np.array_equal(r["chain"][:8], r["chain"][8:])            # different seeds per ensemble
    f = r["chain"][:, 200:].reshape(-1, 2)
    assert np.allclose(f.std(axis=0), 1.0, atol=0.15)
