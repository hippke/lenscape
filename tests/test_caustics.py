"""Caustics, trajectory and magnification maps for figures (lenscape.caustics)."""
import numpy as np
import pytest
from lenscape import caustics as K, fit as F, config as C


def test_caustics_match_vbbl():
    V = pytest.importorskip("VBBinaryLensing")
    v = V.VBBinaryLensing()
    for s, q in ((1.0, 0.01), (2.0, 1e-4), (0.6, 1e-3)):
        ours = np.concatenate(K.caustics(s, q, 6000))
        ref = v.Caustics(s, q)
        R = np.concatenate([np.array(c[0]) + 1j * np.array(c[1]) for c in ref])
        assert len(K.caustics(s, q, 6000)) == len(ref)
        assert max(np.min(abs(ours - r)) for r in R[::7]) < 1e-9


def test_wide_planetary_caustic_position():
    """Wide planet (q << 1): the planetary caustic lies at s - 1/s from the primary, on the planet side."""
    s, q = 2.0, 1e-5
    ca = K.caustics(s, q, 2000)
    xp = K.lens_positions(s, q)[0]
    centres = sorted(np.mean(c[:-1]).real - xp for c in ca)
    assert len(ca) == 2 and abs(centres[1] - (s - 1 / s)) < 1e-3 and abs(centres[0]) < 1e-3


def test_trajectory_matches_light_curve_model():
    """Point-source magnification along trajectory() equals the light-curve model (far from caustics, tiny rho)."""
    p = np.array([0.0, 0.3, 20.0, np.log10(1.3), np.log10(0.02), 0.7, -5.0])
    t = np.linspace(-30, 30, 61)
    y = K.trajectory(p, t)
    from lenscape import nbmag
    A_map = nbmag.mags_point(np.full(len(t), 1.3), np.full(len(t), 0.02), y.real, y.imag)
    A_lc = F.make_backend(C.REFINE_BACKEND).lc(p, t)
    assert np.max(np.abs(A_map / A_lc - 1)) < 1e-4


def test_figures_render(tmp_path):
    """plot_lightcurve / plot_geometry run on a synthetic two-telescope data set and write their files."""
    import matplotlib
    matplotlib.use("Agg")
    from lenscape import data as D, figures as FG
    be = F.make_backend(C.REFINE_BACKEND)
    p = np.array([0.0, 0.05, 20.0, np.log10(1.05), np.log10(0.004), 4.5, -3.0])
    t = np.linspace(-20, 20, 200)
    A = be.lc(p, t)
    idx = np.arange(200) % 2
    f = np.where(idx == 0, 1000 * A + 200, 500 * A - 50.0)
    d = D.Data(t, f, np.full(200, 5.0), idx, ["T1", "T2"])
    mag, err, fs0, fb0 = FG.aligned(be, p, d)
    assert np.allclose([fs0, fb0], [1000, 200], rtol=1e-6)
    _, _, fs1, fb1 = FG.aligned(be, p, D.Data(t, -f, np.full(200, 5.0), idx, ['T1', 'T2']))
    assert np.isclose(fs1, 1000, rtol=1e-6) and fb1 == 0.0          # negative source flux: |fs|, no blend
    assert np.allclose(mag, 22.0 - 2.5 * np.log10(1000 * A + 200), atol=1e-9)   # both sets on T1's system
    FG.plot_lightcurve(be, [p, p], d, insets=[(-1, 1)], path=tmp_path / "lc.png")
    FG.plot_geometry([p], ncols=1, mirror=True, path=tmp_path / "geo.png")
    assert (tmp_path / "lc.png").stat().st_size > 0 and (tmp_path / "geo.png").stat().st_size > 0
