"""Data loading, start-vector estimate, fits and a small map on a synthetic planetary event (no external data)."""
import os, subprocess, sys
import numpy as np
import pytest
from conftest import ROOT
from lenscape import data as D, fit as F, config as C, model as M, maps

TRUE = np.array([0.0, 0.08, 25.0, np.log10(1.1), np.log10(5e-3), 4.2, -2.8])   # t0, u0, tE, log s, log q, alpha, log rho


def synthetic(seed=1, n=250, efac=1.0):
    """Two telescopes with different source / blend fluxes, nightly sampling plus a dense anomaly."""
    rng = np.random.default_rng(seed)
    be = F.make_backend(C.REFINE_BACKEND)
    t = np.sort(np.r_[rng.uniform(-60, 60, n), rng.uniform(-6, 6, n)])
    idx = (np.arange(len(t)) % 2).astype(np.int64)
    A = be.lc(TRUE, t)
    fs, fb = np.array([1000.0, 600.0]), np.array([300.0, -100.0])
    e = np.where(idx == 0, 20.0, 15.0) * efac
    f = fs[idx] * A + fb[idx] + rng.normal(0, e)
    return D.Data(t, f, e, idx, ["T1", "T2"])


def test_data_container_and_loaders(tmp_path):
    d = synthetic()
    sets = {n: (d.t[d.idx == k], d.f[d.idx == k], d.e[d.idx == k]) for k, n in enumerate(d.names)}
    d2 = D.Data.from_sets(sets)
    assert d2.names == ["T1", "T2"] and len(d2) == len(d)
    assert len(d.window(-6, 6)) < len(d) and np.allclose(d.scale_errors({"T1": 2, "T2": 1}).e[d.idx == 0], 40.0)
    for k, n in enumerate(d.names):                       # pySIS layout: HJD, flux, flux error, mag, mag error
        m = d.idx == k
        np.savetxt(tmp_path / f"{n}_I.pysis", np.c_[d.t[m] + 2459000.0, d.f[m], d.e[m], np.zeros(m.sum()),
                                                    np.zeros(m.sum())], header="HJD flux err mag merr")
    d3 = D.read_pysis(str(tmp_path / "*_I.pysis"), t_offset=2459000.0)
    assert d3.names == ["T1", "T2"] and np.allclose(np.sort(d3.t), np.sort(d.t))


def test_seed_and_fit_recover_truth():
    d = synthetic()
    assert abs(F.seed_from_data(d)["t0"] - TRUE[0]) < 2       # brightest point: near the peak
    # strongly perturbed start (u0 -25 %, tE +12 %, rho x2, t0 +0.2 d); alpha is found by the screen
    cfg = dict(C.DEFAULT, seed=dict(t0=0.2, u0=0.06, tE=28.0, logrho=-2.5))
    c, x = F.fit_point(TRUE[3], TRUE[4], d, cfg)
    assert c < len(d) + 5 * np.sqrt(2 * len(d))           # chi2 at the true (s, q) consistent with the noise
    assert abs(x[2] - TRUE[2]) / TRUE[2] < 0.1


def test_missing_seed_is_reported():
    with pytest.raises(ValueError, match="start vector"):
        F.fit_point(TRUE[3], TRUE[4], synthetic(), dict(C.DEFAULT, seed=None))


SCRIPT = r"""
import os, sys; os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.modules["VBBinaryLensing"] = None; sys.modules["VBMicrolensing"] = None   # must work without them
sys.path.insert(0, %r); sys.path.insert(0, %r)
import numpy as np
from test_data_fit_maps import synthetic, TRUE
from lenscape import fit as F, config as C, model as M
d = synthetic()
if sys.argv[1] == "noisy":                       # unrelated work first: results must not depend on history
    be = F.make_backend(C.GRID_BACKEND)
    for k in range(30):
        M.chi2(be, TRUE + 0.01 * k, d)
c, x = F.fit_point(TRUE[3] + 0.05, TRUE[4], d, dict(C.DEFAULT, seed=dict(t0=0.2, u0=0.06, tE=28.0, logrho=-2.5)))
print(repr(float(c)))
""" % (os.path.join(ROOT, "src"), os.path.join(ROOT, "tests"))


def test_no_vbbl_and_determinism():
    out = [subprocess.run([sys.executable, "-c", SCRIPT, m], capture_output=True, text=True, check=True).stdout
           for m in ("clean", "noisy")]
    assert out[0] == out[1]


@pytest.mark.slow
def test_small_map_and_refinement():
    """3 x 3 lattice around the truth: minimum at the true point; refinement adds finer cells and never worsens.
    (~1 min on 2 cores; deselect with -m "not slow")"""
    d = synthetic(n=60, efac=4.0)
    logs = TRUE[3] + np.array([-0.03, 0.0, 0.03])
    logq = TRUE[4] + np.array([-0.15, 0.0, 0.15])
    r = maps.lattice(d, logs, logq, best=TRUE, workers=2, log=lambda *a: None)
    i, j = np.unravel_index(np.nanargmin(r["chi2"]), r["chi2"].shape)
    assert (i, j) == (1, 1) and r["chi2"][1, 1] < len(d) + 5 * np.sqrt(2 * len(d))
    a = maps.refine(r, d, levels=1, best=TRUE, workers=2, log=lambda *a: None)
    assert a["chi2"].min() <= r["chi2_ref"] + 1e-9 and a["level"].max() == 1
