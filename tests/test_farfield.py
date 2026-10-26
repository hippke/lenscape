"""Far-field root solve (nbmag._roots_far): exact where it accepts, falls back where it does not, deterministic.
RESEARCH_point_source_results.md."""
import os
import numpy as np
from numba import njit
from conftest import REF
from lenscape import nbmag

T1 = np.load(os.path.join(REF, "t1_mag.npz"))


@njit(cache=False)
def _far_accept(s_arr, q_arr, y1_arr, y2_arr):
    """Per point: did _roots_far accept, and the point-source magnification with / without it (fresh starts)."""
    n = s_arr.shape[0]
    ok = np.zeros(n, np.bool_)
    A_far = np.empty(n)
    A_reg = np.empty(n)
    c = np.empty(6, np.complex128)
    z = np.empty(5, np.complex128)
    for k in range(n):
        a, m1, m2 = nbmag._setup(s_arr[k], q_arr[k])
        y = complex(y1_arr[k], abs(y2_arr[k])) + a * m1
        nbmag._coefs(a, m1, m2, y, c)
        nbmag._init_roots(z, s_arr[k])
        ok[k] = nbmag._roots_far(a, m1, m2, y, c, z)
        nbmag._init_roots(z, s_arr[k])
        A_far[k] = nbmag._point(a, m1, m2, s_arr[k], q_arr[k], y1_arr[k], y2_arr[k], c, z, False, True)[0]
        nbmag._init_roots(z, s_arr[k])
        A_reg[k] = nbmag._point(a, m1, m2, s_arr[k], q_arr[k], y1_arr[k], y2_arr[k], c, z, False, False)[0]
    return ok, A_far, A_reg


def test_far_field_exact_on_T1():
    """T1 reference set: wherever the far-field solve accepts, the magnification equals the Aberth solution and the
    stored VBBL value; it accepts a substantial fraction of the set."""
    ps = np.load(os.path.join(REF, "t1_ps.npz"))["A"]
    ok, A_far, A_reg = _far_accept(T1["s"], T1["q"], T1["y1"], T1["y2"])
    assert ok.mean() > 0.5
    assert np.max(np.abs(A_far / A_reg - 1)) < 1e-9
    assert np.max(np.abs(A_far / ps - 1)) < 1e-9


def test_far_field_light_curve_agrees_and_is_deterministic():
    """Benchmark-like planetary light curve (resonant s, q = 4e-3), source sizes log rho = -2 and -4: lc_hybrid with
    and without the far-field solve agree to 1e-9 in A and exactly in the branch flags; two runs are bit-identical."""
    t = np.linspace(-40.0, 40.0, 3000)
    for lr in (-2.0, -4.0):
        A0, f0 = nbmag.lc_hybrid(0.002, -2.38, -0.026, 4.4, lr, 35.0, 0.0, t, 1e-2, 1e-3, False)
        A1, f1 = nbmag.lc_hybrid(0.002, -2.38, -0.026, 4.4, lr, 35.0, 0.0, t, 1e-2, 1e-3, True)
        A2, f2 = nbmag.lc_hybrid(0.002, -2.38, -0.026, 4.4, lr, 35.0, 0.0, t, 1e-2, 1e-3, True)
        m = np.isfinite(A0)
        assert np.array_equal(A1, A2, equal_nan=True) and np.array_equal(f1, f2)
        assert np.array_equal(f0, f1)
        assert np.max(np.abs(A1[m] / A0[m] - 1)) < 1e-9


def test_far_field_binary_falls_back():
    """q = 0.5, s = 1 (no planetary far field) and q > 1 (roles swapped in _setup): whatever the far-field solve
    accepts is a complete root set, so the magnifications equal the regular solve's."""
    rng = np.random.default_rng(3)
    n = 2000
    y1, y2 = rng.uniform(-1.5, 1.5, n), rng.uniform(-1.5, 1.5, n)
    for s, q in ((1.0, 0.5), (0.8, 3.0)):
        ok, A_far, A_reg = _far_accept(np.full(n, s), np.full(n, q), y1, y2)
        assert np.max(np.abs(A_far / A_reg - 1)) < 1e-9
