"""Binary-lens magnification vs stored VBBL references (no VBBL needed at test time)."""
import os
import numpy as np
import pytest
from conftest import REF
from lenscape import nbmag, model as M, config as C, fit as F

T1 = np.load(os.path.join(REF, "t1_mag.npz"))


def test_point_source_matches_reference():
    ps = np.load(os.path.join(REF, "t1_ps.npz"))["A"]
    A = nbmag.mags_point(T1["s"], T1["q"], T1["y1"], T1["y2"])
    assert np.max(np.abs(A / ps - 1)) < 1e-9


@pytest.mark.parametrize("which,limit", [("grid_backend", 2e-3), ("refine_backend", 5e-4)])
def test_finite_source_T1(which, limit):
    be = F.make_backend(C.CURRENT[which])
    A = be.mag(T1["s"], T1["q"], T1["y1"], T1["y2"], T1["rho"])
    assert np.max(np.abs(A / T1["A"] - 1)) < limit
    assert be.n_failed == 0


@pytest.mark.parametrize("u", [0.3, 0.6])
@pytest.mark.parametrize("kw,lim_t1,lim_stress", [(dict(tol=1e-2, hexeps=1e-3, ctol=0.1), 2e-3, 2e-4),
                                                   (dict(tol=1e-3, hexeps=1e-4, ctol=0.01), 6e-4, 2e-4)])
def test_limb_darkening(u, kw, lim_t1, lim_stress):
    R = np.load(os.path.join(REF, "ld_ref.npz"))
    rows, n1, ref = R["rows"], int(R["n_t1"]), R[f"A_u{u}"]
    A = M.NumbaHybrid(ld=u, **kw).mag(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4])
    e = np.abs(A / ref - 1)
    assert e[:n1].max() < lim_t1 and e[n1:].max() < lim_stress


def test_ld_zero_is_uniform():
    a = M.NumbaHybrid(ld=0.0).mag(T1["s"], T1["q"], T1["y1"], T1["y2"], T1["rho"])
    b = M.NumbaHybrid().mag(T1["s"], T1["q"], T1["y1"], T1["y2"], T1["rho"])
    assert np.array_equal(a, b)


def test_central_caustic_near_axis_regression():
    """it28: grid point (19, 18) of the box-seeded lattice run (near-axis source at the central caustic).
    (1) fastmath 'nsz' (+ contract + reassoc) gave 33581; (2) an early return in _interval left the image matching
    uninitialised -> segfault; (3) #62: near the critical curve det J had the wrong sign for one image, the parities
    summed to -3, image matching failed and intervals were dropped: 2098 for reltol <= 0.01. Reference: brute-force
    ray shooting 2891.6-2891.8 (spacing 7.8e-6 ... 2e-6); VBBinaryLensing gives 2906.3 here (0.5 % high)."""
    from lenscape import contour
    s, q, y1, y2, rho = 0.35540590371810626, 0.0047196853200182284, -0.0008628772457845972, 5.86736869299754e-05, \
        0.00011903032408317988
    for reltol, n0, maxs in ((0.1, 8, 2048), (0.1 / 3, 32, 4096), (0.01, 8, 8192), (1e-3, 8, 8192)):
        A, status, _ = contour.mag_disk(s, q, y1, y2, rho, reltol, n0, maxs)
        assert status >= 0 and abs(A / 2891.7 - 1) < 2e-3


def test_contour_budget_retry_and_grid_cap():
    """Lens at the source limb (s = 1, q = 3.16e-5, |y| = 0.98 rho): with 2048 samples the ctol 1e-3 refinement stopped
    half-way (A = 385.3); the retry with RETRY_MAXS samples converges to 339.218, which every tolerance reaches with
    enough samples. The grid tolerance converges there only with about 2000 samples (1024: A = 542, 1536: 340.1), so
    the default budget of 2048 must not be cut."""
    from lenscape import contour
    s, q, rho = 1.0, 10 ** -4.5, 0.003938708612378672
    y1, y2 = 0.0009357064347995461, 0.003756647196279318
    a_ref, st_ref, _ = contour.mag_disk(s, q, y1, y2, rho, 1e-3, maxs=32768)
    assert st_ref == 0 and abs(a_ref / 339.218 - 1) < 1e-5
    a, st, ns = contour.mag_disk_ld(s, q, y1, y2, rho, 1e-3, 0.0)          # 2048 samples, then the retry
    assert st >= 0 and abs(a / a_ref - 1) < 1e-4 and ns > 2048
    a, st, ns = contour.mag_disk_ld(s, q, y1, y2, rho, 0.1, 0.0)           # grid tolerance, 2048 samples
    assert st >= 0 and abs(a / a_ref - 1) < 1e-3
