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


def test_fit_full_and_error_factors():
    d = synthetic()
    c, p = F.fit_full(d, TRUE + np.array([0.05, 0.005, 1.0, 0.002, 0.02, 0.01, 0.1]), rounds=2)
    assert c < len(d) + 5 * np.sqrt(2 * len(d)) and abs(p[2] - TRUE[2]) < 1.0
    k = F.error_factors(d.scale_errors({"T1": 0.5, "T2": 1.0}), p)
    assert 1.6 < k["T1"] < 2.4 and 0.8 < k["T2"] < 1.2                    # errors halved -> factor ~2


@pytest.mark.slow
def test_densify_and_alpha_minima():
    """A 2x finer lattice from a coarse one (no new cell fits) and the trajectory-angle minima of a lattice."""
    d = synthetic(n=60, efac=4.0)
    coarse = maps.lattice(d, TRUE[3] + np.array([-0.03, 0.03]), TRUE[4] + np.array([-0.15, 0.15]), best=TRUE,
                          workers=2, log=lambda *a: None)
    fine = maps.densify([coarse], d, TRUE[3] + np.array([-0.03, 0.0, 0.03]), TRUE[4] + np.array([-0.15, 0.0, 0.15]),
                        best=TRUE, workers=2, log=lambda *a: None)
    assert np.unravel_index(np.nanargmin(fine["chi2"]), (3, 3)) == (1, 1)
    m = maps.alpha_minima(fine, d, best=TRUE, workers=2, n_alpha=8, log=lambda *a: None)
    at_true = m[(np.abs(m[:, 0] - TRUE[3]) < 1e-9) & (np.abs(m[:, 1] - TRUE[4]) < 1e-9)]
    assert len(at_true) >= 1 and np.min(at_true[:, 2]) <= fine["chi2"][1, 1] + 1.0


@pytest.mark.slow
def test_alpha_profile_has_minimum_at_truth():
    """(log s, alpha) profile map: a value everywhere, the minimum at the true (log s, alpha)."""
    d = synthetic(n=60, efac=4.0)
    lat = maps.lattice(d, TRUE[3] + np.array([-0.03, 0.03]), TRUE[4] + np.array([-0.15, 0.15]), best=TRUE,
                       workers=2, log=lambda *a: None)
    alphas = TRUE[5] + np.array([-0.1, 0.0, 0.1])
    r = maps.alpha_profile([lat], d, TRUE[3] + np.array([-0.02, 0.0, 0.02]), alphas, best=TRUE, workers=2,
                           log=lambda *a: None)
    assert np.all(np.isfinite(r["chi2"]))
    assert np.unravel_index(np.nanargmin(r["chi2"]), (3, 3)) == (1, 1)
    assert r["chi2"][1, 1] < len(d) + 5 * np.sqrt(2 * len(d))


CROSS = np.array([0.0, -0.025, 35.0, 0.002, -2.4, 4.41, -3.3])      # a resonant caustic crossing
CROSS_T = (-2.4423, 2.1252)                                          # its caustic entry and exit times


def test_caustic_starts_touch_the_caustic_at_the_given_times():
    """Every constructed trajectory is on the caustic at t_in and at t_out (it may cross there or touch it)."""
    from lenscape import caustics as K
    S = np.array(F.caustic_starts(CROSS[3], CROSS[4], CROSS_T, -3.3, 12))
    assert len(S) == 12 * 11
    caus = np.concatenate(K.caustics(10 ** CROSS[3], 10 ** CROSS[4], 20000))
    for x in S:
        p = np.r_[x[:3], CROSS[3], CROSS[4], x[3], x[4]]
        for z in K.trajectory(p, CROSS_T):
            assert np.min(np.abs(caus - z)) < 2e-4


def test_caustic_starts_recover_a_crossing_from_a_rough_start():
    """From a rough start the 20-angle screen misses the caustic crossing; caustic-crossing starts find it."""
    rng = np.random.default_rng(2)
    be = F.make_backend(C.REFINE_BACKEND)
    t = np.sort(np.r_[rng.uniform(-80, 80, 150), rng.uniform(-4, 4, 150)])
    e = np.full(len(t), 30.0)
    d = D.Data(t, 1000 * be.lc(CROSS, t) + 300 + rng.normal(0, e), e, np.zeros(len(t), np.int64), ["T"])
    seed = dict(t0=0.3, u0=0.06, tE=25.0, logrho=-3.0)
    c0, _ = F.fit_point(CROSS[3], CROSS[4], d, dict(C.DEFAULT, seed=seed))
    c1, x1 = F.fit_point(CROSS[3], CROSS[4], d, dict(C.DEFAULT, seed=seed, caustic_crossings=CROSS_T,
                                                    caustic_only=True))
    assert c1 < len(d) + 5 * np.sqrt(2 * len(d)) < c0
    assert abs(x1[2] - CROSS[2]) < 0.5


def test_residual_vectors_reproduce_chi2():
    """model.chi2_residuals: sum of squares = chi2 (unbinned), = chi2 minus a parameter-independent constant (binned)."""
    d = synthetic(n=120)
    be = F.make_backend(C.GRID_BACKEND)
    b = D.bin_wings(d, TRUE[0], TRUE[2])
    off = []
    for dp in (np.zeros(7), np.array([0.3, 0.01, -1.0, 0.002, 0.05, 0.02, 0.1])):
        p = TRUE + dp
        r = M.chi2_residuals(be, p, d)
        assert abs(r @ r - M.chi2(be, p, d)) < 1e-8 * M.chi2(be, p, d)
        rb = M.chi2_residuals(be, p, b)
        off.append(M.chi2(be, p, b) - rb @ rb)
    assert abs(off[0] - off[1]) < 1e-6 * abs(off[0]) + 1e-6


def test_nm_lm_local_fit_and_densify():
    """optimizer="nm_lm" (short Nelder-Mead, then Levenberg-Marquardt) recovers the truth from a perturbed start and
    gives the same densified map minimum as Nelder-Mead."""
    d = synthetic(n=60, efac=4.0)
    cfg = dict(C.DEFAULT, optimizer="nm_lm")
    be = F.make_backend(C.GRID_BACKEND)
    co = F.Coords("teff")
    x0 = np.r_[TRUE[:3], TRUE[5:]] + np.array([0.2, 0.005, 1.0, 0.01, 0.1])
    c, y = F.local_fit(be, d, TRUE[3], TRUE[4], co.from_x(x0), cfg, 0.3, 1000)
    assert c <= M.chi2(be, TRUE, d) + 1.0
    coarse = maps.lattice(d, TRUE[3] + np.array([-0.03, 0.03]), TRUE[4] + np.array([-0.15, 0.15]), best=TRUE,
                          workers=2, log=lambda *a: None)
    fine = maps.densify([coarse], d, TRUE[3] + np.array([-0.03, 0.0, 0.03]), TRUE[4] + np.array([-0.15, 0.0, 0.15]),
                        best=TRUE, cfg=cfg, workers=2, log=lambda *a: None)
    assert np.unravel_index(np.nanargmin(fine["chi2"]), (3, 3)) == (1, 1)
    fine2 = maps.densify([coarse], d, TRUE[3] + np.array([-0.03, 0.0, 0.03]), TRUE[4] + np.array([-0.15, 0.0, 0.15]),
                         best=TRUE, cfg=dict(cfg, interp_start=True), workers=2, log=lambda *a: None)
    assert np.unravel_index(np.nanargmin(fine2["chi2"]), (3, 3)) == (1, 1)
    assert np.nanmin(fine2["chi2"]) <= np.nanmin(fine["chi2"]) + 1.0


def test_interp_start_is_the_mean_of_the_nearest():
    known = [(0.0, 0.0, 10.0, np.array([0.0, 0.1, 20.0, 6.2, -3.0])), (1.0, 0.0, 11.0, np.array([1.0, 0.1, 22.0, 0.1, -3.0])),
             (5.0, 5.0, 9.0, np.array([0.0, -0.1, 20.0, 1.0, -3.0]))]
    dist = np.array([1.0, 1.0, 7.0])
    x = maps._interp_start(known, dist)
    assert abs(x[0] - 0.5) < 1e-12 and abs(np.mod(x[3], 2 * np.pi) - np.mod(0.5 * (6.2 + 0.1 + 2 * np.pi), 2 * np.pi)) < 1e-9
    assert maps._interp_start(known, np.array([3.0, 3.0, 7.0])) is None              # too far


def test_near_mask():
    from lenscape.grid import _near
    m = np.zeros((4, 5), bool)
    m[0, 0] = True
    assert _near(m).sum() == 4 and _near(m)[1, 1] and not _near(m)[0, 4]
    w = _near(m, wrap_j=True)
    assert w.sum() == 6 and w[0, 4] and w[1, 4] and not w[3, 0]


def _logged():
    lines = []
    return lines, lambda *a: lines.append(" ".join(map(str, a)))


def test_relax_changed_only_is_identical():
    """Neighbour refinement checking only cells next to a change gives exactly the result of checking all cells,
    with fewer checks: a 6 x 6 map where only one corner has the good solution, so it spreads over several sweeps."""
    from lenscape import grid as G
    d = synthetic(n=60, efac=4.0)
    cfg = dict(C.DEFAULT, seed=dict(t0=0.0, u0=0.08, tE=25.0, logrho=-2.8))
    be = F.make_backend(C.GRID_BACKEND)
    logs = TRUE[3] + np.linspace(-0.02, 0.02, 6)
    logq = TRUE[4] + np.linspace(-0.1, 0.1, 6)
    good = np.r_[TRUE[:3], TRUE[5:]]
    bad = good + np.array([1.5, 0.02, 6.0, 0.15, 0.3])
    pars = np.tile(bad, (6, 6, 1))
    pars[0, 0] = good
    chi = np.array([[M.chi2(be, F.full_p(pars[i, j], ls, lq), d) for j, lq in enumerate(logq)]
                    for i, ls in enumerate(logs)])
    out = {}
    for flag in (False, True):
        lines, log = _logged()
        c, p = G.relax(chi, pars, logs, logq, cfg, workers=2, data=d, log=log, max_sweeps=12, changed_only=flag)
        nchk = sum(int(l.split(": ")[1].split()[0]) for l in lines if "cells checked" in l)
        out[flag] = (c, p, nchk, len([l for l in lines if "cells checked" in l]))
    assert out[True][3] >= 3                                        # several sweeps
    assert np.array_equal(out[True][0], out[False][0]) and np.array_equal(out[True][1], out[False][1])
    assert out[True][2] < out[False][2]


def test_alpha_profile_changed_only_is_identical():
    d = synthetic(n=60, efac=4.0)
    lat = maps.lattice(d, TRUE[3] + np.array([-0.03, 0.03]), TRUE[4] + np.array([-0.15, 0.15]), best=TRUE,
                       workers=2, log=lambda *a: None)
    logs, alphas = TRUE[3] + np.linspace(-0.03, 0.03, 5), TRUE[5] + np.linspace(-0.3, 0.3, 7)
    r = {flag: maps.alpha_profile([lat], d, logs, alphas, best=TRUE, workers=2, log=lambda *a: None,
                                  changed_only=flag) for flag in (False, True)}
    assert np.array_equal(r[True]["chi2"], r[False]["chi2"], equal_nan=True)
    assert np.array_equal(r[True]["pars"], r[False]["pars"], equal_nan=True)


def test_alpha_maps_with_nm_lm():
    """alpha_profile and alpha_minima with optimizer="nm_lm": minimum at the truth, as with Nelder-Mead."""
    d = synthetic(n=60, efac=4.0)
    cfg = dict(C.DEFAULT, optimizer="nm_lm")
    lat = maps.lattice(d, TRUE[3] + np.array([-0.03, 0.03]), TRUE[4] + np.array([-0.15, 0.15]), best=TRUE,
                       workers=2, log=lambda *a: None)
    alphas = TRUE[5] + np.array([-0.1, 0.0, 0.1])
    r = maps.alpha_profile([lat], d, TRUE[3] + np.array([-0.02, 0.0, 0.02]), alphas, best=TRUE, cfg=cfg, workers=2,
                           log=lambda *a: None)
    assert np.all(np.isfinite(r["chi2"])) and np.unravel_index(np.nanargmin(r["chi2"]), (3, 3)) == (1, 1)
    m = maps.alpha_minima(lat, d, best=TRUE, cfg=cfg, workers=2, n_alpha=8, log=lambda *a: None)
    assert np.min(m[:, 2]) <= np.nanmin(lat["chi2"]) + 1.0


def test_alpha_profile_refine_spreads_the_other_u0_sign():
    """alpha_profile_refine: a better solution with the other sign of u0 (counted at alpha + pi) spreads to its
    neighbours, and the map never gets worse."""
    d = synthetic(n=60, efac=4.0)
    lat = maps.lattice(d, TRUE[3] + np.array([-0.03, 0.03]), TRUE[4] + np.array([-0.15, 0.15]), best=TRUE,
                       workers=2, log=lambda *a: None)
    logs, alphas = TRUE[3] + np.linspace(-0.02, 0.02, 3), TRUE[5] + np.linspace(-0.1, 0.1, 3)
    r = maps.alpha_profile([lat], d, logs, alphas, best=TRUE, workers=2, log=lambda *a: None)
    # spoil all points but the centre by vectors with the other sign of u0, far from any solution
    bad = r["pars"].copy()
    chi = r["chi2"].copy()
    be = F.make_backend(C.GRID_BACKEND)
    for i in range(3):
        for j in range(3):
            if (i, j) != (1, 1):
                p = bad[i, j].copy()
                p[1], p[5], p[0] = -p[1], p[5] + np.pi, p[0] + 2.0
                bad[i, j] = p
                chi[i, j] = M.chi2(be, p, D.bin_wings(d, TRUE[0], TRUE[2]))
    out = maps.alpha_profile_refine(dict(r, chi2=chi, pars=bad), d, best=TRUE, workers=2, log=lambda *a: None)
    assert np.all(out["chi2"] <= chi + 1e-9)
    spoiled = np.ones((3, 3), bool)
    spoiled[1, 1] = False
    assert np.all(out["chi2"][spoiled] < chi[spoiled] - 10.0)            # recovered from the other u0 sign ...
    assert np.all(out["chi2"] <= r["chi2"] + 5.0)                         # ... to about the original values


def test_offset_partner_is_its_own_inverse():
    for p in ([0, -0.0247, 72.5, 0.0256, -2.48, 4.59, -3.1], TRUE, [0, -0.02, 30, -0.1, -3, 4.5, -3]):
        p = np.asarray(p, float)
        q = F.offset_partner(p)
        assert abs(q[3] - p[3]) > 0.01 and np.allclose(F.offset_partner(q), p, atol=1e-9)
    # KMT-2024-BLG-1209 (Han et al. 2025): inner solution log s 0.0256 -> outer 0.0126 below 0, predicted -0.0134
    assert abs(F.offset_partner([0, -0.0247, 72.5, 0.0256, -2.48, 4.59, -3.1])[3] + 0.0134) < 5e-4


def test_box_fit_with_lm_refinement():
    """fit_point_box with cfg box_lm (LM in log s, log q clipped to the cell): finds the truth inside its cell."""
    d = synthetic(n=60, efac=4.0)
    cfg = dict(C.DEFAULT, seed=dict(t0=0.0, u0=0.08, tE=25.0, logrho=-2.8), box_lm=True)
    box = (TRUE[3] - 0.01, TRUE[3] + 0.01, TRUE[4] - 0.05, TRUE[4] + 0.05)
    c, x = F.fit_point_box(box, d, cfg)
    be = F.make_backend(C.GRID_BACKEND)
    assert c <= M.chi2(be, TRUE, d) + 1.0
    assert box[0] <= x[5] <= box[1] and box[2] <= x[6] <= box[3]


def test_alpha_profile_keeps_the_convention_of_a_previous_profile():
    """A previous profile whose best point is a mirror solution (other u0 sign) must not flip the convention."""
    from lenscape.maps import _profile_u0_ref
    d = synthetic(n=60, efac=4.0)
    lat = maps.lattice(d, TRUE[3] + np.array([-0.03, 0.03]), TRUE[4] + np.array([-0.15, 0.15]), best=TRUE,
                       workers=2, log=lambda *a: None)
    logs, alphas = TRUE[3] + np.array([-0.02, 0.0, 0.02]), TRUE[5] + np.array([-0.1, 0.0, 0.1])
    r = maps.alpha_profile([lat], d, logs, alphas, best=TRUE, workers=2, log=lambda *a: None)
    # mirror the best point: (u0, alpha) -> (-u0, -alpha) is the same light curve, slightly "better"
    k = np.unravel_index(np.nanargmin(r["chi2"]), r["chi2"].shape)
    m = dict(r, pars=r["pars"].copy(), chi2=r["chi2"].copy())
    m.pop("u0_ref")
    m["pars"][k][1] *= -1
    m["pars"][k][5] = -m["pars"][k][5]
    m["chi2"][k] -= 1e-6
    assert _profile_u0_ref(m) == np.sign(r["u0_ref"])
    r2 = maps.alpha_profile([m], d, logs, alphas, best=TRUE, workers=2, log=lambda *a: None)
    assert r2["u0_ref"] == np.sign(r["u0_ref"]) or np.sign(r2["u0_ref"]) == np.sign(r["u0_ref"])
    assert np.nanmax(r2["chi2"] - r["chi2"]) < 1.0


def test_rough_mask():
    from lenscape.grid import _rough
    c = np.add.outer(np.arange(6.0), np.arange(5.0))            # smooth plane: no cell is rough
    assert not _rough(c, 0.2).any()
    c[2, 3] += 1.0
    m = _rough(c, 0.2)
    assert m[2, 3] and m.sum() == 1
    w = np.add.outer(np.arange(4.0), np.zeros(6))
    w[1, 0] += 1.0
    assert _rough(w, 0.2, wrap_j=True)[1, 0]


def test_forced_refit_of_rough_cells():
    """grid._relax_rows with a force mask refits a cell from all neighbours' solutions even when the one-chi2
    pre-check (here disabled by a huge margin) would not trigger; relax(smooth=...) never makes a cell worse."""
    from lenscape import grid as G
    d = synthetic(n=60, efac=4.0)
    cfg = dict(C.DEFAULT, seed=dict(t0=0.0, u0=0.08, tE=25.0, logrho=-2.8))
    be = F.make_backend(C.GRID_BACKEND)
    logs = TRUE[3] + np.linspace(-0.01, 0.01, 3)
    logq = TRUE[4] + np.linspace(-0.05, 0.05, 3)
    good = np.r_[TRUE[:3], TRUE[5:]]
    pars = np.tile(good, (3, 3, 1))
    chi = np.array([[M.chi2(be, F.full_p(good, ls, lq), d) for lq in logq] for ls in logs])
    spoil = good + np.array([0.3, 0.0, 0.5, 0.01, 0.0])
    pars[1, 1] = spoil
    chi[1, 1] = M.chi2(be, F.full_p(spoil, logs[1], logq[1]), d)
    G._DATA = d
    force = np.zeros((3, 3), bool)
    force[1, 1] = True
    out = G._relax_rows([1], chi, pars, logs, logq, cfg, 1e9, None, [1], force)
    assert out and out[0][2] < chi[1, 1] - 0.01
    c2, _ = G.relax(chi, pars, logs, logq, cfg, workers=2, data=d, smooth=0.2, log=lambda *a: None)
    assert np.all(c2 <= chi + 1e-9) and c2[1, 1] < chi[1, 1] - 0.01
