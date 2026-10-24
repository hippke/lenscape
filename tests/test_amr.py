"""Adaptive cell refinement (amr.run_contour) on a synthetic chi2 valley (no light curves)."""
import numpy as np
from lenscape import amr


def fake_box(box, cfg, warm):
    s_lo, s_hi, q_lo, q_hi = box
    ls = min(max(0.05 + 0.3 * (np.clip(-1.9, q_lo, q_hi) + 1.9), s_lo), s_hi)   # tilted narrow valley
    lq = min(max(-1.9, q_lo), q_hi)
    best = (np.inf, None)
    for q in np.linspace(q_lo, q_hi, 9):                    # cell minimum on a sub-grid
        s0 = 0.05 + 0.3 * (q + 1.9)
        s = min(max(s0, s_lo), s_hi)
        c = 1000.0 + 2e5 * (s - s0) ** 2 + 300.0 * (q + 1.9) ** 2
        if c < best[0]:
            best = (c, np.r_[756.5, 0.03, 30.0, 2.8, -3.0, s, q])
    return best


def test_relaxation_takes_better_neighbour_solution(monkeypatch):
    """A cell whose own fit missed the valley gets it from a neighbour's start (relaxation, it26)."""
    def flaky_box(box, cfg, warm):
        c, x = fake_box(box, cfg, warm)
        if warm is None and box[0] < 0.06 < box[1]:          # cold fits of the valley column miss the valley
            return c + 500.0, x
        return c, x
    monkeypatch.setattr(amr, "_fit_box", flaky_box)
    monkeypatch.setattr(amr.G, "warmup", lambda cfg, data: None)
    bands = np.array([15.0, 60.0, 135.0, 240.0])
    kw = dict(workers=2, log=lambda *a: None, min_ds=0.2, min_dq=0.5)    # first level only
    a = amr.run_contour((-0.5, 0.5), (-3.0, -1.0), (8, 8), 0, {}, None, bands, relax=False, **kw)
    b = amr.run_contour((-0.5, 0.5), (-3.0, -1.0), (8, 8), 0, {}, None, bands, relax=True, precheck=False, **kw)
    oa, ob = np.lexsort((a["logq"], a["logs"])), np.lexsort((b["logq"], b["logs"]))   # leaf order = finishing order
    a = {k: v[oa] for k, v in a.items() if isinstance(v, np.ndarray) and v.shape[:1] == oa.shape}
    b = {k: v[ob] for k, v in b.items() if isinstance(v, np.ndarray) and v.shape[:1] == ob.shape}
    assert np.allclose(a["logs"], b["logs"]) and np.allclose(a["logq"], b["logq"])
    col = (a["logs"] > 0) & (a["logs"] < 0.125)             # the column whose cold fits missed the valley
    assert np.all(b["chi2"][col] <= a["chi2"][col]) and np.max(a["chi2"][col] - b["chi2"][col]) > 400



def test_seed_grid_gives_first_level_warm_starts(monkeypatch):
    """Idea 60: with a coarse seed grid every first-level cell gets warm starts; the cold-miss column is repaired."""
    def flaky_box(box, cfg, warm):
        c, x = fake_box(box, cfg, warm)
        if not warm and box[0] < 0.06 < box[1] and box[1] - box[0] < 0.2:   # cold first-level fits miss the valley
            return c + 500.0, x
        return c, x
    monkeypatch.setattr(amr, "_fit_box", flaky_box)
    monkeypatch.setattr(amr.G, "warmup", lambda cfg, data: None)
    bands = np.array([15.0, 60.0, 135.0, 240.0])
    kw = dict(workers=2, log=lambda *a: None, min_ds=0.2, min_dq=0.5, relax=False)
    a = amr.run_contour((-0.5, 0.5), (-3.0, -1.0), (8, 8), 0, {}, None, bands, **kw)
    b = amr.run_contour((-0.5, 0.5), (-3.0, -1.0), (8, 8), 0, {}, None, bands, seed=4, **kw)
    assert b["chi2"].min() <= a["chi2"].min()
    assert b["nfit"] == a["nfit"] + 4                       # 2 x 2 seed boxes
    oa, ob = np.lexsort((a["logq"], a["logs"])), np.lexsort((b["logq"], b["logs"]))
    col = (a["logs"][oa] > 0) & (a["logs"][oa] < 0.125)
    assert np.max(a["chi2"][oa][col] - b["chi2"][ob][col]) > 400


def test_resume_from_given_first_level(monkeypatch):
    """start=: level 0 taken from an earlier run (no fits), refinement continues as if level 0 had been fitted."""
    monkeypatch.setattr(amr, "_fit_box", fake_box)
    monkeypatch.setattr(amr.G, "warmup", lambda cfg, data: None)
    bands = np.array([15.0, 60.0, 135.0, 240.0])
    kw = dict(workers=2, log=lambda *a: None, min_ds=0.03, min_dq=0.09, relax=False)
    a = amr.run_contour((-0.5, 0.5), (-3.0, -1.0), (8, 8), 3, {}, None, bands, **kw)
    ws, wq = 1.0 / 8, 2.0 / 8
    start = {(i, j): fake_box((-0.5 + i * ws, -0.5 + (i + 1) * ws, -3.0 + j * wq, -3.0 + (j + 1) * wq), {}, None)
             for i in range(8) for j in range(8)}
    b = amr.run_contour((-0.5, 0.5), (-3.0, -1.0), (8, 8), 3, {}, None, bands, start=start, **kw)
    assert b["nfit"] == a["nfit"] - 64                      # the 64 first-level fits are skipped
    oa, ob = np.lexsort((a["logq"], a["logs"])), np.lexsort((b["logq"], b["logs"]))
    assert np.allclose(a["logs"][oa], b["logs"][ob]) and np.allclose(a["chi2"][oa], b["chi2"][ob])
    assert b["level"].max() >= 1
