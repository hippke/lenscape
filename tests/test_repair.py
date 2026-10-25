"""grid.repair_outliers on a synthetic map (no light curves): a stuck point is found, recalculated, and logged."""
import numpy as np
from lenscape import grid as G


def test_repair_finds_and_fixes_spike(monkeypatch):
    logs, logq = np.linspace(-0.1, 0.1, 5), np.linspace(-3.0, -2.0, 5)
    chi = 1000.0 + 50.0 * (logs[:, None] ** 2 + (logq[None, :] + 2.5) ** 2)
    true = chi.copy()
    chi[2, 3] += 40.0                                      # stuck fit
    pars = np.tile(np.array([756.0, 0.03, 30.0, 4.4, -3.0]), (5, 5, 1))
    monkeypatch.setattr(G.F, "heavy_starts", lambda ls, lq, data, cfg, warm: [np.zeros(5), np.ones(5)])
    monkeypatch.setattr(G.F, "heavy_explore",
                        lambda ls, lq, data, cfg, x0: (float(true[np.argmin(abs(logs - ls)), np.argmin(abs(logq - lq))])
                                                       + x0[0], np.array([756.0, 0.03, 30.0, ls, lq, 4.4, -3.0])))
    monkeypatch.setattr(G, "warmup", lambda cfg, data: None)
    msgs = []
    c2, p2, n = G.repair_outliers(chi, pars, logs, logq, {}, workers=2, log=msgs.append, passes=3)
    assert n == 1 and np.allclose(c2, true)
    assert any("recalculate outliers, pass 1: 1 outliers, 1 improved" in m for m in msgs)
    _, _, n = G.repair_outliers(true, pars, logs, logq, {}, workers=2, log=msgs.append)
    assert n == 0 and msgs[-1] == "recalculate outliers: no outliers found"
