"""(k, h) grid of Hall et al. (2024): exact round trip and the linear form near s = 1."""
import numpy as np
from lenscape import khgrid as KH


def test_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(500):
        ls, lq = rng.uniform(-1.5, 1.5), rng.uniform(-6, 0)
        assert np.allclose(KH.to_sq(*KH.to_kh(ls, lq)), (ls, lq), atol=1e-8)


def test_linear_near_resonance_and_grid_box():
    k, h = KH.to_kh(0.01, -2.4)
    assert np.isclose(k, -2.4 + KH.K_OFF) and np.isclose(h, 0.35)
    g = KH.grid(10, 12)
    assert len(g) > 0 and np.all(np.abs(g[:, 0]) <= 1.5) and np.all((g[:, 1] >= -6) & (g[:, 1] <= 0))
