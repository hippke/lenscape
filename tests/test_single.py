"""Single lens: analytic limits; 1L2S reduces to 1L1S."""
import numpy as np
from lenscape import single as S


def test_espl_centre_exact():
    for rho in (1e-3, 1e-2, 0.1):          # uniform disk centred on the lens: A = sqrt(rho^2 + 4) / rho
        assert abs(S.espl(0.0, rho) / (np.sqrt(rho * rho + 4) / rho) - 1) < 1e-8


def test_espl_point_source_limit():
    for u in (0.01, 0.1, 1.0):
        assert abs(S.espl(u, u * 1e-3) / S.pspl(u) - 1) < 1e-6


def test_espl_continuous_at_switch():
    for rho in (1e-3, 1e-2):               # quadrupole branch (u > 20 rho) vs integral just below
        a, b = S.espl(20.0 * rho * (1 + 1e-9), rho), S.espl(20.0 * rho * (1 - 1e-9), rho)
        assert abs(a / b - 1) < 2e-5


def test_1l2s_reduces_to_1l1s():
    t = np.linspace(740, 770, 500)
    a = S.lc_1l2s(756.0, 0.03, 760.0, 0.1, 30.0, 1e-3, 2e-3, 0.0, t, 0.0)
    b = S.lc_1l1s(756.0, 0.03, 30.0, 1e-3, t, 0.0)
    assert np.allclose(a, b, rtol=1e-14)


def test_espl_vs_independent_quadrature():
    # rho = 0.01; reference: lens-centred 2-D scipy quad (rel 1e-11), stored. Before the kink splits: 1.1e-4
    # (uniform, u = rho) and up to 1e-3 (linear LD with the lens inside the disk)
    for x, ld, ref in ((1.0, 0.0, 127.32819852045297), (0.3, 0.6, 210.22135664081918),
                       (0.9, 0.6, 144.69321323943223), (1.0, 0.6, 122.57103235134996),
                       (1.1, 0.6, 104.02744563751723)):
        assert abs(S.espl(x * 1e-2, 1e-2, ld) / ref - 1) < 1e-5
