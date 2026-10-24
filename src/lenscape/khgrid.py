"""The (k, h) grid of Hall, Yee, Shin, Yang & Zhang (2024, PASP 136, 094401) for searches over binary-lens
separation s and mass ratio q.

k is the logarithm of the width of the central caustic, k = log10[4 q / (s - 1/s)^2], held at its value for
s = s_ref (log s_ref = 0.03) inside |log s| < log s_ref, where k = log q + log10[4 / (s_ref - 1/s_ref)^2].
h runs along a contour of constant k: h = m log s inside |log s| < log s_ref (m = 35); outside,
h = +-(m log s_ref + distance in (log s, log q) from the point of the same k at s = s_ref), + on the wide side,
- on the close side (k is symmetric in s <-> 1/s). An even grid in (k, h) places the points along the curves of
equal central-caustic size. The paper gives no inverse; to_sq solves it numerically.
"""
import numpy as np
from .optimize import bisect

LS_REF = 0.03
M = 35.0
C = M * LS_REF
_S = 10 ** LS_REF
K_OFF = np.log10(4.0 / (_S - 1.0 / _S) ** 2)          # k = log q + K_OFF inside the transition region (2.32)


def _logq_on_contour(ls, k):
    """log q on the contour of constant k at log s = ls (|ls| >= LS_REF)."""
    s = 10.0 ** abs(ls)
    return k + 2.0 * np.log10(s - 1.0 / s) - np.log10(4.0)


def to_kh(ls, lq):
    """(log s, log q) -> (k, h)."""
    ls, lq = float(ls), float(lq)
    if abs(ls) < LS_REF:
        return lq + K_OFF, M * ls
    s = 10.0 ** abs(ls)
    k = np.log10(4.0 * 10.0 ** lq / (s - 1.0 / s) ** 2)
    lq_ref = k - K_OFF
    d = np.hypot(abs(ls) - LS_REF, lq - lq_ref)
    return k, np.sign(ls) * (d + C)


def to_sq(k, h, ls_max=3.0):
    """(k, h) -> (log s, log q); None if the contour ends before the distance |h| - C is reached."""
    k, h = float(k), float(h)
    if abs(h) < C:
        return h / M, k - K_OFF
    d = abs(h) - C
    lq_ref = k - K_OFF
    f = lambda x: np.hypot(x - LS_REF, _logq_on_contour(x, k) - lq_ref) - d
    if f(ls_max) < 0:
        return None
    x = bisect(f, LS_REF, ls_max, xtol=1e-10)
    return np.sign(h) * x, _logq_on_contour(x, k)


def grid(n_k, n_h, ls_range=(-1.5, 1.5), lq_range=(-6.0, 0.0)):
    """Points of an even (k, h) grid (n_k x n_h before cuts) that fall inside the (log s, log q) box: array of
    (log s, log q, k, h). k and h ranges are chosen to cover the box."""
    corners = [to_kh(a, b) for a in np.linspace(*ls_range, 61) for b in np.linspace(*lq_range, 61)]
    ks, hs = np.array([c[0] for c in corners]), np.array([c[1] for c in corners])
    out = []
    for k in np.linspace(ks.min(), ks.max(), n_k):
        for h in np.linspace(hs.min(), hs.max(), n_h):
            sq = to_sq(k, h)
            if sq is not None and ls_range[0] <= sq[0] <= ls_range[1] and lq_range[0] <= sq[1] <= lq_range[1]:
                out.append((sq[0], sq[1], k, h))
    return np.array(out)
