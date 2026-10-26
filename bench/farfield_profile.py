"""Micro-profile of the point-source solve on the benchmark light curves (in-numba timing, ns per point):
_coefs; _roots_newton from the previous point's roots (the kernel's path); _roots (Aberth) from the previous point's
roots; _roots_far from scratch; _point as a whole with FARFIELD on/off (needs a fresh process per setting:
python farfield_profile.py [off|on])."""
import sys, time
import numpy as np
from numba import njit
import farfield_common as FC
from lenscape import nbmag
from lenscape.nbmag import _setup, _coefs, _roots, _roots_newton, _roots_far, _point, _init_roots, FM

mode = sys.argv[1] if len(sys.argv) > 1 else "off"
nbmag.FARFIELD = mode == "on"
import importlib
# module constants are frozen at compile time: force recompilation with the chosen flag by reloading
importlib.reload(nbmag)
nbmag.FARFIELD = mode == "on"
from lenscape.nbmag import _setup, _coefs, _roots, _roots_newton, _roots_far, _point, _init_roots

data, best = FC.load_data()
P, cells, _ = FC.vectors()


@njit(cache=False, error_model='numpy', fastmath=FM)
def run(s, q, y1, y2, which, reps):
    """which: 0 coefs only; 1 coefs + roots_newton (warm); 2 coefs + roots (warm); 3 coefs + roots_far; 4 _point.
    Returns (sum of something to defeat DCE, far acceptance count)."""
    a, m1, m2 = _setup(s, q)
    n = y1.shape[0]
    c = np.empty(6, np.complex128)
    z = np.empty(5, np.complex128)
    acc = 0.0
    nfar = 0
    for r in range(reps):
        _init_roots(z, s)
        for k in range(n):
            y = complex(y1[k], abs(y2[k])) + a * m1
            if which == 4:
                acc += _point(a, m1, m2, s, q, y1[k], y2[k], c, z, True)[0]
                continue
            _coefs(a, m1, m2, y, c)
            if which == 1:
                _roots_newton(c, z)
            elif which == 2:
                _roots(c, z)
            elif which == 3:
                if _roots_far(a, m1, m2, y, c, z):
                    nfar += 1
            acc += c[0].real + z[0].real
    return acc, nfar


names = ["coefs", "coefs+roots_newton(warm)", "coefs+aberth(warm)", "coefs+roots_far", "_point(FARFIELD=%s)" % mode]
tot = np.zeros(5)
for p in P[:8]:
    t0, u0, tE, ls, lq, al, lr = p
    s, q = 10 ** ls, 10 ** lq
    y1, y2 = FC.source_positions(p, data.t_eval)
    for w in range(5):
        run(s, q, y1, y2, w, 1)
        best_ = 1e9
        for _ in range(5):
            t = time.perf_counter(); _, nf = run(s, q, y1, y2, w, 20); dt = (time.perf_counter() - t) / 20 / len(y1)
            best_ = min(best_, dt)
        tot[w] += best_
        if w == 3:
            far_frac = nf / 20 / len(y1)
    print(f"cell {cells[list(map(tuple, P)).index(tuple(p))]}: " + "  ".join(f"{names[w]} {tot[w] * 1e9 / (list(map(tuple, P)).index(tuple(p)) + 1):.0f}" for w in range(5)) + f"  (far accepted {far_frac:.2f})", flush=True)
print("mean ns/point: " + "  ".join(f"{names[w]} {tot[w] * 1e9 / 8:.0f}" for w in range(5)))
