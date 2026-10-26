"""Part 2, step 2 (variant): instruction-level parallel root solvers for one point, the five roots interleaved so
that the out-of-order core overlaps their dependency chains (the existing solvers finish one root before starting
the next). Prototypes: _roots_newton_il (polynomial Newton from the previous point's roots, Jacobi order) and
_roots_il (Aberth-Ehrlich, Jacobi update: all corrections from the current roots, 10 pair inversions reused).
Timed from warm starts along the benchmark light curves; agreement with the sequential solvers reported.
Usage: python part2_ilp.py"""
import time
import numpy as np
from numba import njit
import farfield_common as FC
from lenscape import nbmag
from lenscape.nbmag import _setup, _coefs, _init_roots, _cdiv, _cinv, _cabs2, _roots, _roots_newton, FM


@njit(cache=False, inline="always", error_model="numpy", fastmath=FM)
def _pstep(c, zi):
    p = c[5]
    dp = 0j
    for k in range(4, -1, -1):
        dp = dp * zi + p
        p = p * zi + c[k]
    if dp == 0:
        return zi, np.inf
    step = _cdiv(p, dp)
    zi -= step
    return zi, _cabs2(step)


@njit(cache=False, inline="always", error_model="numpy", fastmath=FM)
def _ptol(zi):
    return 1e-20 * (1.0 + _cabs2(zi) + 2.0 * max(abs(zi.real), abs(zi.imag)))


@njit(cache=False, error_model='numpy', fastmath=FM)
def _roots_newton_il(c, z):
    """As nbmag._roots_newton (same tolerance and fallback rules) with the five roots stepped together."""
    z0, z1, z2, z3, z4 = z[0], z[1], z[2], z[3], z[4]
    c0 = c1 = c2 = c3 = c4 = False
    for it in range(6):
        if not c0:
            z0, s = _pstep(c, z0); c0 = s < _ptol(z0)
        if not c1:
            z1, s = _pstep(c, z1); c1 = s < _ptol(z1)
        if not c2:
            z2, s = _pstep(c, z2); c2 = s < _ptol(z2)
        if not c3:
            z3, s = _pstep(c, z3); c3 = s < _ptol(z3)
        if not c4:
            z4, s = _pstep(c, z4); c4 = s < _ptol(z4)
        if c0 and c1 and c2 and c3 and c4:
            break
    if not (c0 and c1 and c2 and c3 and c4):
        _roots(c, z)
        return
    if (_cabs2(z1 - z0) < 1e-14 * (1.0 + _cabs2(z1)) or _cabs2(z2 - z0) < 1e-14 * (1.0 + _cabs2(z2))
            or _cabs2(z2 - z1) < 1e-14 * (1.0 + _cabs2(z2)) or _cabs2(z3 - z0) < 1e-14 * (1.0 + _cabs2(z3))
            or _cabs2(z3 - z1) < 1e-14 * (1.0 + _cabs2(z3)) or _cabs2(z3 - z2) < 1e-14 * (1.0 + _cabs2(z3))
            or _cabs2(z4 - z0) < 1e-14 * (1.0 + _cabs2(z4)) or _cabs2(z4 - z1) < 1e-14 * (1.0 + _cabs2(z4))
            or _cabs2(z4 - z2) < 1e-14 * (1.0 + _cabs2(z4)) or _cabs2(z4 - z3) < 1e-14 * (1.0 + _cabs2(z4))):
        _roots(c, z)
        return
    z[0] = z0; z[1] = z1; z[2] = z2; z[3] = z3; z[4] = z4


@njit(cache=False, inline="always", error_model="numpy", fastmath=FM)
def _ratio(c, zi):
    p = c[5]
    dp = 0j
    for k in range(4, -1, -1):
        dp = dp * zi + p
        p = p * zi + c[k]
    if p == 0:
        return 0j
    return _cdiv(p, dp) if dp != 0 else complex(1e-3, 1e-3)


@njit(cache=False, inline="always", error_model="numpy", fastmath=FM)
def _sinv(d):
    return _cinv(d) if d != 0 else 0j


@njit(cache=False, error_model='numpy', fastmath=FM)
def _roots_il(c, z):
    """Aberth-Ehrlich with Jacobi updates (all five corrections from the current roots), 10 pair inversions reused,
    same stopping rule as nbmag._roots (max |w|^2 < 1e-26), then one Newton polish per root."""
    z0, z1, z2, z3, z4 = z[0], z[1], z[2], z[3], z[4]
    for it in range(200):
        r0 = _ratio(c, z0); r1 = _ratio(c, z1); r2 = _ratio(c, z2); r3 = _ratio(c, z3); r4 = _ratio(c, z4)
        i01 = _sinv(z0 - z1); i02 = _sinv(z0 - z2); i03 = _sinv(z0 - z3); i04 = _sinv(z0 - z4)
        i12 = _sinv(z1 - z2); i13 = _sinv(z1 - z3); i14 = _sinv(z1 - z4)
        i23 = _sinv(z2 - z3); i24 = _sinv(z2 - z4)
        i34 = _sinv(z3 - z4)
        w0 = _cdiv(r0, 1.0 - r0 * (i01 + i02 + i03 + i04)) if r0 != 0 else 0j
        w1 = _cdiv(r1, 1.0 - r1 * (-i01 + i12 + i13 + i14)) if r1 != 0 else 0j
        w2 = _cdiv(r2, 1.0 - r2 * (-i02 - i12 + i23 + i24)) if r2 != 0 else 0j
        w3 = _cdiv(r3, 1.0 - r3 * (-i03 - i13 - i23 + i34)) if r3 != 0 else 0j
        w4 = _cdiv(r4, 1.0 - r4 * (-i04 - i14 - i24 - i34)) if r4 != 0 else 0j
        z0 -= w0; z1 -= w1; z2 -= w2; z3 -= w3; z4 -= w4
        m = max(max(max(_cabs2(w0), _cabs2(w1)), max(_cabs2(w2), _cabs2(w3))), _cabs2(w4))
        if m < 1e-26:
            break
    z0, _ = _pstep(c, z0); z1, _ = _pstep(c, z1); z2, _ = _pstep(c, z2); z3, _ = _pstep(c, z3); z4, _ = _pstep(c, z4)
    z[0] = z0; z[1] = z1; z[2] = z2; z[3] = z3; z[4] = z4


@njit(cache=False, error_model='numpy', fastmath=FM)
def run(s, q, y1, y2, which, reps, out):
    """which: 0 _roots_newton, 1 _roots, 2 _roots_newton_il, 3 _roots_il; roots of every point into out (n, 5)."""
    a, m1, m2 = _setup(s, q)
    n = y1.shape[0]
    c = np.empty(6, np.complex128)
    z = np.empty(5, np.complex128)
    acc = 0.0
    for r in range(reps):
        _init_roots(z, s)
        for k in range(n):
            y = complex(y1[k], abs(y2[k])) + a * m1
            _coefs(a, m1, m2, y, c)
            if which == 0:
                _roots_newton(c, z)
            elif which == 1:
                _roots(c, z)
            elif which == 2:
                _roots_newton_il(c, z)
            else:
                _roots_il(c, z)
            acc += z[0].real
            if r == 0:
                for i in range(5):
                    out[k, i] = z[i]
    return acc


def match_dev(A, B):
    """Max over points of the max over roots of the distance to the nearest root of the other set."""
    d = np.abs(A[:, :, None] - B[:, None, :]).min(axis=2)
    return d.max()


if __name__ == "__main__":
    data, best = FC.load_data()
    P, cells, _ = FC.vectors()
    names = ["_roots_newton (seq)", "_roots Aberth GS (seq)", "_roots_newton_il", "_roots_il Aberth Jacobi"]
    tot = np.zeros(4)
    devs = np.zeros(4)
    for p in P[:8]:
        t0, u0, tE, ls, lq, al, lr = p
        s, q = 10 ** ls, 10 ** lq
        y1, y2 = FC.source_positions(p, data.t_eval)
        outs = []
        for w in range(4):
            out = np.empty((len(y1), 5), np.complex128)
            run(s, q, y1, y2, w, 1, out)
            b = 1e9
            for _ in range(5):
                t = time.perf_counter(); run(s, q, y1, y2, w, 20, out); b = min(b, (time.perf_counter() - t) / 20 / len(y1))
            tot[w] += b
            outs.append(out.copy())
        for w in range(4):
            devs[w] = max(devs[w], match_dev(outs[w], outs[1]))
    print("ns per point (coefs included), mean over 8 light curves; max root deviation from sequential Aberth:")
    for w in range(4):
        print(f"  {names[w]:26s} {tot[w] / 8 * 1e9:6.0f} ns   max |dz| vs Aberth(seq) {devs[w]:.1e}")
