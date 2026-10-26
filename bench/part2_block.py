"""Part 2, steps 2-3: block-of-8, branch-free root solve. For a block of B consecutive points the quintic
coefficients are stored as real/imag arrays (B, 6) and the roots as (B, 5); K fixed Newton steps per root (or K
fixed Jacobi-Aberth iterations) run with the point index innermost so that LLVM can vectorise across the block
(no branches inside), then a per-point convergence check marks the points that need the scalar fallback. Starts:
(a) every point of the block from the roots of the previous block's last point ("block start"), or (b) linear
extrapolation per point from the previous block's last two points. Also a real/imag scalar Newton (one point,
five roots interleaved) to see whether the complex128 code generation costs anything.
Reports ns per point, fallback rate, root agreement, and whether the hot loop uses ymm/zmm registers.
Usage: python part2_block.py [B] [K]"""
import sys, time
import numpy as np
from numba import njit
import farfield_common as FC
from lenscape.nbmag import _setup, _coefs, _init_roots, _roots, _roots_newton, _cabs2, FM

B = int(sys.argv[1]) if len(sys.argv) > 1 else 8
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4


@njit(cache=False, error_model='numpy', fastmath=FM)
def block_newton(cr, ci, zr, zi, K):
    """K Newton steps on the quintic for every root of every point of the block; cr, ci (B, 6) ascending coefficients,
    zr, zi (B, 5) roots in/out. Point index innermost. Division as reciprocal of |p'|^2 (one divide per step)."""
    nb = cr.shape[0]
    for i in range(5):
        for k in range(K):
            for b in range(nb):
                x = zr[b, i]
                y = zi[b, i]
                pr = cr[b, 5]
                pi = ci[b, 5]
                dr = 0.0
                di = 0.0
                for m in range(4, -1, -1):
                    # dp = dp * z + p ; p = p * z + c[m]
                    tr = dr * x - di * y + pr
                    ti = dr * y + di * x + pi
                    dr = tr
                    di = ti
                    tr = pr * x - pi * y + cr[b, m]
                    ti = pr * y + pi * x + ci[b, m]
                    pr = tr
                    pi = ti
                d = dr * dr + di * di
                inv = 1.0 / d
                # step = p / dp = p * conj(dp) / |dp|^2
                sr = (pr * dr + pi * di) * inv
                si = (pi * dr - pr * di) * inv
                zr[b, i] = x - sr
                zi[b, i] = y - si


@njit(cache=False, error_model='numpy', fastmath=FM)
def block_check(cr, ci, zr, zi, ok):
    """Per point: all five roots have a small final Newton step (|step|^2 < 1e-20 (1 + |z|^2)) and are distinct."""
    nb = cr.shape[0]
    for b in range(nb):
        good = True
        for i in range(5):
            x = zr[b, i]; y = zi[b, i]
            pr = cr[b, 5]; pi = ci[b, 5]; dr = 0.0; di = 0.0
            for m in range(4, -1, -1):
                tr = dr * x - di * y + pr; ti = dr * y + di * x + pi; dr = tr; di = ti
                tr = pr * x - pi * y + cr[b, m]; ti = pr * y + pi * x + ci[b, m]; pr = tr; pi = ti
            d = dr * dr + di * di
            s2 = (pr * pr + pi * pi) / d if d > 0 else np.inf
            if not (s2 < 1e-20 * (1.0 + x * x + y * y)):
                good = False
            for j in range(i):
                if (x - zr[b, j]) ** 2 + (y - zi[b, j]) ** 2 < 1e-14 * (1.0 + x * x + y * y):
                    good = False
        ok[b] = good


@njit(cache=False, error_model='numpy', fastmath=FM)
def block_newton2(cr, ci, zr, zi, K):
    """As block_newton with the step loop outermost and the root loop inside it: the five roots' chains are
    independent within a step, so the core overlaps them (block_newton ran root after root)."""
    nb = cr.shape[0]
    for k in range(K):
        for i in range(5):
            for b in range(nb):
                x = zr[b, i]
                y = zi[b, i]
                pr = cr[b, 5]
                pi = ci[b, 5]
                dr = 0.0
                di = 0.0
                for m in range(4, -1, -1):
                    tr = dr * x - di * y + pr
                    ti = dr * y + di * x + pi
                    dr = tr
                    di = ti
                    tr = pr * x - pi * y + cr[b, m]
                    ti = pr * y + pi * x + ci[b, m]
                    pr = tr
                    pi = ti
                d = dr * dr + di * di
                inv = 1.0 / d
                sr = (pr * dr + pi * di) * inv
                si = (pi * dr - pr * di) * inv
                zr[b, i] = x - sr
                zi[b, i] = y - si


@njit(cache=False, error_model='numpy', fastmath=FM)
def run_block(s, q, y1, y2, B, K, start_mode, reps, out):
    """Light curve solved in blocks of B points. start_mode 0: all points start from the previous block's last roots;
    1: linear extrapolation per point from the previous block's last two points' roots (per root index).
    Returns (fallbacks, total points)."""
    a, m1, m2 = _setup(s, q)
    n = y1.shape[0]
    c = np.empty(6, np.complex128)
    cr = np.empty((B, 6)); ci = np.empty((B, 6))
    zr = np.empty((B, 5)); zi = np.empty((B, 5))
    ok = np.zeros(B, np.bool_)
    z = np.empty(5, np.complex128)
    zprev = np.empty(5, np.complex128)
    nfb = 0
    for r in range(reps):
        _init_roots(z, s)
        for i in range(5):
            zprev[i] = z[i]
        for b0 in range(0, n, B):
            nb = min(B, n - b0)
            for b in range(nb):
                y = complex(y1[b0 + b], abs(y2[b0 + b])) + a * m1
                _coefs(a, m1, m2, y, c)
                for m in range(6):
                    cr[b, m] = c[m].real; ci[b, m] = c[m].imag
                for i in range(5):
                    if start_mode == 0 or start_mode == 2:
                        zr[b, i] = z[i].real; zi[b, i] = z[i].imag
                    else:
                        e = z[i] + (b + 1) * (z[i] - zprev[i])
                        zr[b, i] = e.real; zi[b, i] = e.imag
            if start_mode >= 2:
                block_newton2(cr[:nb], ci[:nb], zr[:nb], zi[:nb], K)
            else:
                block_newton(cr[:nb], ci[:nb], zr[:nb], zi[:nb], K)
            block_check(cr[:nb], ci[:nb], zr[:nb], zi[:nb], ok[:nb])
            for b in range(nb):
                if not ok[b]:
                    nfb += 1
                    # scalar fallback from the previous point's roots (z holds them)
                    for m in range(6):
                        c[m] = complex(cr[b, m], ci[b, m])
                    _roots(c, z)
                    for i in range(5):
                        zr[b, i] = z[i].real; zi[b, i] = z[i].imag
                else:
                    for i in range(5):
                        z[i] = complex(zr[b, i], zi[b, i])
                if r == 0:
                    for i in range(5):
                        out[b0 + b, i] = complex(zr[b, i], zi[b, i])
            # starts for the next block
            for i in range(5):
                zprev[i] = complex(zr[nb - 2, i], zi[nb - 2, i]) if nb >= 2 else zprev[i]
                z[i] = complex(zr[nb - 1, i], zi[nb - 1, i])
            if start_mode == 1 or start_mode == 3:
                for i in range(5):
                    zprev[i] = z[i] - (z[i] - zprev[i])      # slope per point: z_last - z_last-1
    return nfb, n * reps


@njit(cache=False, error_model='numpy', fastmath=FM)
def run_seq(s, q, y1, y2, reps, out):
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
            _roots_newton(c, z)
            acc += z[0].real
            if r == 0:
                for i in range(5):
                    out[k, i] = z[i]
    return acc


def match_dev(A, Bm):
    return np.abs(A[:, :, None] - Bm[:, None, :]).min(axis=2).max()


if __name__ == "__main__":
    data, best = FC.load_data()
    P, cells, _ = FC.vectors()
    print(f"block size {B}, K = {K} fixed Newton steps")
    tot = {"seq": 0.0, "block0": 0.0, "block1": 0.0, "block2": 0.0, "block3": 0.0}
    fbs = {k: [0, 0] for k in ("block0", "block1", "block2", "block3")}
    dev = {k: 0.0 for k in fbs}
    for p in P[:8]:
        t0, u0, tE, ls, lq, al, lr = p
        s, q = 10 ** ls, 10 ** lq
        y1, y2 = FC.source_positions(p, data.t_eval)
        o0 = np.empty((len(y1), 5), np.complex128)
        run_seq(s, q, y1, y2, 1, o0)
        bt = 1e9
        for _ in range(5):
            t = time.perf_counter(); run_seq(s, q, y1, y2, 20, o0); bt = min(bt, (time.perf_counter() - t) / 20 / len(y1))
        tot["seq"] += bt
        for mode in (0, 1, 2, 3):
            ob = np.empty((len(y1), 5), np.complex128)
            run_block(s, q, y1, y2, B, K, mode, 1, ob)
            bt = 1e9
            for _ in range(5):
                t = time.perf_counter(); nfb, npts = run_block(s, q, y1, y2, B, K, mode, 20, ob); bt = min(bt, (time.perf_counter() - t) / 20 / len(y1))
            key = f"block{mode}"
            tot[key] += bt; fbs[key][0] += nfb; fbs[key][1] += npts
            dev[key] = max(dev[key], match_dev(ob, o0))
    print(f"  _roots_newton (scalar, interleaved, warm): {tot['seq'] / 8 * 1e9:.0f} ns/point")
    for key, name in (("block0", "block start, root-outer loop"), ("block1", "linear extrapolation, root-outer loop"), ("block2", "block start, step-outer loop (roots interleaved)"), ("block3", "linear extrapolation, step-outer loop")):
        print(f"  block solve, {name}: {tot[key] / 8 * 1e9:.0f} ns/point, fallbacks {fbs[key][0] / fbs[key][1] * 100:.2f} %, max |dz| vs scalar {dev[key]:.1e}")
    # the vector kernel alone (no filling, checking or fallbacks): ns per point for K steps
    cr = np.random.default_rng(1).standard_normal((B, 6)); ci = np.random.default_rng(2).standard_normal((B, 6))
    zr = np.random.default_rng(3).standard_normal((B, 5)); zi = np.random.default_rng(4).standard_normal((B, 5))
    block_newton2(cr, ci, zr, zi, K)
    reps = 20000
    t = time.perf_counter()
    for _ in range(reps):
        block_newton2(cr, ci, zr, zi, K)
    print(f"  block_newton2 kernel alone: {(time.perf_counter() - t) / reps / B * 1e9:.0f} ns per point for K = {K} steps x 5 roots "
          f"(includes the Python call, ~{(time.perf_counter() - t) / reps * 1e9 / B:.0f} ns/pt gross)")
    asm = block_newton.inspect_asm(block_newton.signatures[0])
    import re
    print(f"  block_newton asm: {len(re.findall(r'zmm', asm))} zmm, {len(re.findall(r'ymm', asm))} ymm, {len(re.findall(r'xmm', asm))} xmm references; "
          f"vfmadd {len(re.findall(r'vfmadd', asm))}, vdivpd {len(re.findall(r'vdivpd', asm))}, vdivsd {len(re.findall(r'vdivsd', asm))}")



@njit(cache=False, error_model='numpy', fastmath=FM)
def block_newton_T(cr, ci, zr, zi, K):
    """block_newton2 with the point index last and contiguous: cr, ci (6, B); zr, zi (5, B) (unit-stride lanes)."""
    nb = cr.shape[1]
    for k in range(K):
        for i in range(5):
            for b in range(nb):
                x = zr[i, b]
                y = zi[i, b]
                pr = cr[5, b]
                pi = ci[5, b]
                dr = 0.0
                di = 0.0
                for m in range(4, -1, -1):
                    tr = dr * x - di * y + pr
                    ti = dr * y + di * x + pi
                    dr = tr
                    di = ti
                    tr = pr * x - pi * y + cr[m, b]
                    ti = pr * y + pi * x + ci[m, b]
                    pr = tr
                    pi = ti
                d = dr * dr + di * di
                inv = 1.0 / d
                sr = (pr * dr + pi * di) * inv
                si = (pi * dr - pr * di) * inv
                zr[i, b] = x - sr
                zi[i, b] = y - si


if __name__ == "__main__":
    import re
    for Bx in (8, 16, 32):
        cr = np.random.default_rng(1).standard_normal((6, Bx)); ci = np.random.default_rng(2).standard_normal((6, Bx))
        zr = np.random.default_rng(3).standard_normal((5, Bx)); zi = np.random.default_rng(4).standard_normal((5, Bx))
        block_newton_T(cr, ci, zr, zi, K)
        reps = 20000
        t = time.perf_counter()
        for _ in range(reps):
            block_newton_T(cr, ci, zr, zi, K)
        dt = (time.perf_counter() - t) / reps
        print(f"  block_newton_T (contiguous lanes) B = {Bx}: {dt / Bx * 1e9:.0f} ns per point for K = {K} steps x 5 roots "
              f"({dt * 1e9:.0f} ns per call incl. ~100 ns dispatch)")
    asm = block_newton_T.inspect_asm(block_newton_T.signatures[0])
    print(f"  block_newton_T asm: {len(re.findall(r'zmm', asm))} zmm, {len(re.findall(r'ymm', asm))} ymm; vfmadd {len(re.findall(r'vfmadd', asm))}, "
          f"vdivpd {len(re.findall(r'vdivpd', asm))}, vgather {len(re.findall(r'vgather', asm))}")


@njit(cache=False, error_model='numpy', fastmath=FM)
def run_block_T(s, q, y1, y2, B, K, reps, out):
    """End-to-end block solve with the contiguous layout: coefficients written straight into (6, B) arrays, starts by
    linear extrapolation from the previous block's last two points, K fixed Newton steps (block_newton_T), the
    convergence check as one more Horner pass, scalar Aberth fallback. Returns (fallbacks, points)."""
    a, m1, m2 = _setup(s, q)
    n = y1.shape[0]
    c = np.empty(6, np.complex128)
    cr = np.empty((6, B)); ci = np.empty((6, B))
    zr = np.empty((5, B)); zi = np.empty((5, B))
    z = np.empty(5, np.complex128)
    zprev = np.empty(5, np.complex128)
    nfb = 0
    for r in range(reps):
        _init_roots(z, s)
        for i in range(5):
            zprev[i] = z[i]
        for b0 in range(0, n, B):
            nb = min(B, n - b0)
            for b in range(nb):
                y = complex(y1[b0 + b], abs(y2[b0 + b])) + a * m1
                _coefs(a, m1, m2, y, c)
                for m in range(6):
                    cr[m, b] = c[m].real
                    ci[m, b] = c[m].imag
                for i in range(5):
                    e = z[i] + (b + 1) * (z[i] - zprev[i])
                    zr[i, b] = e.real
                    zi[i, b] = e.imag
            block_newton_T(cr[:, :nb], ci[:, :nb], zr[:, :nb], zi[:, :nb], K)
            for b in range(nb):
                good = True
                for i in range(5):
                    x = zr[i, b]; yy = zi[i, b]
                    pr = cr[5, b]; pi = ci[5, b]; dr = 0.0; di = 0.0
                    for m in range(4, -1, -1):
                        tr = dr * x - di * yy + pr; ti = dr * yy + di * x + pi; dr = tr; di = ti
                        tr = pr * x - pi * yy + cr[m, b]; ti = pr * yy + pi * x + ci[m, b]; pr = tr; pi = ti
                    d = dr * dr + di * di
                    s2 = (pr * pr + pi * pi) / d if d > 0 else np.inf
                    if not (s2 < 1e-20 * (1.0 + x * x + yy * yy)):
                        good = False
                    for j in range(i):
                        if (x - zr[j, b]) ** 2 + (yy - zi[j, b]) ** 2 < 1e-14 * (1.0 + x * x + yy * yy):
                            good = False
                zlast = z
                if not good:
                    nfb += 1
                    for m in range(6):
                        c[m] = complex(cr[m, b], ci[m, b])
                    _roots(c, z)                      # from the previous point's roots
                    for i in range(5):
                        zr[i, b] = z[i].real; zi[i, b] = z[i].imag
                for i in range(5):
                    zprev[i] = z[i]
                    z[i] = complex(zr[i, b], zi[i, b])
                if r == 0:
                    for i in range(5):
                        out[b0 + b, i] = z[i]
    return nfb, n * reps


if __name__ == "__main__":
    tot = 0.0; fb = [0, 0]; dv = 0.0; tseq = 0.0
    for p in P[:8]:
        t0, u0, tE, ls, lq, al, lr = p
        s, q = 10 ** ls, 10 ** lq
        y1, y2 = FC.source_positions(p, data.t_eval)
        o0 = np.empty((len(y1), 5), np.complex128); run_seq(s, q, y1, y2, 1, o0)
        b = 1e9
        for _ in range(5):
            t = time.perf_counter(); run_seq(s, q, y1, y2, 20, o0); b = min(b, (time.perf_counter() - t) / 20 / len(y1))
        tseq += b
        ob = np.empty((len(y1), 5), np.complex128); run_block_T(s, q, y1, y2, B, K, 1, ob)
        b = 1e9
        for _ in range(5):
            t = time.perf_counter(); nfb, npts = run_block_T(s, q, y1, y2, B, K, 20, ob); b = min(b, (time.perf_counter() - t) / 20 / len(y1))
        tot += b; fb[0] += nfb; fb[1] += npts; dv = max(dv, match_dev(ob, o0))
    print(f"  END-TO-END block solve (contiguous lanes, extrapolated starts) B = {B}, K = {K}: {tot / 8 * 1e9:.0f} ns/point vs scalar "
          f"{tseq / 8 * 1e9:.0f} ns/point; fallbacks {fb[0] / fb[1] * 100:.2f} %; max |dz| vs scalar {dv:.1e}")
