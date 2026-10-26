"""Part 2, step 1: iteration census of the root solvers along the benchmark light curves (counts, not timings):
Aberth iterations per point (_roots from the previous point's roots), per-root Newton iterations and the fallback
rate of _roots_newton, and the root displacement |dz| between consecutive points (max over the 5 roots), for the ~20
benchmark vectors. Instrumented copies of the nbmag solvers. Prints histograms; writes farfield_out/census.npz."""
import numpy as np
from numba import njit
import farfield_common as FC
from lenscape import nbmag
from lenscape.nbmag import _setup, _coefs, _init_roots, _cdiv, _cinv, _cabs2, FM

data, best = FC.load_data()
P, cells, _ = FC.vectors()


@njit(cache=False, error_model='numpy', fastmath=FM)
def aberth_count(c, z):
    n = 5
    nit = 0
    for it in range(200):
        nit += 1
        maxcorr = 0.0
        for i in range(n):
            p = c[5]; dp = 0j
            for k in range(4, -1, -1):
                dp = dp * z[i] + p; p = p * z[i] + c[k]
            if p == 0:
                continue
            ratio = _cdiv(p, dp) if dp != 0 else complex(1e-3, 1e-3)
            sm = 0j
            for j in range(n):
                if j != i:
                    d = z[i] - z[j]
                    if d != 0:
                        sm += _cinv(d)
            w = _cdiv(ratio, 1.0 - ratio * sm)
            z[i] -= w
            aw = _cabs2(w)
            if aw > maxcorr:
                maxcorr = aw
        if maxcorr < 1e-26:
            break
    for i in range(n):
        p = c[5]; dp = 0j
        for k in range(4, -1, -1):
            dp = dp * z[i] + p; p = p * z[i] + c[k]
        if dp != 0:
            z[i] -= _cdiv(p, dp)
    return nit


@njit(cache=False, error_model='numpy', fastmath=FM)
def newton_count(c, z):
    """Returns (total Newton steps over the 5 roots, fell back to Aberth, Aberth iterations if so)."""
    z00, z01, z02, z03, z04 = z[0], z[1], z[2], z[3], z[4]
    tot = 0
    for i in range(5):
        conv = False
        zi = z[i]
        for it in range(6):
            tot += 1
            p = c[5]; dp = 0j
            for k in range(4, -1, -1):
                dp = dp * zi + p; p = p * zi + c[k]
            if dp == 0:
                break
            step = _cdiv(p, dp)
            zi -= step
            if _cabs2(step) < 1e-20 * (1.0 + _cabs2(zi) + 2.0 * max(abs(zi.real), abs(zi.imag))):
                conv = True
                break
        z[i] = zi
        if not conv:
            z[0], z[1], z[2], z[3], z[4] = z00, z01, z02, z03, z04
            return tot, True, aberth_count(c, z)
    for i in range(5):
        for j in range(i):
            if _cabs2(z[i] - z[j]) < 1e-14 * (1.0 + _cabs2(z[i])):
                z[0], z[1], z[2], z[3], z[4] = z00, z01, z02, z03, z04
                return tot, True, aberth_count(c, z)
    return tot, False, 0


@njit(cache=False, error_model='numpy', fastmath=FM)
def census(s, q, y1, y2):
    a, m1, m2 = _setup(s, q)
    n = y1.shape[0]
    c = np.empty(6, np.complex128)
    z = np.empty(5, np.complex128)
    zn = np.empty(5, np.complex128)
    ab = np.empty(n, np.int64); nw = np.empty(n, np.int64); fb = np.zeros(n, np.bool_); fba = np.zeros(n, np.int64)
    dz = np.empty(n)
    _init_roots(z, s)
    _init_roots(zn, s)
    for k in range(n):
        y = complex(y1[k], abs(y2[k])) + a * m1
        _coefs(a, m1, m2, y, c)
        zprev = z.copy()
        ab[k] = aberth_count(c, z)
        # displacement: match each new root to the nearest old one
        mx = 0.0
        for i in range(5):
            dmin = 1e300
            for j in range(5):
                d = _cabs2(z[i] - zprev[j])
                if d < dmin:
                    dmin = d
            mx = max(mx, np.sqrt(dmin))
        dz[k] = mx
        nw[k], fb[k], fba[k] = newton_count(c, zn)
    return ab, nw, fb, fba, dz


AB, NW, FB, FBA, DZ = [], [], [], [], []
for p in P:
    t0, u0, tE, ls, lq, al, lr = p
    y1, y2 = FC.source_positions(p, data.t_eval)
    ab, nw, fb, fba, dz = census(10 ** ls, 10 ** lq, y1, y2)
    AB.append(ab); NW.append(nw); FB.append(fb); FBA.append(fba); DZ.append(dz)
AB, NW, FB, FBA, DZ = (np.concatenate(v) for v in (AB, NW, FB, FBA, DZ))
print(f"{len(AB)} points ({len(P)} light curves of {len(data.t_eval)})")
print(f"Aberth (warm start from the previous point): mean {AB.mean():.2f} iterations/point, median {np.median(AB):.0f}, "
      f"percentiles 90/99/max {np.percentile(AB, 90):.0f}/{np.percentile(AB, 99):.0f}/{AB.max()}")
h = np.bincount(AB)
print("  histogram (iterations: count):", {i: int(h[i]) for i in range(len(h)) if h[i]})
print(f"_roots_newton: mean {NW.mean():.2f} Newton steps/point over 5 roots ({NW[~FB].mean() / 5:.2f} per root without "
      f"fallback), fallback rate {FB.mean() * 100:.2f} % ({FB.sum()} points; those then take {FBA[FB].mean() if FB.any() else 0:.1f} Aberth iterations)")
print(f"root displacement between consecutive points |dz| (max over roots): median {np.median(DZ[1:]):.1e}, "
      f"90 % {np.percentile(DZ[1:], 90):.1e}, 99 % {np.percentile(DZ[1:], 99):.1e}, max {DZ[1:].max():.1e}")
for thr in (1e-3, 1e-2, 3e-2, 1e-1):
    print(f"  fraction of steps with |dz| < {thr:.0e}: {np.mean(DZ[1:] < thr):.3f}")
np.savez(FC.os.path.join(FC.ensure_out(), "census.npz"), AB=AB, NW=NW, FB=FB, FBA=FBA, DZ=DZ)
