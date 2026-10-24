"""Finite-source (uniform disk) binary-lens magnification by contour integration, numba.

A = sum over image tracks of parity * (1/2) * integral Im(conj(z) dz) / (pi rho^2), where the image tracks z_j(theta)
are the images of the source boundary y0 + rho e^{i theta}. Between neighbouring samples each track is integrated as
a cubic Hermite curve built from positions and analytic tangents dz/dtheta (3-point Gauss-Legendre, exact for the
cubic). Where the boundary crosses a caustic the image number changes by two; the created/annihilated (+, -) pair is
joined by a chord and that interval is bisected down to a tiny width. Intervals are refined adaptively (largest
error estimate first) until the estimated error is below reltol * area. Geometry as nbmag (VBBL frame).

Returns (A, status, nsamples): status 0 ok, 1 sample budget exhausted (A still returned), -1 inconsistent (caller falls back).
"""
import numpy as np
from numba import njit

# no zero-division checks and 'safe' fast-math (no nnan/ninf: the code relies on inf/nan semantics)
FM = {'nsz', 'arcp', 'contract', 'afn', 'reassoc'}  # nsz is safe since the parity-sum fix (contour._fix_parity)
from . import nbmag
from .nbmag import _cabs, _cabs2, _cinv, _cdiv

MAXS = 4096
TRACK = True                             # track images for refinement samples
NEWTON_TOL = 1e-10                       # relative image-position tolerance of the tracked Newton steps
RICHARDSON = False                       # error estimate from the change of the area on splitting
RICH_FACTOR = 1.0
LD_CTOL_MAX = 0.03                       # contour tolerance cap inside the limb-darkening integral
GL_T = np.array([0.5 - 0.5 * np.sqrt(0.6), 0.5, 0.5 + 0.5 * np.sqrt(0.6)])
GL_W = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])


@njit(cache=True, error_model='numpy', fastmath=FM)
def _fix_parity(a, m1, m2, oz, op, n):
    """Two point masses: n_plus - n_minus = -1 for 3 and 5 images. Images next to a critical curve have det J ~ 0
    and their sign is decided by rounding (5 images with parities summing to -3 -> no consistent image
    matching -> intervals silently dropped, A 28 % low). Flip the least reliable sign (smallest |det J|) until the
    rule holds."""
    if n != 3 and n != 5:
        return
    tot = 0.0
    for i in range(n):
        tot += op[i]
    for _ in range(2):
        if tot == -1.0:
            return
        want = 1.0 if tot < -1.0 else -1.0       # flip an image of parity -want to +want
        bi = -1
        bd = 1e300
        for i in range(n):
            if op[i] == -want:
                zz = oz[i]
                J1 = m1 * _cinv((zz - a) * (zz - a)) + m2 * _cinv(zz * zz)
                d = abs(1.0 - (J1 * J1.conjugate()).real)
                if d < bd:
                    bd, bi = d, i
        if bi < 0:
            return
        op[bi] = want
        tot += 2.0 * want


@njit(cache=True, error_model='numpy', fastmath=FM)
def _images(a, m1, m2, y, dy, c, z, oz, op, odz, og):
    """True images of source y (shifted VBBL frame) with parity and tangent dz for a source displacement dy.
    og[0] = lens-equation residual of the better ghost root (3-image case; ~ sqrt(distance to the caustic)), else inf."""
    nbmag._coefs(a, m1, m2, y, c)
    nbmag._roots(c, z)
    w1 = w2 = w3 = -1
    g1 = g2 = g3 = -1.0
    for i in range(5):
        zz = z[i]
        zc = zz.conjugate()
        g = _cabs((y - zz) + m1 * _cinv(zc - a) + m2 * _cinv(zc))
        if g > g1:
            w3, g3, w2, g2, w1, g1 = w2, g2, w1, g1, i, g
        elif g > g2:
            w3, g3, w2, g2 = w2, g2, i, g
        elif g > g3:
            w3, g3 = i, g
    three = g2 * nbmag.DLMIN > g3 + 1e-12
    og[0] = g2 if three else np.inf
    n = 0
    for i in range(5):
        if three and (i == w1 or i == w2):
            continue
        zz = z[i]
        J1 = m1 * _cinv((zz - a) * (zz - a)) + m2 * _cinv(zz * zz)
        J1c = J1.conjugate()
        dJ = 1.0 - (J1 * J1c).real
        oz[n] = zz
        op[n] = 1.0 if dJ > 0 else -1.0
        odz[n] = (dy - J1c * dy.conjugate()) / dJ
        n += 1
    _fix_parity(a, m1, m2, oz, op, n)
    return n


@njit(cache=True, error_model='numpy', fastmath=FM)
def _newton_image(a, m1, m2, y, zz, cnt):
    """Newton on the lens equation from zz. Returns (z, detJ, J1c, converged)."""
    tol2 = NEWTON_TOL * NEWTON_TOL
    for it in range(10):
        ia = _cinv(zz - a)                 # 1/conj(w) = conj(1/w): two inversions per step
        ib = _cinv(zz)
        r = y - (zz - m1 * ia.conjugate() - m2 * ib.conjugate())
        J1 = m1 * ia * ia + m2 * ib * ib
        J1c = J1.conjugate()
        dJ = 1.0 - (J1 * J1c).real
        step = (r - J1c * r.conjugate()) / dJ
        zz = zz + step
        if cnt is not None:
            cnt[4] += 1
        # |step| < tol (1 + |z|) without sqrt: (1 + |z|)^2 >= 1 + |z|^2 + (|x| + |y|) (slightly stricter)
        az = abs(zz.real) + abs(zz.imag)
        if _cabs2(step) < tol2 * (1.0 + zz.real * zz.real + zz.imag * zz.imag + az):
            ia = _cinv(zz - a)
            ib = _cinv(zz)
            J1 = m1 * ia * ia + m2 * ib * ib
            J1c = J1.conjugate()
            return zz, 1.0 - (J1 * J1c).real, J1c, True
    return zz, 0.0, 0j, False


@njit(cache=True, error_model='numpy', fastmath=FM)
def _images_tracked(a, m1, m2, y, dy, c, nL, PRED, PL, zall, oz, op, odz, og, cnt=None):
    """Images at source y from the neighbour's images (prediction ZL + DL*dth, Newton on the lens
    equation); ghosts from the quintic deflated by three images (closed-form quadratic). Handles image-number changes
    without a full solve: 3 -> 5 (the ghost pair became real: polish it), 5 -> 3 (the annihilated pair fails to track;
    keep the 3 consistent images). Returns n, or -1 -> caller does a full solve."""
    if nL != 3 and nL != 5:
        return -1
    nbmag._coefs(a, m1, m2, y, c)
    nok = 0
    for i in range(nL):
        zz, dJ, J1c, conv = _newton_image(a, m1, m2, y, PRED[i], cnt)
        good = conv and dJ * PL[i] > 0 and abs(dJ) >= 1e-9
        if good:
            for j in range(nok):
                if _cabs(oz[j] - zz) < 1e-8 * (1.0 + _cabs(zz)):
                    good = False
                    break
        if not good:
            if nL == 3:
                return -1
            continue
        oz[nok] = zz
        op[nok] = PL[i]
        odz[nok] = (dy - J1c * dy.conjugate()) / dJ
        nok += 1
    if nL == 5:
        if nok == 5:
            og[0] = np.inf
            for i in range(5):
                zall[i] = oz[i]
            return 5
        if nok != 3 or op[0] + op[1] + op[2] != -1.0:
            return -1
    # deflate the quintic by the three images -> quadratic p2 z^2 + p1 z + p0 (its roots are the ghosts)
    b0, b1, b2, b3, b4, b5 = c[0], c[1], c[2], c[3], c[4], c[5]
    for i in range(3):
        r_ = oz[i]
        q4 = b5
        q3 = b4 + r_ * q4
        q2 = b3 + r_ * q3
        q1 = b2 + r_ * q2
        q0 = b1 + r_ * q1
        b0, b1, b2, b3, b4, b5 = q0, q1, q2, q3, q4, 0j
    p0, p1, p2 = b0, b1, b2
    disc = np.sqrt(p1 * p1 - 4.0 * p2 * p0 + 0j)
    g_a = (-p1 + disc) / (2.0 * p2)
    g_b = (-p1 - disc) / (2.0 * p2)
    best = 1e300
    for zz in (g_a, g_b):
        zc = zz.conjugate()
        g = _cabs((y - zz) + m1 * _cinv(zc - a) + m2 * _cinv(zc))
        if g < best:
            best = g
    if best * nbmag.DLMIN <= 1e-12:          # the ghost pair is real: image number 3 -> 5
        if nL == 5:
            return -1
        for zz0 in (g_a, g_b):
            zz, dJ, J1c, conv = _newton_image(a, m1, m2, y, zz0, cnt)
            if not conv or abs(dJ) < 1e-9:
                return -1
            for j in range(nok):
                if _cabs(oz[j] - zz) < 1e-8 * (1.0 + _cabs(zz)):
                    return -1
            oz[nok] = zz
            op[nok] = 1.0 if dJ > 0 else -1.0
            odz[nok] = (dy - J1c * dy.conjugate()) / dJ
            nok += 1
        if op[3] + op[4] != 0.0:
            return -1
        og[0] = np.inf
        for i in range(5):
            zall[i] = oz[i]
        return 5
    og[0] = best
    zall[0], zall[1], zall[2], zall[3], zall[4] = oz[0], oz[1], oz[2], g_a, g_b
    return 3


@njit(cache=True, error_model='numpy', fastmath=FM)
def _herm_area(P, Q, T0, T1):
    """(1/2) integral Im(conj(z) z') dt over the cubic Hermite curve P -> Q with end tangents T0, T1 (t in [0, 1])."""
    s = 0.0
    for k in range(3):
        t = GL_T[k]
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        d00 = 6 * t2 - 6 * t
        d10 = 3 * t2 - 4 * t + 1
        d01 = -6 * t2 + 6 * t
        d11 = 3 * t2 - 2 * t
        zt = h00 * P + h10 * T0 + h01 * Q + h11 * T1
        dzt = d00 * P + d10 * T0 + d01 * Q + d11 * T1
        s += GL_W[k] * (zt.conjugate() * dzt).imag
    return 0.5 * s


@njit(cache=True, error_model='numpy', fastmath=FM)
def _interval(nL, ZL, PL, DL, nR, ZR, PR, DR, dth, mrow):
    """Contribution and error estimate of one theta interval. Returns (area, err, ok). mrow (the image matching,
    read later for midpoint predictions) is reset first: an early return used to leave it uninitialised (np.empty),
    and the refinement then indexed Z[j, garbage] -> segfault."""
    for i in range(5):
        mrow[i] = -1
    usedR = 0                                  # bitmask of matched right images
    m0 = m1_ = m2_ = m3 = m4 = -1              # match of left image i (no allocations)
    nm = min(nL, nR)
    # greedy matching on predicted positions, same parity only (squared distances)
    for _ in range(nm):
        best = 1e300
        bi = -1
        bj = -1
        for i in range(nL):
            mi = m0 if i == 0 else (m1_ if i == 1 else (m2_ if i == 2 else (m3 if i == 3 else m4)))
            if mi >= 0:
                continue
            pred = ZL[i] + DL[i] * dth
            for j in range(nR):
                if (usedR >> j) & 1 or PR[j] != PL[i]:
                    continue
                dd = pred - ZR[j]
                d2 = dd.real * dd.real + dd.imag * dd.imag
                if d2 < best:
                    best, bi, bj = d2, i, j
        if bi < 0:
            return 0.0, 0.0, False
        if bi == 0:
            m0 = bj
        elif bi == 1:
            m1_ = bj
        elif bi == 2:
            m2_ = bj
        elif bi == 3:
            m3 = bj
        else:
            m4 = bj
        usedR |= 1 << bj
    area = 0.0
    err = 0.0
    for i in range(nL):
        j = m0 if i == 0 else (m1_ if i == 1 else (m2_ if i == 2 else (m3 if i == 3 else m4)))
        mrow[i] = j
        if j < 0:
            continue
        P, Q = ZL[i], ZR[j]
        T0, T1 = DL[i] * dth, DR[j] * dth
        h = _herm_area(P, Q, T0, T1)
        ch = 0.5 * (P.conjugate() * Q).imag
        # error scale: Hermite - chord, damped by the relative tangent change (smooth arcs converge as dth^2)
        r = _cabs(T1 - T0) / (_cabs(T0) + _cabs(T1) + 1e-300)
        area += PL[i] * h
        err += abs(h - ch) * min(1.0, r * r) + 0.1 * _cabs(Q - P) * _cabs(Q - P - 0.5 * (T0 + T1))
    if nL != nR:
        # pair created (extra images on the right) or annihilated (on the left): join the (+, -) pair by a chord
        if nR > nL:
            ip = -1
            im = -1
            for j in range(nR):
                if not (usedR >> j) & 1:
                    if PR[j] > 0:
                        ip = j
                    else:
                        im = j
            if ip < 0 or im < 0:
                return 0.0, 0.0, False
            ch = 0.5 * (ZR[im].conjugate() * ZR[ip]).imag
            area += ch
            err += _cabs2(ZR[ip] - ZR[im])
        else:
            ip = -1
            im = -1
            for i in range(nL):
                mi = m0 if i == 0 else (m1_ if i == 1 else (m2_ if i == 2 else (m3 if i == 3 else m4)))
                if mi < 0:
                    if PL[i] > 0:
                        ip = i
                    else:
                        im = i
            if ip < 0 or im < 0:
                return 0.0, 0.0, False
            ch = 0.5 * (ZL[ip].conjugate() * ZL[im]).imag
            area += ch
            err += _cabs2(ZL[ip] - ZL[im])
    return area, err, True


@njit(cache=True, error_model='numpy', fastmath=FM)
def mag_disk(s, q, y1, y2, rho, reltol, n0=16, maxs=2048, zwarm=None, cnt=None):
    """Uniform-disk magnification at source (y1, y2) with radius rho. zwarm: roots of the disk centre (warm start)."""
    a, m1, m2 = nbmag._setup(s, q)
    y0 = complex(y1, abs(y2)) + a * m1
    c = np.empty(6, np.complex128)
    z = np.empty(5, np.complex128)
    if zwarm is None:
        nbmag._init_roots(z, s)
    else:
        for i in range(5):
            z[i] = zwarm[i]
    TH = np.empty(maxs)
    NI = np.empty(maxs, np.int64)
    Z = np.empty((maxs, 5), np.complex128)
    P = np.empty((maxs, 5))
    DZ = np.empty((maxs, 5), np.complex128)
    ZA = np.empty((maxs, 5), np.complex128)  # all 5 roots per sample (warm starts)
    G = np.empty(maxs)                       # ghost residual per sample (caustic proximity)
    MT = np.empty((maxs, 5), np.int64)       # matching of interval [k, NX[k]]: left image i -> right image MT[k, i]
    PRED = np.empty(5, np.complex128)
    NX = np.empty(maxs, np.int64)            # next sample index (circular list, ordered in theta)
    AR = np.empty(maxs)                      # area of interval [k, NX[k]] (written before read)
    ER = np.empty(maxs)
    ns = 0
    for k in range(n0):
        th = 2 * np.pi * k / n0
        e = complex(np.cos(th), np.sin(th))
        TH[k] = th
        nt = -1
        if TRACK and k > 0:                  # track sequentially around the circle from the previous sample
            for i in range(NI[k - 1]):
                PRED[i] = Z[k - 1, i] + DZ[k - 1, i] * (2 * np.pi / n0)
            nt = _images_tracked(a, m1, m2, y0 + rho * e, 1j * rho * e, c, NI[k - 1], PRED, P[k - 1],
                                 ZA[k], Z[k], P[k], DZ[k], G[k:k + 1], cnt)
        if cnt is not None:
            cnt[0 if nt > 0 else 1] += 1
        if nt > 0:
            NI[k] = nt
        else:
            if k > 0:
                for i in range(5):
                    z[i] = ZA[k - 1, i]
            NI[k] = _images(a, m1, m2, y0 + rho * e, 1j * rho * e, c, z, Z[k], P[k], DZ[k], G[k:k + 1])
            for i in range(5):
                ZA[k, i] = z[i]
        NX[k] = (k + 1) % n0
    ns = n0
    for k in range(n0):
        j = NX[k]
        dth = (TH[j] - TH[k]) % (2 * np.pi)
        ar, er, ok = _interval(NI[k], Z[k], P[k], DZ[k], NI[j], Z[j], P[j], DZ[j], dth, MT[k])
        AR[k], ER[k] = ar, (er if ok else 1e300)
        if NI[k] == 3 and NI[j] == 3 and min(G[k], G[j]) ** 2 < rho * dth:
            ER[k] = 1e300                    # the boundary may dip into the caustic inside this interval
    status = 0
    unresolved = False
    tot = 0.0
    while True:
        tot = 0.0
        terr = 0.0
        k = 0
        for _ in range(ns):
            tot += AR[k]
            terr += ER[k]
            k = NX[k]
        if cnt is not None:
            cnt[3] += 1
        if terr < reltol * abs(tot):
            break
        if ns >= maxs:
            status = 1
            break
        # one pass: split every interval whose error exceeds its share of the budget
        thresh = 0.5 * reltol * abs(tot) / ns
        start = 0
        k = start
        nint = ns
        for _ in range(nint):
            j = NX[k]
            if ER[k] > thresh and ns < maxs:
                dth = (TH[j] - TH[k]) % (2 * np.pi)
                if dth < 1e-9:                   # cannot refine further (e.g. tangent at a cusp)
                    if AR[k] == 0.0 and ER[k] >= 1e300:
                        unresolved = True        # failed image matching: its area would be dropped silently
                    ER[k] = 0.0
                else:
                    th = TH[k] + 0.5 * dth
                    e = complex(np.cos(th), np.sin(th))
                    new = ns
                    TH[new] = th
                    nt = -1
                    if TRACK:
                        # cubic-Hermite midpoint prediction from both ends (left-tangent step if unmatched)
                        for i in range(NI[k]):
                            jm = MT[k, i]
                            if jm >= 0:
                                PRED[i] = 0.5 * (Z[k, i] + Z[j, jm]) + 0.125 * dth * (DZ[k, i] - DZ[j, jm])
                            else:
                                PRED[i] = Z[k, i] + DZ[k, i] * (0.5 * dth)
                        nt = _images_tracked(a, m1, m2, y0 + rho * e, 1j * rho * e, c, NI[k], PRED, P[k],
                                             ZA[new], Z[new], P[new], DZ[new], G[new:new + 1], cnt)
                    if cnt is not None:
                        cnt[0 if nt > 0 else 1] += 1
                    if nt > 0:
                        NI[new] = nt
                    else:
                        for i in range(5):       # full solve, warm start from the left neighbour's roots
                            z[i] = ZA[k, i]
                        NI[new] = _images(a, m1, m2, y0 + rho * e, 1j * rho * e, c, z, Z[new], P[new], DZ[new],
                                          G[new:new + 1])
                        for i in range(5):
                            ZA[new, i] = z[i]
                    NX[new] = j
                    NX[k] = new
                    ns += 1
                    a_whole = AR[k]
                    smooth = True
                    for (u, v) in ((k, new), (new, j)):
                        d = (TH[v] - TH[u]) % (2 * np.pi)
                        ar, er, ok = _interval(NI[u], Z[u], P[u], DZ[u], NI[v], Z[v], P[v], DZ[v], d, MT[u])
                        if cnt is not None:
                            cnt[2] += 1
                        AR[u] = ar
                        ER[u] = er if ok else (1e300 if d > 1e-9 else 0.0)
                        if not ok and d <= 1e-9:
                            unresolved = True
                        if not ok or NI[u] != NI[v]:
                            smooth = False
                        if NI[u] == 3 and NI[v] == 3 and min(G[u], G[v]) ** 2 < rho * d and d > 1e-9:
                            ER[u] = 1e300
                            smooth = False
                    if RICHARDSON and smooth:
                        # error of the two halves from the change on splitting (cubic Hermite: ~h^5 -> /15)
                        de = abs(a_whole - AR[k] - AR[new]) * RICH_FACTOR
                        ER[k] = min(ER[k], de)
                        ER[new] = min(ER[new], de)
            k = j
    A = tot / (np.pi * rho * rho)
    if not np.isfinite(A) or A < 0.9 or unresolved:
        return A, -1, ns
    return A, status, ns


@njit(cache=True, error_model='numpy', fastmath=FM)
def _ld_f(s, q, y1, y2, rho, phi, reltol, n0, maxs, zwarm):
    """Integrand M(rho sin phi) sin phi / pi of the limb-darkening integral (M = magnified area of the disk)."""
    sp = np.sin(phi)
    r = rho * sp
    if r <= 0.0:
        return 0.0, 0
    Ak, stk, _ = mag_disk(s, q, y1, y2, r, reltol, n0, maxs, zwarm)
    return Ak * r * r * sp, stk


@njit(cache=True, error_model='numpy', fastmath=FM)
def mag_disk_ld(s, q, y1, y2, rho, reltol, ld, n0=8, maxs=2048, zwarm=None):
    """Linear limb darkening I(r) ~ 1 - u (1 - sqrt(1 - r^2/rho^2)) from uniform-disk contour integrations:
    with M(r) = A_uniform(r) * pi r^2 (magnified area of the disk of radius r),
    A_LD = [(1 - u) M(rho) + u * int_0^{pi/2} M(rho sin phi) sin phi dphi] / [pi rho^2 (1 - u/3)]
    (integration by parts; r = rho sin(phi) removes the sqrt singularity of the law at the limb).
    The phi integral is adaptive Simpson (M(r) has kinks where an annulus crosses the caustic: a fixed 8-node
    Gauss-Legendre rule was off by up to 5e-3 at caustic crossings); tolerance 1e-2 * reltol relative."""
    if ld != 0.0:
        reltol = min(reltol, LD_CTOL_MAX)     # LD amplifies per-annulus contour bias (grid ctol 0.1: 5e-3 at caustics)
    A1, st, ns = mag_disk(s, q, y1, y2, rho, reltol, n0, maxs, zwarm)
    if ld == 0.0 or st < 0:
        return A1, st, ns
    tol = 1e-2 * reltol * A1 * rho * rho
    # explicit stack of Simpson intervals: (a, b, fa, fm, fb, whole, depth)
    SA = np.empty(64); SB = np.empty(64); FA = np.empty(64); FM_ = np.empty(64); FB = np.empty(64)
    WH = np.empty(64); DP = np.empty(64, np.int64)
    a0, b0 = 0.0, 0.5 * np.pi
    fa, s1 = _ld_f(s, q, y1, y2, rho, a0, reltol, n0, maxs, zwarm)
    fm, s2 = _ld_f(s, q, y1, y2, rho, 0.5 * (a0 + b0), reltol, n0, maxs, zwarm)
    fb, s3 = _ld_f(s, q, y1, y2, rho, b0, reltol, n0, maxs, zwarm)
    if s1 < 0 or s2 < 0 or s3 < 0:
        return np.nan, -1, ns
    top = 0
    SA[0], SB[0], FA[0], FM_[0], FB[0], DP[0] = a0, b0, fa, fm, fb, 0
    WH[0] = (b0 - a0) / 6.0 * (fa + 4.0 * fm + fb)
    acc = 0.0
    while top >= 0:
        a, b, fa, fm, fb, whole, dep = SA[top], SB[top], FA[top], FM_[top], FB[top], WH[top], DP[top]
        top -= 1
        m = 0.5 * (a + b)
        flm, s1 = _ld_f(s, q, y1, y2, rho, 0.5 * (a + m), reltol, n0, maxs, zwarm)
        frm, s2 = _ld_f(s, q, y1, y2, rho, 0.5 * (m + b), reltol, n0, maxs, zwarm)
        ns += 2
        if s1 < 0 or s2 < 0:
            return np.nan, -1, ns
        left = (m - a) / 6.0 * (fa + 4.0 * flm + fm)
        right = (b - m) / 6.0 * (fm + 4.0 * frm + fb)
        if dep >= 12 or top >= 60 or abs(left + right - whole) < 15.0 * tol * (b - a) / (0.5 * np.pi):
            acc += left + right + (left + right - whole) / 15.0
        else:
            top += 1
            SA[top], SB[top], FA[top], FM_[top], FB[top], WH[top], DP[top] = m, b, fm, frm, fb, right, dep + 1
            top += 1
            SA[top], SB[top], FA[top], FM_[top], FB[top], WH[top], DP[top] = a, m, fa, flm, fm, left, dep + 1
    return ((1.0 - ld) * A1 * rho * rho + ld * acc) / (rho * rho * (1.0 - ld / 3.0)), st, ns


EXTRAPOLATE = 2      # 0 off, 1 linear (1.06x), 2 quadratic


@njit(cache=True, error_model='numpy', fastmath=FM)
def lc_full(ls, lq, u0, alpha, lr, tE, t0, t, tol, hexeps, ctol, ld=0.0):
    """Complete numba light curve: point source / quadrupole / hexadecapole (nbmag) and our contour integration
    for the rest. flag: 0 point source, 1 quad/hex, 3 own contour, 2 contour failed (A = nan, caller falls back)."""
    s = 10.0 ** ls
    q = 10.0 ** lq
    rho = 10.0 ** lr
    a, m1, m2 = nbmag._setup(s, q)
    sa, ca = np.sin(alpha), np.cos(alpha)
    n = t.shape[0]
    A = np.empty(n)
    flag = np.zeros(n, np.int8)
    c = np.empty(6, np.complex128)
    z = np.empty(5, np.complex128)
    zs = np.empty(5, np.complex128)
    imgs = np.empty(5, np.complex128)
    work = np.empty(5, np.complex128)
    par = np.empty(5)
    zp = np.empty(5, np.complex128)                  # roots two points back (for the extrapolated start)
    zpp = np.empty(5, np.complex128)                 # ... and three points back
    zk1 = np.empty(5, np.complex128)
    nbmag._init_roots(z, s)
    y2m1 = y2m2 = y2m3 = 0.0
    for k in range(n):
        tn = (t[k] - t0) / tE
        y1 = u0 * sa - tn * ca
        y2 = -u0 * ca - tn * sa
        for i in range(5):
            zk1[i] = z[i]
        # start Newton from the roots extrapolated in time: quadratic through the last three points, else
        # linear through two; _point works at |y2|, so only while y2 keeps its sign, and only for comparable steps
        if EXTRAPOLATE and k >= 2 and y2 * y2m1 > 0.0 and y2m1 * y2m2 > 0.0:
            dt0 = t[k - 1] - t[k - 2]
            dt1 = t[k] - t[k - 1]
            if dt0 > 0.0 and dt1 <= 3.0 * dt0:
                if EXTRAPOLATE == 2 and k >= 3 and y2m2 * y2m3 > 0.0 and t[k - 2] - t[k - 3] > 0.0 \
                        and dt1 <= 3.0 * (t[k - 2] - t[k - 3]):
                    x0, x1, x2, x = t[k - 3], t[k - 2], t[k - 1], t[k]
                    l0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
                    l1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
                    l2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))
                    for i in range(5):
                        z[i] = l0 * zpp[i] + l1 * zp[i] + l2 * z[i]
                else:
                    r = dt1 / dt0
                    for i in range(5):
                        z[i] = z[i] + r * (z[i] - zp[i])
        A[k], flag[k] = nbmag._one(a, m1, m2, s, q, rho, y1, y2, c, z, zs, tol, hexeps, imgs, par, work, ld)
        if flag[k] == 2:
            Ak, st, _ = mag_disk_ld(s, q, y1, y2, rho, ctol, ld, 8, 2048, z)
            if st >= 0:
                A[k], flag[k] = Ak, 3
        for i in range(5):
            zpp[i] = zp[i]
            zp[i] = zk1[i]
        y2m3, y2m2, y2m1 = y2m2, y2m1, y2
    return A, flag


@njit(cache=True, error_model='numpy', fastmath=FM)
def mags_full(s_arr, q_arr, y1_arr, y2_arr, rho_arr, tol, hexeps, ctol, ld=0.0):
    n = s_arr.shape[0]
    A = np.empty(n)
    flag = np.zeros(n, np.int8)
    for k in range(n):
        a, b = nbmag.mags_hybrid(s_arr[k:k + 1], q_arr[k:k + 1], y1_arr[k:k + 1], y2_arr[k:k + 1], rho_arr[k:k + 1],
                                 tol, hexeps, ld)
        A[k], flag[k] = a[0], b[0]
        if flag[k] == 2:
            Ak, st, _ = mag_disk_ld(s_arr[k], q_arr[k], y1_arr[k], y2_arr[k], rho_arr[k], ctol, ld, 8, 2048)
            if st >= 0:
                A[k], flag[k] = Ak, 3
    return A, flag


@njit(cache=True, error_model='numpy', fastmath=FM)
def lc_traj(s_arr, q, rho, y1_arr, y2_arr, tol, hexeps, ctol, ld=0.0):
    """Light curve along a general trajectory (parallax, orbital motion): per-point separation s_arr and source
    position (y1, y2) in the centre-of-mass frame; roots warm-started from the previous point as in lc_full."""
    n = s_arr.shape[0]
    A = np.empty(n)
    flag = np.zeros(n, np.int8)
    c = np.empty(6, np.complex128)
    z = np.empty(5, np.complex128)
    zs = np.empty(5, np.complex128)
    imgs = np.empty(5, np.complex128)
    work = np.empty(5, np.complex128)
    par = np.empty(5)
    nbmag._init_roots(z, s_arr[0])
    for k in range(n):
        s = s_arr[k]
        a, m1, m2 = nbmag._setup(s, q)
        A[k], flag[k] = nbmag._one(a, m1, m2, s, q, rho, y1_arr[k], y2_arr[k], c, z, zs, tol, hexeps, imgs, par,
                                   work, ld)
        if flag[k] == 2:
            Ak, st, _ = mag_disk_ld(s, q, y1_arr[k], y2_arr[k], rho, ctol, ld, 8, 2048, z)
            if st >= 0:
                A[k], flag[k] = Ak, 3
    return A, flag

