"""Numba binary-lens magnification: point source + VBBL quadrupole test + Gould (2008) hexadecapole.

Geometry and coefficients follow VBBinaryLensing 3.7 (BinaryMag0 / NewImages / BinaryMag2) exactly:
centre-of-mass frame, primary on the left, planet on the right, y2 -> |y2| by symmetry.
Points that fail both the point-source test and the hexadecapole test are flagged for the
full contour integration (VBBL BinaryMag2) by the caller.
"""
import numpy as np
from numba import njit

# no zero-division checks and 'safe' fast-math (no nnan/ninf: the code relies on inf/nan semantics)
FM = {'nsz', 'arcp', 'contract', 'afn', 'reassoc'}  # nsz is safe since the parity-sum fix (contour._fix_parity)

DLMIN = 1.0e-4


@njit(cache=True, inline="always", error_model="numpy", fastmath=FM)
def _cabs(z):
    """|z| without libm hypot (perf: hypot was 22 % of runtime); magnitudes here are far from overflow."""
    return np.sqrt(z.real * z.real + z.imag * z.imag)


@njit(cache=True, inline="always", error_model="numpy", fastmath=FM)
def _cabs2(z):
    return z.real * z.real + z.imag * z.imag


@njit(cache=True, inline="always", error_model="numpy", fastmath=FM)
def _cinv(b):
    """1/b as conj(b)/|b|^2: numba's complex division is an out-of-line, overflow-safe routine (perf: ~4 % + call
    overhead in every hot function); our magnitudes are moderate."""
    d = b.real * b.real + b.imag * b.imag
    return complex(b.real / d, -b.imag / d)


@njit(cache=True, inline="always", error_model="numpy", fastmath=FM)
def _cdiv(a, b):
    return a * _cinv(b)
HEX_GHOST_MAX = 0.01
QUAD_EPS = 0.003
AQUAD_EPS = 1e-3                        # analytic-quadrupole acceptance (stress set: 95 % of quad/hex points, max err 6e-4)


@njit(cache=True, error_model='numpy', fastmath=FM)
def _setup(s, q):
    """Return a, m1, m2 (complex) as VBBL: q<1 -> a = -s; q>1 -> a = s, q -> 1/q."""
    if q < 1.0:
        a = complex(-s, 0.0)
        qq = q
    else:
        a = complex(s, 0.0)
        qq = 1.0 / q
    m1 = 1.0 / (1.0 + qq)
    m2 = qq * m1
    return a, complex(m1, 0.0), complex(m2, 0.0)


@njit(cache=True, error_model='numpy', fastmath=FM)
def _coefs(a, m1, m2, y, c):
    """Quintic coefficients c[0..5] (ascending powers) for source position y (already shifted)."""
    yc = y.conjugate()
    a2 = a * a
    a3 = a2 * a
    c[0] = a2 * m2 * m2 * y
    c[1] = a * m2 * (a * (m1 + y * (2 * yc - a)) - 2 * y)
    c[2] = y * (1 - a3 * yc) - a * (m1 + 2 * y * yc * (1 + m2)) + a2 * (yc * (m1 - m2) + y * (1 + m2 + yc * yc))
    c[3] = 2 * y * yc + a3 * yc + a2 * (yc * (2 * y - yc) - m1) - a * (y + 2 * yc * (yc * y - m2))
    c4 = yc * (2 * a + y)
    c[4] = yc * (c4 - 1) - a * (c4 - m1)
    c[5] = yc * (a - yc)


@njit(cache=True, error_model='numpy', fastmath=FM)
def _roots(c, z):
    """Aberth-Ehrlich for the quintic, warm-started from z (in/out), then Newton polish."""
    n = 5
    for it in range(200):
        maxcorr = 0.0
        for i in range(n):
            p = c[5]
            dp = 0j
            for k in range(4, -1, -1):
                dp = dp * z[i] + p
                p = p * z[i] + c[k]
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
    for i in range(n):                      # polish with the full polynomial
        for it in range(1):
            p = c[5]
            dp = 0j
            for k in range(4, -1, -1):
                dp = dp * z[i] + p
                p = p * z[i] + c[k]
            if dp == 0:
                break
            z[i] -= _cdiv(p, dp)


@njit(cache=True, error_model='numpy', fastmath=FM)
def _roots_newton(c, z):
    """Per-root Newton from the previous point's roots (they move little between consecutive data points).
    Falls back to Aberth if a root does not converge in 6 steps or two roots coincide. Tests on squared moduli
    (no sqrt): |step| < 1e-10 (1 + |z|) <=> |step|^2 < 1e-20 (1 + |z|)^2."""
    z00, z01, z02, z03, z04 = z[0], z[1], z[2], z[3], z[4]
    for i in range(5):
        conv = False
        zi = z[i]
        for it in range(6):
            p = c[5]
            dp = 0j
            for k in range(4, -1, -1):
                dp = dp * zi + p
                p = p * zi + c[k]
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
            _roots(c, z)
            return
    for i in range(5):
        for j in range(i):
            if _cabs2(z[i] - z[j]) < 1e-14 * (1.0 + _cabs2(z[i])):
                z[0], z[1], z[2], z[3], z[4] = z00, z01, z02, z03, z04
                _roots(c, z)
                return


NEWTON_ROOTS = True


@njit(cache=True, error_model='numpy', fastmath=FM)
def _point(a, m1, m2, s, q, y1, y2, c, z, need_test):
    """Point-source magnification at (y1, |y2|). Returns (mag, nimg, corrquad, corrquad2, safedist, ok, lap);
    lap = source-plane Laplacian of the total magnification (need_test only)."""
    y = complex(y1, abs(y2)) + a * m1
    yc = y.conjugate()
    _coefs(a, m1, m2, y, c)
    if NEWTON_ROOTS:
        _roots_newton(c, z)
    else:
        _roots(c, z)
    # lens-equation residuals; track the three worst roots without allocating (w1 worst)
    w1 = w2 = w3 = -1
    g1 = g2 = g3 = -1.0
    for i in range(5):
        zz = z[i]
        zc = zz.conjugate()
        g = _cabs2((y - zz) + m1 * _cinv(zc - a) + m2 * _cinv(zc))      # squared residual (ranking only)
        if g > g1:
            w3, g3, w2, g2, w1, g1 = w2, g2, w1, g1, i, g
        elif g > g2:
            w3, g3, w2, g2 = w2, g2, i, g
        elif g > g3:
            w3, g3 = i, g
    three = np.sqrt(g2) * DLMIN > np.sqrt(g3) + 1e-12
    mag = 0.0
    corrquad = 0.0
    corrquad2 = 0.0
    lap = 0.0
    checkjac = 0
    nimg = 0
    for i in range(5):
        if three and (i == w1 or i == w2):
            continue
        zz = z[i]
        ia = _cinv(zz - a)                       # two inversions per image; powers by multiplication
        ib = _cinv(zz)
        ia2 = ia * ia
        ib2 = ib * ib
        J1 = m1 * ia2 + m2 * ib2
        J1c = J1.conjugate()
        dJ = 1.0 - (J1 * J1c).real
        mag += 1.0 / abs(dJ)
        nimg += 1
        checkjac += (1 if dJ > 0 else -1) if abs(dJ) > 1e-7 else 10
        if need_test:
            J2 = -2.0 * (m1 * ia2 * ia + m2 * ib2 * ib)
            J3 = 6.0 * (m1 * ia2 * ia2 + m2 * ib2 * ib2)
            # Laplacian of mu = 1/detJ in the source plane (implicit differentiation of zeta = z - g(conj z)):
            # d_zeta mu = mu^3 G, G = J2 J1* - J1^2 J2*, d_zetabar = mu (d_zbar - J1* d_z), lap = 4 d_zeta d_zetabar mu
            J2c = J2.conjugate()
            J3c = J3.conjugate()
            mu = 1.0 / dJ
            mu3 = mu * mu * mu
            G = J2 * J1c - J1 * J1 * J2c
            dzF = 3.0 * mu3 * mu * J2 * J1c * G + mu3 * (J3 * J1c - 2.0 * J1 * J2 * J2c)
            dzbF = 3.0 * mu3 * mu * J1 * J2c * G + mu3 * (J2 * J2c - J1 * J1 * J3c)
            lap += (1.0 if dJ > 0 else -1.0) * 4.0 * (mu * (dzbF - J1c * dzF)).real
            dJ2 = dJ * dJ
            J1c2 = J1c * J1c
            J3 = J3 * J1c2
            ob2 = (J2.real * J2.real + J2.imag * J2.imag) * (6 - 6 * dJ + dJ2)
            J2b = J2 * J2 * J1c2 * J1c
            corrquad += 0.5 * (abs(ob2 - 6.0 * J2b.real - 2.0 * J3.real * dJ) + 3 * abs(J2b.imag)) / abs(dJ * dJ2 * dJ2)
    if need_test and three:
        for i in (w1, w2):
            zz = z[i]
            dza = zz - a
            za2 = dza * dza
            zb2 = zz * zz
            J1 = m1 * _cinv(za2) + m2 * _cinv(zb2)
            J1c = J1.conjugate()
            dJ = 1.0 - (J1 * J1c).real
            J2 = -2.0 * (m1 * _cinv(za2 * dza) + m2 * _cinv(zb2 * zz))
            zaltc = yc + m1 * _cinv(dza) + m2 * _cinv(zz)
            zb = zaltc - a
            Jaltc = m1 * _cinv(zb * zb) + m2 * _cinv(zaltc * zaltc)
            Jalt = Jaltc.conjugate()
            JJalt2 = 1 - J1c * Jalt
            J3 = J2 * J1c * JJalt2
            J3 = _cdiv(J3 - J3.conjugate() * Jalt, JJalt2 * JJalt2 * dJ)
            cq = 0.0 if dJ < -100 else (J3.real * J3.real + J3.imag * J3.imag)
            if cq > corrquad2:
                corrquad2 = cq
    ok = True
    if checkjac != -1:
        if abs(mag - 1.0) < 1e-2:           # far field: VBBL accepts and zeroes corrquad
            corrquad = 0.0
        else:
            ok = False
    safedist = 10.0
    qq = q if q < 1 else 1.0 / q
    if qq < 0.01:
        safedist = (y1 + (a * m1).real - 1.0 / a.real) ** 2 + y2 * y2 - 36.0 * q / (s * s)
    return mag, nimg, corrquad, corrquad2, safedist, ok, lap


@njit(cache=True, error_model='numpy', fastmath=FM)
def _true_images(a, m1, m2, y, z, imgs, par):
    """Select the true images among the 5 roots z (same rule as _point); fill imgs/par (sign of det J). Returns n."""
    w1 = w2 = w3 = -1
    g1 = g2 = g3 = -1.0
    for i in range(5):
        zz = z[i]
        zc = zz.conjugate()
        g = _cabs2((y - zz) + m1 * _cinv(zc - a) + m2 * _cinv(zc))      # squared residual (ranking only)
        if g > g1:
            w3, g3, w2, g2, w1, g1 = w2, g2, w1, g1, i, g
        elif g > g2:
            w3, g3, w2, g2 = w2, g2, i, g
        elif g > g3:
            w3, g3 = i, g
    three = np.sqrt(g2) * DLMIN > np.sqrt(g3) + 1e-12
    n = 0
    for i in range(5):
        if three and (i == w1 or i == w2):
            continue
        zz = z[i]
        J1 = m1 * _cinv((zz - a) * (zz - a)) + m2 * _cinv(zz * zz)
        imgs[n] = zz
        par[n] = 1.0 if 1.0 - (J1 * J1.conjugate()).real > 0 else -1.0
        n += 1
    return n


@njit(cache=True, error_model='numpy', fastmath=FM)
def _tracked_mag(a, m1, m2, y, dy, imgs, par, n, work):
    """Magnification at source y + dy by following the n images of y: first-order step + Newton on the
    lens equation. Returns (mag, ok); ok = False if an image does not converge, flips parity or two merge."""
    mag = 0.0
    for i in range(n):
        zz = imgs[i]
        J1 = m1 * _cinv((zz - a) * (zz - a)) + m2 * _cinv(zz * zz)
        J1c = J1.conjugate()
        dJ = 1.0 - (J1 * J1c).real
        zz = zz + (dy - J1c * dy.conjugate()) / dJ                  # first-order image displacement
        target = y + dy
        conv = False
        for it in range(8):
            zc = zz.conjugate()
            r = target - (zz - m1 * _cinv(zc - a) - m2 * _cinv(zc))
            J1 = m1 * _cinv((zz - a) * (zz - a)) + m2 * _cinv(zz * zz)
            J1c = J1.conjugate()
            dJ = 1.0 - (J1 * J1c).real
            step = (r - J1c * r.conjugate()) / dJ
            zz = zz + step
            if _cabs(step) < 1e-12 * (1.0 + _cabs(zz)):
                conv = True
                break
        if not conv:
            return 0.0, False
        J1 = m1 * _cinv((zz - a) * (zz - a)) + m2 * _cinv(zz * zz)
        dJ = 1.0 - (J1 * J1.conjugate()).real
        if dJ * par[i] <= 0 or abs(dJ) < 1e-9:
            return 0.0, False
        for j in range(i):
            if _cabs(work[j] - zz) < 1e-9 * (1.0 + _cabs(zz)):
                return 0.0, False
        work[i] = zz
        mag += 1.0 / abs(dJ)
    return mag, True


@njit(cache=True, error_model='numpy', fastmath=FM)
def _hex_tracked(a, m1, m2, rho, y1, y2, z, mag0, hexeps, imgs, par, work, gam=0.0):
    """Hexadecapole from tracked images. Returns (A, status): 1 accepted, 0 rejected by hexeps, -1 tracking failed."""
    y = complex(y1, abs(y2)) + a * m1
    sg = 1.0 if y2 >= 0 else -1.0
    n = _true_images(a, m1, m2, y, z, imgs, par)
    # quadrupole tier. The mean of 4 samples at r = rho/sqrt(2) is A2 rho^2/2 + A4 rho^4/4, i.e. the
    # uniform-disk value up to A4 rho^4/12; accept when that correction is small (|M| < QUAD_EPS * A0; on the
    # 118k-point stress set 98 % of guarded hexadecapole points, max error 4e-4)
    r = rho * 0.7071067811865476
    sq = 0.0
    for j in range(4):
        ang = 0.5 * np.pi * j
        m, ok = _tracked_mag(a, m1, m2, y, complex(r * np.cos(ang), sg * r * np.sin(ang)), imgs, par, n, work)
        if not ok:
            return 0.0, -1
        sq += m
    M = 0.25 * sq - mag0
    if abs(M) < QUAD_EPS * mag0:
        return mag0 + M * (1.0 - 0.2 * gam), 1
    sp = 0.0
    sx = 0.0
    sh = 0.0
    for j in range(4):
        ang = 0.5 * np.pi * j
        ca, sa = np.cos(ang), np.sin(ang)
        m, ok = _tracked_mag(a, m1, m2, y, complex(rho * ca, sg * rho * sa), imgs, par, n, work)
        if not ok:
            return 0.0, -1
        sp += m
        m, ok = _tracked_mag(a, m1, m2, y, complex(0.5 * rho * ca, sg * 0.5 * rho * sa), imgs, par, n, work)
        if not ok:
            return 0.0, -1
        sh += m
        cx, sx_ = np.cos(ang + 0.25 * np.pi), np.sin(ang + 0.25 * np.pi)
        m, ok = _tracked_mag(a, m1, m2, y, complex(rho * cx, sg * rho * sx_), imgs, par, n, work)
        if not ok:
            return 0.0, -1
        sx += m
    Ap = 0.25 * sp - mag0
    Ax = 0.25 * sx - mag0
    Ah = 0.25 * sh - mag0
    A2 = (16.0 * Ah - Ap) / 3.0
    A4 = 0.5 * (Ap + Ax) - A2
    if abs(A4 / 3.0) < hexeps * mag0:
        return mag0 + 0.5 * A2 * (1.0 - 0.2 * gam) + A4 / 3.0 * (1.0 - 11.0 * gam / 35.0), 1
    return 0.0, 0


FAST_HEX = True


@njit(cache=True, error_model='numpy', fastmath=FM)
def _one(a, m1, m2, s, q, rho, y1, y2, c, z, zs, tol, hexeps, imgs, par, work, ld=0.0, fast=True):
    """Hybrid magnification of one point. Returns (A, flag): 0 point source, 1 hexadecapole, 2 contour."""
    rho2 = rho * rho
    mag0, nimg, cq, cq2_raw, sd, ok, lap = _point(a, m1, m2, s, q, y1, y2, c, z, True)
    cq *= 6 * (rho2 + 1e-4 * tol)
    cq2 = cq2_raw * (rho + 1e-3)
    if ok and cq < tol and cq2 < 1 and sd > 4 * rho2:
        return mag0, 0
    # analytic quadrupole tier (uniform disk: A0 + rho^2/8 * lap A); remainder ~ A4 rho^4 / 3
    # limb darkening (linear law, coefficient u = ld): Gould (2008) Gamma = 2u/(3-u); the rho^2 term scales by
    # (1 - Gamma/5), the rho^4 term by (1 - 11 Gamma/35)
    gam = 2.0 * ld / (3.0 - ld)
    corr = 0.125 * rho2 * lap * (1.0 - 0.2 * gam)
    if AQUAD_EPS > 0 and ok and cq2_raw * rho2 < HEX_GHOST_MAX and abs(corr) < AQUAD_EPS * mag0:
        return mag0 + corr, 1
    # hexadecapole (Gould 2008), uniform source; reject if any sample changes image number, and near cusps:
    # the caustic interior can reach into the disk between the 12 samples (3-image case, errors up to 24 %);
    # the ghost-image term separates those (all bad cases had cq2_raw*rho^2 >= 0.021 in a 118k-point stress test)
    if hexeps > 0 and ok and cq2_raw * rho2 < HEX_GHOST_MAX:
        if fast:
            A_h, st = _hex_tracked(a, m1, m2, rho, y1, y2, z, mag0, hexeps, imgs, par, work, gam)
            if st == 1:
                return A_h, 1
            if st == 0:
                return np.nan, 2
        for i in range(5):
            zs[i] = z[i]
        bad = False
        sp = 0.0
        sx = 0.0
        sh = 0.0
        for j in range(4):
            ang = 0.5 * np.pi * j
            ca, sa = np.cos(ang), np.sin(ang)
            m, nn, _, _, _, ok2, _ = _point(a, m1, m2, s, q, y1 + rho * ca, y2 + rho * sa, c, zs, False)
            bad = bad or nn != nimg or not ok2
            sp += m
            m, nn, _, _, _, ok2, _ = _point(a, m1, m2, s, q, y1 + 0.5 * rho * ca, y2 + 0.5 * rho * sa, c, zs, False)
            bad = bad or nn != nimg or not ok2
            sh += m
            cx, sx_ = np.cos(ang + 0.25 * np.pi), np.sin(ang + 0.25 * np.pi)
            m, nn, _, _, _, ok2, _ = _point(a, m1, m2, s, q, y1 + rho * cx, y2 + rho * sx_, c, zs, False)
            bad = bad or nn != nimg or not ok2
            sx += m
        if not bad:
            Ap = 0.25 * sp - mag0
            Ax = 0.25 * sx - mag0
            Ah = 0.25 * sh - mag0
            A2 = (16.0 * Ah - Ap) / 3.0
            A4 = 0.5 * (Ap + Ax) - A2
            if abs(A4 / 3.0) < hexeps * mag0:
                return mag0 + 0.5 * A2 * (1.0 - 0.2 * gam) + A4 / 3.0 * (1.0 - 11.0 * gam / 35.0), 1
    return np.nan, 2


@njit(cache=True, error_model='numpy', fastmath=FM)
def _init_roots(z, s):
    for i in range(5):
        z[i] = complex(np.cos(2 * np.pi * i / 5 + 0.3), np.sin(2 * np.pi * i / 5 + 0.3)) * (1.0 + s)


@njit(cache=True, error_model='numpy', fastmath=FM)
def lc_hybrid(ls, lq, u0, alpha, lr, tE, t0, t, tol, hexeps):
    """Light curve. Returns (A, flag): flag 0 point source, 1 hexadecapole, 2 needs contour (A = nan)."""
    s = 10.0 ** ls
    q = 10.0 ** lq
    rho = 10.0 ** lr
    a, m1, m2 = _setup(s, q)
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
    _init_roots(z, s)
    for k in range(n):
        tn = (t[k] - t0) / tE
        A[k], flag[k] = _one(a, m1, m2, s, q, rho, u0 * sa - tn * ca, -u0 * ca - tn * sa, c, z, zs, tol, hexeps,
                             imgs, par, work)
    return A, flag


@njit(cache=True, error_model='numpy', fastmath=FM)
def mags_hybrid(s_arr, q_arr, y1_arr, y2_arr, rho_arr, tol, hexeps, ld=0.0):
    """Hybrid magnifications for independent samples (fresh root guesses each)."""
    n = s_arr.shape[0]
    A = np.empty(n)
    flag = np.zeros(n, np.int8)
    c = np.empty(6, np.complex128)
    z = np.empty(5, np.complex128)
    zs = np.empty(5, np.complex128)
    imgs = np.empty(5, np.complex128)
    work = np.empty(5, np.complex128)
    par = np.empty(5)
    for k in range(n):
        a, m1, m2 = _setup(s_arr[k], q_arr[k])
        _init_roots(z, s_arr[k])
        A[k], flag[k] = _one(a, m1, m2, s_arr[k], q_arr[k], rho_arr[k], y1_arr[k], y2_arr[k], c, z, zs, tol, hexeps,
                             imgs, par, work, ld)
    return A, flag


@njit(cache=True, error_model='numpy', fastmath=FM)
def mags_point(s_arr, q_arr, y1_arr, y2_arr):
    """Point-source magnifications for arrays (testing)."""
    n = s_arr.shape[0]
    out = np.empty(n)
    c = np.empty(6, np.complex128)
    z = np.empty(5, np.complex128)
    for k in range(n):
        a, m1, m2 = _setup(s_arr[k], q_arr[k])
        for i in range(5):
            z[i] = complex(np.cos(2 * np.pi * i / 5 + 0.3), np.sin(2 * np.pi * i / 5 + 0.3)) * (1.0 + s_arr[k])
        out[k] = _point(a, m1, m2, s_arr[k], q_arr[k], y1_arr[k], y2_arr[k], c, z, False)[0]
    return out
