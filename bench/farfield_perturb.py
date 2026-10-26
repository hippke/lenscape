"""Far-field (perturbative) binary-lens magnification, prototype for RESEARCH_PLAN_point_source.md part 1.

Frame of nbmag: primary (mass m1) at z = a (a = -s for q < 1), planet (mass m2 = q m1) at z = 0, source y in that
frame (y = (y1, |y2|) + a m1 from the centre-of-mass position). Lens equation zeta = z - m1/conj(z - a) - m2/conj(z).

Approximations, in increasing accuracy:
  single : the primary alone (mass m1), A = PSPL(|eta| / sqrt(m1)), eta = y - a (source relative to the primary).
  pert0  : first-order perturbation theory. The two single-lens images w+- of the primary are displaced by the
           planet's deflection eps = m2 / conj(z0) at the image: dw = (eps - kappa conj eps) / (1 - |kappa|^2),
           kappa = m1 / conj(w0)^2 (the linearised single-lens Jacobian); the planet's own image sits at
           z = -a - m2 / conj(u), u = eta - eta_p, eta_p = -a - m1/(-a) (the planet's position mapped to the source
           plane; Chang-Refsdal far field of a point mass m2 with the primary's shear). Magnifications from the exact
           binary Jacobian at the three approximate image positions.
  pertK  : pert0 followed by K Newton steps on the full lens equation for each of the three images (quadratic
           convergence far from the caustics), then the exact Jacobians.
  far_roots: pertK with convergence test, plus the two remaining roots of the quintic by deflation (closed form)
           so that nbmag._point's ghost tests (three-image test, corrquad2) can run unchanged on the 5 roots.
"""
import numpy as np
from numba import njit
from lenscape import nbmag
from lenscape.nbmag import _cinv, _cabs2, _coefs, FM


@njit(cache=True, error_model='numpy', fastmath=FM)
def pspl(u):
    return (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))


@njit(cache=True, error_model='numpy', fastmath=FM)
def single_primary(a, m1, y):
    eta = y - a
    r = np.sqrt(_cabs2(eta))
    return pspl(r / np.sqrt(m1.real))


@njit(cache=True, error_model='numpy', fastmath=FM)
def _starts(a, m1, m2, y, z):
    """Three approximate images (first-order perturbation theory) into z[0..2]. Returns False if degenerate."""
    eta = y - a
    r2 = _cabs2(eta)
    if r2 < 1e-30:
        return False
    r = np.sqrt(r2)
    sq = np.sqrt(r2 + 4.0 * m1.real)
    ehat = eta / r
    for k in range(2):
        w0 = (0.5 * (r + sq) if k == 0 else 0.5 * (r - sq)) * ehat
        z0 = w0 + a
        eps = m2 * _cinv(z0.conjugate())                 # planet's deflection at the unperturbed image
        kappa = m1 * _cinv((w0 * w0).conjugate())
        dj = 1.0 - _cabs2(kappa)
        dw = (eps - kappa * eps.conjugate()) / dj
        z[k] = z0 + dw
    wp = -a                                              # planet position relative to the primary
    eta_p = wp - m1 * _cinv(wp.conjugate())
    u = eta - eta_p
    if _cabs2(u) < 1e-30:
        return False
    z[2] = -a * 0.0 - m2 * _cinv(u.conjugate())          # planet at z = 0 in this frame
    return True


@njit(cache=True, error_model='numpy', fastmath=FM)
def _newton(a, m1, m2, y, zz, maxit, tol2):
    """Newton on the lens equation from zz. Returns (z, converged, iterations)."""
    for it in range(maxit):
        zc = zz.conjugate()
        ia = _cinv(zc - a.conjugate())
        ib = _cinv(zc)
        r = y - (zz - m1 * ia - m2 * ib)
        J1c = m1 * ia * ia + m2 * ib * ib                 # conj(J1)
        dJ = 1.0 - _cabs2(J1c)
        step = (r - J1c * r.conjugate()) / dJ
        zz = zz + step
        if _cabs2(step) < tol2 * (1.0 + _cabs2(zz)):
            return zz, True, it + 1
    return zz, False, maxit


@njit(cache=True, error_model='numpy', fastmath=FM)
def _mag3(a, m1, m2, z):
    """Sum of |1/det J| over z[0..2] and the parity sum."""
    mag = 0.0
    par = 0
    for i in range(3):
        ia = _cinv(z[i] - a)
        ib = _cinv(z[i])
        J1 = m1 * ia * ia + m2 * ib * ib
        dJ = 1.0 - _cabs2(J1)
        mag += 1.0 / abs(dJ)
        par += 1 if dJ > 0 else -1
    return mag, par


@njit(cache=True, error_model='numpy', fastmath=FM)
def pert_mag(a, m1, m2, y, z, npolish):
    """pert0 (npolish = 0) or pertK magnification. Returns (A, ok, total Newton iterations)."""
    if not _starts(a, m1, m2, y, z):
        return 0.0, False, 0
    nit = 0
    if npolish > 0:
        for i in range(3):
            zi, conv, k = _newton(a, m1, m2, y, z[i], npolish, 1e-24)
            z[i] = zi
            nit += k
    mag, par = _mag3(a, m1, m2, z)
    return mag, True, nit


@njit(cache=True, error_model='numpy', fastmath=FM)
def far_roots(a, m1, m2, y, c, z, maxit, tol2, gmin):
    """Five quintic roots from the far-field starts: three images by Newton on the lens equation (converged to
    |step|^2 < tol2 (1 + |z|^2), at most maxit steps each), the other two by deflating the quintic. Accepts only
    when the three images are distinct, their parities sum to -1 (3-image topology: +, -, -), and the two deflated
    roots have lens-equation residuals above gmin (ghosts, i.e. the source is outside the caustic by a margin).
    Returns (ok, Newton iterations). On failure z is left unspecified."""
    if not _starts(a, m1, m2, y, z):
        return False, 0
    nit = 0
    for i in range(3):
        zi, conv, k = _newton(a, m1, m2, y, z[i], maxit, tol2)
        nit += k
        if not conv:
            return False, nit
        z[i] = zi
    for i in range(3):
        for j in range(i):
            if _cabs2(z[i] - z[j]) < 1e-16 * (1.0 + _cabs2(z[i])):
                return False, nit
    par = 0
    for i in range(3):
        ia = _cinv(z[i] - a)
        ib = _cinv(z[i])
        J1 = m1 * ia * ia + m2 * ib * ib
        dJ = 1.0 - _cabs2(J1)
        par += 1 if dJ > 0 else -1
    if par != -1:
        return False, nit
    # deflate: p(z) / ((z - z0)(z - z1)(z - z2)) by three synthetic divisions (coefficients ascending in c)
    _coefs(a, m1, m2, y, c)
    b5, b4, b3, b2, b1, b0 = c[5], c[4], c[3], c[2], c[1], c[0]
    # division by (z - r): descending Horner, q_{n-1} = b_n, q_{k} = b_{k+1} + r q_{k+1}
    r = z[0]
    q4 = b5; q3 = b4 + r * q4; q2 = b3 + r * q3; q1 = b2 + r * q2; q0 = b1 + r * q1
    r = z[1]
    p3 = q4; p2 = q3 + r * p3; p1 = q2 + r * p2; p0 = q1 + r * p1
    r = z[2]
    s2 = p3; s1 = p2 + r * s2; s0 = p1 + r * s1
    # quadratic s2 z^2 + s1 z + s0
    disc = s1 * s1 - 4.0 * s2 * s0
    sq = np.sqrt(disc)
    # stable pair: the larger-magnitude root first, the other by Vieta
    if (s1.real * sq.real + s1.imag * sq.imag) >= 0.0:
        t = -0.5 * (s1 + sq)
    else:
        t = -0.5 * (s1 - sq)
    if _cabs2(t) < 1e-300:
        return False, nit
    z[3] = t * _cinv(s2)
    z[4] = s0 * _cinv(t)
    # ghost residuals: both must be clearly non-images
    for i in (3, 4):
        zc = z[i].conjugate()
        g = _cabs2((y - z[i]) + m1 * _cinv(zc - a) + m2 * _cinv(zc))
        if not (g > gmin * gmin):
            return False, nit
    return True, nit


@njit(cache=True, error_model='numpy', fastmath=FM)
def eval_grid(s, q, y1, y2, npolish, maxit, tol2, gmin):
    """For arrays of centre-of-mass source positions at fixed (s, q): exact point-source A (nbmag._point with fresh
    starts, regular solve), the single-primary A, the pert(npolish) A, and the magnification from the package's
    nbmag._roots_far with its ok flag (maxit, tol2, gmin arguments are kept for the prototype far_roots above but
    unused here). Returns (A_exact, A_single, A_pert, A_far, ok_far, nit_far (0), g_ghost_min)."""
    n = y1.shape[0]
    a, m1, m2 = nbmag._setup(s, q)
    c = np.empty(6, np.complex128)
    z = np.empty(5, np.complex128)
    zf = np.empty(5, np.complex128)
    Ae = np.empty(n); As = np.empty(n); Ap = np.empty(n); Af = np.empty(n)
    ok = np.zeros(n, np.bool_); nit = np.zeros(n, np.int64); gg = np.empty(n)
    for k in range(n):
        nbmag._init_roots(z, s)
        Ae[k] = nbmag._point(a, m1, m2, s, q, y1[k], y2[k], c, z, False, False)[0]
        y = complex(y1[k], abs(y2[k])) + a * m1
        As[k] = single_primary(a, m1, y)
        Ap[k], okp, _ = pert_mag(a, m1, m2, y, zf, npolish)
        if not okp:
            Ap[k] = np.nan
        okk = nbmag._roots_far(a, m1, m2, y, c, zf)              # the package's solve (maxit, tol2, gmin: nbmag.FAR_*)
        nk = 0
        ok[k] = okk; nit[k] = nk
        if okk:
            # the magnification as _point would compute it from these 5 roots (true images by the residual ranking)
            for i in range(5):
                z[i] = zf[i]
            # _point re-solves; instead evaluate the 3-image sum directly (the 5 roots are checked in _point later)
            Af[k], _ = _mag3(a, m1, m2, zf)
            gmn = 1e300
            for i in (3, 4):
                zc = zf[i].conjugate()
                g = _cabs2((y - zf[i]) + m1 * _cinv(zc - a) + m2 * _cinv(zc))
                gmn = min(gmn, g)
            gg[k] = np.sqrt(gmn)
        else:
            Af[k] = np.nan; gg[k] = np.nan
    return Ae, As, Ap, Af, ok, nit, gg
