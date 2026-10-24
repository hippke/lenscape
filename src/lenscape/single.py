"""Single lens: point source, finite source (uniform and linear limb darkening), and the binary-source model 1L2S.

Uniform disk (Lee et al. 2009): with int_0^R A_ps(r) r dr = R sqrt(R^2 + 4) / 2, the magnified disk area is a 1-D
integral over the polar angle about the lens:
  u > rho:  A = 1/(pi rho^2) int_0^thmax [u2 sqrt(u2^2+4) - u1 sqrt(u1^2+4)] dth,  u1,2 = u cos th -+ sqrt(rho^2 - u^2 sin^2 th)
  u <= rho: A = 1/(pi rho^2) int_0^pi u2 sqrt(u2^2+4) dth
For u > rho the substitution sin th = (rho/u) sin phi removes the sqrt singularity at thmax. Linear limb darkening
uses the same annulus formula as the binary-lens code (contour.mag_disk_ld).

1L2S parameters p = (t0_1, u0_1, t0_2, u0_2, tE, log10 rho_1, log10 rho_2, log10 qF); A = (A1 + qF A2) / (1 + qF).
"""
import numpy as np
from numba import njit
from . import model

FM = {'nsz', 'arcp', 'contract', 'afn', 'reassoc'}  # nsz is safe since the parity-sum fix (contour._fix_parity)
_X, _W = np.polynomial.legendre.leggauss(96)
GX = 0.5 * (_X + 1.0)                 # nodes on [0, 1]
GW = 0.5 * _W
_XL, _WL = np.polynomial.legendre.leggauss(8)
LD_X = 0.5 * (_XL + 1.0)              # nodes on [0, 1] for the annulus integral (per sub-interval)
LD_WX = 0.5 * _WL


@njit(cache=True, error_model='numpy', fastmath=FM)
def pspl(u):
    return (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))


@njit(cache=True, error_model='numpy', fastmath=FM)
def _F(x):
    return x * np.sqrt(x * x + 4.0)


@njit(cache=True, error_model='numpy', fastmath=FM)
def espl_uniform(u, rho):
    """Uniform-disk single-lens magnification. For u > 20 rho the analytic quadrupole A + rho^2/8 lap(A) with
    lap(A) = A'' + A'/u = 8/(u^3 (u^2+4)^1.5) + 24/(u (u^2+4)^2.5) (next term ~ (rho/u)^4, < 1e-5)."""
    u = abs(u)
    if u > 20.0 * rho:
        w = u * u + 4.0
        lap = 8.0 / (u ** 3 * w ** 1.5) + 24.0 / (u * w ** 2.5)
        return pspl(u) + 0.125 * rho * rho * lap
    acc = 0.0
    if u > rho:
        a = rho / u
        for k in range(GX.shape[0]):
            phi = 0.5 * np.pi * GX[k]
            sth = a * np.sin(phi)
            cth = np.sqrt(1.0 - sth * sth)
            h = rho * np.cos(phi)                      # sqrt(rho^2 - u^2 sin^2 th)
            u1, u2 = u * cth - h, u * cth + h
            dth = a * np.cos(phi) / cth * 0.5 * np.pi  # dtheta/dphi * dphi/dx
            acc += GW[k] * (_F(u2) - _F(u1)) * dth
    else:
        # split at th = pi/2: at u = rho the integrand has a kink there (|cos th|; one interval: 1.1e-4 error)
        for h in range(2):
            for k in range(GX.shape[0]):
                th = 0.5 * np.pi * (GX[k] + h)
                st = np.sin(th)
                u2 = u * np.cos(th) + np.sqrt(max(rho * rho - u * u * st * st, 0.0))
                acc += GW[k] * _F(u2) * 0.5 * np.pi
    return acc / (np.pi * rho * rho)


@njit(cache=True, error_model='numpy', fastmath=FM)
def espl(u, rho, ld=0.0):
    """Finite-source single-lens magnification with linear limb darkening coefficient ld."""
    A1 = espl_uniform(u, rho)
    if ld == 0.0:
        return A1
    # annuli r = rho sin(phi); M(r) = r^2 A_uniform(u, r) has a kink at r = u (lens on the annulus edge): split the
    # phi integral there (one 8-node interval: up to 1e-3 error with the lens inside the disk)
    u = abs(u)
    acc = 0.0
    pu = np.arcsin(u / rho) if u < rho else 0.5 * np.pi
    for h in range(2):
        a, b = (0.0, pu) if h == 0 else (pu, 0.5 * np.pi)
        if b <= a:
            continue
        for k in range(LD_X.shape[0]):
            phi = a + (b - a) * LD_X[k]
            sp = np.sin(phi)
            r = rho * sp
            acc += (b - a) * LD_WX[k] * espl_uniform(u, r) * r * r * sp
    return ((1.0 - ld) * A1 * rho * rho + ld * acc) / (rho * rho * (1.0 - ld / 3.0))


@njit(cache=True, error_model='numpy', fastmath=FM)
def lc_1l2s(t0a, u0a, t0b, u0b, tE, rhoa, rhob, qf, t, ld):
    n = t.shape[0]
    A = np.empty(n)
    for k in range(n):
        ta = (t[k] - t0a) / tE
        tb = (t[k] - t0b) / tE
        ua = np.sqrt(u0a * u0a + ta * ta)
        ub = np.sqrt(u0b * u0b + tb * tb)
        A[k] = (espl(ua, rhoa, ld) + qf * espl(ub, rhob, ld)) / (1.0 + qf)
    return A


@njit(cache=True, error_model='numpy', fastmath=FM)
def lc_1l1s(t0, u0, tE, rho, t, ld):
    n = t.shape[0]
    A = np.empty(n)
    for k in range(n):
        tn = (t[k] - t0) / tE
        A[k] = espl(np.sqrt(u0 * u0 + tn * tn), rho, ld)
    return A


def chi2_1l2s(p, data, ld=0.0):
    """The chi2 of the 1L2S model with linearly profiled fluxes; p = (t0_1, u0_1, t0_2, u0_2, tE, lrho1, lrho2, lqF)."""
    t0a, u0a, t0b, u0b, tE, lra, lrb, lqf = p
    pen = 0.0
    for lr in (lra, lrb):                     # same physical box as the binary model
        c = min(max(lr, model.LOGRHO_RANGE[0]), model.LOGRHO_RANGE[1])
        pen += (lr - c) ** 2
    lra = min(max(lra, model.LOGRHO_RANGE[0]), model.LOGRHO_RANGE[1])
    lrb = min(max(lrb, model.LOGRHO_RANGE[0]), model.LOGRHO_RANGE[1])
    tEc = min(max(abs(tE), model.TE_RANGE[0]), model.TE_RANGE[1])
    A = lc_1l2s(t0a, u0a, t0b, u0b, tEc, 10.0 ** lra, 10.0 ** lrb, 10.0 ** lqf, data.t, ld)
    fs, fb = model.fluxes(A, data)
    r = (data.f - fs[data.idx] * A - fb[data.idx]) / data.e
    return float((r * r).sum()) + 1e4 * pen
