"""Higher-order 2L1S effects: annual microlens parallax and linear lens orbital motion.

Parameters p = (t0, u0, tE, log10 s, log10 q, alpha, log10 rho, piEN, piEE, ds/dt, dalpha/dt), times in days
(HJD - tref), ds/dt in 1/yr and dalpha/dt in rad/yr (KMTNet/Han convention), both referenced to t0.

Parallax (Gould 2004): with Delta s(t) the Sun's position seen from Earth, projected on the sky (north, east; AU),
minus its value and linear motion at t0_par,
    tau  = (t - t0)/tE + piE . Delta s,    beta = u0 + piE x Delta s,
and the source position (y1, y2) = (beta sin alpha - tau cos alpha, -beta cos alpha - tau sin alpha), the same
centre-of-mass frame as the static model (model.py). The (u0, alpha) <-> (-u0, -alpha) ecliptic degeneracy is
explored by fitting both signs of u0.
Orbital motion: s(t) = s + ds/dt (t - t0), alpha(t) = alpha + dalpha/dt (t - t0) (VBBinaryLensing sign: its
circular-orbit model w1 = (ds/dt)/s, w2 = dalpha/dt agrees to first order; validated in tests).
Solar ephemeris: low-precision formulae of the Astronomical Almanac (~0.01 deg; parallax offsets to ~1e-4 AU).
"""
import numpy as np
from . import contour, model

DEG = np.pi / 180.0
YR = 365.25


def parse_coords(ra, dec):
    """'hh:mm:ss.s', '+dd:mm:ss.s' -> (RA, Dec) in radians."""
    h, m, s = (float(x) for x in ra.split(":"))
    sg = -1.0 if dec.strip().startswith("-") else 1.0
    d, dm, ds = (abs(float(x)) for x in dec.split(":"))
    return (h + m / 60 + s / 3600) * 15 * DEG, sg * (d + dm / 60 + ds / 3600) * DEG


def sun_vector(jd):
    """Geocentric equatorial position of the Sun (AU), shape (n, 3)."""
    n = np.asarray(jd, float) - 2451545.0
    L = (280.460 + 0.9856474 * n) * DEG
    g = (357.528 + 0.9856003 * n) * DEG
    lam = L + (1.915 * np.sin(g) + 0.020 * np.sin(2 * g)) * DEG
    R = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)
    eps = (23.439 - 4e-7 * n) * DEG
    return np.stack([R * np.cos(lam), R * np.cos(eps) * np.sin(lam), R * np.sin(eps) * np.sin(lam)], axis=-1)


def sky_basis(ra, dec):
    """Unit vectors north, east on the sky at (ra, dec)"""
    north = np.array([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)])
    east = np.array([-np.sin(ra), np.cos(ra), 0.0])
    return north, east


def parallax_offsets(jd, jd0par, ra, dec, h=0.1):
    """Delta s = (north, east) components in AU of the projected Sun position minus its position and velocity at
    t0_par (velocity by central difference with step h days)."""
    N, E = sky_basis(ra, dec)
    proj = lambda j: np.stack([sun_vector(j) @ N, sun_vector(j) @ E], axis=-1)
    s0 = proj(np.array([jd0par]))[0]
    v0 = (proj(np.array([jd0par + h]))[0] - proj(np.array([jd0par - h]))[0]) / (2 * h)
    return proj(jd) - s0 - (np.asarray(jd) - jd0par)[:, None] * v0


class Trajectory:
    """Precomputed parallax geometry for one data set's times. t: HJD - tref; t0par in the same time frame."""

    def __init__(self, t, tref, ra, dec, t0par):
        self.t, self.t0par = np.asarray(t, float), float(t0par)
        if isinstance(ra, str):
            ra, dec = parse_coords(ra, dec)
        self.ds = parallax_offsets(self.t + tref, t0par + tref, ra, dec)       # (n, 2): north, east

    def positions(self, p):
        """(s(t), y1, y2) for the parameter vector p (length 7, 9 or 11)."""
        t0, u0, tE, ls, lq, al, lr = p[:7]
        piN, piE = (p[7], p[8]) if len(p) > 7 else (0.0, 0.0)
        dsdt, dadt = (p[9], p[10]) if len(p) > 9 else (0.0, 0.0)
        dt = self.t - t0
        tau = dt / tE + piN * self.ds[:, 0] + piE * self.ds[:, 1]
        beta = u0 + piN * self.ds[:, 1] - piE * self.ds[:, 0]
        a = al + dadt * dt / YR
        s = 10.0 ** ls + dsdt * dt / YR
        return np.maximum(s, 1e-3), beta * np.sin(a) - tau * np.cos(a), -beta * np.cos(a) - tau * np.sin(a)


class HighOrder:
    """Magnification for p with parallax and/or orbital motion, using the own numba stack (same accuracy settings
    as model.NumbaHybrid)."""

    def __init__(self, traj, tol=1e-2, hexeps=1e-3, ctol=0.1, ld=0.0):
        self.traj = traj
        self.base = model.NumbaHybrid(tol=tol, hexeps=hexeps, ctol=ctol, ld=ld)
        self.tol, self.hexeps, self.ctol, self.ld = tol, hexeps, ctol, ld

    def mag(self, p):
        q, rho = 10.0 ** p[4], 10.0 ** p[6]
        s, y1, y2 = self.traj.positions(p)
        A, flag = contour.lc_traj(s, q, rho, y1, y2, self.tol, self.hexeps, self.ctol, self.ld)
        if (flag == 2).any():
            n = len(s)
            A = self.base._repair(A, flag, s, np.full(n, q), y1, y2, np.full(n, rho))
        return A


def chi2(ho, p, data):
    """The chi2 with profiled fluxes; physical box on the first 7 parameters as in model.chi2."""
    c, out = model.clamp(p[:7])
    pc = np.concatenate([c, p[7:]])
    A = ho.mag(pc)
    fs, fb = model.fluxes(A, data)
    r = (data.f - fs[data.idx] * A - fb[data.idx]) / data.e
    return float((r * r).sum()) + 1e4 * out
