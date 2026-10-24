"""Binary-lens geometry for figures: critical curves and caustics, source trajectory, point-source magnification maps.

Frame: centre of mass, primary on the left (x = -s q / (1 + q)), secondary on the right (x = s / (1 + q)), as in
nbmag (VBBinaryLensing); the trajectory follows the light-curve model (model.lc): y1 = u0 sin(alpha) - tau cos(alpha),
y2 = -u0 cos(alpha) - tau sin(alpha), tau = (t - t0) / tE.
"""
import numpy as np


def lens_positions(s, q):
    """(x of the primary, x of the secondary) in the centre-of-mass frame."""
    return -s * q / (1.0 + q), s / (1.0 + q)


def critical_curves(s, q, n=1000):
    """Critical curves and caustics: lists of closed complex arrays (one per curve, in drawing order).
    |m1 / (z - z1)^2 + m2 / (z - z2)^2| = 1 with z1, z2 the lens positions and m1 + m2 = 1: for each phase phi,
    the four roots of e^{i phi} (z - z1)^2 (z - z2)^2 - m1 (z - z2)^2 - m2 (z - z1)^2 = 0; roots are tracked from
    phase to phase (nearest neighbour) so that each root traces one curve segment."""
    s, q = float(s), float(q)
    z1, z2 = lens_positions(s, q)
    m1, m2 = 1.0 / (1.0 + q), q / (1.0 + q)
    p1, p2 = np.polymul([1.0, -z1], [1.0, -z1]), np.polymul([1.0, -z2], [1.0, -z2])
    base = np.polymul(p1, p2)
    rest = np.r_[0.0, 0.0, np.polyadd(m1 * p2, m2 * p1)]
    phis = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    Z = np.empty((n, 4), complex)
    for k, ph in enumerate(phis):
        r = np.roots(np.exp(1j * ph) * base - rest)
        if k:                                            # keep each root on its own track
            order, free = [], list(range(4))
            for j in range(4):
                i = min(free, key=lambda i: abs(r[i] - Z[k - 1, j]))
                order.append(i)
                free.remove(i)
            r = r[order]
        Z[k] = r
    # one track per root; the tracks of a curve join up after 2 pi (possibly as a permutation): chain them
    tracks, used, crit = [Z[:, j] for j in range(4)], set(), []
    for j in range(4):
        if j in used:
            continue
        curve, cur = [tracks[j]], j
        used.add(j)
        while True:
            end = curve[-1][-1]
            nxt = min(range(4), key=lambda i: abs(tracks[i][0] - end))
            if nxt in used:
                break
            used.add(nxt)
            curve.append(tracks[nxt])
        c = np.concatenate(curve)
        crit.append(np.r_[c, c[:1]])
    lens = lambda z: z - m1 / np.conj(z - z1) - m2 / np.conj(z - z2)
    return crit, [lens(c) for c in crit]


def caustics(s, q, n=1000):
    """Caustics only (list of closed complex arrays, centre-of-mass frame)."""
    return critical_curves(s, q, n)[1]


def trajectory(p, t):
    """Source positions (complex, centre-of-mass frame) at times t for p = (t0, u0, tE, log s, log q, alpha, log rho)."""
    t0, u0, tE, al = p[0], p[1], p[2], p[5]
    tau = (np.asarray(t, float) - t0) / tE
    return (u0 * np.sin(al) - tau * np.cos(al)) + 1j * (-u0 * np.cos(al) - tau * np.sin(al))


def magnification_map(s, q, x, y):
    """Point-source magnification on the grid x (nx,) times y (ny,) (source plane, centre-of-mass frame);
    returns (ny, nx)."""
    from . import nbmag
    X, Y = np.meshgrid(np.asarray(x, float), np.asarray(y, float))
    n = X.size
    return nbmag.mags_point(np.full(n, float(s)), np.full(n, float(q)), X.ravel(), Y.ravel()).reshape(X.shape)
