"""Magnification back ends and chi2 with linearly profiled source/blend fluxes.

Parameter vector p = (t0, u0, tE, log10 s, log10 q, alpha, log10 rho), VBBL geometry convention (centre-of-mass
frame). The package has no dependency on VBBinaryLensing: it is used only as a test reference, registered from the
test/benchmark helpers via `register_backend("vbbl", ...)`.
"""
import math
import numpy as np
from numba import njit

LN10 = np.log(10.0)
BACKENDS = {}


def register_backend(kind, cls):
    """Make a backend class available to `fit.make_backend(dict(kind=...))` (e.g. a reference code in tests)."""
    BACKENDS[kind] = cls


def fluxes(A, data):
    """Weighted linear least squares F = fs*A + fb per data set; returns fs, fb arrays (nsets)."""
    w, f, idx, n = data.w, data.f, data.idx, data.nsets
    S = np.bincount(idx, w, n)
    Sa = np.bincount(idx, w * A, n)
    Saa = np.bincount(idx, w * A * A, n)
    Sf = np.bincount(idx, w * f, n)
    Saf = np.bincount(idx, w * A * f, n)
    det = S * Saa - Sa ** 2
    return (S * Saf - Sa * Sf) / det, (Saa * Sf - Sa * Saf) / det


# Physical parameter box. Unbounded optimisers drifted to log10 rho = -12.9 (where VBBinaryLensing segfaults);
# physical sources have rho ~ 1e-4 .. 1e-2.
LOGRHO_RANGE = (-5.0, -0.5)
TE_RANGE = (0.5, 1000.0)
U0_MAX = 5.0


def clamp(p):
    """Clamp p into the physical box; returns (clamped p, squared distance outside the box). Python floats instead
    of np.clip on scalars (6 -> ~1 us per call)."""
    t0, u0, tE, ls, lq, al, lr = p
    u0c = min(max(u0, -U0_MAX), U0_MAX)
    tEc = min(max(tE, TE_RANGE[0]), TE_RANGE[1])
    lrc = min(max(lr, LOGRHO_RANGE[0]), LOGRHO_RANGE[1])
    out = (u0 - u0c) ** 2 + (math.log(tE / tEc) if tE > 0 else 10.0) ** 2 + (lr - lrc) ** 2
    return (t0, float(u0c), float(tEc), ls, lq, al, float(lrc)), out


def magnification(backend, p, data):
    p, _ = clamp(p)
    if getattr(backend, "single_call", True):
        return backend.lc(p, data.t)
    A = np.empty_like(data.t)                    # one call per data set (baseline reference behaviour)
    for sl in data.slices:
        A[sl] = backend.lc(p, data.t[sl])
    return A


def residuals(backend, p, data):
    A = magnification(backend, p, data)
    fs, fb = fluxes(A, data)
    return (data.f - fs[data.idx] * A - fb[data.idx]) / data.e


@njit(cache=True, error_model='numpy')
def chi2_profiled(A, f, w, idx, nsets):
    """The chi2 with fs, fb per data set from weighted linear least squares (as fluxes()), in two passes: sums, then
    residuals (a one-pass expansion would cancel catastrophically: sum w f^2 >> chi2)."""
    S = np.zeros(nsets)
    Sa = np.zeros(nsets)
    Saa = np.zeros(nsets)
    Sf = np.zeros(nsets)
    Saf = np.zeros(nsets)
    for k in range(A.shape[0]):
        j, wk, ak = idx[k], w[k], A[k]
        S[j] += wk
        Sa[j] += wk * ak
        Saa[j] += wk * ak * ak
        Sf[j] += wk * f[k]
        Saf[j] += wk * ak * f[k]
    fs = np.empty(nsets)
    fb = np.empty(nsets)
    for j in range(nsets):
        det = S[j] * Saa[j] - Sa[j] ** 2
        fs[j] = (S[j] * Saf[j] - Sa[j] * Sf[j]) / det
        fb[j] = (Saa[j] * Sf[j] - Sa[j] * Saf[j]) / det
    c = 0.0
    for k in range(A.shape[0]):
        j = idx[k]
        r = f[k] - fs[j] * A[k] - fb[j]
        c += r * r * w[k]
    return c


@njit(cache=True, error_model='numpy')
def chi2_profiled_binned(A, kpos, kf, kw, kidx, bm, bp, bh, bset, W, Sf, Sff, Stf, Stt, nsets):
    """As chi2_profiled, for data.Binned: kept points as usual; per bin the model is linear, A0 + A1 tau, with value and
    slope from A(t_bar - h), A(t_bar + h); chi2 of a bin from its sufficient statistics."""
    S = np.zeros(nsets)
    Sa = np.zeros(nsets)
    Saa = np.zeros(nsets)
    Sfv = np.zeros(nsets)
    Saf = np.zeros(nsets)
    for k in range(kpos.shape[0]):
        j, wk, ak = kidx[k], kw[k], A[kpos[k]]
        S[j] += wk
        Sa[j] += wk * ak
        Saa[j] += wk * ak * ak
        Sfv[j] += wk * kf[k]
        Saf[j] += wk * ak * kf[k]
    nb = bm.shape[0]
    A0 = np.empty(nb)
    A1 = np.empty(nb)
    for b in range(nb):
        am, ap = A[bm[b]], A[bp[b]]
        A0[b] = 0.5 * (am + ap)
        A1[b] = (ap - am) / (2.0 * bh[b])
        j = bset[b]
        S[j] += W[b]
        Sa[j] += W[b] * A0[b]
        Saa[j] += W[b] * A0[b] * A0[b] + A1[b] * A1[b] * Stt[b]
        Sfv[j] += Sf[b]
        Saf[j] += A0[b] * Sf[b] + A1[b] * Stf[b]
    fs = np.empty(nsets)
    fb = np.empty(nsets)
    for j in range(nsets):
        det = S[j] * Saa[j] - Sa[j] ** 2
        fs[j] = (S[j] * Saf[j] - Sa[j] * Sfv[j]) / det
        fb[j] = (Saa[j] * Sfv[j] - Sa[j] * Saf[j]) / det
    c = 0.0
    for k in range(kpos.shape[0]):
        j = kidx[k]
        r = kf[k] - fs[j] * A[kpos[k]] - fb[j]
        c += r * r * kw[k]
    for b in range(nb):
        j = bset[b]
        c0 = fs[j] * A0[b] + fb[j]
        c1 = fs[j] * A1[b]
        c += Sff[b] - 2.0 * c0 * Sf[b] - 2.0 * c1 * Stf[b] + c0 * c0 * W[b] + c1 * c1 * Stt[b]
    return c


def chi2(backend, p, data):
    pc, out = clamp(p)
    if getattr(data, "binned", False):
        d = data
        A = backend.lc(pc, d.t_eval)
        return chi2_profiled_binned(A, d.k_pos, d.k_f, d.k_w, d.k_idx, d.b_m, d.b_p, d.b_h, d.b_set, d.b_W, d.b_Sf,
                                    d.b_Sff, d.b_Stf, d.b_Stt, d.nsets) + 1e4 * out
    if getattr(backend, "single_call", True):
        A = backend.lc(pc, data.t)
    else:
        A = magnification(backend, pc, data)
    # penalty pushes optimisers back into the box
    return chi2_profiled(A, data.f, data.w, data.idx, data.nsets) + 1e4 * out



@njit(cache=True, error_model='numpy')
def resid_profiled(A, f, w, idx, nsets):
    """Residuals (f - fs A - fb) sqrt(w) with fs, fb per data set as in chi2_profiled; sum of squares = chi2."""
    S = np.zeros(nsets)
    Sa = np.zeros(nsets)
    Saa = np.zeros(nsets)
    Sf = np.zeros(nsets)
    Saf = np.zeros(nsets)
    for k in range(A.shape[0]):
        j, wk, ak = idx[k], w[k], A[k]
        S[j] += wk
        Sa[j] += wk * ak
        Saa[j] += wk * ak * ak
        Sf[j] += wk * f[k]
        Saf[j] += wk * ak * f[k]
    out = np.empty(A.shape[0] + 1)
    for j in range(nsets):
        det = S[j] * Saa[j] - Sa[j] ** 2
        fs = (S[j] * Saf[j] - Sa[j] * Sf[j]) / det
        fb = (Saa[j] * Sf[j] - Sa[j] * Saf[j]) / det
        S[j], Sa[j] = fs, fb
    for k in range(A.shape[0]):
        j = idx[k]
        out[k] = (f[k] - S[j] * A[k] - Sa[j]) * np.sqrt(w[k])
    out[A.shape[0]] = 0.0
    return out


@njit(cache=True, error_model='numpy')
def resid_profiled_binned(A, kpos, kf, kw, kidx, bm, bp, bh, bset, W, Sf, Sff, Stf, Stt, nsets):
    """Residual vector for data.Binned: kept points as resid_profiled, two per bin, sqrt(W) (c0 - Sf/W) and
    sqrt(Stt) (c1 - Stf/Stt) (tau is centred on the bin's weighted mean, so the bin's chi2 is the sum of their squares
    plus a constant, Sff - Sf^2/W - Stf^2/Stt). Fluxes as in chi2_profiled_binned."""
    S = np.zeros(nsets)
    Sa = np.zeros(nsets)
    Saa = np.zeros(nsets)
    Sfv = np.zeros(nsets)
    Saf = np.zeros(nsets)
    for k in range(kpos.shape[0]):
        j, wk, ak = kidx[k], kw[k], A[kpos[k]]
        S[j] += wk
        Sa[j] += wk * ak
        Saa[j] += wk * ak * ak
        Sfv[j] += wk * kf[k]
        Saf[j] += wk * ak * kf[k]
    nb = bm.shape[0]
    A0 = np.empty(nb)
    A1 = np.empty(nb)
    for b in range(nb):
        am, ap = A[bm[b]], A[bp[b]]
        A0[b] = 0.5 * (am + ap)
        A1[b] = (ap - am) / (2.0 * bh[b])
        j = bset[b]
        S[j] += W[b]
        Sa[j] += W[b] * A0[b]
        Saa[j] += W[b] * A0[b] * A0[b] + A1[b] * A1[b] * Stt[b]
        Sfv[j] += Sf[b]
        Saf[j] += A0[b] * Sf[b] + A1[b] * Stf[b]
    fs = np.empty(nsets)
    fb = np.empty(nsets)
    for j in range(nsets):
        det = S[j] * Saa[j] - Sa[j] ** 2
        fs[j] = (S[j] * Saf[j] - Sa[j] * Sfv[j]) / det
        fb[j] = (Saa[j] * Sfv[j] - Sa[j] * Saf[j]) / det
    nk = kpos.shape[0]
    out = np.empty(nk + 2 * nb + 1)
    for k in range(nk):
        j = kidx[k]
        out[k] = (kf[k] - fs[j] * A[kpos[k]] - fb[j]) * np.sqrt(kw[k])
    for b in range(nb):
        j = bset[b]
        out[nk + 2 * b] = np.sqrt(W[b]) * (fs[j] * A0[b] + fb[j] - Sf[b] / W[b])
        out[nk + 2 * b + 1] = np.sqrt(Stt[b]) * (fs[j] * A1[b] - Stf[b] / Stt[b]) if Stt[b] > 0 else 0.0
    out[nk + 2 * nb] = 0.0
    return out


def chi2_residuals(backend, p, data):
    """Residual vector whose sum of squares is model.chi2(backend, p, data) up to a constant (binned data); the last
    element carries the box penalty. For least-squares optimisers (fit._lm)."""
    pc, out = clamp(p)
    if getattr(data, "binned", False):
        d = data
        r = resid_profiled_binned(backend.lc(pc, d.t_eval), d.k_pos, d.k_f, d.k_w, d.k_idx, d.b_m, d.b_p, d.b_h,
                                  d.b_set, d.b_W, d.b_Sf, d.b_Sff, d.b_Stf, d.b_Stt, d.nsets)
    else:
        A = backend.lc(pc, data.t) if getattr(backend, "single_call", True) else magnification(backend, pc, data)
        r = resid_profiled(A, data.f, data.w, data.idx, data.nsets)
    r[-1] = 100.0 * math.sqrt(out)
    return r

class NumbaHybrid:
    """Own magnification code: numba point source, analytic quadrupole, sampled quadrupole/hexadecapole and contour
    integration (contour.py). No external magnification library.

    tol: point-source acceptance (VBBL-style quadrupole test), hexeps: hexadecapole acceptance, ctol: contour
    tolerance, ld: linear limb-darkening coefficient u (0 = uniform source)."""

    def __init__(self, tol=1e-2, hexeps=1e-3, ctol=0.1, ld=0.0, single_call=True, reltol=None):
        from . import nbmag, contour
        self.nb, self.ct = nbmag, contour
        self.tol, self.hexeps, self.ctol, self.ld = tol, hexeps, ctol, ld
        self.single_call = True
        self.n_retry = 0                         # contour integrations that needed the dense retry
        self.n_failed = 0                        # ... and that still failed (point-source value used, counted)

    def _repair(self, A, flag, s, q, y1, y2, rho):
        """Flag 2 = contour integration reported an inconsistency: retry densely, else point source + count."""
        idx = np.nonzero(flag == 2)[0]
        for k in idx:
            self.n_retry += 1
            a, st, _ = self.ct.mag_disk_ld(s[k], q[k], y1[k], y2[k], rho[k], self.ctol / 3, self.ld, 32, 4096)
            if st >= 0:
                A[k] = a
            else:
                self.n_failed += 1
                A[k] = self.nb.mags_point(s[k:k + 1], q[k:k + 1], y1[k:k + 1], y2[k:k + 1])[0]
        return A

    def lc(self, p, t):
        t0, u0, tE, ls, lq, al, lr = p
        A, flag = self.ct.lc_full(ls, lq, u0, al, lr, tE, t0, t, self.tol, self.hexeps, self.ctol, self.ld)
        if (flag == 2).any():
            n = len(t)
            tn = (t - t0) / tE
            y1, y2 = u0 * np.sin(al) - tn * np.cos(al), -u0 * np.cos(al) - tn * np.sin(al)
            A = self._repair(A, flag, np.full(n, 10.0 ** ls), np.full(n, 10.0 ** lq), y1, y2, np.full(n, 10.0 ** lr))
        return A

    def mag(self, s, q, y1, y2, rho):
        A, flag = self.ct.mags_full(s, q, y1, y2, rho, self.tol, self.hexeps, self.ctol, self.ld)
        return self._repair(A, flag, s, q, y1, y2, rho)


register_backend("nb", NumbaHybrid)
