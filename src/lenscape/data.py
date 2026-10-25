"""Photometry: the Data container (several data sets with their own source and blend fluxes), loaders for plain
text and KMTNet pySIS files, error rescaling, and per-night binning of the light-curve wings.

Time is any linear time axis (e.g. HJD - 2450000); the model uses the same axis for t0. Fluxes are linear
(difference-imaging fluxes may be negative); each data set gets its own source and blend flux in every fit.
"""
import glob, os
import numpy as np


class Data:
    """Concatenated data of all sets: t, flux f, flux error e, set index idx (0 .. nsets-1), set names."""

    def __init__(self, t, f, e, idx, names):
        self.t, self.f, self.e, self.idx, self.names = t, f, e, idx, names
        self.w = 1.0 / e ** 2
        self.nsets = len(names)
        self.slices = [np.where(idx == k)[0] for k in range(self.nsets)]

    @classmethod
    def from_sets(cls, sets):
        """Data from {name: (t, flux, flux_err)} (in this order of data sets)."""
        ts, fs, es, ids, names = [], [], [], [], []
        for k, (name, (t, f, e)) in enumerate(sets.items()):
            t, f, e = (np.asarray(a, float) for a in (t, f, e))
            ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0)
            ts.append(t[ok]); fs.append(f[ok]); es.append(e[ok]); ids.append(np.full(ok.sum(), k))
            names.append(name)
        return cls(np.concatenate(ts), np.concatenate(fs), np.concatenate(es), np.concatenate(ids), names)

    def select(self, mask):
        """New Data with the points where mask is True (e.g. a time window or an outlier clip)."""
        m = np.asarray(mask, bool)
        return Data(self.t[m], self.f[m], self.e[m], self.idx[m], list(self.names))

    def window(self, t1, t2):
        return self.select((self.t >= t1) & (self.t <= t2))

    def scale_errors(self, k):
        """New Data with the flux errors of each set multiplied by k[name] (dict) or by one factor."""
        fac = np.array([k[n] if isinstance(k, dict) else k for n in self.names], float)
        return Data(self.t, self.f, self.e * fac[self.idx], self.idx, list(self.names))

    def __len__(self):
        return len(self.t)

    def __repr__(self):
        n = {name: len(sl) for name, sl in zip(self.names, self.slices)}
        span = f", t {self.t.min():.2f} .. {self.t.max():.2f}" if len(self.t) else ""
        return f"Data({len(self.t)} points{span}, sets {n})"


def read_text(path, cols=(0, 1, 2), t_offset=0.0, mag=False, zp=22.0, comments="#", **kw):
    """(t, flux, flux_err) from a whitespace-separated text file: columns cols = (time, value, error), time minus
    t_offset. mag=True: value and error are magnitudes, converted to flux with flux = 10^(-0.4 (mag - zp))."""
    d = np.loadtxt(path, comments=comments, usecols=cols, **kw)
    t, v, e = d[:, 0] - t_offset, d[:, 1], d[:, 2]
    if mag:
        f = 10 ** (-0.4 * (v - zp))
        return t, f, f * 0.4 * np.log(10) * e
    return t, v, e


def read_pysis(paths, t_offset=2450000.0, window=None):
    """KMTNet pySIS files (HJD, difference flux, flux error, mag, mag error, ...) -> Data; one data set per file,
    named after the file up to the first '_' (KMTC22_I.pysis -> KMTC22). paths: list of files or a glob pattern.
    The time column is full HJD in some files and HJD - 2450000 in others; both are recognised. The returned axis
    is HJD - t_offset (default HJD - 2450000). window = (t1, t2) on that axis."""
    files = sorted(glob.glob(paths)) if isinstance(paths, str) else list(paths)
    if not files:
        raise FileNotFoundError(f"no files match {paths!r}")
    sets = {}
    for f in files:
        t, fl, fe = read_text(f, (0, 1, 2))
        t = t - (t_offset if np.nanmedian(t) > 2.4e6 else t_offset - 2450000.0)
        if window is not None:
            m = (t >= window[0]) & (t <= window[1])
            t, fl, fe = t[m], fl[m], fe[m]
        sets[os.path.basename(f).split("_")[0]] = (t, fl, fe)
    return Data.from_sets(sets)


def rescale(fe, emag, k, smin):
    """Yee et al. (2012): sigma_mag' = k * sqrt(sigma_mag^2 + sigma_min^2), applied to flux errors."""
    return fe * k * np.sqrt(emag ** 2 + smin ** 2) / emag


class Binned:
    """Data with the wings binned per night. Points with |t - t0| > K tE are grouped per data set into runs
    without gaps > `gap` days; each run of >= 2 points is replaced by its sufficient statistics (sum w, w f, w f^2,
    w tau f, w tau^2 with tau = t - weighted mean time) and the model is evaluated at t_bar +- h (value and slope),
    so chi2 is exact for a model that is linear in time within a night (test event, K = 1: |error| <= 0.03 over 261
    parameter vectors, 64 % of the points evaluated, chi2 1.27x). The original arrays (t, f, e, idx, w, ...) stay
    available for residuals and plots; model.chi2 uses t_eval and the statistics.

    model = (backend, p), or a list of such pairs: models of the event (its best fit, and e.g. solutions from other
    parts of the map). Every run is checked against each: where the binned chi2 of the run differs from its exact
    chi2 by more than `tol` for any of them, the run is kept at full cadence. Without this check an anomaly farther
    than K tE from t0 (a planetary caustic crossed at tau > 1) is binned with a straight line through the caustic
    feature. `n_unbinned` counts the runs kept this way, `max_bin_error` is the largest remaining per-run error."""
    binned = True

    def __init__(self, data, t0, tE, K=1.0, gap=0.3, model=None, tol=0.05):
        self.__dict__.update(data.__dict__)
        t, w, f, idx = data.t, data.w, data.f, data.idx
        keep = np.abs(t - t0) <= K * tE
        groups = []
        for j in range(data.nsets):
            sel = np.where((idx == j) & ~keep)[0]
            sel = sel[np.argsort(t[sel], kind="stable")]
            for g in (np.split(sel, np.where(np.diff(t[sel]) > gap)[0] + 1) if len(sel) else []):
                if len(g) < 2:
                    keep[g] = True
                else:
                    groups.append(g)
        self.n_unbinned, self.max_bin_error = 0, 0.0
        if model is not None and groups:
            models = model if isinstance(model, list) else [model]
            err = np.max(np.abs(np.array([self._errors(groups, m) for m in models])), axis=0)
            bad = err > tol
            for g in [g for g, b in zip(groups, bad) if b]:
                keep[g] = True
            self.n_unbinned = int(bad.sum())
            self.max_bin_error = float(np.max(err[~bad])) if (~bad).any() else 0.0
            groups = [g for g, b in zip(groups, bad) if not b]
        B = np.array([self._stats(g, t, w, f, idx) for g in groups]).reshape(-1, 8)
        kp = np.where(keep)[0]
        tt = np.r_[t[kp], B[:, 1] - B[:, 2], B[:, 1] + B[:, 2]]
        order = np.argsort(tt, kind="stable")                  # sorted: root warm starts along the trajectory
        pos = np.empty(len(tt), np.int64)
        pos[order] = np.arange(len(tt))
        nk, nb = len(kp), len(B)
        self.t_eval = tt[order]
        self.k_pos, self.k_f, self.k_w, self.k_idx = pos[:nk], f[kp], w[kp], idx[kp]
        self.b_m, self.b_p = pos[nk:nk + nb], pos[nk + nb:]
        self.b_h, self.b_set = B[:, 2].copy(), B[:, 0].astype(np.int64)
        self.b_W, self.b_Sf, self.b_Sff, self.b_Stf, self.b_Stt = (B[:, k].copy() for k in range(3, 8))

    @staticmethod
    def _stats(g, t, w, f, idx):
        W = w[g].sum()
        tb = (w[g] * t[g]).sum() / W
        tau = t[g] - tb
        return (idx[g[0]], tb, max(np.abs(tau).max(), 1e-3), W, (w[g] * f[g]).sum(), (w[g] * f[g] ** 2).sum(),
                (w[g] * tau * f[g]).sum(), (w[g] * tau ** 2).sum())

    def _errors(self, groups, model):
        """Binned minus exact chi2 of every run at the model (backend, p), with the fluxes of the exact fit."""
        from . import model as M
        backend, p = model
        p = tuple(float(v) for v in np.asarray(p, float))
        t, w, f, idx = self.t, self.w, self.f, self.idx
        A = backend.lc(p, t)
        fs, fb = M.fluxes(A, self)
        S = np.array([self._stats(g, t, w, f, idx) for g in groups])
        te = np.r_[S[:, 1] - S[:, 2], S[:, 1] + S[:, 2]]
        o = np.argsort(te, kind="stable")
        Ae = np.empty(len(te))
        Ae[o] = backend.lc(p, te[o])
        n = len(groups)
        A0, A1 = 0.5 * (Ae[:n] + Ae[n:]), (Ae[n:] - Ae[:n]) / (2 * S[:, 2])
        j = S[:, 0].astype(int)
        c0, c1 = fs[j] * A0 + fb[j], fs[j] * A1
        binned = S[:, 5] - 2 * c0 * S[:, 4] - 2 * c1 * S[:, 6] + c0 ** 2 * S[:, 3] + c1 ** 2 * S[:, 7]
        exact = np.array([np.sum(w[g] * (f[g] - fs[idx[g]] * A[g] - fb[idx[g]]) ** 2) for g in groups])
        return binned - exact


def bin_wings(data, t0, tE, K=1.0, gap=0.3, model=None, tol=0.05):
    """Speed option for maps: the wings binned per night (Binned). model = (backend, p) checks every run against
    the model p and keeps runs that the linear-in-time model misrepresents (an anomaly beyond K tE)."""
    return Binned(data, t0, tE, K, gap, model, tol)
