"""Helpers for the Han et al. (2025) notebook: rebuild the data behind the paper's figures from its arXiv source.

The paper (A&A, arXiv:2505.22951) does not publish its photometry. Its light-curve figures (Fig. 1, Fig. 5) are
vector graphics: every plotted point and error bar is stored with its position. We read them (lenscape.eps), map
device units to time and magnitude with the frames and tick marks of each panel, and add the public KMTNet points
outside the plotted window (lenscape.datasets.kmtnet), mapped onto the paper's flux scale. Nothing derived from the
paper is stored in this repository; it is rebuilt here from the paper's own source files.
"""
import os
import numpy as np
from lenscape import eps, data as D, model as M

ZP = 22.0                                      # magnitude zero point of the rebuilt fluxes (arbitrary)


def _merge(zoom, overall, t1, t2, dt, zoom_only=False):
    """Per data set: zoom-panel points, plus overall-panel points that are not the same point shown in the zoom.
    zoom_only: drop every overall-panel point inside the zoom window (where markers overlap too densely to be
    separated in the overall panel)."""
    out = {}
    for name in sorted({p[0] for p in zoom + overall}):
        z = [p for p in zoom if p[0] == name]
        zt = np.array([p[1] for p in z])
        keep = z + [p for p in overall if p[0] == name and
                    not (t1 < p[1] < t2 and (zoom_only or (len(zt) and np.min(np.abs(zt - p[1])) < 1.5 * dt)))]
        out[name] = np.array(sorted((p[1], p[2], p[3]) for p in keep))
    return out


def _to_data(sets, order):
    """{name: (t, mag, mag_err)} -> Data with fluxes on the zero point ZP."""
    arr = {}
    for name in order:
        t, m, e = sets[name].T
        f = 10 ** (-0.4 * (m - ZP))
        arr[name] = (t, f, f * 0.4 * np.log(10) * e)
    return D.Data.from_sets(arr)


def moa091_plotted(src):
    """MOA-2022-BLG-091, Fig. 1: KMTNet (re-reduced) and MOA points with error bars; time HJD - 2450000."""
    subs = eps.subpaths(os.path.join(src, "f1.eps"))
    col = {(255, 99, 71): "KMTC22", (30, 144, 255): "KMTS22", (100, 180, 100): "KMTA22", (100, 100, 100): "MOA22"}
    zoom = eps.points_with_errors(subs, eps.calibrate(292, 9652, 471, 9653), eps.calibrate(1251, 18, 1490, 17),
                                  (292, 2261, 1203, 1823), col)
    over = eps.points_with_errors(subs, eps.calibrate(654, 9640, 678.17, 9641), eps.calibrate(347, 20, 457, 19),
                                  (292, 2261, 292, 875), col)
    return _to_data(_merge(zoom, over, 9652, 9663, 1 / 24.17), ["KMTA22", "KMTC22", "KMTS22", "MOA22"])


def kb1209_plotted(src):
    """KMT-2024-BLG-1209, Fig. 5: KMTNet, OGLE and LCO points with error bars; time HJD - 2460000."""
    subs = eps.subpaths(os.path.join(src, "f5.eps"))
    col = {(255, 99, 71): "KMTC", (30, 144, 255): "KMTS", (100, 180, 100): "KMTA", (139, 69, 19): "OGLE",
           (122, 103, 238): "LCOA", (224, 102, 255): "LCOC"}
    # time: zoom panel 0.5 d per minor tick (123.06 units), HJD' 488 at x = 538 -- fitted to the public KMTNet epochs
    # (all 48 KMTNet points within 0.005 d); overall panel 410.5 units per 20 d, HJD' 460 at x = 620.
    # magnitude: the zoom panel's labels (517 units per mag, 16.6 at y = 1772) and the overall panel's tick spacing
    # (162 units per mag) are not consistent with each other: the same points differ by 0.06-0.09 mag between the
    # panels. A joint fit of the two scales and a zoom offset, at either published solution (Table 3), gives
    # 517 / 175.8 units per mag and a zoom offset of +0.082 mag (chi2 = 582 and 585 for 580 points); without it,
    # chi2 = 1184. This calibration therefore uses the published model, which a check of the paper cannot avoid.
    zoom = eps.points_with_errors(subs, eps.calibrate(538, 488, 1030.25, 490),
                                  eps.calibrate(1772, 16.6 + 0.082, 1255, 17.6 + 0.082), (292, 2261, 1203, 1823), col)
    over = eps.points_with_errors(subs, eps.calibrate(620, 460, 1030.5, 480), eps.calibrate(729, 17, 729 - 175.8, 18),
                                  (292, 2261, 292, 875), col)
    return _to_data(_merge(zoom, over, 487, 495, 20 / 410.5, zoom_only=True),
                    ["KMTA", "KMTC", "KMTS", "OGLE", "LCOA", "LCOC"])


def extend_baseline(plotted, public, best, backend, match=None):
    """Add the public points outside the plotted time range of each KMTNet set.
    plotted, public: Data; match: {plotted name: public name} (default: same names). Each public set is mapped onto
    the plotted set's flux scale by a linear relation fitted on the shared epochs, its errors scaled by the median
    ratio of the errors there; the added points are then normalised separately (errors scaled to chi2/dof = 1
    against the best model of the plotted data, 5-sigma clipped). Returns (Data, notes)."""
    match = match or {n: n for n in public.names if n in plotted.names}
    sets, notes = {}, []
    A_pl = backend.lc(np.asarray(best, float), plotted.t)
    fs, fb = M.fluxes(A_pl, plotted)
    for k, name in enumerate(plotted.names):
        sel = plotted.idx == k
        t, f, e = plotted.t[sel], plotted.f[sel], plotted.e[sel]
        if name not in match:
            sets[name] = (t, f, e)
            continue
        j = public.names.index(match[name])
        pt, pf, pe = (a[public.idx == j] for a in (public.t, public.f, public.e))
        near = np.array([int(np.argmin(np.abs(pt - ti))) for ti in t])
        ok = np.abs(pt[near] - t) < 0.01
        (a, c), *_ = np.linalg.lstsq(np.vstack([pf[near[ok]], np.ones(ok.sum())]).T, f[ok], rcond=None)
        ratio = np.median(e[ok] / (abs(a) * pe[near[ok]]))
        out = (pt < t.min() - 0.01) | (pt > t.max() + 0.01)
        at, af, ae = pt[out], a * pf[out] + c, ratio * abs(a) * pe[out]
        r = (af - (fs[k] * backend.lc(np.asarray(best, float), at) + fb[k])) / ae
        keep = np.abs(r) < 5 * 1.4826 * np.median(np.abs(r))
        knorm = np.sqrt(np.mean(r[keep] ** 2))
        sets[name] = (np.r_[t, at[keep]], np.r_[f, af[keep]], np.r_[e, knorm * ae[keep]])
        notes.append(f"{name}: {ok.sum()} shared epochs, added {keep.sum()} public points "
                     f"(clipped {(~keep).sum()}), their error factor {knorm:.2f}")
    return D.Data.from_sets(sets), notes


def paper_map(src, fig, frame, x_map, y_map, size=(6, 16)):
    """Coloured cells of a Delta chi2 map panel: array of (x, y, band) with band 1 (best) .. 5.
    frame = (x0, x1, y0, y1) in device units; x_map, y_map: device -> data (lenscape.eps.calibrate)."""
    bands = {(238, 118, 0): 1, (255, 215, 0): 2, (124, 252, 0): 3, (126, 192, 238): 4, (65, 105, 225): 5}
    out = []
    for cx, cy, w, h, c in eps.markers(eps.subpaths(os.path.join(src, fig)), size, min_points=4):
        if frame[0] < cx < frame[1] and frame[2] < cy < frame[3]:
            b = bands.get(c, 1 if c[0] > 200 and c[1] < 180 else 0)
            if b:
                out.append((float(x_map(cx)), float(y_map(cy)), b))
    return np.array(out)


# ---------------------------------------------------------------------------------------------------- figures
BANDS = (4.0, 8.0, 12.0, 16.0, 20.0)                     # Delta chi2 < k n^2, n = 2 (captions of Figs. 2 and 6)
COLORS = {"KMTC22": "#e8463a", "KMTS22": "#1f8fff", "KMTA22": "#4caf50", "MOA22": "#666666",
          "KMTC": "#e8463a", "KMTS": "#1f8fff", "KMTA": "#4caf50", "OGLE": "#8b4513", "LCOA": "#7a67ee",
          "LCOC": "#e066ff"}


def plot_two_panels(backend, models, data, zoom, overall, ref, title="", ylim_zoom=None, ylim_all=None):
    """Light curve as in Figs. 1 and 5: zoom panel (top) and the whole light curve (bottom), data aligned with the
    first model. models: list of (label, line style, function t -> magnitude) or (label, style, p) for binary-lens
    vectors p."""
    import matplotlib.pyplot as plt
    from lenscape import figures
    p0 = models[0][2]
    mag, err, fs0, fb0 = figures.aligned(backend, p0, data, ref)
    curves = []
    for label, ls, m in models:
        if callable(m):
            curves.append((label, ls, m))
        else:
            _, _, fs, fb = figures.aligned(backend, m, data, ref)
            curves.append((label, ls, lambda t, m=m, fs=fs, fb=fb: figures.model_mag(backend, m, t, fs, fb)))
    fig, ax = plt.subplots(2, 1, figsize=(7, 7.5), gridspec_kw=dict(height_ratios=[1, 1]))
    for a, (t1, t2), yl in zip(ax, (zoom, overall), (ylim_zoom, ylim_all)):
        sel = (data.t >= t1) & (data.t <= t2)
        for k, name in enumerate(data.names):
            j = sel & (data.idx == k)
            a.errorbar(data.t[j], mag[j], err[j], fmt="o", ms=3, lw=0.7, color=COLORS.get(name, None), label=name)
        tm = np.linspace(t1, t2, 4000)
        for label, ls, f in curves:
            a.plot(tm, f(tm), ls, color="k", lw=1, label=label)
        a.set_xlim(t1, t2)
        if yl:
            a.set_ylim(*yl)
        else:
            a.invert_yaxis()
        a.set_ylabel("Magnitude")
    ax[0].legend(fontsize=7, ncol=2, loc="lower center")
    ax[0].set_title(title)
    ax[1].set_xlabel("HJD - 2450000" if data.t.mean() > 5000 else "HJD - 2460000")
    fig.tight_layout()
    return fig


def compare_maps(paper_left, paper_right, ours, cell, alpha_height, xlim, qlim, alim, title=""):
    """Figs. 2 / 6: the paper's map (redrawn from its vector graphics) next to lenscape's, both in the paper's
    colour bands, in the (log s, log q) and (log s, alpha) planes. ours: dict with logs, logq, chi2, chi2_ref, alpha
    (alpha of the best fit at every lattice point). cell = (d log s, d log q) of the lattice."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from lenscape.figures import BAND_COLORS
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.6), sharex=True)

    def cells(ax, pts, w, h):
        for b in range(5, 0, -1):
            sel = [(x, y) for x, y, bb in pts if bb == b]
            pc = PatchCollection([plt.Rectangle((x - 0.51 * w, y - 0.51 * h), 1.02 * w, 1.02 * h) for x, y in sel],
                                 facecolor=BAND_COLORS[b - 1], lw=0)
            ax.add_collection(pc)

    d = ours["chi2"] - ours["chi2_ref"]
    band = np.searchsorted(np.asarray(BANDS), d, side="right") + 1
    LS = np.repeat(ours["logs"][:, None], len(ours["logq"]), 1)
    LQ = np.repeat(ours["logq"][None, :], len(ours["logs"]), 0)
    mine = np.c_[LS.ravel(), LQ.ravel(), band.ravel()]
    mine_a = np.c_[LS.ravel(), ours["alpha"].ravel(), band.ravel()]
    cells(axs[0], paper_left, *cell)
    cells(axs[1], [r for r in mine if r[2] <= 5], *cell)
    cells(axs[2], paper_right, cell[0], alpha_height)
    cells(axs[3], [r for r in mine_a if r[2] <= 5], cell[0], alpha_height)
    for a, t in zip(axs, ("Han et al. 2025", "lenscape", "Han et al. 2025", "lenscape")):
        a.set_title(t); a.set_xlim(*xlim); a.set_xlabel("log s")
    for a in axs[:2]:
        a.set_ylim(*qlim); a.set_ylabel("log q")
    for a in axs[2:]:
        a.set_ylim(*alim); a.set_ylabel("alpha (rad)")
    fig.suptitle(title + f"  (bands: dchi2 < {', '.join(f'{b:g}' for b in BANDS)}; orange = best)")
    fig.tight_layout()
    return fig
