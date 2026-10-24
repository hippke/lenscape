"""Paper figures of a fitted event: model light curves with residuals and zoom insets, and lens-system geometry
(caustics, source trajectory, equal-magnification contours). matplotlib is imported only here.

Solutions are parameter vectors p = (t0, u0, tE, log s, log q, alpha, log rho) on the data's time axis.
Magnitudes: every data set is mapped onto the flux system of the reference set (index `ref`) with the solution's
own source and blend fluxes, f_ref = (f - fb_k) / fs_k * fs_ref + fb_ref, and m = zp - 2.5 log10(f_ref).
Difference-imaging fluxes (e.g. KMTNet pySIS) can have fs_ref < 0 and no absolute zero point; then the magnified
source alone is shown: f_ref = (f - fb_k) / fs_k * |fs_ref| (blend set to zero).
"""
import numpy as np
from . import model as M, caustics as K

STYLES = ("-", ":", "--", "-.")
COLORS = ("#e8463a", "#1f8fff", "#4caf50", "#555555", "#9b59b6", "#f39c12")


def fluxes(backend, p, data):
    """Source and blend flux per data set (fs, fb) for the solution p."""
    return M.fluxes(backend.lc(M.clamp(p)[0], data.t), data)


def aligned(backend, p, data, ref=0, zp=22.0):
    """Data magnitudes and errors on the reference set's system for solution p: (mag, err, fs_ref, fb_ref), with
    the reference source and blend flux used (see the module notes for fs_ref < 0); points with a non-positive
    aligned flux get NaN."""
    fs, fb = fluxes(backend, p, data)
    fs0, fb0 = (fs[ref], fb[ref]) if fs[ref] > 0 else (-fs[ref], 0.0)
    k = data.idx
    f = (data.f - fb[k]) / fs[k] * fs0 + fb0
    e = data.e / np.abs(fs[k]) * fs0
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = f > 0
        mag = np.where(ok, zp - 2.5 * np.log10(np.where(ok, f, 1.0)), np.nan)
        err = np.where(ok, 2.5 / np.log(10) * e / np.where(ok, f, 1.0), np.nan)
    return mag, err, fs0, fb0


def model_mag(backend, p, t, fs_ref, fb_ref, zp=22.0):
    with np.errstate(invalid="ignore", divide="ignore"):
        return zp - 2.5 * np.log10(fs_ref * backend.lc(M.clamp(p)[0], np.asarray(t, float)) + fb_ref)


def plot_lightcurve(backend, solutions, data, labels=None, window=None, insets=(), ref=0, zp=22.0, t_offset=0.0,
                    colors=COLORS, n_model=4000, figsize=(7.0, 7.5), path=None, xlabel="HJD - 2450000", res_lim=0.3):
    """Model light curves of several solutions over the data (aligned with the first solution), zoom insets
    [(t1, t2), ...] in the top panel, and one residual panel per solution (data - model, each with its own
    alignment). window = (t1, t2) on the data's time axis; t_offset is added to the axis labels only."""
    import matplotlib.pyplot as plt
    sols = [np.asarray(p, float) for p in solutions]
    labels = labels or [f"Sol {chr(65 + i)}" for i in range(len(sols))]
    t1, t2 = window or (data.t.min(), data.t.max())
    sel = (data.t >= t1) & (data.t <= t2)
    tm = np.linspace(t1, t2, n_model)
    ns = len(sols)
    fig, axs = plt.subplots(1 + ns, 1, figsize=figsize, sharex=True,
                            gridspec_kw=dict(height_ratios=[3.2] + [1] * ns, hspace=0))
    top = axs[0]

    def data_panel(ax, mag, err, m):
        for k, name in enumerate(data.names):
            j = m & (data.idx == k)
            ax.errorbar(data.t[j] + t_offset, mag[j], err[j], fmt="o", ms=3, lw=0.8, color=colors[name] if isinstance(colors, dict) else colors[k % len(colors)],
                        label=name, zorder=2)

    mag, err, _, _ = aligned(backend, sols[0], data, ref, zp)
    data_panel(top, mag, err, sel)
    curves, lines = [], []
    for p, lab, ls in zip(sols, labels, STYLES * 4):
        _, _, fsp, fbp = aligned(backend, p, data, ref, zp)
        mm = model_mag(backend, p, tm, fsp, fbp, zp)
        curves.append((mm, fsp, fbp))
        lines += top.plot(tm + t_offset, mm, ls, color="k", lw=1.0, label=lab, zorder=3)
    top.invert_yaxis()
    top.set_ylabel("Magnitude")
    top.legend(handles=lines, loc="lower center", frameon=False, fontsize=8)
    for n_in, (a, b) in enumerate(insets):
        ax = top.inset_axes([0.05 + 0.6 * (n_in % 2), 0.62, 0.33, 0.33])
        m = (data.t >= a) & (data.t <= b)
        data_panel(ax, mag, err, m)
        tz = np.linspace(a, b, 800)
        for p, ls, (mm, fsp, fbp) in zip(sols, STYLES * 4, curves):
            ax.plot(tz + t_offset, model_mag(backend, p, tz, fsp, fbp, zp), ls, color="k", lw=0.9)
        ax.set_xlim(a + t_offset, b + t_offset)
        yy = mag[m]
        if yy.size:
            ax.set_ylim(np.nanmax(yy) + 0.15, np.nanmin(yy) - 0.15)
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        top.indicate_inset_zoom(ax, edgecolor="0.5")
    for ax, p, lab, (mm, fsp, fbp) in zip(axs[1:], sols, labels, curves):
        mg, er, _, _ = aligned(backend, p, data, ref, zp)
        res = mg - model_mag(backend, p, data.t, fsp, fbp, zp)
        data_panel(ax, res, er, sel)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylim(-res_lim, res_lim)
        ax.set_yticks([-0.2 * res_lim / 0.3, 0, 0.2 * res_lim / 0.3])
        ax.set_ylabel("Residual", fontsize=8)
        ax.text(0.01, 0.8, lab, transform=ax.transAxes, fontsize=8)
    axs[-1].set_xlim(t1 + t_offset, t2 + t_offset)
    axs[-1].set_xlabel(xlabel)
    axs[-1].ticklabel_format(axis="x", useOffset=False, style="plain")
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_geometry(solutions, labels=None, extent=(-0.3, 0.5, -0.2, 0.2), ncols=2, mirror=False, centre="primary",
                  levels=None, figsize=None, path=None, t_arrow=None):
    """Lens-system configuration per solution: caustics (red), source trajectory with direction arrow (black),
    lens positions (blue), equal-magnification contours of the point-source magnification (grey).
    centre: 'primary' (origin at the primary lens, as in most papers) or 'com' (centre of mass, our frame);
    mirror: reflect y (the (u0, alpha) -> (-u0, -alpha) symmetry; some papers draw the source above the axis)."""
    import matplotlib.pyplot as plt
    sols = [np.asarray(p, float) for p in solutions]
    labels = labels or [f"Sol {chr(65 + i)}" for i in range(len(sols))]
    nrows = -(-len(sols) // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize or (3.5 * ncols, 1.9 * nrows + 0.4), squeeze=False,
                            sharex=True, sharey=True)
    sy = -1.0 if mirror else 1.0
    for ax, p, lab in zip(axs.ravel(), sols, labels):
        s, q = 10 ** p[3], 10 ** p[4]
        x0 = K.lens_positions(s, q)[0] if centre == "primary" else 0.0
        x = np.linspace(extent[0], extent[1], 400) + x0
        y = np.linspace(extent[2], extent[3], 240)
        A = K.magnification_map(s, q, x, y * sy)
        lv = levels if levels is not None else np.geomspace(1.5, 300, 28)
        ax.contour(x - x0, y, np.log(A), levels=np.log(lv), colors="0.6", linewidths=0.5)
        for c in K.caustics(s, q, 2000):
            ax.plot(c.real - x0, sy * c.imag, color="red", lw=1.2)
        t = np.linspace(p[0] - 3 * p[2], p[0] + 3 * p[2], 2000)
        tr = K.trajectory(p, t)
        ax.plot(tr.real - x0, sy * tr.imag, color="k", lw=0.9)
        ta = p[0] if t_arrow is None else t_arrow
        a, b = K.trajectory(p, [ta, ta + 0.02 * p[2]])
        ax.annotate("", xy=(b.real - x0, sy * b.imag), xytext=(a.real - x0, sy * a.imag),
                    arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
        for xl in K.lens_positions(s, q):
            if extent[0] <= xl - x0 <= extent[1]:
                ax.plot(xl - x0, 0, "o", color="blue", ms=5)
        ax.text(0.04, 0.88, lab, transform=ax.transAxes, fontsize=9)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
    for ax in axs[-1]:
        ax.set_xlabel("$x$")
    for ax in axs[:, 0]:
        ax.set_ylabel("$y$")
    for ax in axs.ravel()[len(sols):]:
        ax.set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig
