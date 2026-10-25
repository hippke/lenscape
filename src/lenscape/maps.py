"""Delta chi2 maps over (log s, log q): the public entry points.

lattice()  profile chi2 at every point of a (log s, log q) lattice (s and q fixed, the other parameters fitted),
           computed in stages so that fits do not get stuck in local minima:
             1. cell fits: best fit anywhere inside each lattice cell (s, q free inside the cell), started from a
                coarse seed grid and refined from the neighbours' solutions;
             2. lattice-point fits started from the cell solutions nearby;
             3. neighbour refinement: points are refitted from their neighbours' solutions until none improves;
             4. outlier recalculation: points worse than their neighbours get a deep search.
refine()   continues from a lattice() result: cells crossed by a Delta chi2 band edge are split 2 x 2, `levels`
           times (cell minima, as in stage 1).

Both run one process per fit (`workers` at a time) and print numbered stages and progress bars (tqdm, on a
terminal; LENSCAPE_PROGRESS=0/1 forces them off/on).
"""
import os, time
import numpy as np
from .optimize import minimize, least_squares_lm
from . import config as C, grid as G, data as D, fit as F, model as M, amr

BANDS = (4.0, 16.0, 36.0, 64.0, 100.0)


def _workers(n):
    return n or max(1, (os.cpu_count() or 2) - 1)


def _point_from_starts(ls, lq, starts, cfg, lm_start=None):
    """Lattice point (s, q fixed) from the given start vectors (t0, u0, tE, alpha, log rho, ...). lm_start: an
    extra start close to the solution (interpolated from neighbouring known solutions), fitted by Levenberg-Marquardt
    alone (~20 light curves)."""
    be = F.make_backend(cfg["grid_backend"])
    co = F.Coords(cfg.get("coords", "teff"))
    fx = lambda y: M.chi2(be, F.full_p(co.to_x(y), ls, lq), G._DATA)
    best = (np.inf, None)
    if lm_start is not None and co.kind == "teff":
        y = co.from_x(np.asarray(lm_start, float))
        fr = lambda y: M.chi2_residuals(be, np.asarray(F.full_p(co.to_x(y), ls, lq), float), G._DATA)
        q = least_squares_lm(fr, y, F.LM_STEP, F.LM_CAP)
        c = fx(q.x)
        if np.isfinite(c):
            best = (c, co.to_x(q.x))
    for x0 in starts:
        y = co.from_x(np.asarray(x0[:5], float))
        c = fx(y)
        if cfg.get("optimizer", "nm") != "nm":
            c1, y1 = F.local_fit(be, G._DATA, ls, lq, y, cfg, 0.3, 1000)
            if c1 < c:
                c, y = c1, y1
            if c < best[0]:
                best = (c, co.to_x(y))
            continue
        for step in (0.3, 0.1):
            sim = np.vstack([y] + [y + np.eye(5)[k] * s for k, s in enumerate(co.steps(y, step))])
            r = minimize(fx, y, method="Nelder-Mead", options=dict(maxfev=1000, adaptive=True, initial_simplex=sim,
                                                                 fatol=1e-2))
            if r.fun < c:
                c, y = r.fun, r.x
        if c < best[0]:
            best = (c, co.to_x(y))
    return best


def _interp_start(known, dist, tie=0.05, max_dist=1.5):
    """Mean (in fit coordinates, alpha unwrapped) of the equally nearest known solutions: on a lattice refined by 2,
    the bilinear interpolation at a cell centre (4 corners) or edge midpoint (2 ends), the point itself if known.
    None if they are farther than max_dist (new-lattice steps), differ in the sign of u0 or spread in alpha > 0.3."""
    k = np.argsort(dist)[:8]
    k = k[dist[k] <= dist[k[0]] + tie]
    if dist[k[0]] > max_dist:
        return None
    P = np.array([known[m][3] for m in k], float)
    if len(set(np.sign(P[:, 1]))) > 1:
        return None
    co = F.Coords("teff")
    Y = np.array([co.from_x(p) for p in P])
    Y[:, 3] = Y[0, 3] + (Y[:, 3] - Y[0, 3] + np.pi) % (2 * np.pi) - np.pi
    if np.ptp(Y[:, 3]) > 0.3:
        return None
    return co.to_x(Y.mean(0))


def _prepare(data, cfg, best, bin_wings):
    cfg = dict(C.DEFAULT if cfg is None else cfg)
    if best is not None:
        best = np.asarray(best, float)
        cfg["seed"] = cfg.get("seed") or dict(t0=float(best[0]), u0=float(best[1]), tE=float(best[2]),
                                              logrho=float(best[6]))
    else:
        cfg["seed"] = cfg.get("seed") or F.seed_from_data(data)
    fitdata = D.bin_wings(data, best[0], best[2]) if (bin_wings and best is not None) else data
    G._DATA = fitdata
    return cfg, fitdata


def lattice(data, logs, logq, best=None, cfg=None, workers=None, bands=BANDS, seed_grid=2, bin_wings=True,
            log=print):
    """Profile chi2 on the lattice logs x logq (1-D arrays, uniform steps).
    data: Data; best: best-fit vector (t0, u0, tE, log s, log q, alpha, log rho) of the event, used as the start of
    the fits and to bin the far wings (bin_wings); without it, a start is estimated from the data. bands: Delta chi2
    levels whose edges guide the cell refinement. Returns a dict: logs, logq, chi2 (ns, nq), pars (ns, nq, 5:
    t0, u0, tE, alpha, log rho), chi2_ref (lowest chi2 found), cells (stage-1 cell fits), wall-clock times."""
    workers = _workers(workers)
    logs, logq = np.asarray(logs, float), np.asarray(logq, float)
    dls, dlq = logs[1] - logs[0], logq[1] - logq[0]
    cfg, fitdata = _prepare(data, cfg, best, bin_wings)
    seed = seed_grid if min(len(logs), len(logq)) >= 2 * seed_grid else 0
    ST = G.Stages(5 if seed else 4, log)
    log(f"lattice {len(logs)} x {len(logq)} = {len(logs) * len(logq)} points; N = {len(data.t)}"
        + (f" ({len(fitdata.t_eval)} evaluated after binning the wings)" if hasattr(fitdata, "t_eval") else ""))
    t0 = time.time()
    B = amr.run_contour((logs[0] - dls / 2, logs[-1] + dls / 2), (logq[0] - dlq / 2, logq[-1] + dlq / 2),
                        (len(logs), len(logq)), 0, cfg, fitdata, np.asarray(bands, float), workers=workers, seed=seed,
                        min_ds=np.inf, min_dq=np.inf, stage=ST, log=log)
    t1 = time.time()
    ST("Lattice point fits", "s and q fixed at each lattice point, started from the cell fits")
    tasks = {}
    for i, ls in enumerate(logs):
        for j, lq in enumerate(logq):
            near = np.where((np.abs(B["logs"] - ls) <= 1.5 * dls) & (np.abs(B["logq"] - lq) <= 1.5 * dlq))[0]
            near = near[np.argsort(B["chi2"][near])][:3]
            starts = [B["x"][k].copy() for k in near if np.all(np.isfinite(B["x"][k][:5]))]
            tasks[(i, j)] = (float(ls), float(lq), starts, cfg)
    G.warmup(cfg, fitdata)
    res, _ = G.run_isolated(_point_from_starts, tasks, workers, log=log, desc="lattice point fits")
    chi = np.full((len(logs), len(logq)), np.nan)
    pars = np.full((len(logs), len(logq), 5), np.nan)
    for (i, j), v in res.items():
        if v is not None and v[1] is not None:
            chi[i, j], pars[i, j] = v
    t2 = time.time()
    ST("Neighbour refinement", "refit points whose neighbours found a better solution")
    chi2, pars2 = G.relax(chi, pars, logs, logq, cfg, workers=workers, data=fitdata, log=log)
    ST("Recalculate outliers", "deep search for points that are worse than their neighbours")
    chi2, pars2, _ = G.repair_outliers(chi2, pars2, logs, logq, cfg, workers=workers, data=fitdata, log=log, passes=10)
    t3 = time.time()
    ref = float(min(np.nanmin(chi2), np.min(B["chi2"])))
    log(f"done in {t3 - t0:.0f}s; lowest chi2 {ref:.2f}")
    return dict(logs=logs, logq=logq, chi2=chi2, pars=pars2, chi2_ref=ref, chi2_first=chi, cells=B,
                wall=t3 - t0, wall_cells=t1 - t0, wall_points=t2 - t1, wall_refinement=t3 - t2)


def refine(result, data, levels=2, best=None, cfg=None, workers=None, bands=BANDS, bin_wings=True, log=print):
    """Finer cells along the Delta chi2 band edges of a lattice() result: each level halves the cell size where a
    band edge runs between neighbouring cells. The lattice's cell fits are the first level (each improved by its
    lattice point where that is lower). Returns the leaves: logs, logq (cell centres), hs, hq (half widths), chi2
    (best chi2 inside the cell), x (best fit: t0, u0, tE, alpha, log rho, log s, log q), level, chi2_ref."""
    workers = _workers(workers)
    logs, logq, chi, pars, B = result["logs"], result["logq"], result["chi2"], result["pars"], result["cells"]
    dls, dlq = logs[1] - logs[0], logq[1] - logq[0]
    cfg, fitdata = _prepare(data, cfg, best, bin_wings)
    start = {}
    for ls, lq, c, x in zip(B["logs"], B["logq"], B["chi2"], B["x"]):
        i, j = int(round((ls - logs[0]) / dls)), int(round((lq - logq[0]) / dlq))
        start[(i, j)] = (float(c), x if np.all(np.isfinite(x)) else None)
    for i in range(len(logs)):
        for j in range(len(logq)):
            if np.isfinite(chi[i, j]) and chi[i, j] < start.get((i, j), (np.inf,))[0]:
                start[(i, j)] = (float(chi[i, j]), np.r_[pars[i, j], logs[i], logq[j]])
    ST = G.Stages(levels, log)
    A = amr.run_contour((logs[0] - dls / 2, logs[-1] + dls / 2), (logq[0] - dlq / 2, logq[-1] + dlq / 2),
                        (len(logs), len(logq)), levels, cfg, fitdata, np.asarray(bands, float), workers=workers,
                        start=start, min_ds=0.99 * dls / 2 ** levels, min_dq=0.99 * dlq / 2 ** levels, stage=ST,
                        log=log)
    A["chi2_ref"] = float(min(result["chi2_ref"], A["chi2"].min()))
    return A


def _known_solutions(previous):
    """(log s, log q, chi2, 5-vector) of all solutions in earlier lattice() / refine() / densify() results."""
    rows = []
    for r in previous:
        if np.ndim(r["chi2"]) == 2:                                    # lattice points
            for i, ls in enumerate(r["logs"]):
                for j, lq in enumerate(r["logq"]):
                    if np.isfinite(r["chi2"][i, j]) and np.all(np.isfinite(r["pars"][i, j])):
                        rows.append((ls, lq, r["chi2"][i, j], r["pars"][i, j]))
        else:                                                          # cells: best fit inside the cell
            for x, c in zip(r["x"], r["chi2"]):
                if np.isfinite(c) and np.all(np.isfinite(x)):
                    rows.append((x[5], x[6], c, x[:5]))
    return rows


def densify(previous, data, logs, logq, best=None, cfg=None, workers=None, n_starts=3, bin_wings=True, max_sweeps=30,
            log=print):
    """A (usually finer) lattice started from known solutions instead of from scratch: every new point is fitted
    (s, q fixed) from the n_starts nearest solutions of `previous` (a list of lattice(), refine() or densify()
    results), then neighbour refinement and outlier recalculation as in lattice(). No seed grid and no cell fits,
    so a 2x finer lattice costs about 4x the lattice-point stage of the original map rather than a new map.
    Returns a dict like lattice() (without "cells")."""
    workers = _workers(workers)
    logs, logq = np.asarray(logs, float), np.asarray(logq, float)
    cfg, fitdata = _prepare(data, cfg, best, bin_wings)
    interp = bool(cfg.get("interp_start", False))
    known = _known_solutions(previous)
    K = np.array([(a, b) for a, b, _, _ in known])
    # distances in units of the new lattice step, so that s and q count alike
    ds, dq = abs(logs[1] - logs[0]), abs(logq[1] - logq[0])
    ST = G.Stages(3, log)
    ST("Lattice point fits", f"{len(logs) * len(logq)} points started from the {n_starts} nearest of "
                             f"{len(known)} known solutions")
    t0 = time.time()
    tasks = {}
    for i, ls in enumerate(logs):
        for j, lq in enumerate(logq):
            dist = np.hypot((K[:, 0] - ls) / ds, (K[:, 1] - lq) / dq)
            near = np.argsort(dist)[:4 * n_starts]
            # the nearest known solution always (a point of an earlier lattice keeps its own solution as a start),
            # then the best chi2 among the close ones
            ns_ = n_starts - 1 if interp else n_starts
            near = [near[0]] + sorted(near[1:], key=lambda k: (dist[k] > 2.0, known[k][2]))[:ns_ - 1]
            tasks[(i, j)] = (float(ls), float(lq), [np.asarray(known[k][3], float) for k in near], cfg,
                             _interp_start(known, dist) if interp else None)
    G.warmup(cfg, fitdata)
    res, _ = G.run_isolated(_point_from_starts, tasks, workers, log=log, desc="lattice point fits")
    chi = np.full((len(logs), len(logq)), np.nan)
    pars = np.full((len(logs), len(logq), 5), np.nan)
    for (i, j), v in res.items():
        if v is not None and v[1] is not None:
            chi[i, j], pars[i, j] = v
    t1 = time.time()
    ST("Neighbour refinement", "refit points whose neighbours found a better solution")
    chi2, pars2 = G.relax(chi, pars, logs, logq, cfg, workers=workers, data=fitdata, log=log, max_sweeps=max_sweeps)
    ST("Recalculate outliers", "deep search for points that are worse than their neighbours")
    chi2, pars2, _ = G.repair_outliers(chi2, pars2, logs, logq, cfg, workers=workers, data=fitdata, log=log, passes=10)
    t2 = time.time()
    ref = float(min(np.nanmin(chi2), min(c for _, _, c, _ in known)))
    log(f"done in {t2 - t0:.0f}s; lowest chi2 {ref:.2f}")
    return dict(logs=logs, logq=logq, chi2=chi2, pars=pars2, chi2_ref=ref, chi2_first=chi, wall=t2 - t0,
                wall_points=t1 - t0, wall_refinement=t2 - t1)


def polish(result, data, previous=(), best=None, cfg=None, workers=None, max_sweeps=30, bin_wings=True, log=print):
    """Improve a lattice result in place of recomputing it: at points that coincide with points of `previous`
    lattice results, keep the better solution; then neighbour refinement (up to max_sweeps) and outlier
    recalculation. Returns a new result dict."""
    workers = _workers(workers)
    cfg, fitdata = _prepare(data, cfg, best, bin_wings)
    logs, logq = result["logs"], result["logq"]
    chi, pars = result["chi2"].copy(), result["pars"].copy()
    n_old = 0
    for p in previous:
        for a, ls in enumerate(p["logs"]):
            i = int(np.argmin(np.abs(logs - ls)))
            if abs(logs[i] - ls) > 1e-9:
                continue
            for b, lq in enumerate(p["logq"]):
                j = int(np.argmin(np.abs(logq - lq)))
                if abs(logq[j] - lq) < 1e-9 and p["chi2"][a, b] < chi[i, j]:
                    chi[i, j], pars[i, j] = p["chi2"][a, b], p["pars"][a, b]
                    n_old += 1
    log(f"{n_old} points taken from earlier lattices")
    ST = G.Stages(2, log)
    ST("Neighbour refinement", "refit points whose neighbours found a better solution")
    chi, pars = G.relax(chi, pars, logs, logq, cfg, workers=workers, data=fitdata, log=log, max_sweeps=max_sweeps)
    ST("Recalculate outliers", "deep search for points that are worse than their neighbours")
    chi, pars, _ = G.repair_outliers(chi, pars, logs, logq, cfg, workers=workers, data=fitdata, log=log, passes=10)
    return dict(result, chi2=chi, pars=pars, chi2_ref=float(min(result["chi2_ref"], np.nanmin(chi))))


def _alpha_minima_point(ls, lq, x0, cfg, n_alpha, tol_alpha):
    be = F.make_backend(cfg["grid_backend"])
    co = F.Coords(cfg.get("coords", "teff"))
    fx = lambda y: M.chi2(be, F.full_p(co.to_x(y), ls, lq), G._DATA)
    found = []
    for a in np.arange(n_alpha) * 2 * np.pi / n_alpha:
        x = np.array(x0, float)
        x[3] = a
        y = co.from_x(x)
        if cfg.get("optimizer", "nm") != "nm" and co.kind == "teff":
            # exploration from this angle as before (600), then Levenberg-Marquardt instead of the second Nelder-Mead
            fr = lambda y: M.chi2_residuals(be, np.asarray(F.full_p(co.to_x(y), ls, lq), float), G._DATA)
            c, y = F.nm_then_lm(fx, fr, y, None, 600)
            found.append((float(c), co.to_x(y)))
            continue
        r = minimize(fx, y, method="Nelder-Mead", options=dict(maxfev=600, adaptive=True))
        y = r.x
        sim = np.vstack([y] + [y + np.eye(5)[k] * s for k, s in enumerate(co.steps(y, 0.1))])
        r = minimize(fx, y, method="Nelder-Mead", options=dict(maxfev=800, adaptive=True, initial_simplex=sim,
                                                             fatol=1e-2))
        found.append((float(r.fun), co.to_x(r.x)))
    found.sort(key=lambda v: v[0])
    out = []
    for c, x in found:                                                 # distinct minima (in alpha, mod 2 pi)
        al = np.mod(x[3], 2 * np.pi)
        if all(abs((al - np.mod(y[3], 2 * np.pi) + np.pi) % (2 * np.pi) - np.pi) > tol_alpha for _, y in out):
            out.append((c, x))
    return out


def alpha_minima(result, data, best=None, cfg=None, workers=None, n_alpha=20, tol_alpha=0.02, bin_wings=True,
                 log=print):
    """All local minima in the trajectory angle at every point of a lattice() / densify() result: at each (s, q),
    fits from n_alpha angles (the other parameters from that point's best fit) and keeps every distinct minimum
    (|delta alpha| > tol_alpha). This is how (log s, alpha) maps of papers are usually drawn: every local solution
    at its log s and alpha. Returns an array with rows (log s, log q, chi2, t0, u0, tE, alpha, log rho)."""
    workers = _workers(workers)
    cfg, fitdata = _prepare(data, cfg, best, bin_wings)
    logs, logq, chi, pars = result["logs"], result["logq"], result["chi2"], result["pars"]
    tasks = {(i, j): (float(logs[i]), float(logq[j]), pars[i, j], cfg, n_alpha, tol_alpha)
             for i in range(len(logs)) for j in range(len(logq)) if np.all(np.isfinite(pars[i, j]))}
    G.Stages(1, log)("Trajectory-angle minima", f"{n_alpha} starting angles at each of {len(tasks)} points")
    G.warmup(cfg, fitdata)
    res, _ = G.run_isolated(_alpha_minima_point, tasks, workers, log=log, desc="alpha minima")
    rows = [(logs[i], logq[j], c, *x) for (i, j), v in res.items() if v for c, x in v]
    return np.array(rows, float).reshape(-1, 8)


# ---------------------------------------------------------------------------------------------- (log s, alpha) maps
def _sa_vector(y, ls, al):
    """(t0, t_eff, ln tE, log q, log t_*) at fixed (log s, alpha) -> the 7-vector (t0, u0, tE, log s, log q, alpha,
    log rho)."""
    t0, te, ltE, lq, lts = y
    tE = np.exp(ltE)
    return np.array([t0, te / tE, tE, ls, lq, al, lts - np.log10(tE)])


def _sa_coords(p):
    t0, u0, tE = p[0], p[1], abs(p[2]) + 1e-3
    return np.array([t0, u0 * tE, np.log(tE), p[4], p[6] + np.log10(tE)])


# Levenberg-Marquardt steps and caps in the (log s, alpha) coordinates (t0, t_eff, ln tE, log q, log t_*)
SA_LM_STEP = np.array([1e-4, 1e-5, 1e-4, 1e-4, 1e-3])
SA_LM_CAP = np.array([0.3, 0.05, 0.1, 0.05, 0.2])


def _sa_fit(ls, al, starts, cfg, cand=None, cval=np.inf, margin=1.0):
    """Best fit at fixed (log s, alpha) from 7-vector starts (their log s and alpha are replaced). cand/cval: in
    refinement sweeps, refit only if a start already beats cval - margin with one chi2 evaluation."""
    be = F.make_backend(cfg["grid_backend"])
    als = np.broadcast_to(np.asarray(al, float), (len(starts),))    # model alpha per start (u0 signs may differ)
    ys = [(_sa_coords(np.asarray(s, float)), a) for s, a in zip(starts, als)]
    if cand is not None:
        ys = [(y, a) for y, a in ys if M.chi2(be, _sa_vector(y, ls, a), G._DATA) < cval - margin]
        if not ys:
            return (np.inf, None)                      # nothing to refit (None is reserved for a lost task)
    best = (np.inf, None)
    for y, al in ys:
        c, y1 = _sa_fit_one(be, ls, al, y, cfg)
        if c < best[0]:
            best = (c, _sa_vector(y1, ls, al))
    return best


def _sa_fit_one(be, ls, al, y, cfg):
    """One start y (coordinates of _sa_coords) at fixed (log s, model alpha al): (chi2, y)."""
    f = lambda y: M.chi2(be, _sa_vector(y, ls, al), G._DATA)
    if cfg.get("optimizer", "nm") != "nm":
        fr = lambda y: M.chi2_residuals(be, _sa_vector(y, ls, al), G._DATA)
        st = np.array([0.05, 0.1 * abs(y[1]) + 1e-3, 0.1, 0.1, 0.1]) * 0.3
        sim = np.vstack([y] + [y + np.eye(5)[k] * st[k] for k in range(5)])
        c, y1 = F.nm_then_lm(f, fr, y, sim, cfg.get("explore_nfev", 100), SA_LM_STEP, SA_LM_CAP)
        c0 = f(y)
        return (c0, y) if c0 < c else (c, y1)
    c = f(y)
    for step in (0.3, 0.1):
        st = np.array([0.05, 0.1 * abs(y[1]) + 1e-3, 0.1, 0.1, 0.1]) * step
        sim = np.vstack([y] + [y + np.eye(5)[k] * st[k] for k in range(5)])
        r = minimize(f, y, method="Nelder-Mead", options=dict(maxfev=1000, adaptive=True, initial_simplex=sim,
                                                             fatol=1e-2))
        if r.fun < c:
            c, y = r.fun, r.x
    return c, y


def _alpha_sweeps(chi, pars, logs, alphas, u0_ref, cfg, workers, margin, max_sweeps, changed_only, log, smooth=None):
    """Neighbour refinement of a (log s, alpha) profile map (alpha periodic): a point is refitted from the solutions
    of neighbours better by > margin, each at its own model alpha (a neighbour with the other sign of u0 is fitted
    at alpha - pi or alpha + pi, so solutions of both signs spread)."""
    model_alpha = lambda p, al: al if np.sign(p[1]) == np.sign(u0_ref) else al - np.pi
    smooth = cfg.get("smooth") if smooth is None else smooth
    ns, na = chi.shape
    # alpha is periodic only if the grid covers the full circle (a partial range, e.g. 3.6 .. 5.6, has two edges)
    wrap = na > 1 and (alphas[-1] - alphas[0]) + (alphas[1] - alphas[0]) >= 2 * np.pi - 1e-9
    chi, pars = chi.copy(), pars.copy()
    stale = np.ones(chi.shape, bool)                 # 3 x 3 neighbourhood changed since the point was last checked
    for sweep in range(max_sweeps):
        # after the first sweep only stale points (alpha periodic): the others would repeat a rejected refit with
        # the same inputs (see grid.relax)
        need = stale if (sweep > 0 and changed_only) else np.ones(chi.shape, bool)
        force = G._rough(chi, smooth, wrap) & need if smooth is not None else np.zeros(chi.shape, bool)
        checked = need.copy()
        tasks = {}
        for i in range(ns):
            for j in range(na):
                if not need[i, j]:
                    continue
                nb = [(i + a, (j + b) % na) for a in (-1, 0, 1) for b in (-1, 0, 1)
                      if (a or b) and 0 <= i + a < ns and (wrap or 0 <= j + b < na)]
                if force[i, j]:                         # outlier: refit from all neighbours, no pre-check
                    cand = [pars[k] for k in nb if np.isfinite(chi[k]) and np.all(np.isfinite(pars[k]))]
                    if cand:
                        tasks[(i, j)] = (float(logs[i]), [model_alpha(c, alphas[j]) for c in cand], cand, cfg)
                    continue
                cand = [pars[k] for k in nb if np.isfinite(chi[k]) and (not np.isfinite(chi[i, j])
                                                                        or chi[k] < chi[i, j] - margin)]
                if cand:
                    als = [model_alpha(c, alphas[j]) for c in cand]
                    cv = float(chi[i, j]) if np.isfinite(chi[i, j]) else np.inf
                    tasks[(i, j)] = (float(logs[i]), als, cand, cfg, True, cv, margin)
        if not tasks:
            break
        res, _ = G.run_isolated(_sa_fit, tasks, workers, log=log, desc=f"neighbour refinement, pass {sweep + 1}")
        n = 0
        changed = np.zeros(chi.shape, bool)
        for k in tasks:
            if k not in res or res[k] is None:          # task lost (crash): check again next sweep
                checked[k] = False
        for k, v in res.items():
            if v is not None and v[1] is not None and (v[0] < chi[k] - (0.01 if force[k] else 1e-9)
                                                       or not np.isfinite(chi[k])):
                chi[k], pars[k] = v
                changed[k] = True
                n += 1
        stale = (stale & ~checked) | G._near(changed, wrap)
        log(f"neighbour refinement, pass {sweep + 1}: {len(tasks)} points checked, {n} improved")
        if n == 0:
            break
    return chi, pars


def alpha_profile_refine(result, data, best=None, cfg=None, workers=None, margin=1.0, max_sweeps=30, bin_wings=True,
                         log=print, changed_only=True, smooth=None):
    """More neighbour-refinement sweeps for an alpha_profile result (e.g. one stopped at max_sweeps). Returns a new
    result dict."""
    workers = _workers(workers)
    cfg, fitdata = _prepare(data, cfg, best, bin_wings)
    chi, pars = np.asarray(result["chi2"], float), np.asarray(result["pars"], float)
    u0_ref = _profile_u0_ref(result)
    t0 = time.time()
    G.warmup(cfg, fitdata)
    chi, pars = _alpha_sweeps(chi, pars, result["logs"], result["alphas"], u0_ref, cfg, workers, margin, max_sweeps,
                              changed_only, log, smooth)
    ref = float(min(np.nanmin(chi), result.get("chi2_ref", np.inf)))
    log(f"done in {time.time() - t0:.0f}s; lowest chi2 {ref:.2f}")
    return dict(logs=np.asarray(result["logs"]), alphas=np.asarray(result["alphas"]), chi2=chi, pars=pars,
                chi2_ref=ref, wall=time.time() - t0, u0_ref=float(u0_ref))


def _profile_u0_ref(result):
    """The sign of u0 that a (log s, alpha) profile counts at its plotted alpha (the other sign at alpha + pi). Stored
    as u0_ref; for older results: the sign at the points whose model alpha equals the plotted alpha (not the sign of
    the best point: after refinement it may be the mirror solution, counted at alpha + pi)."""
    if "u0_ref" in result:
        return float(result["u0_ref"])
    pars, alphas = np.asarray(result["pars"], float), np.asarray(result["alphas"], float)
    d = np.abs((pars[..., 5] - alphas[None, :] + np.pi) % (2 * np.pi) - np.pi)
    same = np.isfinite(d) & (d < 1e-6)
    return float(np.sign(np.sum(np.sign(pars[..., 1][same]))))


def _plot_alpha(p, u0_ref):
    """alpha as drawn in (log s, alpha) panels: a solution with the opposite sign of u0 counts at alpha + pi."""
    return np.mod(p[5] + (np.pi if np.sign(p[1]) != np.sign(u0_ref) else 0.0), 2 * np.pi)


def alpha_profile(previous, data, logs, alphas, best=None, cfg=None, workers=None, n_starts=3, margin=1.0,
                  max_sweeps=30, bin_wings=True, log=print, changed_only=True, u0_ref=None, smooth=None):
    """Profile chi2 over (log s, alpha): at every grid point, chi2 minimised over all other parameters including
    log q, so the map has a value everywhere (unlike a projection of lattice solutions). alpha is in the convention
    of (log s, alpha) panels: solutions with the opposite sign of u0 to the best fit are counted at alpha + pi, and
    the fit at such a point keeps that sign (alpha - pi in the model). Starts: the n_starts nearest known solutions
    of `previous` (lattice / densify results, refine cells, alpha_minima rows); then neighbour refinement sweeps.
    u0_ref: the sign of u0 counted at the plotted alpha; default: that of an alpha_profile result in `previous`,
    else of the best known solution (pass it when `previous` holds solutions of both signs, e.g. a refined profile
    converted to rows: its best point may be the mirror solution).
    Returns dict(logs, alphas, chi2 (ns, na), pars (ns, na, 7) model vectors, chi2_ref, wall, u0_ref)."""
    workers = _workers(workers)
    cfg, fitdata = _prepare(data, cfg, best, bin_wings)
    logs, alphas = np.asarray(logs, float), np.asarray(alphas, float)
    known = []                                                         # 7-vectors (t0, u0, tE, ls, lq, al, lr), chi2
    for r in previous:
        if isinstance(r, np.ndarray):                                  # alpha_minima rows
            for row in r:
                ls_, lq_, c, t0, u0, tE, al, lr = row
                known.append((np.array([t0, u0, tE, ls_, lq_, al, lr]), c))
        elif "alphas" in r:                                            # an alpha_profile result (7-vectors)
            P, C = np.asarray(r["pars"], float).reshape(-1, 7), np.asarray(r["chi2"], float).ravel()
            for p, c in zip(P, C):
                if np.isfinite(c) and np.all(np.isfinite(p)):
                    known.append((p, c))
        elif np.ndim(r["chi2"]) == 2:
            for i, ls_ in enumerate(r["logs"]):
                for j, lq_ in enumerate(r["logq"]):
                    p = r["pars"][i, j]
                    if np.isfinite(r["chi2"][i, j]) and np.all(np.isfinite(p)):
                        known.append((np.array([p[0], p[1], p[2], ls_, lq_, p[3], p[4]]), r["chi2"][i, j]))
        else:
            for x, c in zip(r["x"], r["chi2"]):
                if np.isfinite(c) and np.all(np.isfinite(x)):
                    known.append((np.array([x[0], x[1], x[2], x[5], x[6], x[3], x[4]]), c))
    if u0_ref is None:
        prof = [r for r in previous if isinstance(r, dict) and "alphas" in r]
        u0_ref = _profile_u0_ref(prof[0]) if prof else min(known, key=lambda v: v[1])[0][1]
    K = np.array([(p[3], _plot_alpha(p, u0_ref)) for p, _ in known])
    ds, da = abs(logs[1] - logs[0]), abs(alphas[1] - alphas[0])

    def model_alpha(p, al):                                            # alpha used in the model for a start p
        return al if np.sign(p[1]) == np.sign(u0_ref) else al - np.pi

    ST = G.Stages(2, log)
    ST("Fits at fixed (log s, alpha)", f"{len(logs) * len(alphas)} points from the {n_starts} nearest of "
                                       f"{len(known)} known solutions")
    t0 = time.time()
    tasks = {}
    for i, ls in enumerate(logs):
        for j, al in enumerate(alphas):
            dal = (K[:, 1] - al + np.pi) % (2 * np.pi) - np.pi
            dist = np.hypot((K[:, 0] - ls) / ds, dal / da)
            near = np.argsort(dist)[:4 * n_starts]
            near = [near[0]] + sorted(near[1:], key=lambda k: (dist[k] > 2.0, known[k][1]))[:n_starts - 1]
            # starts of one u0 sign per point (the nearest one's), fitted at the matching model alpha
            sg = np.sign(known[near[0]][0][1])
            starts = [known[k][0] for k in near if np.sign(known[k][0][1]) == sg]
            tasks[(i, j)] = (float(ls), float(model_alpha(known[near[0]][0], al)), starts, cfg)
    G.warmup(cfg, fitdata)
    res, _ = G.run_isolated(_sa_fit, tasks, workers, log=log, desc="(log s, alpha) fits")
    chi = np.full((len(logs), len(alphas)), np.nan)
    pars = np.full((len(logs), len(alphas), 7), np.nan)
    for (i, j), v in res.items():
        if v is not None and v[1] is not None:
            chi[i, j], pars[i, j] = v
    ST("Neighbour refinement", "refit points whose neighbours found a better solution")
    chi, pars = _alpha_sweeps(chi, pars, logs, alphas, u0_ref, cfg, workers, margin, max_sweeps, changed_only, log,
                              smooth)
    ref = float(min(np.nanmin(chi), min(c for _, c in known)))
    log(f"done in {time.time() - t0:.0f}s; lowest chi2 {ref:.2f}")
    return dict(logs=logs, alphas=alphas, chi2=chi, pars=pars, chi2_ref=ref, wall=time.time() - t0,
                u0_ref=float(u0_ref))
