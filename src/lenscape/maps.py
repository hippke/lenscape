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


_same_start = F.same_start


def _point_from_starts(ls, lq, starts, cfg, lm_start=None, c_ref=np.inf, smooth=False, c_cut=np.inf):
    """Lattice point (s, q fixed) from the given start vectors (t0, u0, tE, alpha, log rho, ...).
    lm_start: a start interpolated from the known solutions around the point; c_ref: the interpolation of those
    solutions' chi2; smooth: those solutions agree (same Delta chi2 band, spread < 0.1 dex). Steps, each taken only
    if the previous one did not settle the point (a result settles it when it lies within cfg["lm_gate"], default
    1.0, of c_ref and beats every other start's own chi2):
      1. smooth and cfg["adaptive"] (off in config.DEFAULT: measured no gain, the refinement then repairs the
         slightly worse values): one light curve at the interpolated start (its chi2 is a true profile value at the
         point, an upper bound of the minimum);
      2. Levenberg-Marquardt alone from the interpolated start (~20 light curves; MOA-2022-BLG-091, 2x -> 4x: this
         start is a median 0.06 above the solution, the nearest known solution 44);
      3. the starts, each a short Nelder-Mead and Levenberg-Marquardt (fit.local_fit), in the order of their chi2
         at the point, skipping a start in the basin of one already fitted (cfg["dedupe"], default on: the nearest
         known solutions of a converged coarse map are usually one and the same solution); only the best start
         when every start's chi2 at the point lies above c_cut (the point is far above the displayed bands)."""
    be = F.make_backend(cfg["grid_backend"])
    co = F.Coords(cfg.get("coords", "teff"))
    fx = lambda y: M.chi2(be, F.full_p(co.to_x(y), ls, lq), G._DATA)
    best = (np.inf, None)
    pre = []
    for x0 in starts:
        y = co.from_x(np.asarray(x0[:5], float))
        pre.append((fx(y), y))
    if cfg.get("rho_starts") and co.kind == "teff":
        # source-size screen: each start with log rho set to these values (one chi2 each); a variant that beats its
        # start becomes an extra start. Separate small- and large-source local minima exist (MOA-2022-BLG-091: in its
        # wide-side arm log rho -4 ... -5 is 8-15 better than the best fit's -2.9 at otherwise equal parameters), and
        # neither the angle screen nor the neighbour refinement moves a whole region from one to the other
        for c0, y in list(pre):
            ltE = np.log10(np.exp(y[2]))                      # y[4] = log t_* = log rho + log tE
            vs = [np.r_[y[:4], lr + ltE] for lr in cfg["rho_starts"]]
            cv = [fx(v) for v in vs]
            if cfg.get("rho_fit"):                            # fit every variant (the small source wins only
                pre += list(zip(cv, vs))                      # after the other parameters adjust to it)
                continue
            k = int(np.argmin(cv))
            if cv[k] < c0:
                pre.append((cv[k], vs[k]))
    if cfg.get("point_screen") and co.kind == "teff" and pre:
        # valley enumeration at the point: (t0, u0, tE) are pinned by the single-lens part of the light curve, so
        # the distinct valleys differ in the trajectory angle and the source size. The best start's (t0, t_eff,
        # ln tE) with cfg["point_screen"] angles x the source-size values, one chi2 each; the best
        # cfg["point_screen_keep"] (3) outside the basins of the existing starts become extra starts
        c0, y0 = min(pre, key=lambda v: v[0])
        ltE = np.log10(np.exp(y0[2]))
        rhos = [y0[4]] + [lr + ltE for lr in (cfg.get("rho_starts") or ())]
        cand = [np.r_[y0[:3], a, lr] for a in np.linspace(0, 2 * np.pi, int(cfg["point_screen"]), endpoint=False)
                for lr in rhos]
        cand = sorted(((fx(v), v) for v in cand), key=lambda v: v[0])
        kept = 0
        for c, v in cand:
            if kept >= cfg.get("point_screen_keep", 3):
                break
            if np.isfinite(c) and not any(_same_start(v, y) for _, y in pre):
                pre.append((c, v))
                kept += 1
    gate = cfg.get("lm_gate", 1.0)
    if cfg.get("rho_fit") and cfg.get("rho_starts"):
        gate = None                       # the rho variants are to be fitted: no early settling on the interpolated start
    settled = lambda c: gate is not None and c < c_ref + gate and all(c < c0 for c0, _ in pre)
    if lm_start is not None and co.kind == "teff":
        y = co.from_x(np.asarray(lm_start, float))
        if smooth and cfg.get("adaptive", True):
            c = fx(y)
            if np.isfinite(c):
                best = (c, np.asarray(lm_start, float))
                if settled(c):
                    return best
        fr = lambda y: M.chi2_residuals(be, np.asarray(F.full_p(co.to_x(y), ls, lq), float), G._DATA)
        jac = F.lm_jacobian(be, G._DATA, ls, lq, co) if cfg.get("lm_jac", "fd") == "analytic" else None
        q = least_squares_lm(fr, y, F.LM_STEP, F.LM_CAP, tol=cfg.get("lm_tol", 1e-3), jac=jac)
        c = fx(q.x)
        if np.isfinite(c) and c < best[0]:
            best = (c, co.to_x(q.x))
            if settled(c):
                return best
    done = []                                        # starts and results already explored (fit coordinates)
    found = []                                       # every fit result (chi2, y): the point's solution branches
    order = sorted(pre, key=lambda v: v[0])
    if order and order[0][0] > c_cut:
        order = order[:1]
    top = cfg.get("precise_top")             # only the best `top` starts get the (precise) cfg fit; the others a
    cheap = dict(cfg, explore_nfev=100, lm_tol=0.03, lm_maxit=50)    # short one, and the best result a final polish
    for rank, (c, y) in enumerate(order):
        if cfg.get("dedupe", True) and any(_same_start(y, z) for z in done):
            continue
        done.append(y.copy())
        if cfg.get("optimizer", "nm") != "nm":
            c1, y1 = F.local_fit(be, G._DATA, ls, lq, y, cheap if (top is not None and rank >= top) else cfg, 0.3, 1000)
            if c1 < c:
                c, y = c1, y1
            done.append(np.asarray(y))
            found.append((c, np.asarray(y)))
            if c < best[0]:
                best = (c, co.to_x(y))
            continue
        for step in (0.3, 0.1):
            sim = np.vstack([y] + [y + np.eye(5)[k] * s for k, s in enumerate(co.steps(y, step))])
            r = minimize(fx, y, method="Nelder-Mead", options=dict(maxfev=1000, adaptive=True, initial_simplex=sim,
                                                                 fatol=1e-2))
            if r.fun < c:
                c, y = r.fun, r.x
        done.append(np.asarray(y))
        found.append((c, np.asarray(y)))
        if c < best[0]:
            best = (c, co.to_x(y))
    if top is not None and best[1] is not None and cfg.get("optimizer", "nm") != "nm" and co.kind == "teff":
        c1, y1 = F.local_fit(be, G._DATA, ls, lq, co.from_x(np.asarray(best[1], float)), cfg, 0.1, 1000)
        if c1 < best[0]:
            best = (c1, co.to_x(y1))
            found.append((c1, np.asarray(y1)))
    if cfg.get("branches"):
        # multi-branch map: the best cfg["branches"] distinct solutions are kept per point and propagated by the
        # refinement, so a branch that loses here can still seed the points where it wins
        return best + ([(c, co.to_x(y)) for c, y in F.top_distinct(found, co, cfg["branches"])],)
    return best


def _grid_knn(Kx, Ky, xs, ys, m, period_y=None):
    """For every point (xs[i], ys[j]) of a regular grid, the m known points (Kx, Ky) nearest in grid-step units
    (y periodic with period_y if given): a function (i, j) -> (indices, distances), sorted by distance (ties by index).
    Known points are bucketed by grid cell once; each query searches a growing window of cells until the m-th
    distance is certain, so a map of 1e5 points with 1e5 known solutions takes seconds instead of a full sort per
    point."""
    dx, dy = abs(xs[1] - xs[0]), abs(ys[1] - ys[0])
    nx, ny = len(xs), len(ys)
    Kx, Ky = np.asarray(Kx, float), np.asarray(Ky, float)
    if period_y is not None:                     # representative of each known y nearest to the grid's centre
        yc = 0.5 * (ys[0] + ys[-1])
        Ky = (Ky - yc + period_y / 2) % period_y - period_y / 2 + yc
    u, v = (Kx - xs[0]) / dx, (Ky - ys[0]) / dy
    bu = np.clip(np.round(u).astype(np.int64), 0, nx - 1)    # clamped: points outside stay findable at the edge
    bv = np.clip(np.round(v).astype(np.int64), 0, ny - 1)
    order = np.lexsort((np.arange(len(u)), bv, bu))
    keys = bu[order] * ny + bv[order]
    uk, start = np.unique(keys, return_index=True)
    stop = np.r_[start[1:], len(keys)]
    buckets = {int(k): order[a:b] for k, a, b in zip(uk, start, stop)}

    def query(i, j):
        r = 1
        while True:
            cells = [buckets.get(a * ny + b) for a in range(max(0, i - r), min(nx, i + r + 1))
                     for b in range(max(0, j - r), min(ny, j + r + 1))]
            cand = np.concatenate([c for c in cells if c is not None] or [np.zeros(0, np.int64)])
            covers_all = r >= max(nx, ny)
            if len(cand) >= m or covers_all:
                dyv = Ky[cand] - ys[j]
                if period_y is not None:
                    dyv = (dyv + period_y / 2) % period_y - period_y / 2
                d = np.hypot((Kx[cand] - xs[i]) / dx, dyv / dy)
                o = np.lexsort((cand, d))[:m]
                if covers_all or (len(o) == m and d[o[-1]] <= r - 0.5):
                    return cand[o], d[o]
            r *= 2
    return query


def _interp_start(known, dist, tie=0.05, max_dist=1.5):
    """dist: (indices, distances) of the nearest known solutions (from _grid_knn). Mean (in fit coordinates, alpha
    unwrapped) of the equally nearest known solutions, the mean of their chi2 and their chi2 values: on a lattice
    refined by 2, the bilinear interpolation at a cell centre (4 corners) or edge midpoint (2 ends), the point
    itself if known.
    None if they are farther than max_dist (new-lattice steps), differ in the sign of u0 or spread in alpha > 0.3."""
    idx, d = dist                                  # nearest known solutions and their distances, sorted
    k = idx[d <= d[0] + tie][:8]
    if d[0] > max_dist:
        return None
    P = np.array([known[m][3] for m in k], float)
    if len(set(np.sign(P[:, 1]))) > 1:
        return None
    co = F.Coords("teff")
    Y = np.array([co.from_x(p) for p in P])
    Y[:, 3] = Y[0, 3] + (Y[:, 3] - Y[0, 3] + np.pi) % (2 * np.pi) - np.pi
    if np.ptp(Y[:, 3]) > 0.3:
        return None
    c = np.array([known[m][2] for m in k])
    return co.to_x(Y.mean(0)), float(c.mean()), c


def _check_models(known, best, n=6, cut=500.0):
    """Models for the per-night binning check besides best: up to n known solutions (7-vectors) spread in chi2 over
    those within `cut` of the lowest, so that nights the map's other solutions bend are kept unbinned too."""
    rows = [(c, p) for p, c in known if np.isfinite(c) and np.all(np.isfinite(p))]
    if not rows:
        return []
    rows.sort(key=lambda v: v[0])
    rows = [r for r in rows if r[0] - rows[0][0] < cut]
    idx = np.unique(np.linspace(0, len(rows) - 1, n).round().astype(int))
    return [np.asarray(rows[k][1], float) for k in idx]


def _prepare(data, cfg, best, bin_wings, models=()):
    """cfg with a seed, and the data to fit: the wings binned per night when best is given (bin_wings), each
    night checked against best and the extra `models` (7-vectors)."""
    cfg = dict(C.DEFAULT if cfg is None else cfg)
    if best is not None:
        best = np.asarray(best, float)
        cfg["seed"] = cfg.get("seed") or dict(t0=float(best[0]), u0=float(best[1]), tE=float(best[2]),
                                              logrho=float(best[6]))
    else:
        cfg["seed"] = cfg.get("seed") or F.seed_from_data(data)
    if bin_wings and best is not None:
        # runs that a linear model within a night misrepresents at the best fit (an anomaly beyond one tE) stay unbinned
        be = F.make_backend(cfg["grid_backend"])
        K = cfg.get("bin_K", 0.5)
        if not len(models):                      # only the best fit to check against: bin no closer than one tE
            K = max(K, 1.0)
        fitdata = D.bin_wings(data, best[0], best[2], K=K, model=[(be, best)] + [(be, m) for m in models],
                              tol=cfg.get("bin_tol", 0.05))
    else:
        fitdata = data
    G._DATA = fitdata
    return cfg, fitdata


def lattice(data, logs, logq, best=None, cfg=None, workers=None, bands=BANDS, seed_grid=2, bin_wings=True,
            max_sweeps=30, dchi2_cut=None, log=print):
    """Profile chi2 on the lattice logs x logq (1-D arrays, uniform steps).
    data: Data; best: best-fit vector (t0, u0, tE, log s, log q, alpha, log rho) of the event, used as the start of
    the fits and to bin the far wings (bin_wings); without it, a start is estimated from the data. bands: Delta chi2
    levels whose edges guide the cell refinement. max_sweeps: neighbour-refinement passes (as in densify; a map
    stopped there is not converged and says so). dchi2_cut: in units of the last band edge, points above
    dchi2_cut * bands[-1] over the minimum get one start, no refinement and no outlier search unless a neighbour
    is below (None, the default here: everything everywhere; the cut judges a point by its starts before any fit,
    which also skipped visible points: MOA-2022-BLG-091 stage 1, points worse by > 1 than a converged map 19 % with
    the cut, 12.5 % without). cfg: default config.LATTICE (DEFAULT with source-size starts and precise local fits:
    the fits of a map from scratch start far from the solution). Returns a dict: logs, logq, chi2 (ns, nq), pars (ns, nq, 5:
    t0, u0, tE, alpha, log rho), chi2_ref (lowest chi2 found), cells (stage-1 cell fits), wall-clock times."""
    workers = _workers(workers)
    logs, logq = np.asarray(logs, float), np.asarray(logq, float)
    dls, dlq = logs[1] - logs[0], logq[1] - logq[0]
    cfg, fitdata = _prepare(data, C.LATTICE if cfg is None else cfg, best, bin_wings)
    seed = seed_grid if min(len(logs), len(logq)) >= 2 * seed_grid else 0
    ST = G.Stages(5 if seed else 4, log)
    log(f"lattice {len(logs)} x {len(logq)} = {len(logs) * len(logq)} points; N = {len(data.t)}"
        + (f" ({len(fitdata.t_eval)} evaluated after binning the wings"
           + (f"; {fitdata.n_unbinned} runs kept unbinned" if fitdata.n_unbinned else "") + ")"
           if hasattr(fitdata, "t_eval") else ""))
    t0 = time.time()
    B = amr.run_contour((logs[0] - dls / 2, logs[-1] + dls / 2), (logq[0] - dlq / 2, logq[-1] + dlq / 2),
                        (len(logs), len(logq)), 0, cfg, fitdata, np.asarray(bands, float), workers=workers, seed=seed,
                        min_ds=np.inf, min_dq=np.inf, stage=ST, log=log)
    t1 = time.time()
    ST("Lattice point fits", "s and q fixed at each lattice point, started from the cell fits")
    cut = None if dchi2_cut is None else dchi2_cut * float(bands[-1])
    c_cut = np.inf if cut is None else float(np.min(B["chi2"])) + cut
    tasks = {}
    for i, ls in enumerate(logs):
        for j, lq in enumerate(logq):
            near = np.where((np.abs(B["logs"] - ls) <= 1.5 * dls) & (np.abs(B["logq"] - lq) <= 1.5 * dlq))[0]
            near = near[np.argsort(B["chi2"][near])][:3]
            starts = [B["x"][k].copy() for k in near if np.all(np.isfinite(B["x"][k][:5]))]
            tasks[(i, j)] = (float(ls), float(lq), starts, cfg, None, np.inf, False, c_cut)
    # lowest cell chi2 first: fits near the minimum take longest, and a long task at the end idles the other workers
    tasks = dict(sorted(tasks.items(), key=lambda kv: min([B["chi2"][k] for k in np.where(
        (np.abs(B["logs"] - kv[1][0]) <= 1.5 * dls) & (np.abs(B["logq"] - kv[1][1]) <= 1.5 * dlq))[0]] or [np.inf])))
    G.warmup(cfg, fitdata)
    res, _ = G.run_isolated(_point_from_starts, tasks, workers, log=log, desc="lattice point fits")
    chi = np.full((len(logs), len(logq)), np.nan)
    pars = np.full((len(logs), len(logq), 5), np.nan)
    alts = {} if cfg.get("branches") else None
    for (i, j), v in res.items():
        if v is not None and v[1] is not None:
            chi[i, j], pars[i, j] = v[:2]
            if alts is not None and len(v) > 2:
                alts[(i, j)] = v[2]
    t2 = time.time()
    ST("Neighbour refinement", "refit points whose neighbours found a better solution")
    chi2, pars2 = G.relax(chi, pars, logs, logq, cfg, workers=workers, data=fitdata, log=log, max_sweeps=max_sweeps,
                          dchi2_cut=cut, alts=alts)
    ST("Recalculate outliers", "deep search for points that are worse than their neighbours")
    chi2, pars2, _ = G.repair_outliers(chi2, pars2, logs, logq, cfg, workers=workers, data=fitdata, log=log, passes=10,
                                       dchi2_cut=cut)
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
            dchi2_cut=5.0, log=print):
    """A (usually finer) lattice started from known solutions instead of from scratch: every new point is fitted
    (s, q fixed) from the n_starts nearest solutions of `previous` (a list of lattice(), refine() or densify()
    results), then neighbour refinement and outlier recalculation as in lattice(). No seed grid and no cell fits,
    so a 2x finer lattice costs about 4x the lattice-point stage of the original map rather than a new map.
    dchi2_cut: as in lattice(), in units of BANDS[-1] = 100. cfg: default config.DENSIFY (the LATTICE settings
    without the source-size starts). Returns a dict like lattice() (without "cells")."""
    workers = _workers(workers)
    logs, logq = np.asarray(logs, float), np.asarray(logq, float)
    known = _known_solutions(previous)
    cfg, fitdata = _prepare(data, C.DENSIFY if cfg is None else cfg, best, bin_wings,
                            _check_models([(np.array([x[0], x[1], x[2], a, b, x[3], x[4]]), c) for a, b, c, x in known], best)
                            if best is not None else ())
    interp = bool(cfg.get("interp_start", True))
    ref0 = min(c for _, _, c, _ in known)
    band = lambda dc: int(np.searchsorted(np.asarray(BANDS, float), dc))
    cut = None if dchi2_cut is None else dchi2_cut * float(BANDS[-1])
    c_cut = np.inf if cut is None else ref0 + cut
    K = np.array([(a, b) for a, b, _, _ in known])
    # distances in units of the new lattice step, so that s and q count alike
    ds, dq = abs(logs[1] - logs[0]), abs(logq[1] - logq[0])
    knn = _grid_knn(K[:, 0], K[:, 1], logs, logq, max(4 * n_starts, 8))
    ST = G.Stages(3, log)
    ST("Lattice point fits", f"{len(logs) * len(logq)} points started from the {n_starts} nearest of "
                             f"{len(known)} known solutions")
    t0 = time.time()
    tasks = {}
    for i, ls in enumerate(logs):
        for j, lq in enumerate(logq):
            nn, dn = knn(i, j)
            dist = dict(zip(nn.tolist(), dn.tolist()))
            near = nn[:4 * n_starts]
            # the nearest known solution always (a point of an earlier lattice keeps its own solution as a start),
            # then the best chi2 among the close ones
            near = [near[0]] + sorted(near[1:], key=lambda k: (dist[k] > 2.0, known[k][2]))[:n_starts - 1]
            ip = _interp_start(known, (nn, dn)) if interp else None
            smooth = False
            if ip is not None:                       # the surrounding solutions agree: same band, spread < 0.1 dex
                dc = np.sort(ip[2]) - ref0
                smooth = bool(band(dc[0]) == band(dc[-1]) and np.log10(dc[-1] + 1e-3) - np.log10(dc[0] + 1e-3) < 0.1)
            tasks[(i, j)] = (float(ls), float(lq), [np.asarray(known[k][3], float) for k in near], cfg,
                             ip[0] if ip else None, ip[1] if ip else np.inf, smooth, c_cut)
    tasks = dict(sorted(tasks.items(), key=lambda kv: kv[1][5]))      # lowest reference chi2 first (see lattice)
    G.warmup(cfg, fitdata)
    res, _ = G.run_isolated(_point_from_starts, tasks, workers, log=log, desc="lattice point fits")
    chi = np.full((len(logs), len(logq)), np.nan)
    pars = np.full((len(logs), len(logq), 5), np.nan)
    for (i, j), v in res.items():
        if v is not None and v[1] is not None:
            chi[i, j], pars[i, j] = v
    t1 = time.time()
    ST("Neighbour refinement", "refit points whose neighbours found a better solution")
    chi2, pars2 = G.relax(chi, pars, logs, logq, cfg, workers=workers, data=fitdata, log=log, max_sweeps=max_sweeps,
                          dchi2_cut=cut)
    ST("Recalculate outliers", "deep search for points that are worse than their neighbours")
    chi2, pars2, _ = G.repair_outliers(chi2, pars2, logs, logq, cfg, workers=workers, data=fitdata, log=log, passes=10,
                                       dchi2_cut=cut)
    t2 = time.time()
    ref = float(min(np.nanmin(chi2), min(c for _, _, c, _ in known)))
    log(f"done in {t2 - t0:.0f}s; lowest chi2 {ref:.2f}")
    return dict(logs=logs, logq=logq, chi2=chi2, pars=pars2, chi2_ref=ref, chi2_first=chi, wall=t2 - t0,
                wall_points=t1 - t0, wall_refinement=t2 - t1)


def polish(result, data, previous=(), best=None, cfg=None, workers=None, max_sweeps=30, bin_wings=True, dchi2_cut=5.0,
           log=print):
    """Improve a lattice result in place of recomputing it: at points that coincide with points of `previous`
    lattice results, keep the better solution; then neighbour refinement (up to max_sweeps) and outlier
    recalculation. Returns a new result dict."""
    workers = _workers(workers)
    known = _known_solutions([result] + list(previous))
    cfg, fitdata = _prepare(data, C.DENSIFY if cfg is None else cfg, best, bin_wings,
                            _check_models([(np.array([x[0], x[1], x[2], a, b, x[3], x[4]]), c) for a, b, c, x in known], best)
                            if best is not None else ())
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
    cut = None if dchi2_cut is None else dchi2_cut * float(BANDS[-1])
    chi, pars = G.relax(chi, pars, logs, logq, cfg, workers=workers, data=fitdata, log=log, max_sweeps=max_sweeps,
                        dchi2_cut=cut)
    ST("Recalculate outliers", "deep search for points that are worse than their neighbours")
    chi, pars, _ = G.repair_outliers(chi, pars, logs, logq, cfg, workers=workers, data=fitdata, log=log, passes=10,
                                     dchi2_cut=cut)
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


def _sa_fit(ls, al, starts, cfg, cand=None, cval=np.inf, margin=1.0, top=None):
    """Best fit at fixed (log s, alpha) from 7-vector starts (their log s and alpha are replaced). cand/cval: in
    refinement sweeps, refit only if a start already beats cval - margin with one chi2 evaluation. top: fit only
    the `top` starts with the lowest chi2 at the point (forced refits of rough points: all of up to eight were
    fitted before)."""
    be = F.make_backend(cfg["grid_backend"])
    als = np.broadcast_to(np.asarray(al, float), (len(starts),))    # model alpha per start (u0 signs may differ)
    ys = [(_sa_coords(np.asarray(s, float)), a) for s, a in zip(starts, als)]
    if cand is not None or top is not None:
        c0 = [M.chi2(be, _sa_vector(y, ls, a), G._DATA) for y, a in ys]
        if cand is not None:
            ys = [v for v, c in zip(ys, c0) if c < cval - margin]
        else:
            ys = [v for _, v in sorted(zip(c0, ys), key=lambda z: z[0])[:top]]
        if not ys:
            return (np.inf, None)                      # nothing to refit (None is reserved for a lost task)
    best = (np.inf, None)
    for y, al in ys:
        c, y1 = _sa_fit_one(be, ls, al, y, cfg)
        if c < best[0]:
            best = (c, _sa_vector(y1, ls, al))
    return best


def _same_sa(y, z):
    """_same_start for the (log s, alpha) fit coordinates (t0, t_eff, ln tE, log q, log t_*)."""
    tE = np.exp(0.5 * (y[2] + z[2]))
    return (abs(y[0] - z[0]) < 0.01 * tE and abs(y[1] - z[1]) < 0.02 * abs(y[1]) + 1e-4 * tE
            and abs(y[2] - z[2]) < 0.02 and abs(y[3] - z[3]) < 0.02 and abs(y[4] - z[4]) < 0.05)


def _sa_point(ls, al, starts, cfg, lm_start=None, c_ref=np.inf):
    """First-stage fit of a (log s, alpha) profile point from 7-vector starts (all of one u0 sign, fitted at the
    model alpha al), as _point_from_starts does for lattice points: Levenberg-Marquardt alone from the interpolated
    start lm_start (fit coordinates) settles the point when its chi2 lies within cfg["lm_gate"] of c_ref (the mean
    chi2 of the interpolated solutions) and below every start's own chi2; otherwise the starts are fitted in the
    order of their chi2 at the point, skipping starts in the basin of one already fitted (cfg["dedupe"])."""
    be = F.make_backend(cfg["grid_backend"])
    f = lambda y: M.chi2(be, _sa_vector(y, ls, al), G._DATA)
    pre = []
    for s_ in starts:
        y = _sa_coords(np.asarray(s_, float))
        pre.append((f(y), y))
    best = (np.inf, None)
    gate = cfg.get("lm_gate", 1.0)
    if lm_start is not None and cfg.get("profile_interp", True):
        y = np.asarray(lm_start, float)
        fr = lambda y: M.chi2_residuals(be, _sa_vector(y, ls, al), G._DATA)
        q = least_squares_lm(fr, y, SA_LM_STEP, SA_LM_CAP, tol=cfg.get("lm_tol", 1e-3))
        c = f(q.x)
        if np.isfinite(c):
            best = (c, _sa_vector(q.x, ls, al))
            if gate is not None and c < c_ref + gate and all(c < c0 for c0, _ in pre):
                return best
    done = []
    for c, y in sorted(pre, key=lambda v: v[0]):
        if cfg.get("profile_dedupe", True) and any(_same_sa(y, z) for z in done):
            continue
        done.append(y.copy())
        c1, y1 = _sa_fit_one(be, ls, al, y, cfg)
        done.append(np.asarray(y1))
        if c1 < c:
            c, y = c1, y1
        if c < best[0]:
            best = (c, _sa_vector(y, ls, al))
    return best


def _sa_interp(known, near, dn, sg, tie=0.05, max_dist=1.5):
    """Interpolated start of a profile point: the mean (fit coordinates) of the equally nearest known solutions
    with the sign sg of u0, and the mean of their chi2; None if too far or if their log q spreads by > 0.5."""
    k = [m for m, d in zip(near, dn) if d <= dn[0] + tie and np.sign(known[m][0][1]) == sg][:8]
    if not k or dn[0] > max_dist:
        return None
    Y = np.array([_sa_coords(np.asarray(known[m][0], float)) for m in k])
    if np.ptp(Y[:, 3]) > 0.5:
        return None
    return Y.mean(0), float(np.mean([known[m][1] for m in k]))


def _sa_fit_one(be, ls, al, y, cfg):
    """One start y (coordinates of _sa_coords) at fixed (log s, model alpha al): (chi2, y)."""
    f = lambda y: M.chi2(be, _sa_vector(y, ls, al), G._DATA)
    # Nelder-Mead by default: nm_lm was worse here from distant starts (MOA-2022-BLG-091 profile block: 10 % of the
    # points with Delta chi2 < 4 worse by > 1; test with spoiled other-sign starts), opt in with profile_optimizer
    if cfg.get("profile_optimizer", "nm") != "nm":
        fr = lambda y: M.chi2_residuals(be, _sa_vector(y, ls, al), G._DATA)
        st = np.array([0.05, 0.1 * abs(y[1]) + 1e-3, 0.1, 0.1, 0.1]) * 0.3
        sim = np.vstack([y] + [y + np.eye(5)[k] * st[k] for k in range(5)])
        c, y1 = F.nm_then_lm(f, fr, y, sim, cfg.get("explore_nfev", 100), SA_LM_STEP, SA_LM_CAP,
                             cfg.get("lm_tol", 1e-3), cfg.get("nm_fatol", 1e-2))
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


def _alpha_sweeps(chi, pars, logs, alphas, u0_ref, cfg, workers, margin, max_sweeps, changed_only, log, smooth=None,
                  one_sign=False):
    """Neighbour refinement of a (log s, alpha) profile map (alpha periodic): a point is refitted from the solutions
    of neighbours better by > margin, each at its own model alpha (a neighbour with the other sign of u0 is fitted
    at alpha - pi or alpha + pi, so solutions of both signs spread). cfg["relax_mode"] "queue" (default): the
    asynchronous engine grid.refine_queue (improvements spread as fast as the fits finish); "sweeps": Jacobi sweeps
    with a barrier each (changed_only: re-check only points next to a change)."""
    model_alpha = lambda p, al: al if np.sign(p[1]) == np.sign(u0_ref) else al - np.pi
    smooth = cfg.get("smooth") if smooth is None else smooth
    ns, na = chi.shape
    # alpha is periodic only if the grid covers the full circle (a partial range, e.g. 3.6 .. 5.6, has two edges)
    wrap = na > 1 and (alphas[-1] - alphas[0]) + (alphas[1] - alphas[0]) >= 2 * np.pi - 1e-9
    if cfg.get("relax_mode", "queue") != "sweeps":
        ok_sign = lambda p: not one_sign or np.sign(p[1]) == np.sign(u0_ref)

        def make_args(i, j, chi, pars, forced):
            nb = [(i + a, (j + b) % na) for a in (-1, 0, 1) for b in (-1, 0, 1)
                  if (a or b) and 0 <= i + a < ns and (wrap or 0 <= j + b < na)]
            if forced:
                cand = [pars[k] for k in nb if np.isfinite(chi[k]) and np.all(np.isfinite(pars[k])) and ok_sign(pars[k])]
                if not cand:
                    return None
                return (float(logs[i]), [model_alpha(c, alphas[j]) for c in cand], cand, cfg, None, np.inf, margin, 3)
            cand = [pars[k] for k in nb if np.isfinite(chi[k]) and (not np.isfinite(chi[i, j])
                                                                    or chi[k] < chi[i, j] - margin) and ok_sign(pars[k])]
            if not cand:
                return None
            cv = float(chi[i, j]) if np.isfinite(chi[i, j]) else np.inf
            return (float(logs[i]), [model_alpha(c, alphas[j]) for c in cand], cand, cfg, True, cv, margin)

        return G.refine_queue(chi, pars, _sa_fit, make_args, workers, margin, log, "neighbour refinement",
                              wrap_j=wrap, smooth=smooth, max_refits=max_sweeps)
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
                    cand = [pars[k] for k in nb if np.isfinite(chi[k]) and np.all(np.isfinite(pars[k]))
                            and (not one_sign or np.sign(pars[k][1]) == np.sign(u0_ref))]
                    if cand:
                        tasks[(i, j)] = (float(logs[i]), [model_alpha(c, alphas[j]) for c in cand], cand, cfg,
                                         None, np.inf, margin, 3)
                    continue
                cand = [pars[k] for k in nb if np.isfinite(chi[k]) and (not np.isfinite(chi[i, j])
                                                                        or chi[k] < chi[i, j] - margin)
                        and (not one_sign or np.sign(pars[k][1]) == np.sign(u0_ref))]
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
        small = 0
        for k, v in res.items():
            if v is not None and v[1] is not None and (v[0] < chi[k] - (0.01 if force[k] else 1e-9)
                                                       or not np.isfinite(chi[k])):
                if chi[k] - v[0] > margin or not np.isfinite(chi[k]):   # gains below the margin do not propagate
                    changed[k] = True
                    n += 1
                else:
                    small += 1
                chi[k], pars[k] = v
        stale = (stale & ~checked) | G._near(changed, wrap)
        log(f"neighbour refinement, pass {sweep + 1}: {len(tasks)} points checked, {n} improved"
            + (f" ({small} by less than {margin:g})" if small else ""))
        if n == 0:
            break
    else:
        if max_sweeps and n:
            log(f"WARNING: refinement stopped at max_sweeps={max_sweeps} with {n} points still improving; the map is "
                "not converged (fronts of better solutions leave straight edges). Continue with alpha_profile_refine().")
    return chi, pars


def alpha_profile_refine(result, data, best=None, cfg=None, workers=None, margin=1.0, max_sweeps=30, bin_wings=True,
                         log=print, changed_only=True, smooth=None, one_sign=None):
    """More neighbour-refinement sweeps for an alpha_profile result (e.g. one stopped at max_sweeps). Returns a new
    result dict."""
    workers = _workers(workers)
    cfg, fitdata = _prepare(data, cfg, best, bin_wings)
    chi, pars = np.asarray(result["chi2"], float), np.asarray(result["pars"], float)
    u0_ref = _profile_u0_ref(result)
    t0 = time.time()
    G.warmup(cfg, fitdata)
    one_sign = bool(result.get("one_sign", False)) if one_sign is None else one_sign
    chi, pars = _alpha_sweeps(chi, pars, result["logs"], result["alphas"], u0_ref, cfg, workers, margin, max_sweeps,
                              changed_only, log, smooth, one_sign)
    ref = float(min(np.nanmin(chi), result.get("chi2_ref", np.inf)))
    log(f"done in {time.time() - t0:.0f}s; lowest chi2 {ref:.2f}")
    return dict(logs=np.asarray(result["logs"]), alphas=np.asarray(result["alphas"]), chi2=chi, pars=pars,
                chi2_ref=ref, wall=time.time() - t0, u0_ref=float(u0_ref), one_sign=bool(one_sign))


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
                  max_sweeps=30, bin_wings=True, log=print, changed_only=True, u0_ref=None, smooth=None,
                  one_sign=False):
    """Profile chi2 over (log s, alpha): at every grid point, chi2 minimised over all other parameters including
    log q, so the map has a value everywhere (unlike a projection of lattice solutions). alpha is in the convention
    of (log s, alpha) panels: solutions with the opposite sign of u0 to the best fit are counted at alpha + pi, and
    the fit at such a point keeps that sign (alpha - pi in the model). Starts: the n_starts nearest known solutions
    of `previous` (lattice / densify results, refine cells, alpha_minima rows); then neighbour refinement sweeps.
    u0_ref: the sign of u0 counted at the plotted alpha; default: that of an alpha_profile result in `previous`,
    else of the best known solution (pass it when `previous` holds solutions of both signs, e.g. a refined profile
    converted to rows: its best point may be the mirror solution).
    one_sign: only solutions with the sign of u0_ref (known solutions of the other sign are mirrored, (u0, alpha) ->
    (-u0, -alpha), the same light curve): each solution appears once, as in maps of codes that keep u0 > 0. Without
    it, every solution also appears mirrored about alpha = 3 pi / 2.
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
    if one_sign:                           # mirror the other sign: (u0, alpha) -> (-u0, -alpha) is the same light curve
        known = [(p if np.sign(p[1]) == np.sign(u0_ref) else np.r_[p[0], -p[1], p[2:5], -p[5], p[6]], c)
                 for p, c in known]
    K = np.array([(p[3], _plot_alpha(p, u0_ref)) for p, _ in known])
    ds, da = abs(logs[1] - logs[0]), abs(alphas[1] - alphas[0])
    knn = _grid_knn(K[:, 0], K[:, 1], logs, alphas, 4 * n_starts, period_y=2 * np.pi)

    def model_alpha(p, al):                                            # alpha used in the model for a start p
        return al if np.sign(p[1]) == np.sign(u0_ref) else al - np.pi

    ST = G.Stages(2, log)
    ST("Fits at fixed (log s, alpha)", f"{len(logs) * len(alphas)} points from the {n_starts} nearest of "
                                       f"{len(known)} known solutions")
    t0 = time.time()
    tasks = {}
    for i, ls in enumerate(logs):
        for j, al in enumerate(alphas):
            nn_all, dn_all = knn(i, j)
            near, dn = nn_all, dn_all
            dist = dict(zip(near.tolist(), dn.tolist()))
            near = [near[0]] + sorted(near[1:], key=lambda k: (dist[k] > 2.0, known[k][1]))[:n_starts - 1]
            # starts of one u0 sign per point (the nearest one's), fitted at the matching model alpha
            sg = np.sign(known[near[0]][0][1])
            starts = [known[k][0] for k in near if np.sign(known[k][0][1]) == sg]
            ip = _sa_interp(known, nn_all, dn_all, sg) if cfg.get("profile_interp", True) else None
            tasks[(i, j)] = (float(ls), float(model_alpha(known[near[0]][0], al)), starts, cfg,
                             ip[0] if ip else None, ip[1] if ip else np.inf)
    G.warmup(cfg, fitdata)
    res, _ = G.run_isolated(_sa_point, tasks, workers, log=log, desc="(log s, alpha) fits")
    chi = np.full((len(logs), len(alphas)), np.nan)
    pars = np.full((len(logs), len(alphas), 7), np.nan)
    for (i, j), v in res.items():
        if v is not None and v[1] is not None:
            chi[i, j], pars[i, j] = v
    ST("Neighbour refinement", "refit points whose neighbours found a better solution")
    chi, pars = _alpha_sweeps(chi, pars, logs, alphas, u0_ref, cfg, workers, margin, max_sweeps, changed_only, log,
                              smooth, one_sign)
    ref = float(min(np.nanmin(chi), min(c for _, c in known)))
    log(f"done in {time.time() - t0:.0f}s; lowest chi2 {ref:.2f}")
    return dict(logs=logs, alphas=alphas, chi2=chi, pars=pars, chi2_ref=ref, wall=time.time() - t0,
                u0_ref=float(u0_ref), one_sign=bool(one_sign))


def multilevel(data, logs, logq, best=None, cfg=None, workers=None, levels=1, bands=BANDS, bin_wings=True, log=print):
    """A map by levels: lattice() with seed grid and cell fits on every 2**levels-th point of logs x logq, then
    densify() to each twice finer lattice down to the full one (each level started from all coarser results).
    Cell fits cost about ten lattice points each and are needed only to find narrow valleys between points; the
    fine levels inherit the coarse solutions. The coarse lattice must still resolve the map's valleys (spacing a
    few times finer than the narrowest valley): a 17 x 17 test block of MOA-2022-BLG-091 from a 5 x 5 coarse lattice
    (levels=2) took twice as long as the direct lattice and ended worse at 13 % of the points, while the 50 x 49
    paper lattice densified to 8x (the same route by hand) is the workflow of the research note. Requires
    len(logs) - 1 and len(logq) - 1 to be multiples of 2**levels. Returns the finest result (as densify(); "cells"
    of the coarse lattice, "levels": the results of all levels, coarse first)."""
    logs, logq = np.asarray(logs, float), np.asarray(logq, float)
    step = 2 ** levels
    if (len(logs) - 1) % step or (len(logq) - 1) % step:
        raise ValueError(f"len(logs) - 1 and len(logq) - 1 must be multiples of {step} for levels={levels}")
    t0 = time.time()
    r = lattice(data, logs[::step], logq[::step], best=best, cfg=cfg, workers=workers, bands=bands,
                bin_wings=bin_wings, log=log)
    out, previous = [r], [r, r["cells"]]
    for lev in range(levels - 1, -1, -1):
        s = 2 ** lev
        log(f"\nlevel {levels - lev}/{levels}: {len(logs[::s])} x {len(logq[::s])} points")
        r = densify(previous, data, logs[::s], logq[::s], best=best, cfg=cfg, workers=workers, bin_wings=bin_wings,
                    log=log)
        out.append(r)
        previous = [r] + previous
    return dict(r, cells=out[0]["cells"], levels=out, wall=time.time() - t0)
