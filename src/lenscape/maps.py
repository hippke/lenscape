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
from scipy.optimize import minimize
from . import config as C, grid as G, data as D, fit as F, model as M, amr

BANDS = (4.0, 16.0, 36.0, 64.0, 100.0)


def _workers(n):
    return n or max(1, (os.cpu_count() or 2) - 1)


def _point_from_starts(ls, lq, starts, cfg):
    """Lattice point (s, q fixed) from the given start vectors (t0, u0, tE, alpha, log rho, ...)."""
    be = F.make_backend(cfg["grid_backend"])
    co = F.Coords(cfg.get("coords", "teff"))
    fx = lambda y: M.chi2(be, F.full_p(co.to_x(y), ls, lq), G._DATA)
    best = (np.inf, None)
    for x0 in starts:
        y = co.from_x(np.asarray(x0[:5], float))
        c = fx(y)
        for step in (0.3, 0.1):
            sim = np.vstack([y] + [y + np.eye(5)[k] * s for k, s in enumerate(co.steps(y, step))])
            r = minimize(fx, y, method="Nelder-Mead", options=dict(maxfev=1000, adaptive=True, initial_simplex=sim,
                                                                 fatol=1e-2))
            if r.fun < c:
                c, y = r.fun, r.x
        if c < best[0]:
            best = (c, co.to_x(y))
    return best


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
