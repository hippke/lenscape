"""Crash-tolerant (log s, log q) grid driver.

VBBinaryLensing can segfault for some parameter vectors. With multiprocessing.Pool the lost task made
imap_unordered hang forever (baseline run); with a ProcessPoolExecutor a deterministic crasher kills
every rebuilt pool before other cells finish. So each task runs in its own forked
process (cheap: ms per fork vs ~10 s per cell); a crash costs only that task, which is retried once and
then reported with its exit signal.
"""
import warnings
import os, time, pickle
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import wait
from . import data as D, fit as F

_DATA = None


def _child(conn, fn, args):
    try:
        conn.send(("ok", fn(*args)))
    except Exception as e:                       # Python-level errors are reported, not retried
        conn.send(("err", repr(e)))
    conn.close()


def _default_data():
    """Data used when run_grid / relax are called without data (none in lenscape; applications may replace this)."""
    raise ValueError("no data: pass data=...")


def warmup(cfg, data):
    """Load (or compile) the kernels in the parent, so forked tasks inherit them instead of each loading them from the
    numba disk cache (~0.1 s per task). The warm-up trajectory crosses a central caustic in the middle of the data,
    so that the point-source, hexadecapole and contour paths are all compiled."""
    from . import model as M
    be = F.make_backend(cfg.get("grid_backend", cfg))       # full strategy config or a backend config
    t = getattr(data, "t_eval", data.t)
    M.chi2(be, np.array([float(np.median(t)), 0.005, 20.0, 0.0, -2.0, 1.0, -2.5]), data)


PROGRESS = {"0": False, "1": True}.get(os.environ.get("LENSCAPE_PROGRESS", ""), None)   # None: only on a terminal


def progress_bar(total, desc):
    """Tqdm progress bar (one tick per finished task) when PROGRESS is True, or None and stderr is a terminal;
    None if switched off or tqdm is not installed."""
    import sys
    on = PROGRESS if PROGRESS is not None else sys.stderr.isatty()
    if not on:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=desc, unit="fit", dynamic_ncols=True, smoothing=0.05, leave=True)


class Stages:
    """Numbered stage headers for people watching a run: s = Stages(4); s("Cell fits", "what it does")
    prints 'Stage 1/4: Cell fits - what it does'."""

    def __init__(self, n, log=print):
        self.n, self.i, self.log = n, 0, log

    def __call__(self, name, what=""):
        self.i += 1
        self.log(f"\nStage {self.i}/{self.n}: {name}" + (f" - {what}" if what else ""))


def run_isolated(fn, tasks, workers=19, retries=1, log=print, desc=None):
    """Run fn(*args) for each task in its own forked process. Returns {k: result or None}, crashes.
    desc: label of the progress bar (see progress_bar)."""
    ctx = mp.get_context("fork")
    bar = progress_bar(len(tasks), desc or fn.__name__.strip("_"))
    pending = list(tasks.items())
    tries = {k: 0 for k in tasks}
    running, results, crashes = {}, {}, {}
    while pending or running:
        while pending and len(running) < workers:
            k, args = pending.pop(0)
            r, w = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_child, args=(w, fn, args))
            p.start()
            w.close()
            running[r] = (k, p)
        for r in wait(list(running)):
            k, p = running.pop(r)
            try:
                status, val = r.recv()
            except EOFError:
                status, val = "crash", None
            p.join()
            if bar is not None and (status != "crash" or tries[k] + 1 > retries):
                bar.update(1)
            if status == "ok":
                results[k] = val
            elif status == "err":
                log(f"task {k}: python error {val}")
                results[k] = None
            else:
                tries[k] += 1
                crashes[k] = p.exitcode
                if tries[k] <= retries:
                    log(f"task {k}: worker died (exit {p.exitcode}), retrying")
                    pending.append((k, tasks[k]))
                else:
                    log(f"task {k}: worker died (exit {p.exitcode}) again -> NaN")
                    results[k] = None
                if bar is not None:
                    bar.set_postfix(crashes=len(crashes))
    if bar is not None:
        bar.close()
    return results, crashes


def _cell(i, j, ls, lq, cfg, warm=None):
    t = time.process_time()
    c, x = F.fit_point(ls, lq, _DATA, cfg, warm)
    return c, x, time.process_time() - t


def run_grid(logs, logq, cfg, workers=19, out=None, log=print, data=None):
    global _DATA
    if data is not None:
        _DATA = data
    if _DATA is None:
        _DATA = _default_data()
    chi = np.full((len(logs), len(logq)), np.nan)
    pars = np.full((len(logs), len(logq), 5), np.nan)
    cpu = np.full((len(logs), len(logq)), np.nan)
    t0 = time.time()
    tasks = {(i, j): (i, j, float(logs[i]), float(logq[j]), cfg)
             for i in range(len(logs)) for j in range(len(logq))}
    warmup(cfg, _DATA)
    res, crashes = run_isolated(_cell, tasks, workers, log=log, desc=f"grid {len(logs)}x{len(logq)}")
    for (i, j), v in res.items():
        if v is not None:
            chi[i, j], pars[i, j], cpu[i, j] = v
    wall = time.time() - t0
    if out:
        np.savez(out, logs=logs, logq=logq, chi2=chi, pars=pars, cpu=cpu, wall=wall,
                 crashes=np.array([[*np.atleast_1d(k), v] for k, v in crashes.items()], dtype=object))
    return chi, pars, cpu, wall


def _shift(chi, di, dj, wrap_j=False):
    """Array whose [i, j] is chi[i + di, j + dj] (NaN outside; second axis periodic if wrap_j)."""
    ni, nj = chi.shape
    out = np.full((ni, nj), np.nan)
    src = chi[max(0, di):ni + min(0, di)]
    if wrap_j:
        out[max(0, -di):ni + min(0, -di)] = np.roll(src, -dj, axis=1)
    else:
        out[max(0, -di):ni + min(0, -di), max(0, -dj):nj + min(0, -dj)] = src[:, max(0, dj):nj + min(0, dj)]
    return out


def _rough(chi, tau, wrap_j=False):
    """Cells whose chi2 exceeds the median of the means of opposite neighbour pairs (along both axes and both
    diagonals, i.e. the linear interpolations through the cell) by more than tau: on a smooth map no cell does (also
    not on a slope or at the edges); a fit that stopped in a worse local minimum than its neighbours' does."""
    mids = []
    for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
        mids.append(0.5 * (_shift(chi, di, dj, wrap_j) + _shift(chi, -di, -dj, wrap_j)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # cells without any complete pair (corners)
        med = np.nanmedian(np.array(mids), axis=0)
    return np.isfinite(chi) & np.isfinite(med) & (chi > med + tau)


def _relax_rows(rows, chi, pars, logs, logq, cfg, margin, active=None, cols=None, force=None):
    """For cells in `rows` (columns `cols`, default all), try the 8 neighbours' solutions; refine when one
    beats the current chi2. Reads only the previous sweep's arrays (Jacobi), so the split into tasks does not change
    the result. force: cells refitted from all neighbours' solutions (no pre-check), kept if better by > 0.01."""
    from . import model
    be = F.make_backend(cfg["grid_backend"])
    co = F.Coords(cfg.get("coords", "teff"))
    out = []
    ni, nj = chi.shape
    for i in rows:
        for j in (range(nj) if cols is None else cols):
            if active is not None and not active[i, j]:
                continue
            ls, lq = float(logs[i]), float(logq[j])
            fx = lambda x: model.chi2(be, F.full_p(x, ls, lq), _DATA)
            best_c = chi[i, j] if np.isfinite(chi[i, j]) else np.inf
            forced = force is not None and force[i, j]
            mg = 0.01 if forced else margin
            cands = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if (di or dj) and 0 <= a < ni and 0 <= b < nj and np.all(np.isfinite(pars[a, b])):
                        if forced:
                            cands.append((chi[a, b], pars[a, b]))
                            continue
                        c = fx(pars[a, b])
                        if c < best_c - margin:
                            cands.append((c, pars[a, b]))
            cands.sort(key=lambda z: z[0])
            new = None
            for c0, x0 in cands[:8 if forced else cfg.get("relax_tries", 2)]:
                if cfg.get("optimizer", "nm") != "nm":
                    rc, ry = F.local_fit(be, _DATA, ls, lq, co.from_x(x0), cfg, cfg.get("refine_step", 0.3),
                                         cfg.get("relax_nfev", 600))
                else:
                    f = lambda y: fx(co.to_x(y))
                    f.coords = co
                    r = F._nm(f, co.from_x(x0), cfg.get("relax_nfev", 600), cfg.get("refine_step", 0.3), 1e-2)
                    rc, ry = r.fun, r.x
                if rc < best_c - mg:
                    best_c, new = rc, co.to_x(ry)
            if new is not None:
                out.append((i, j, best_c, new))
    return out


def _near(mask, wrap_j=False):
    """mask dilated by one cell in each direction (8-neighbourhood); wrap_j: second axis periodic."""
    out = mask.copy()
    ni, nj = mask.shape
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if not (di or dj):
                continue
            src = mask[max(0, -di):ni - max(0, di)]
            if wrap_j:
                src = np.roll(src, dj, axis=1)
                out[max(0, di):ni - max(0, -di)] |= src
            else:
                src = src[:, max(0, -dj):nj - max(0, dj)]
                out[max(0, di):ni - max(0, -di), max(0, dj):nj - max(0, -dj)] |= src
    return out


def relax(chi, pars, logs, logq, cfg, workers=19, margin=1.0, max_sweeps=10, log=print, data=None, dchi2_cut=None,
          seg=5, changed_only=True, smooth=None):
    """Repeat neighbour-relaxation sweeps (rows in parallel, isolated processes) until no cell improves.
    dchi2_cut: only relax cells with Delta chi2 < cut (vs the current minimum) and their 8 neighbours.
    changed_only: after the first sweep, check only cells whose 3 x 3 neighbourhood changed since they were last
    checked. A cell's refit depends only on its own and its neighbours'
    values and the fits are deterministic, so the other cells would give the same (rejected) result again: identical
    output, far fewer refits in late sweeps.
    smooth: also refit cells whose chi2 exceeds the median of their neighbours by more than `smooth` from all
    neighbours' solutions, without the one-chi2 pre-check and margin (fits that stopped in a slightly worse local
    minimum than their neighbours' leave a speckle of 1-3 in chi2 that the margin-1 trigger never catches)."""
    global _DATA
    if data is not None:
        _DATA = data
    if _DATA is None:
        _DATA = _default_data()
    chi, pars = chi.copy(), pars.copy()
    smooth = cfg.get("smooth") if smooth is None else smooth
    stale = np.ones(chi.shape, bool)                 # inputs changed since the cell was last checked
    for sweep in range(max_sweeps):
        active = None
        if dchi2_cut is not None:
            low = (chi - np.nanmin(chi)) < dchi2_cut
            active = low.copy()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    active |= np.roll(np.roll(low, di, 0), dj, 1)
        if changed_only and sweep > 0:
            active = stale if active is None else active & stale
        force = None
        if smooth is not None:
            force = _rough(chi, smooth) & (stale if (changed_only and sweep > 0) else True)
            if active is not None:
                active = active | force
        if active is not None:
            if not active.any():
                break
            rows_act = np.where(active.any(axis=1))[0]
        else:
            rows_act = np.arange(len(logs))
        checked = np.ones(chi.shape, bool) if active is None else active.copy()
        # short row segments (seg cells), expensive first (cells near the minimum need the most refits), instead
        # of one chunk of rows per worker: the chunks' costs differed so much that most cores idled at every sweep
        nj = len(logq)
        segs = [(i, list(range(j0, min(j0 + seg, nj)))) for i in rows_act for j0 in range(0, nj, seg)]
        if active is not None:
            segs = [(i, c) for i, c in segs if active[i, c].any()]
        lowc = np.nan_to_num(chi - np.nanmin(chi), nan=1e30)
        segs.sort(key=lambda t: lowc[t[0], t[1]].min())
        tasks = {k: ([i], chi, pars, logs, logq, cfg, margin, active, cols, force) for k, (i, cols) in enumerate(segs)}
        warmup(cfg, _DATA)
        res, _ = run_isolated(_relax_rows, tasks, workers, log=log, desc=f"neighbour refinement, pass {sweep + 1}")
        n = 0
        changed = np.zeros(chi.shape, bool)
        for k, (i, cols) in enumerate(segs):
            if res.get(k) is None:                   # task lost (crash): its cells count as not checked
                checked[i, cols] = False
        for v in res.values():
            for i, j, c, x in v or []:
                if c < chi[i, j] or not np.isfinite(chi[i, j]):
                    chi[i, j], pars[i, j] = c, x
                    changed[i, j] = True
                    n += 1
        stale = (stale & ~checked) | _near(changed)
        log(f"neighbour refinement, pass {sweep + 1}: {int(checked.sum())} cells checked, {n} improved")
        if n == 0:
            break
    else:
        if max_sweeps and n:
            log(f"WARNING: neighbour refinement stopped at max_sweeps={max_sweeps} with {n} cells still improving; "
                "the map is not converged (fronts of better solutions leave straight edges). Continue with polish().")
    return chi, pars


def _outliers(chi, thr):
    """Points that stand out above their neighbours along s or along q by more than thr, by either of two rules:
    above the larger of the two adjacent values (isolated spikes and one-cell-wide stripes), or above the quadratic
    extrapolation of the surface through the two neighbours on either side (a stuck fit on a slope or next to other
    stuck fits; one neighbour on a side: that value). Checked on MOA-2022-BLG-091 against later, better maps: of the
    points the quadratic rule adds, 57 % (50 x 49 map) and 93 % (8x map) were lowered by > 1 later; of the points
    only the adjacent rule finds, 52 % / none, so both are kept. Where no axis has neighbours on both sides (corners),
    the point must exceed all available extrapolations."""
    fin = np.isfinite(chi)
    preds, both = [], np.zeros(chi.shape, bool)
    flag = np.zeros(chi.shape, bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for di, dj in ((1, 0), (0, 1)):
            side, adj = [], []
            for sg in (-1, 1):
                n1, n2 = _shift(chi, sg * di, sg * dj), _shift(chi, 2 * sg * di, 2 * sg * dj)
                adj.append(n1)
                side.append(np.where(np.isfinite(n2), 2 * n1 - n2, n1))
            b = np.isfinite(adj[0]) & np.isfinite(adj[1])
            flag |= b & ((chi - np.fmax(adj[0], adj[1]) > thr) | (chi - np.fmax(side[0], side[1]) > thr))
            both |= b
            preds += side
        allmax = np.nanmax(np.array(preds), axis=0)
    flag |= ~both & np.isfinite(allmax) & (chi - allmax > thr)
    return flag & fin


def repair_outliers(chi, pars, logs, logq, cfg, workers=19, thr=8.0, passes=3, log=print, data=None, min_gain=1.0):
    """Stuck fits: points above their neighbours along s or along q by more than thr (see _outliers; e.g. MOA-2022-BLG-091: 60.4 -> 6.9 and 54.2 -> 28.2, matching their neighbours, while the
    neighbour relaxation had missed them) get a deep search (fit_point_heavy) with the neighbours' and the point's
    own solutions as extra starts. Any improvement is kept, but only an improvement by more than min_gain counts as
    a repair: a point whose search gained less is a feature of the profile and is not searched again unless a
    neighbour (or the point itself) changed by more than min_gain since; passes run until no such point is left.
    Returns chi, pars, n_fixed (points improved by more than min_gain)."""
    global _DATA
    if data is not None:
        _DATA = data
    chi, pars = chi.copy(), pars.copy()
    nfix = 0
    stale = np.ones(chi.shape, bool)                 # not searched since the neighbourhood last changed by > min_gain
    ni, nj = chi.shape
    for npass in range(passes):
        cand = _outliers(chi, thr) & stale
        tasks = {}
        for i, j in zip(*np.nonzero(cand)):
            nb = [(i + a, j + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= i + a < ni and 0 <= j + b < nj]
            tasks[(int(i), int(j))] = (float(logs[i]), float(logq[j]), _DATA, cfg,
                                       [pars[k] for k in nb if np.all(np.isfinite(pars[k]))] + [pars[i, j]])
        if not tasks:
            if npass == 0:
                log("recalculate outliers: no outliers found")
            break
        warmup(cfg, _DATA)
        res, _ = run_isolated(F.fit_point_heavy, tasks, workers, log=log, desc=f"recalculate outliers, pass {npass + 1}")
        n = 0
        changed = np.zeros(chi.shape, bool)
        searched = np.zeros(chi.shape, bool)
        for (i, j), v in res.items():
            searched[i, j] = v is not None               # a lost task is searched again next pass
            if v is not None and v[0] < chi[i, j]:
                if chi[i, j] - v[0] > min_gain:
                    changed[i, j] = True
                    n += 1
                chi[i, j], pars[i, j] = v[0], v[1][[0, 1, 2, 5, 6]] if len(v[1]) == 7 else v[1]
        nfix += n
        stale = (stale & ~searched) | _near(changed)
        log(f"recalculate outliers, pass {npass + 1}: {len(tasks)} outliers, {n} improved")
        if n == 0:
            break
    return chi, pars, nfix
