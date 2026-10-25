"""Crash-tolerant (log s, log q) grid driver.

VBBinaryLensing can segfault for some parameter vectors. With multiprocessing.Pool the lost task made
imap_unordered hang forever (baseline run); with a ProcessPoolExecutor a deterministic crasher kills
every rebuilt pool before other cells finish. So each task runs in its own forked
process (cheap: ms per fork vs ~10 s per cell); a crash costs only that task, which is retried once and
then reported with its exit signal.
"""
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


def _relax_rows(rows, chi, pars, logs, logq, cfg, margin, active=None, cols=None):
    """For cells in `rows` (columns `cols`, default all), try the 8 neighbours' solutions; refine when one
    beats the current chi2. Reads only the previous sweep's arrays (Jacobi), so the split into tasks does not change
    the result."""
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
            cands = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if (di or dj) and 0 <= a < ni and 0 <= b < nj and np.all(np.isfinite(pars[a, b])):
                        c = fx(pars[a, b])
                        if c < best_c - margin:
                            cands.append((c, pars[a, b]))
            cands.sort(key=lambda z: z[0])
            new = None
            for c0, x0 in cands[:cfg.get("relax_tries", 2)]:
                if cfg.get("optimizer", "nm") != "nm":
                    rc, ry = F.local_fit(be, _DATA, ls, lq, co.from_x(x0), cfg, cfg.get("refine_step", 0.3),
                                         cfg.get("relax_nfev", 600))
                else:
                    f = lambda y: fx(co.to_x(y))
                    f.coords = co
                    r = F._nm(f, co.from_x(x0), cfg.get("relax_nfev", 600), cfg.get("refine_step", 0.3), 1e-2)
                    rc, ry = r.fun, r.x
                if rc < best_c - margin:
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
          seg=5, changed_only=True):
    """Repeat neighbour-relaxation sweeps (rows in parallel, isolated processes) until no cell improves.
    dchi2_cut: only relax cells with Delta chi2 < cut (vs the current minimum) and their 8 neighbours.
    changed_only: after the first sweep, check only cells whose 3 x 3 neighbourhood changed since they were last
    checked. A cell's refit depends only on its own and its neighbours'
    values and the fits are deterministic, so the other cells would give the same (rejected) result again: identical
    output, far fewer refits in late sweeps."""
    global _DATA
    if data is not None:
        _DATA = data
    if _DATA is None:
        _DATA = _default_data()
    chi, pars = chi.copy(), pars.copy()
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
        tasks = {k: ([i], chi, pars, logs, logq, cfg, margin, active, cols) for k, (i, cols) in enumerate(segs)}
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
    return chi, pars


def repair_outliers(chi, pars, logs, logq, cfg, workers=19, thr=8.0, passes=3, log=print, data=None):
    """Cells whose chi2 exceeds both neighbours along s or along q by > thr are stuck fits in practice (e.g. MOA-2022-BLG-091: 60.4 -> 6.9
    and 54.2 -> 28.2, matching their neighbours, while the neighbour relaxation had missed them): deep search
    (fit_point_heavy) with the neighbours' and the cell's own solutions as extra starts. Returns chi, pars, n_fixed."""
    global _DATA
    if data is not None:
        _DATA = data
    chi, pars = chi.copy(), pars.copy()
    nfix = 0
    for npass in range(passes):
        tasks = {}
        ni, nj = chi.shape
        for i in range(ni):
            for j in range(nj):
                nb = [(i + a, j + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= i + a < ni and 0 <= j + b < nj]
                if not nb or not np.isfinite(chi[i, j]):
                    continue
                # above both s-neighbours or both q-neighbours: isolated cells and one-cell-wide stripes (a column of
                # stuck fits has stuck neighbours above/below, e.g. MOA-2022-BLG-091: 51.7 -> 14.1 with neighbours 8-10);
                # edge cells (a top-row outlier was never checked before): an axis with both neighbours, or all neighbours
                ridge_s = 0 < i < ni - 1 and chi[i, j] - max(chi[i - 1, j], chi[i + 1, j]) > thr
                ridge_q = 0 < j < nj - 1 and chi[i, j] - max(chi[i, j - 1], chi[i, j + 1]) > thr
                above_all = chi[i, j] - max(chi[k] for k in nb) > thr
                if ridge_s or ridge_q or above_all:
                    tasks[(i, j)] = (float(logs[i]), float(logq[j]), _DATA, cfg,
                                     [pars[k] for k in nb if np.all(np.isfinite(pars[k]))] + [pars[i, j]])
        if not tasks:
            if npass == 0:
                log("recalculate outliers: no outliers found")
            break
        warmup(cfg, _DATA)
        res, _ = run_isolated(F.fit_point_heavy, tasks, workers, log=log, desc=f"recalculate outliers, pass {npass + 1}")
        n = 0
        for (i, j), v in res.items():
            if v is not None and v[0] < chi[i, j]:
                chi[i, j], pars[i, j] = v[0], v[1][[0, 1, 2, 5, 6]] if len(v[1]) == 7 else v[1]
                n += 1
        nfix += n
        log(f"recalculate outliers, pass {npass + 1}: {len(tasks)} outliers, {n} improved")
        if n == 0:
            break
    return chi, pars, nfix

