"""Crash-tolerant (log s, log q) grid driver.

VBBinaryLensing can segfault for some parameter vectors. With multiprocessing.Pool the lost task made
imap_unordered hang forever (baseline run); with a ProcessPoolExecutor a deterministic crasher kills
every rebuilt pool before other cells finish. So each task runs in its own forked
process (cheap: ms per fork vs ~10 s per cell); a crash costs only that task, which is retried once and
then reported with its exit signal.
"""
import warnings
import os, time, pickle
from collections import deque
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


def _worker(conn, fn):
    """Pool worker: runs fn(*args) for every (k, args) received until None; a Python error is reported, a crash
    (segfault) ends the process and the parent sees the broken pipe."""
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        k, args = msg
        try:
            conn.send((k, "ok", fn(*args)))
        except Exception as e:
            conn.send((k, "err", repr(e)))
    conn.close()


class Pool:
    """Forked worker processes that each run fn(*args) for many tasks. submit(k, args) queues a task (dispatched
    as soon as a worker is idle); next() waits for the next finished task and returns (k, status, value) with
    status "ok", "err" (Python error, value = repr) or "crash" (the worker died twice on it; it is re-forked and the
    task retried once before that); close() stops the workers. Workers inherit the parent's state at construction
    (kernels, module globals such as _DATA), so large inputs belong there, not in the task arguments."""

    def __init__(self, fn, workers, log=print, retries=1):
        self.fn, self.log, self.retries = fn, log, retries
        self.ctx = mp.get_context("fork")
        self.pool = [self._spawn() for _ in range(max(1, workers))]
        self.idle = list(range(len(self.pool)))
        self.busy = {}                               # connection -> (slot, k, args)
        self.queue = deque()
        self.tries = {}
        self.crashes = {}
        self.done = deque()

    def _spawn(self):
        a, b = self.ctx.Pipe(duplex=True)
        p = self.ctx.Process(target=_worker, args=(b, self.fn))
        p.start()
        b.close()
        return [a, p]

    @property
    def n_idle(self):
        return len(self.idle)

    @property
    def active(self):
        return bool(self.queue or self.busy or self.done)

    def submit(self, k, args):
        self.tries.setdefault(k, 0)
        self.queue.append((k, args))
        self._dispatch()

    def _dispatch(self):
        while self.queue and self.idle:
            slot = self.idle.pop()
            k, args = self.queue.popleft()
            self.pool[slot][0].send((k, args))
            self.busy[self.pool[slot][0]] = (slot, k, args)

    def next(self):
        """(k, status, value) of the next finished task, or None when nothing is queued or running."""
        self._dispatch()
        while not self.done:
            if not self.busy:
                return None
            for conn in wait(list(self.busy)):
                slot, k, args = self.busy.pop(conn)
                try:
                    _, status, val = conn.recv()
                except EOFError:
                    status, val = "crash", None
                    p = self.pool[slot][1]
                    p.join()
                    self.crashes[k] = p.exitcode
                    self.pool[slot] = self._spawn()
                self.idle.append(slot)
                if status == "crash":
                    self.tries[k] += 1
                    if self.tries[k] <= self.retries:
                        self.log(f"task {k}: worker died (exit {self.crashes[k]}), retrying")
                        self.queue.append((k, args))
                        continue
                    self.log(f"task {k}: worker died (exit {self.crashes[k]}) again -> NaN")
                elif status == "err":
                    self.log(f"task {k}: python error {val}")
                self.done.append((k, status, val))
            self._dispatch()
        return self.done.popleft()

    def close(self):
        for conn, p in self.pool:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            conn.close()
        for conn, p in self.pool:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_isolated(fn, tasks, workers=19, retries=1, log=print, desc=None):
    """Run fn(*args) for each task {k: args} in forked worker processes, `workers` at a time. Returns
    ({k: result or None}, crashes). A task whose worker dies is retried once in a fresh worker, then reported with
    the exit signal and given None. desc: label of the progress bar (see progress_bar).
    The workers are forked once per call (Pool: they inherit the parent's state, kernels and module globals; large
    inputs belong in module globals such as _DATA, not in the task arguments, which travel through a pipe) and
    each runs many tasks; LENSCAPE_POOL=0 forks one process per task instead (the earlier behaviour)."""
    if os.environ.get("LENSCAPE_POOL", "1") == "0":
        return _run_forked(fn, tasks, workers, retries, log, desc)
    bar = progress_bar(len(tasks), desc or fn.__name__.strip("_"))
    results = {}
    with Pool(fn, min(workers, max(1, len(tasks))), log, retries) as pool:
        for k, args in tasks.items():
            pool.submit(k, args)
        while True:
            out = pool.next()
            if out is None:
                break
            k, status, val = out
            results[k] = val if status == "ok" else None
            if bar is not None:
                bar.update(1)
                if pool.crashes:
                    bar.set_postfix(crashes=len(pool.crashes))
        crashes = dict(pool.crashes)
    if bar is not None:
        bar.close()
    return results, crashes


def _run_forked(fn, tasks, workers=19, retries=1, log=print, desc=None):
    """One forked process per task (LENSCAPE_POOL=0)."""
    ctx = mp.get_context("fork")
    bar = progress_bar(len(tasks), desc or fn.__name__.strip("_"))
    pending = deque(tasks.items())
    tries = {k: 0 for k in tasks}
    running, results, crashes = {}, {}, {}
    while pending or running:
        while pending and len(running) < workers:
            k, args = pending.popleft()
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
    """Cells whose chi2 exceeds the median of the interpolations through their neighbours (along both axes and both
    diagonals) by more than tau: the quadratic interpolation through the two neighbours on either side where those
    exist (three points next to an edge), else the mean of the two adjacent neighbours. On a smooth map no cell does, also not on a slope, at the
    edges or, with the 4-point rule, on a curved surface (a coarse lattice's ridges and valley walls passed the
    linear test by their curvature alone and were refit in every sweep); a fit that stopped in a worse local
    minimum than its neighbours' does."""
    mids = []
    for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
        n1, p1 = _shift(chi, -di, -dj, wrap_j), _shift(chi, di, dj, wrap_j)
        n2, p2 = _shift(chi, -2 * di, -2 * dj, wrap_j), _shift(chi, 2 * di, 2 * dj, wrap_j)
        lin = 0.5 * (n1 + p1)
        quad = (-n2 + 4.0 * n1 + 4.0 * p1 - p2) / 6.0                 # through -2, -1, +1, +2
        q_r = n1 / 3.0 + p1 - p2 / 3.0                                 # through -1, +1, +2 (second row from an edge)
        q_l = p1 / 3.0 + n1 - n2 / 3.0
        one = np.where(np.isfinite(q_r), q_r, q_l)
        mids.append(np.where(np.isfinite(quad), quad, np.where(np.isfinite(one), one, lin)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # cells without any complete pair (corners)
        med = np.nanmedian(np.array(mids), axis=0)
    return np.isfinite(chi) & np.isfinite(med) & (chi > med + tau)


def _relax_rows(rows, chi, pars, logs, logq, cfg, margin, active=None, cols=None, force=None):
    """For cells in `rows` (columns `cols`, default all), try the 8 neighbours' solutions; refine when one
    beats the current chi2. Reads only the previous sweep's arrays (Jacobi), so the split into tasks does not change
    the result. force: cells refitted from the neighbours' solutions ranked by their one chi2 without the margin
    (the best relax_tries + 1 of them; all eight were fitted before), kept if better by > 0.01."""
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
                        c = fx(pars[a, b])
                        if forced or c < best_c - margin:     # forced: no margin, but ranked by the same chi2
                            cands.append((c, pars[a, b]))
            cands.sort(key=lambda z: z[0])
            new = None
            tries = cfg.get("relax_tries", 2)
            for c0, x0 in cands[:tries + 1 if forced else tries]:
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


def _rough_at(chi, i, j, tau, wrap_j=False):
    """_rough for one point, from its 5 x 5 neighbourhood."""
    ni, nj = chi.shape
    w = np.full((5, 5), np.nan)
    for a in range(-2, 3):
        if not 0 <= i + a < ni:
            continue
        for b in range(-2, 3):
            jj = (j + b) % nj if wrap_j else j + b
            if 0 <= jj < nj:
                w[a + 2, b + 2] = chi[i + a, jj]
    return bool(_rough(w, tau)[2, 2])


def _relax_point(ls, lq, own, nb_pars, cfg, margin, forced):
    """One point of the queue refinement: the neighbours' solutions nb_pars evaluated at (ls, lq); the best
    cfg["relax_tries"] (+1 if forced) that beat own - margin (forced: all, ranked) are refined (fit.local_fit).
    Returns (chi2, x) if one beats own by more than margin (forced: 0.01), else None."""
    from . import model
    be = F.make_backend(cfg["grid_backend"])
    co = F.Coords(cfg.get("coords", "teff"))
    fx = lambda x: model.chi2(be, F.full_p(x, ls, lq), _DATA)
    best_c = own if np.isfinite(own) else np.inf
    cands = []
    for x0 in nb_pars:
        c = fx(x0)
        if forced or c < best_c - margin:
            cands.append((c, x0))
    cands.sort(key=lambda z: z[0])
    tries = cfg.get("relax_tries", 2)
    mg = 0.01 if forced else margin
    new = None
    for c0, x0 in cands[:tries + 1 if forced else tries]:
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
    return (best_c, new) if new is not None else None


def refine_queue(chi, pars, fn, make_args, workers=19, margin=1.0, log=print, desc="neighbour refinement",
                 wrap_j=False, smooth=None, active=None, max_refits=30, stats=None):
    """Asynchronous neighbour refinement of a map (chi, pars updated copies returned). Every point is a task
    (fn(*make_args(i, j, chi, pars, forced)) -> (chi2, x) or None); a point whose result improves it by more than
    `margin` (or that was unknown) makes its eight neighbours stale, and stale points are queued, the largest gap to
    the best neighbour first. Workers (grid.Pool) pull from the queue as they finish, so an improvement spreads as
    fast as the fits complete instead of one lattice step per sweep with a barrier between sweeps; the final
    state satisfies the same rule as the sweeps (no stale point left), reached in a timing-dependent order.
    Forced points (rough by `smooth`, grid._rough) are refit even without a better neighbour; their gains below the
    margin are kept but do not propagate. active: mask of points to consider (None: all). A point is refit at most
    max_refits times."""
    import heapq
    chi, pars = chi.copy(), pars.copy()
    ni, nj = chi.shape
    stale = np.ones(chi.shape, bool) if active is None else active.copy()
    inflight, dirty, refits = {}, set(), np.zeros(chi.shape, np.int64)
    heap, seq = [], 0

    def nb_min(i, j):
        m = np.inf
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                if not (a or b) or not 0 <= i + a < ni:
                    continue
                jj = (j + b) % nj if wrap_j else j + b
                if 0 <= jj < nj and np.isfinite(chi[i + a, jj]):
                    m = min(m, chi[i + a, jj])
        return m

    def push(i, j):
        nonlocal seq
        if active is not None and not active[i, j]:
            return
        if (i, j) in inflight:
            dirty.add((i, j))
            return
        if stale[i, j] and refits[i, j] < max_refits:
            own = chi[i, j] if np.isfinite(chi[i, j]) else np.inf
            gap = own - nb_min(i, j)                 # largest gap to the best neighbour first (steep fronts)
            heapq.heappush(heap, (-gap if np.isfinite(gap) else -np.inf, seq, i, j))
            seq += 1

    for i in range(ni):
        for j in range(nj):
            push(i, j)
    bar = progress_bar(len(heap), desc)
    n_task = n_imp = n_small = n_capped = 0
    with Pool(fn, workers, log) as pool:
        def dispatch():
            nonlocal n_task
            while heap and pool.n_idle > 0:
                _, _, i, j = heapq.heappop(heap)
                if not stale[i, j] or (i, j) in inflight:
                    continue
                forced = smooth is not None and _rough_at(chi, i, j, smooth, wrap_j)
                args = make_args(i, j, chi, pars, forced)
                stale[i, j] = False
                if args is None:                     # nothing to try at this point
                    continue
                inflight[(i, j)] = forced
                refits[i, j] += 1
                n_task += 1
                pool.submit((i, j), args)

        dispatch()
        while True:
            out = pool.next()
            if out is None:
                if heap:
                    dispatch()
                    continue
                break
            (i, j), status, v = out
            forced = inflight.pop((i, j))
            if bar is not None:
                bar.update(1)
            if status != "ok":
                stale[i, j] = True                   # lost task: check again
            elif v is not None and v[1] is not None and (v[0] < chi[i, j] or not np.isfinite(chi[i, j])):
                gain = chi[i, j] - v[0]
                unknown = not np.isfinite(chi[i, j])
                chi[i, j], pars[i, j] = v[0], v[1]
                if gain > margin or unknown:
                    n_imp += 1
                    for a in (-1, 0, 1):
                        for b in (-1, 0, 1):
                            jj = (j + b) % nj if wrap_j else j + b
                            if 0 <= i + a < ni and 0 <= jj < nj:
                                if a or b:
                                    stale[i + a, jj] = True
                                push(i + a, jj)
                else:
                    n_small += 1
            if (i, j) in dirty:
                dirty.discard((i, j))
                stale[i, j] = True
                push(i, j)
            if refits[i, j] >= max_refits and stale[i, j]:
                n_capped += 1
            if bar is not None and heap:
                bar.total = n_task + len(heap)
            dispatch()
    if bar is not None:
        bar.close()
    if stats is not None:
        stats.update(tasks=n_task, improved=n_imp, small=n_small)
    log(f"{desc}: {n_task} points checked, {n_imp} improved"
        + (f" ({n_small} by less than {margin:g})" if n_small else "")
        + (f"; WARNING: {n_capped} points reached max_refits={max_refits} (not converged)" if n_capped else ""))
    return chi, pars


_RELAX = None          # (chi, pars, logs, logq, cfg, margin, active, force) of the current relax sweep (inherited by the workers)


def _relax_task(rows, cols):
    chi, pars, logs, logq, cfg, margin, active, force = _RELAX
    return _relax_rows(rows, chi, pars, logs, logq, cfg, margin, active, cols, force)


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
          seg=5, changed_only=True, smooth=None, mode=None):
    """Neighbour refinement of a lattice map until no point has a neighbour solution that beats it by > margin.
    mode (default cfg["relax_mode"], "queue"): "queue" = asynchronous, one task per stale point, improvements
    propagate as fast as the fits finish (refine_queue); "sweeps" = the earlier Jacobi sweeps (relax_sweeps; a
    front moves one lattice step per sweep, with a barrier between sweeps). max_sweeps caps the sweeps, or the
    refits of one point in queue mode. Both: smooth (default cfg["smooth"]) also refits rough points."""
    global _DATA
    if data is not None:
        _DATA = data
    if _DATA is None:
        _DATA = _default_data()
    mode = cfg.get("relax_mode", "queue") if mode is None else mode
    if mode == "sweeps":
        return relax_sweeps(chi, pars, logs, logq, cfg, workers, margin, max_sweeps, log, data, dchi2_cut, seg,
                            changed_only, smooth)
    smooth = cfg.get("smooth") if smooth is None else smooth
    active = None
    if dchi2_cut is not None:
        low = (chi - np.nanmin(chi)) < dchi2_cut
        active = _near(low)
    ni, nj = chi.shape

    def make_args(i, j, chi, pars, forced):
        nb = [pars[a, b] for a in (i - 1, i, i + 1) for b in (j - 1, j, j + 1)
              if (a != i or b != j) and 0 <= a < ni and 0 <= b < nj and np.all(np.isfinite(pars[a, b]))]
        if not nb:
            return None
        return (float(logs[i]), float(logq[j]), float(chi[i, j]), nb, cfg, margin, forced)

    warmup(cfg, _DATA)
    return refine_queue(chi, pars, _relax_point, make_args, workers, margin, log, "neighbour refinement",
                        smooth=smooth, active=active, max_refits=max_sweeps)


def relax_sweeps(chi, pars, logs, logq, cfg, workers=19, margin=1.0, max_sweeps=10, log=print, data=None,
                 dchi2_cut=None, seg=5, changed_only=True, smooth=None):
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
        global _RELAX
        _RELAX = (chi, pars, logs, logq, cfg, margin, active, force)
        tasks = {k: ([i], cols) for k, (i, cols) in enumerate(segs)}
        warmup(cfg, _DATA)
        res, _ = run_isolated(_relax_task, tasks, workers, log=log, desc=f"neighbour refinement, pass {sweep + 1}")
        n = 0
        changed = np.zeros(chi.shape, bool)
        for k, (i, cols) in enumerate(segs):
            if res.get(k) is None:                   # task lost (crash): its cells count as not checked
                checked[i, cols] = False
        small = 0
        for v in res.values():
            for i, j, c, x in v or []:
                if c < chi[i, j] or not np.isfinite(chi[i, j]):
                    # a forced refit may gain less than the margin: kept, but it is no reason to re-check the
                    # neighbours (noise-level gains cascaded through the map until max_sweeps)
                    if chi[i, j] - c > margin or not np.isfinite(chi[i, j]):
                        changed[i, j] = True
                        n += 1
                    else:
                        small += 1
                    chi[i, j], pars[i, j] = c, x
        stale = (stale & ~checked) | _near(changed)
        log(f"neighbour refinement, pass {sweep + 1}: {int(checked.sum())} cells checked, {n} improved"
            + (f" ({small} by less than {margin:g})" if small else ""))
        if n == 0:
            break
    else:
        if max_sweeps and n:
            log(f"WARNING: neighbour refinement stopped at max_sweeps={max_sweeps} with {n} cells still improving; "
                "the map is not converged (fronts of better solutions leave straight edges). Continue with polish().")
    return chi, pars


def _heavy_starts_task(ls, lq, cfg, warm):
    return F.heavy_starts(ls, lq, _DATA, cfg, warm)


def _heavy_explore_task(ls, lq, cfg, x0):
    return F.heavy_explore(ls, lq, _DATA, cfg, x0)


def _outliers(chi, thr):
    """Points that stand out above their neighbours along s or along q by more than thr, by either of two rules:
    above the larger of the two adjacent values (isolated spikes and one-cell-wide stripes), or above the quadratic
    extrapolation of the surface through the two neighbours on either side (a stuck fit on a slope or next to other
    stuck fits; one neighbour on a side: that value), or, at an edge of the map, above the one-sided quadratic
    extrapolation from the interior. Checked on MOA-2022-BLG-091 against later, better maps: of the
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
            # edge along this axis: above both the adjacent value and the one-sided quadratic extrapolation from
            # the interior by more than thr (the extrapolation alone is pulled down by a spike two cells in); an edge
            # point next to a much better neighbour had escaped both rules and stayed 16 above the map's value
            one = np.isfinite(adj[0]) ^ np.isfinite(adj[1])
            a1 = np.where(np.isfinite(adj[0]), adj[0], adj[1])
            s1 = np.where(np.isfinite(side[0]), side[0], side[1])
            flag |= one & (chi - np.fmax(s1, a1) > thr)
            both |= b
            preds += side
        allmax = np.nanmax(np.array(preds), axis=0)
    flag |= ~both & np.isfinite(allmax) & (chi - allmax > thr)
    return flag & fin


def repair_outliers(chi, pars, logs, logq, cfg, workers=19, thr=8.0, passes=3, log=print, data=None, min_gain=1.0,
                    dchi2_cut=None):
    """Stuck fits: points above their neighbours along s or along q by more than thr (see _outliers; e.g. MOA-2022-BLG-091: 60.4 -> 6.9 and 54.2 -> 28.2, matching their neighbours, while the
    neighbour relaxation had missed them) get a deep search (fit_point_heavy) with the neighbours' and the point's
    own solutions as extra starts. Any improvement is kept, but only an improvement by more than min_gain counts as
    a repair: a point whose search gained less is a feature of the profile and is not searched again unless a
    neighbour (or the point itself) changed by more than min_gain since; passes run until no such point is left.
    dchi2_cut: only points within dchi2_cut of the map's minimum, and their neighbours, are searched.
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
        if dchi2_cut is not None:
            cand &= _near((chi - np.nanmin(chi)) < dchi2_cut)
        tasks = {}
        for i, j in zip(*np.nonzero(cand)):
            nb = [(i + a, j + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= i + a < ni and 0 <= j + b < nj]
            tasks[(int(i), int(j))] = (float(logs[i]), float(logq[j]), cfg,
                                       [pars[k] for k in nb if np.all(np.isfinite(pars[k]))] + [pars[i, j]])
        if not tasks:
            if npass == 0:
                log("recalculate outliers: no outliers found")
            break
        warmup(cfg, _DATA)
        # the deep search of a point in two task layers: its screened starts, then one task per exploration (a few
        # points with 30 explorations each would otherwise occupy a few workers for a minute while the rest idle)
        st, _ = run_isolated(_heavy_starts_task, tasks, workers, log=log, desc=f"outlier starts, pass {npass + 1}")
        ex = {(k, m): (tasks[k][0], tasks[k][1], tasks[k][2], x0) for k, xs in st.items() if xs for m, x0 in enumerate(xs)}
        exr, _ = run_isolated(_heavy_explore_task, ex, workers, log=log, desc=f"recalculate outliers, pass {npass + 1}")
        res = {k: None for k in tasks}
        for (k, m), v in exr.items():
            if v is not None and (res[k] is None or v[0] < res[k][0]):
                res[k] = v
        n = 0
        changed = np.zeros(chi.shape, bool)
        searched = np.zeros(chi.shape, bool)
        for (i, j), v in res.items():
            searched[i, j] = v is not None               # a lost task is searched again next pass
            if v is not None and v[0] < chi[i, j]:
                if chi[i, j] - v[0] > min_gain:
                    changed[i, j] = True
                    n += 1
                x = np.asarray(v[1])
                chi[i, j], pars[i, j] = v[0], x[[0, 1, 2, 5, 6]] if len(x) == 7 else x
        nfix += n
        stale = (stale & ~searched) | _near(changed)
        log(f"recalculate outliers, pass {npass + 1}: {len(tasks)} outliers, {n} improved")
        if n == 0:
            break
    return chi, pars, nfix
