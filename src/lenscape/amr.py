"""Adaptive mesh refinement of the (log s, log q) profile map.

Every cell is fitted with s and q free inside the cell (fit.fit_point_box), so its value is the minimum of the
profile over the cell: a narrow valley crossing the cell is found even if it misses the cell centre (the centre-value
version had to split almost everything). A cell is split into 2 x 2 children if its minimum is below `cut` (vs the
current global minimum) or if a band boundary runs between it and a neighbour below `cut`. Children are warm-started
from the parent's and neighbours' solutions. Leaves are shown in the band of their cell minimum.
"""
import time
import numpy as np
from . import fit as F, grid as G


def _fit_box(box, cfg, warm):
    return F.fit_point_box(box, G._DATA, cfg, warm)


def _relax_box(box, cfg, warm, cval, margin):
    """Refit a cell only from neighbour solutions that, placed inside the cell (as fit_point_box places warm
    starts), already beat the cell value by > margin (one chi2 each). Returns (chi2, x, refitted)."""
    from . import model as M
    s_lo, s_hi, q_lo, q_hi = box
    be = F.make_backend(cfg["grid_backend"])
    good = []
    for w in warm[1:]:
        ls = s_lo + (s_hi - s_lo) * min(max((w[5] - s_lo) / (s_hi - s_lo), 0.02), 0.98)
        lq = q_lo + (q_hi - q_lo) * min(max((w[6] - q_lo) / (q_hi - q_lo), 0.02), 0.98)
        if M.chi2(be, F.full_p(w[:5], ls, lq), G._DATA) < cval - margin:
            good.append(w)
    if not good:
        return None
    c, x = F.fit_point_box(box, G._DATA, cfg, [warm[0]] + good)
    return c, x


def _relax_queue(cells, boxf, nx, cfg_relax, top, workers, log, margin, relax_nb, mirror, precheck, label):
    """_relax through grid.refine_queue: one task per stale cell, improvements propagate as they arrive."""
    ny = max(j for _, j in cells) + 1
    chi = np.full((nx, ny), np.nan)
    pars = np.full((nx, ny, 7), np.nan)
    for (i, j), (c, x) in cells.items():
        if x is not None and np.isfinite(c):
            chi[i, j], pars[i, j] = c, x
    state = {"cmin": float(np.nanmin(chi))}

    def make_args(i, j, chi, pars, forced):
        c = chi[i, j]
        if not np.isfinite(c):
            return None
        state["cmin"] = min(state["cmin"], float(np.nanmin(chi[max(0, i - 1):i + 2, max(0, j - 1):j + 2])))
        cand = []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                if (a or b) and 0 <= i + a < nx and 0 <= j + b < ny and np.isfinite(chi[i + a, j + b]) \
                        and chi[i + a, j + b] < c - margin and chi[i + a, j + b] - state["cmin"] < 2 * top:
                    cand.append((chi[i + a, j + b], pars[i + a, j + b]))
        km = nx - 1 - i
        if mirror and np.isfinite(chi[km, j]) and chi[km, j] < c - margin:
            y = pars[km, j].copy()
            y[5] = -y[5]
            cand.append((chi[km, j], y))
        cand.sort(key=lambda v: v[0])
        if relax_nb:
            cand = cand[:relax_nb]
        if not cand:
            return None
        w = [pars[i, j]] + [v[1] for v in cand]
        return (boxf(i, j), cfg_relax, w, float(c), margin) if precheck else (boxf(i, j), cfg_relax, w)

    st = {}
    chi2, pars2 = G.refine_queue(chi, pars, _relax_box if precheck else _fit_box, make_args, workers, margin,
                                 log, f"{label}neighbour refinement", stats=st)
    for (i, j) in cells:
        if np.isfinite(chi2[i, j]) and chi2[i, j] < cells[(i, j)][0]:
            cells[(i, j)] = (float(chi2[i, j]), pars2[i, j].copy())
    log(f"  {label}neighbour refinement: {st.get('tasks', 0)} cells checked, {st.get('improved', 0)} improved by > {margin}")
    return st.get("tasks", 0), st.get("improved", 0)


def _relax(cells, boxf, nx, cfg_relax, top, workers, log, margin=1.0, sweeps=6, relax_nb=None, mirror=False,
           precheck=True, label=""):
    """Relaxation of one level: cells = {(i, j): (chi2, x)} (updated in place); nx = cells per row (for the
    s <-> 1/s mirror). cfg["relax_mode"] "queue" (default): grid.refine_queue, one task per stale cell;
    "sweeps": the earlier Jacobi sweeps (at most `sweeps`). Returns the number of worker tasks and of full refits."""
    if cfg_relax.get("relax_mode", "queue") != "sweeps":
        return _relax_queue(cells, boxf, nx, cfg_relax, top, workers, log, margin, relax_nb, mirror, precheck, label)
    ntask = nref = 0
    for sweep in range(sweeps):
        cmin = min(v[0] for v in cells.values())
        rt = {}
        for (i, j), (c, x) in cells.items():
            if x is None:
                continue
            cand = [cells[(i + di, j + dj)] for di in (-1, 0, 1) for dj in (-1, 0, 1)
                    if (di or dj) and (i + di, j + dj) in cells and cells[(i + di, j + dj)][1] is not None
                    and cells[(i + di, j + dj)][0] < c - margin and cells[(i + di, j + dj)][0] - cmin < 2 * top]
            km = (nx - 1 - i, j)
            if mirror and km in cells and cells[km][1] is not None and cells[km][0] < c - margin:
                y = np.array(cells[km][1], float)
                y[5] = -y[5]
                cand.append((cells[km][0], y))
            cand.sort(key=lambda v: v[0])
            if relax_nb:                                 # only the best relax_nb candidates
                cand = cand[:relax_nb]
            if cand:
                w = [x] + [v[1] for v in cand]
                rt[(i, j)] = (boxf(i, j), cfg_relax, w, c, margin) if precheck else (boxf(i, j), cfg_relax, w)
        if not rt:
            break
        res, _ = G.run_isolated(_relax_box if precheck else _fit_box, rt, workers, log=lambda *a: None,
                                desc=f"{label}neighbour refinement, pass {sweep + 1}")
        ntask += len(rt)
        gain, nr = [], 0
        for k, v in res.items():
            if v is None:
                continue
            nr += 1
            if np.isfinite(v[0]) and v[0] < cells[k][0]:
                gain.append(cells[k][0] - v[0])
                cells[k] = v
        nref += nr
        gain = np.array(gain)
        log(f"  {label}neighbour refinement, pass {sweep + 1}: {len(rt)} cells checked, {nr} refitted, "
            f"{np.sum(gain > margin)} improved by > {margin}"
            + (f" (max {gain.max():.1f})" if len(gain) else ""))
        if not np.any(gain > margin):
            break
    return ntask, nref


def _leaves(cells, box):
    leaf = []
    for (l, i, j), (c, x) in cells.items():
        if (l + 1, 2 * i, 2 * j) in cells:
            continue
        b = box(l, i, j)
        xs = x if x is not None else np.full(7, np.nan)
        leaf.append((0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[1] - b[0]), 0.5 * (b[3] - b[2]), c, l,
                     xs[5], xs[6], *xs))
    return np.array(leaf)


def run_contour(s_range, q_range, n0, levels, cfg, data, bands, workers=19, log=print, snapshot=None,
                min_tol=0.1, max_fits=None, min_ds=0.004, min_dq=0.01, max_nb=None, relax=True, relax_margin=1.0,
                relax_sweeps=6, relax_nb=None, precheck=True, seed=0, start=None, stage=None):
    """Goal-driven stopping rule (contours, not just the minimum):
      - while the global minimum still improves by > min_tol per level, cells whose minimum lies in the lowest
        band are split (to locate the minimum);
      - always: a cell is split only if a displayed contour runs through it: its band differs from a neighbour's
        band, or from its parent's (children disagree), with at least one of them inside the displayed bands;
      - cells whose 4 children all share the parent's band are uniform -> not split again;
      - target resolution: a cell is not split once its width would drop below (min_ds, min_dq) in (log s, log q);
        default 0.004 = ~1/5 of the narrowest ridge seen (Delta chi2 < 240 band ~0.02 wide in log s at log q -2.1);
        `levels` is only a safety cap;
      - stop when nothing qualifies, or at `max_fits`.
    A snapshot of the leaves is written after every level (`snapshot` path).
    relax: after each level, cells whose same-level neighbour (or s <-> 1/s mirror cell, if the s range is
    symmetric) is lower by > relax_margin are refitted from those solutions (no alpha screen), in sweeps until no
    cell improves; first-level box fits without warm starts were often far from their minimum (8 x 6 test: 40 of 48
    cells improved, up to 2424; 24 band changes at n = 45).
    start: {(i, j): (chi2, x)} first-level cells from an earlier run (e.g. a lattice run's box fits, improved by its
    repaired grid points), x = (t0, u0, tE, alpha, log rho, log s, log q): level 0 is taken as given (no seed grid,
    no fits, no relaxation) and the refinement continues from there.
    stage: optional grid.Stages; a numbered header is printed for the seed grid and for every level fitted."""
    G._DATA = data
    G.warmup(cfg, data)
    ws0 = (s_range[1] - s_range[0]) / n0[0]
    mirror_ok = abs(s_range[0] + s_range[1]) < 1e-9
    cfg_relax = dict(cfg, n_screen=0)
    # cells with warm starts (seeded level-0 cells, children): cfg["n_screen_seeded"] angle-screen starts instead of
    # cfg["n_screen"] (the seed grid has screened the angles already; default: the same)
    cfg_warm = dict(cfg, n_screen=cfg.get("n_screen_seeded", cfg.get("n_screen", 4)))
    with_warm = lambda warm: cfg_warm if warm else cfg
    wq0 = (q_range[1] - q_range[0]) / n0[1]
    top = bands[-1]

    def box(l, i, j):
        ws, wq = ws0 / 2 ** l, wq0 / 2 ** l
        return (s_range[0] + i * ws, s_range[0] + (i + 1) * ws, q_range[0] + j * wq, q_range[0] + (j + 1) * wq)

    cells = {}
    t0, nfit, per_level, prev_min = time.time(), 0, [], np.inf
    seeds = {}
    if seed and start is None:                           # coarse seed grid (seed x seed first-level cells)
        if stage:
            stage("Coarse seed grid", f"quick fits on {seed} x {seed} merged cells, used as starting points")
        nsx, nsy = -(-n0[0] // seed), -(-n0[1] // seed)
        sbox = lambda i, j: (s_range[0] + i * seed * ws0, s_range[0] + (i + 1) * seed * ws0,
                             q_range[0] + j * seed * wq0, q_range[0] + (j + 1) * seed * wq0)
        res, _ = G.run_isolated(_fit_box, {(i, j): (sbox(i, j), cfg, None) for i in range(nsx) for j in range(nsy)},
                                workers, log=lambda *a: None, desc=f"seed grid {nsx} x {nsy}")
        seeds = {k: (v if v is not None and np.isfinite(v[0]) else (np.inf, None)) for k, v in res.items()}
        nt, nr = _relax(seeds, sbox, nsx, cfg_relax, top, workers, log, relax_margin, relax_sweeps, relax_nb,
                        mirror_ok and n0[0] % seed == 0, precheck, label="seed grid: ")
        nfit += len(seeds) + nr
        log(f"seed grid {nsx} x {nsy}: {len(seeds) + nr} fits, {time.time() - t0:.0f}s, "
            f"min {min(v[0] for v in seeds.values()):.2f}")

    def seed_warm(i, j):
        if not seeds:
            return None
        k = (i // seed, j // seed)
        nb = sorted((seeds[(k[0] + a, k[1] + b)] for a in (-1, 0, 1) for b in (-1, 0, 1)
                     if (a or b) and (k[0] + a, k[1] + b) in seeds and seeds[(k[0] + a, k[1] + b)][1] is not None),
                    key=lambda v: v[0])[:2]
        return [w for w in [seeds[k][1]] + [v[1] for v in nb] if w is not None]

    tasks = {(0, i, j): (box(0, i, j), with_warm(seed_warm(i, j)), seed_warm(i, j)) for i in range(n0[0]) for j in range(n0[1])}
    for lev in range(levels + 1):
        gb_prev = min(cells.values(), key=lambda v: v[0]) if cells else (np.inf, None)
        resumed = lev == 0 and start is not None
        if resumed:
            res = {(0, i, j): (float(c), None if x is None else np.asarray(x, float)) for (i, j), (c, x) in start.items()}
        else:
            if stage:
                stage("Cell fits" if lev == 0 else f"Finer cells, level {lev}",
                      "best fit inside each cell" if lev == 0 else
                      f"{len(tasks) // 4} cells on band edges, each split into 4")
            res, _ = G.run_isolated(_fit_box, tasks, workers, log=lambda *a: None,
                                    desc="cell fits" if lev == 0 else f"finer cells, level {lev}")
            nfit += len(tasks)
        n_inh = 0
        for k, v in res.items():
            v = v if v is not None and np.isfinite(v[0]) else (np.inf, None)
            # monotonicity: a cell minimum can never be worse than an already evaluated point inside the cell
            # (the parent's best point or the global best); a child fit can miss the needle-like valley
            bx = tasks[k][0]
            l, i, j = k
            for cand in ((cells.get((l - 1, i // 2, j // 2)) if l > 0 else None), gb_prev):
                if cand is not None and cand[1] is not None and cand[0] < v[0]:
                    ls_c, lq_c = cand[1][5], cand[1][6]
                    if bx[0] <= ls_c <= bx[1] and bx[2] <= lq_c <= bx[3]:
                        v = cand
                        n_inh += 1
            cells[k] = v
        per_level.append(0 if resumed else len(tasks))
        if relax and not resumed:
            lvl = {(i, j): v for (l, i, j), v in cells.items() if l == lev}
            nt, nr = _relax(lvl, lambda i, j: box(lev, i, j), n0[0] * 2 ** lev, cfg_relax, top, workers, log,
                            relax_margin, relax_sweeps, relax_nb, mirror_ok, precheck, label="" if lev == 0 else f"level {lev}: ")
            for (i, j), v in lvl.items():
                cells[(lev, i, j)] = v
            nfit += nr
            per_level[-1] += nr
        cmin = min(v[0] for v in cells.values())
        log(f"{'cells' if lev == 0 else f'level {lev}'}: {0 if resumed else len(tasks)} fits, {time.time() - t0:.0f}s, "
            f"best chi2 {cmin:.2f}"
            + (f" ({n_inh} cells inherited a better point)" if n_inh else ""))
        if snapshot:
            lf = _leaves(cells, box)
            np.savez(snapshot, logs=lf[:, 0], logq=lf[:, 1], hs=lf[:, 2], hq=lf[:, 3], chi2=lf[:, 4], level=lf[:, 5],
                     best_ls=lf[:, 6], best_lq=lf[:, 7], x=lf[:, 8:], nfit=nfit, per_level=np.array(per_level),
                     wall=time.time() - t0, last_level=lev)
        min_moving = prev_min - cmin > min_tol
        prev_min = cmin
        if lev == levels or (max_fits and nfit >= max_fits):
            break
        band = lambda c: int(np.searchsorted(bands, c - cmin))
        leaves = {k: v for k, v in cells.items() if k[0] == lev}
        split = []
        child_ws, child_wq = ws0 / 2 ** (lev + 1), wq0 / 2 ** (lev + 1)
        if child_ws < min_ds and child_wq < min_dq:
            log("  -> target resolution reached")
            break
        for (l, i, j), (c, x) in leaves.items():
            b = band(c)
            do = min_moving and b == 0
            if not do:
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = leaves.get((l, i + di, j + dj))
                    if nb is not None and min(nb[0], c) - cmin < top and band(nb[0]) != b:
                        do = True
                        break
            if not do and l > 0:
                par = cells.get((l - 1, i // 2, j // 2))
                if par is not None and band(par[0]) != b and min(par[0], c) - cmin < top:
                    do = True
            if do:
                split.append((l, i, j))
        tasks = {}
        gbest = min(cells.values(), key=lambda v: v[0])[1]
        for (l, i, j) in split:
            warm = [cells[(l, i, j)][1], gbest]
            nbs = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = cells.get((l, i + di, j + dj))
                    if (di or dj) and nb is not None and nb[1] is not None and nb[0] - cmin < 2 * top:
                        nbs.append(nb)
            nbs.sort(key=lambda v: v[0])                 # only the best `max_nb` neighbours
            warm += [v[1] for v in nbs[:max_nb]]
            for a in (0, 1):
                for b2 in (0, 1):
                    tasks[(lev + 1, 2 * i + a, 2 * j + b2)] = (box(lev + 1, 2 * i + a, 2 * j + b2), with_warm(warm), warm)
        log(f"  -> {len(split)} cells on band edges will be split (best fit {'still improving' if min_moving else 'converged'})")
        if not tasks:
            break
    lf = _leaves(cells, box)
    return dict(logs=lf[:, 0], logq=lf[:, 1], hs=lf[:, 2], hq=lf[:, 3], chi2=lf[:, 4], level=lf[:, 5],
                best_ls=lf[:, 6], best_lq=lf[:, 7], x=lf[:, 8:], nfit=nfit, per_level=np.array(per_level),
                wall=time.time() - t0)


def run(s_range, q_range, n0, levels, cfg, data, cut, bands, workers=19, log=print):
    G._DATA = data
    G.warmup(cfg, data)
    ws0 = (s_range[1] - s_range[0]) / n0[0]
    wq0 = (q_range[1] - q_range[0]) / n0[1]

    def box(l, i, j):
        ws, wq = ws0 / 2 ** l, wq0 / 2 ** l
        return (s_range[0] + i * ws, s_range[0] + (i + 1) * ws, q_range[0] + j * wq, q_range[0] + (j + 1) * wq)

    cells = {}                                         # (level, i, j) -> (chi2, 7-vector)
    tasks = {(0, i, j): (box(0, i, j), cfg, None) for i in range(n0[0]) for j in range(n0[1])}
    t0, nfit, per_level = time.time(), 0, []
    for lev in range(levels + 1):
        resumed = lev == 0 and start is not None
        if resumed:
            res = {(0, i, j): (float(c), None if x is None else np.asarray(x, float)) for (i, j), (c, x) in start.items()}
        else:
            if stage:
                stage("Cell fits" if lev == 0 else f"Finer cells, level {lev}",
                      "best fit inside each cell" if lev == 0 else
                      f"{len(tasks) // 4} cells on band edges, each split into 4")
            res, _ = G.run_isolated(_fit_box, tasks, workers, log=lambda *a: None,
                                    desc="cell fits" if lev == 0 else f"finer cells, level {lev}")
            nfit += len(tasks)
        for k, v in res.items():
            cells[k] = v if v is not None else (np.inf, None)
        per_level.append(len(tasks))
        log(f"level {lev}: {len(tasks)} fits, {time.time() - t0:.0f}s, min {min(v[0] for v in cells.values()):.2f}")
        if lev == levels:
            break
        cmin = min(v[0] for v in cells.values())
        band = lambda c: int(np.searchsorted(bands, c - cmin))
        leaves = {k: v for k, v in cells.items() if k[0] == lev}
        split = []
        for (l, i, j), (c, x) in leaves.items():
            interesting = c - cmin < cut
            if not interesting:
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = leaves.get((l, i + di, j + dj))
                    if nb is not None and min(nb[0], c) - cmin < cut and band(nb[0]) != band(c):
                        interesting = True
                        break
            if interesting:
                split.append((l, i, j))
        tasks = {}
        gbest = min(cells.values(), key=lambda v: v[0])[1]       # global best so far: one extra start everywhere
        for (l, i, j) in split:
            warm = [cells[(l, i, j)][1], gbest]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = cells.get((l, i + di, j + dj))
                    if (di or dj) and nb is not None and nb[1] is not None and nb[0] - cmin < 2 * cut:
                        warm.append(nb[1])
            for a in (0, 1):
                for b in (0, 1):
                    tasks[(lev + 1, 2 * i + a, 2 * j + b)] = (box(lev + 1, 2 * i + a, 2 * j + b), cfg, warm)
        if not tasks:
            break
    leaf = []
    for (l, i, j), (c, x) in cells.items():
        if (l + 1, 2 * i, 2 * j) in cells:
            continue
        b = box(l, i, j)
        xs = x if x is not None else np.full(7, np.nan)
        leaf.append((0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[1] - b[0]), 0.5 * (b[3] - b[2]), c, l,
                     xs[5], xs[6]))
    leaf = np.array(leaf)
    return dict(logs=leaf[:, 0], logq=leaf[:, 1], hs=leaf[:, 2], hq=leaf[:, 3], chi2=leaf[:, 4], level=leaf[:, 5],
                best_ls=leaf[:, 6], best_lq=leaf[:, 7], nfit=nfit, per_level=np.array(per_level),
                wall=time.time() - t0)

