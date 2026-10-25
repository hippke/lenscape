"""Grid-point optimisation strategies: s, q fixed; (t0, u0, tE, alpha, log rho) optimised."""
import numpy as np
from .optimize import minimize, least_squares_lm
from . import model

SEED = None          # default start vector dict(t0, u0, tE, logrho); per fit: cfg["seed"] (see seed_from_data)


def seed_from_data(data, u0=0.05, tE=20.0, logrho=-3.0):
    """Rough start vector for the fits: t0 at the brightest point (in units of each data set's scatter around its
    median); u0, tE, log rho from the arguments. Every fit screens 20 trajectory angles and explores from there,
    so the start only has to be in the right neighbourhood (t0 within ~tE)."""
    z = np.empty(len(data.t))
    for j in range(data.nsets):
        k = data.idx == j
        f = data.f[k]
        mad = np.median(np.abs(f - np.median(f))) * 1.4826 or 1.0
        z[k] = (f - np.median(f)) / mad
    return dict(t0=float(data.t[int(np.argmax(z))]), u0=float(u0), tE=float(tE), logrho=float(logrho))


def _seed(seed):
    sd = seed or SEED
    if sd is None:
        raise ValueError("no start vector: set cfg['seed'] = dict(t0=..., u0=..., tE=..., logrho=...), "
                         "e.g. lenscape.fit.seed_from_data(data)")
    return sd
ALPHAS = np.arange(20) * 2 * np.pi / 20


def make_backend(cfg):
    """Backend from a config dict; kind defaults to our own code ('nb'). Reference backends (e.g. 'vbbl' in the
    test helpers) and experimental ones register themselves in model.BACKENDS."""
    cfg = dict(cfg)
    kind = cfg.pop("kind", "nb")
    if kind not in model.BACKENDS:
        raise KeyError(f"backend {kind!r} not registered (reference backends such as VBBinaryLensing "
                       f"register themselves with model.register_backend)")
    return model.BACKENDS[kind](**cfg)


def full_p(x, ls, lq):
    return (x[0], x[1], abs(x[2]) + 1e-3, ls, lq, x[3], x[4])


def fit_point_baseline(ls, lq, data, cfg):
    """Iteration-0 strategy: 20 alpha seeds x Nelder-Mead(200), refine best 2 with NM(1500)."""
    be = make_backend(cfg["grid_backend"])
    f = lambda x: model.chi2(be, full_p(x, ls, lq), data)
    trials = []
    sd = _seed(cfg.get("seed"))
    for a in ALPHAS:
        x0 = [sd["t0"], sd["u0"], sd["tE"], a, sd["logrho"]]
        r = minimize(f, x0, method="Nelder-Mead", options=dict(maxfev=200, adaptive=True))
        trials.append((r.fun, r.x))
    trials.sort(key=lambda z: z[0])
    best = trials[0]
    for _, x in trials[:2]:
        r = minimize(f, x, method="Nelder-Mead", options=dict(maxfev=1500, adaptive=True, fatol=1e-2))
        if r.fun < best[0]:
            best = (r.fun, r.x)
    return best[0], np.asarray(best[1])


def screen_alphas(f, k, seed=None):
    """The chi2 of all 20 alpha seeds at the start vector; return the k best start vectors."""
    if k <= 0:                                       # warm starts only (AMR relaxation refits)
        return []
    sd = _seed(seed)
    starts = [np.array([sd["t0"], sd["u0"], sd["tE"], a, sd["logrho"]]) for a in ALPHAS]
    c = np.array([f(x) for x in starts])
    return [starts[i] for i in np.argsort(c)[:k]]


def _caustic_curves(ls, lq):
    """Caustics of (s, q) as (points, cumulative arc length) per closed curve."""
    from . import caustics as K
    out = []
    for c in K.caustics(10.0 ** ls, 10.0 ** lq, 4000):
        z = c[:-1]
        out.append((c, np.r_[0.0, np.cumsum(np.abs(np.diff(c)))]))
    return out


def _on_caustic(curve, a):
    """Point at arc position a (periodic) on a closed caustic curve."""
    c, arc = curve
    a = np.mod(a, arc[-1])
    return complex(np.interp(a, arc, c.real), np.interp(a, arc, c.imag))


def _chord(z_in, z_out, t_in, t_out, logrho):
    """(t0, u0, tE, alpha, log rho) of the straight trajectory on z_in at t_in and on z_out at t_out."""
    dz = z_out - z_in
    L = abs(dz)
    if L < 1e-9:
        return None
    d = dz / L                                         # direction of motion = (-cos alpha, -sin alpha)
    al = np.arctan2(-d.imag, -d.real)
    u0 = z_in.real * np.sin(al) - z_in.imag * np.cos(al)   # y(tau) = u0 (sin a, -cos a) + tau d
    tau_in = z_in.real * d.real + z_in.imag * d.imag
    tE = (t_out - t_in) / L
    return np.array([t_in - tau_in * tE, u0, tE, np.mod(al, 2 * np.pi), logrho])


def caustic_starts(ls, lq, crossings, logrho=-3.0, n=24, with_arcs=False):
    """Start vectors (t0, u0, tE, alpha, log rho) of straight trajectories that are on the caustics of (s, q) at
    crossings[0] (entry) and crossings[1] (exit): the caustic-crossing parameterisation (Cassan 2008; MulensModel's
    uniform caustic sampling). Every ordered pair of n points per caustic, uniform in arc length on the same
    caustic, fixes alpha, u0, tE = (t_out - t_in) / |z_out - z_in| and t0. with_arcs: also return
    (curve index, arc_in, arc_out) for each vector (see screen_caustic). Frame as in model.lc."""
    t_in, t_out = float(crossings[0]), float(crossings[1])
    out, arcs = [], []
    for ci, curve in enumerate(_caustic_curves(ls, lq)):
        pos = np.linspace(0, curve[1][-1], n, endpoint=False)
        pts = [_on_caustic(curve, a) for a in pos]
        for a in range(n):
            for b in range(n):
                if a != b:
                    x = _chord(pts[a], pts[b], t_in, t_out, logrho)
                    if x is not None:
                        out.append(x)
                        arcs.append((ci, pos[a], pos[b]))
    return (out, arcs) if with_arcs else out


def screen_caustic(fx, ls, lq, crossings, k, logrho=-3.0, n=24, refine=True):
    """The k best caustic_starts by chi2 (fx: chi2 of a 5-vector at fixed s, q); several crossing pairs may be
    given as a list of (t_in, t_out). refine: the best 2 k candidates are first optimised in their own coordinates,
    the entry and exit positions along the caustic (two parameters; the trajectory stays on the caustic at the
    given times), which turns a coarse candidate into an accurate start before the k best are returned."""
    pairs = [crossings] if np.ndim(crossings) == 1 else list(crossings)
    cand, info = [], []
    for pr in pairs:
        xs, arcs = caustic_starts(ls, lq, pr, logrho, n, with_arcs=True)
        cand += xs
        info += [(pr, a) for a in arcs]
    if not cand or k <= 0:
        return []
    c = np.array([fx(x) for x in cand])
    order = np.argsort(c)
    if not refine:
        return [cand[i] for i in order[:k]]
    curves = _caustic_curves(ls, lq)
    done = []
    for i in order[:2 * k]:
        (t_in, t_out), (ci, a0, b0) = info[i]
        curve = curves[ci]
        step = curve[1][-1] / n
        g = lambda v: (lambda x: fx(x) if x is not None else np.inf)(
            _chord(_on_caustic(curve, v[0]), _on_caustic(curve, v[1]), float(t_in), float(t_out), logrho))
        r = minimize(g, [a0, b0], method="Nelder-Mead",
                     options=dict(maxfev=120, initial_simplex=[[a0, b0], [a0 + 0.5 * step, b0], [a0, b0 + 0.5 * step]],
                                  xatol=1e-4 * step, fatol=1.0))
        x = _chord(_on_caustic(curve, r.x[0]), _on_caustic(curve, r.x[1]), float(t_in), float(t_out), logrho)
        if x is not None:
            done.append((r.fun, x))
    done.sort(key=lambda v: v[0])
    return [x for _, x in done[:k]]


def _all_starts(fx, ls, lq, cfg):
    """Angle screen and, if cfg["caustic_crossings"] is set, caustic-crossing starts."""
    starts = [] if cfg.get("caustic_only") else screen_alphas(fx, cfg.get("n_screen", 4), cfg.get("seed"))
    if cfg.get("caustic_crossings") is not None:
        lr = (cfg.get("seed") or {}).get("logrho", -3.0)
        starts += screen_caustic(fx, ls, lq, cfg["caustic_crossings"], cfg.get("n_caustic", 4), lr,
                                 cfg.get("caustic_samples", 24), cfg.get("caustic_refine", True))
    return starts


class Coords:
    """Optimiser coordinates.
    'native': (t0, u0, tE, alpha, log rho)
    'teff'  : (t0, t_eff = u0*tE, ln tE, alpha, log10 t_*), t_* = rho*tE  (what a blended high-mag event measures)
    Steps define a scaled initial simplex (the default simplex is +-5 % of each value, i.e. +-38 d in t0)."""

    def __init__(self, kind):
        self.kind = kind

    def to_x(self, y):
        if self.kind == "native":
            return np.asarray(y)
        t0, te, ltE, al, lts = y
        tE = np.exp(ltE)
        return np.array([t0, te / tE, tE, al, lts - np.log10(tE)])

    def from_x(self, x):
        if self.kind == "native":
            return np.asarray(x)
        t0, u0, tE, al, lr = x
        tE = abs(tE) + 1e-3
        return np.array([t0, u0 * tE, np.log(tE), al, lr + np.log10(tE)])

    def steps(self, y, scale):
        if self.kind == "native":
            st = [0.05, 0.1 * abs(y[1]) + 1e-3, 0.1 * abs(y[2]), 0.05, 0.1]
        else:
            st = [0.05, 0.1 * abs(y[1]) + 1e-3, 0.1, 0.05, 0.1]
        return np.array(st) * scale


def _nm(f, y0, maxfev, step_scale, fatol=None):
    if step_scale is None:                       # default simplex (+-5 %)
        return minimize(f, y0, method="Nelder-Mead", options=dict(maxfev=maxfev, adaptive=True,
                                                                   **({"fatol": fatol} if fatol else {})))
    sim = np.vstack([y0] + [y0 + np.eye(5)[k] * st for k, st in enumerate(f.coords.steps(y0, step_scale))])
    return minimize(f, y0, method="Nelder-Mead", options=dict(maxfev=maxfev, adaptive=True, initial_simplex=sim,
                                                               **({"fatol": fatol} if fatol else {})))


# Levenberg-Marquardt in the t_eff coordinates (t0, t_eff, ln tE, alpha, log t_*): forward-difference steps (well above
# the integration noise of the grid backend) and largest step per iteration
LM_STEP = np.array([1e-4, 1e-5, 1e-4, 1e-4, 1e-3])
LM_CAP = np.array([0.3, 0.05, 0.1, 0.05, 0.2])


def nm_then_lm(f, fr, y0, sim, n_explore, h=LM_STEP, cap=LM_CAP):
    """A short Nelder-Mead exploration from y0 (initial simplex sim, or the default simplex if None; n_explore
    evaluations) to reach the right valley, then Levenberg-Marquardt on the residual vector fr to converge (from
    distant starts LM alone stops in a nearer minimum). f: chi2; fr: residuals with sum of squares = chi2 + const.
    Returns (chi2, y)."""
    opts = dict(maxfev=n_explore, adaptive=True, fatol=1e-2)
    if sim is not None:
        opts["initial_simplex"] = sim
    r = minimize(f, y0, method="Nelder-Mead", options=opts)
    q = least_squares_lm(fr, r.x, h, cap)
    if q.fun < r.fun or not np.isfinite(r.fun):
        c = f(q.x)                      # chi2 itself (binned data: the residuals' sum of squares omits a constant)
        if c < r.fun:
            return c, q.x
    return r.fun, r.x


def local_fit(be, data, ls, lq, y0, cfg, step, maxfev):
    """Local fit at fixed (s, q) from y0 (optimiser coordinates). cfg["optimizer"]:
    "nm" (default): Nelder-Mead, maxfev evaluations, initial simplex scaled by step.
    "nm_lm": nm_then_lm with min(maxfev, cfg["explore_nfev"]) (default 100) Nelder-Mead evaluations, then
    Levenberg-Marquardt; 2-3x fewer light curves than Nelder-Mead at the same quality (MOA-2022-BLG-091: densified
    block 3.1x less CPU, 99.8 % of points in the same Delta chi2 band). Only for the "teff" coordinates.
    Returns (chi2, y)."""
    co = Coords(cfg.get("coords", "teff"))
    f = lambda y: model.chi2(be, full_p(co.to_x(y), ls, lq), data)
    f.coords = co
    if cfg.get("optimizer", "nm") == "nm" or co.kind != "teff":
        r = _nm(f, y0, maxfev, step, 1e-2)
        return r.fun, r.x
    fr = lambda y: model.chi2_residuals(be, np.asarray(full_p(co.to_x(y), ls, lq), float), data)
    sim = None if step is None else np.vstack([y0] + [y0 + np.eye(5)[k] * st for k, st in enumerate(co.steps(y0, step))])
    return nm_then_lm(f, fr, y0, sim, min(maxfev, cfg.get("explore_nfev", 100)))


def fit_point_screen_d(ls, lq, data, cfg, warm=None):
    """Screen strategy in configurable coordinates with a scaled initial simplex.
    warm: optional neighbour solutions (t0, u0, tE, alpha, log rho) added as extra starts."""
    be = make_backend(cfg["grid_backend"])
    co = Coords(cfg.get("coords", "teff"))
    sc = cfg.get("step_scale", 1.0)
    fx = lambda x: model.chi2(be, full_p(x, ls, lq), data)
    f = lambda y: fx(co.to_x(y))
    f.coords = co
    trials = []
    starts = _all_starts(fx, ls, lq, cfg)
    for w in (warm or []):
        if w is not None and np.all(np.isfinite(w)):
            starts.append(np.asarray(w))
    for x0 in starts:
        r = _nm(f, co.from_x(x0), cfg.get("n_explore", 200), cfg.get("explore_scale", sc))
        trials.append((r.fun, r.x))
    trials.sort(key=lambda z: z[0])
    best = trials[0]
    lm = cfg.get("optimizer", "nm") != "nm" and co.kind == "teff"
    if lm:
        fr = lambda y: model.chi2_residuals(be, np.asarray(full_p(co.to_x(y), ls, lq), float), data)
    for _, y in trials[:2]:
        rs = cfg.get("refine_step", None if sc is None else 0.3 * sc)
        if lm:                                          # refinement by nm_then_lm (see local_fit)
            sim = None if rs is None else np.vstack([y] + [y + np.eye(5)[k] * st for k, st in enumerate(co.steps(y, rs))])
            c, yr = nm_then_lm(f, fr, y, sim, cfg.get("explore_nfev", 100))
        else:
            r = _nm(f, y, cfg.get("n_refine", 1500), rs, 1e-2)
            c, yr = r.fun, r.x
        if c < best[0]:
            best = (c, yr)
    return best[0], co.to_x(best[1])


def fit_point_heavy(ls, lq, data, cfg, warm=None):
    """Exhaustive reference search for single cells: screen ~1080 diverse starts (alpha x tE x u0 x sign x t0),
    explore the best n_heavy with NM (default simplex) and refine each in t_eff coordinates."""
    be = make_backend(cfg["grid_backend"])
    sd = _seed(cfg.get("seed"))
    co = Coords("teff")
    fx = lambda x: model.chi2(be, full_p(x, ls, lq), data)
    g = lambda y: fx(co.to_x(y))
    g.coords = co
    starts = [np.array([sd["t0"] + dt, sg * u, sd["tE"] * ft, a, sd["logrho"]])
              for a in ALPHAS for ft in (0.5, 1.0, 2.0) for u in (0.3 * abs(sd["u0"]) + 1e-3, abs(sd["u0"]), 3 * abs(sd["u0"]))
              for sg in (1, -1) for dt in (-1.0, 0.0, 1.0)]
    starts += [np.asarray(w) for w in (warm or []) if w is not None and np.all(np.isfinite(w))]
    c0 = np.array([fx(x) for x in starts])
    best = (np.inf, None)
    for k in np.argsort(c0)[:cfg.get("n_heavy", 30)]:
        r = _nm(g, co.from_x(starts[k]), 300, None)
        r = _nm(g, r.x, 1500, 0.3, 1e-3)
        if r.fun < best[0]:
            best = (r.fun, co.to_x(r.x))
    return best[0], np.asarray(best[1])


STRATEGIES = {"baseline": fit_point_baseline, "screen_d": fit_point_screen_d, "heavy": fit_point_heavy}
WARM_STRATEGIES = {"screen_d", "heavy"}          # strategies that accept neighbour solutions as extra starts


def _sig(u):
    """Logistic map for the box fit, clipped to |u| <= 30 (sigma(30) = 1 - 1e-13): a cell minimum on the box edge
    otherwise drove u to |u| > 700 (exp overflow; flat tail where Nelder-Mead wanders without effect)."""
    u = min(max(u, -30.0), 30.0)
    return 1.0 / (1.0 + np.exp(-u))


def _logit(p):
    p = min(max(p, 0.02), 0.98)          # keep warm starts off the flat ends of the sigmoid
    return np.log(p / (1 - p))


def fit_point_box(box, data, cfg, warm=None):
    """Minimum of chi2 over a (log s, log q) cell box = (s_lo, s_hi, q_lo, q_hi): s and q float inside the
    box via a sigmoid, so a narrow valley crossing the cell is found even if it misses the cell centre.
    warm: full solutions (t0, u0, tE, alpha, log rho, log s, log q). Returns (chi2, full 7-vector)."""
    s_lo, s_hi, q_lo, q_hi = box
    be = make_backend(cfg["grid_backend"])
    co = Coords(cfg.get("coords", "teff"))
    to_s = lambda u: s_lo + (s_hi - s_lo) * _sig(u)
    to_q = lambda u: q_lo + (q_hi - q_lo) * _sig(u)

    def f(y):                                           # y = (t0, t_eff, ln tE, alpha, log t_*, us, uq)
        x = co.to_x(y[:5])
        return model.chi2(be, full_p(x, to_s(y[5]), to_q(y[6])), data)

    def nm(y0, maxfev, step):
        if step is None:
            return minimize(f, y0, method="Nelder-Mead", options=dict(maxfev=maxfev, adaptive=True))
        st = np.r_[co.steps(y0[:5], step), 0.5, 0.5]
        sim = np.vstack([y0] + [y0 + np.eye(7)[k] * st[k] for k in range(7)])
        return minimize(f, y0, method="Nelder-Mead",
                        options=dict(maxfev=maxfev, adaptive=True, initial_simplex=sim, fatol=1e-2))

    lsc, lqc = 0.5 * (s_lo + s_hi), 0.5 * (q_lo + q_hi)
    fx = lambda x: model.chi2(be, full_p(x, lsc, lqc), data)
    starts = [np.r_[co.from_x(x0), 0.0, 0.0] for x0 in _all_starts(fx, lsc, lqc, cfg)]
    for w in (warm or []):
        if w is not None and np.all(np.isfinite(w)):
            us = _logit((w[5] - s_lo) / (s_hi - s_lo))
            uq = _logit((w[6] - q_lo) / (q_hi - q_lo))
            starts.append(np.r_[co.from_x(w[:5]), us, uq])
    trials = []
    for y0 in starts:
        r = nm(y0, cfg.get("n_explore_box", 300), None)
        trials.append((r.fun, r.x))
    trials.sort(key=lambda z: z[0])
    best = trials[0]
    # refinement by Nelder-Mead also with optimizer="nm_lm": Levenberg-Marquardt in the sigmoid coordinates of s, q
    # was worse (MOA-2022-BLG-091, 30 cells: 13 worse by > 1 than Nelder-Mead). cfg["box_lm"] (experimental):
    # short Nelder-Mead, then LM in (y, log s, log q) with s, q clipped to the cell
    lm_best = (np.inf, None)
    for _, y in trials[:2]:
        if cfg.get("box_lm") and co.kind == "teff":
            r = nm(y, cfg.get("explore_nfev", 100), cfg.get("refine_step", 0.3))
            z0 = np.r_[r.x[:5], to_s(r.x[5]), to_q(r.x[6])]
            clip = lambda z: (min(max(z[5], s_lo), s_hi), min(max(z[6], q_lo), q_hi))
            fr = lambda z: model.chi2_residuals(be, np.asarray(full_p(co.to_x(z[:5]), *clip(z)), float), data)
            qz = least_squares_lm(fr, z0, np.r_[LM_STEP, 1e-5, 1e-4], np.r_[LM_CAP, 0.25 * (s_hi - s_lo),
                                                                                0.25 * (q_hi - q_lo)])
            sc, qc = clip(qz.x)
            c = model.chi2(be, full_p(co.to_x(qz.x[:5]), sc, qc), data)
            if c < min(r.fun, best[0], lm_best[0]):
                lm_best = (c, np.r_[co.to_x(qz.x[:5]), sc, qc])
            if r.fun < best[0]:
                best = (r.fun, r.x)
            continue
        r = nm(y, cfg.get("n_refine", 1500), cfg.get("refine_step", 0.3))
        if r.fun < best[0]:
            best = (r.fun, r.x)
    if lm_best[0] < best[0]:
        return lm_best
    y = best[1]
    return best[0], np.r_[co.to_x(y[:5]), to_s(y[5]), to_q(y[6])]


def fit_point(ls, lq, data, cfg, warm=None):
    strat = cfg["strategy"]
    if strat not in STRATEGIES:
        raise KeyError(f"strategy {strat!r} not registered")
    if strat in WARM_STRATEGIES:
        if strat == "screen_d" and not cfg.get("warm"):
            warm = None
        return STRATEGIES[strat](ls, lq, data, cfg, warm)
    return STRATEGIES[strat](ls, lq, data, cfg)


def fit_full(data, p0, backend=None, rounds=3, maxfev=6000):
    """Best fit of all seven parameters (t0, u0, tE, log s, log q, alpha, log rho) from the start p0: Nelder-Mead,
    restarted `rounds` times. backend: a backend or a settings dict (default config.REFINE_BACKEND).
    Returns (chi2, p). For a new event, find p0 with a map (maps.lattice) or with fit_point at a few (s, q)."""
    from . import config
    be = make_backend(backend or config.REFINE_BACKEND) if backend is None or isinstance(backend, dict) else backend
    p = np.asarray(p0, float)
    f = lambda x: model.chi2(be, x, data)
    r = None
    for _ in range(rounds):
        r = minimize(f, p, method="Nelder-Mead", options=dict(maxfev=maxfev, xatol=1e-8, fatol=1e-3, adaptive=True))
        p = r.x
    return float(r.fun), p


def offset_partner(p):
    """Start vector for the offset-degenerate partner of the solution p = (t0, u0, tE, log s, log q, alpha, log rho)
    (Zhang et al. 2022; the reflection used by RTModel, Bozza 2024): the separation s' at which the trajectory
    crosses the binary axis at the same place relative to the other caustic; u0 adjusted so that this crossing
    point stays fixed, other parameters kept. Its own inverse. Predicts the partner's log s to ~0.001 for
    KMT-2024-BLG-1209 (inner/outer) and MOA-2022-BLG-091 (solutions C/D); fit_full from it found the partner in 3
    of these 4 cases. Not meaningful for alpha near 0 or pi (trajectory along the binary axis)."""
    t0, u0, tE, ls, lq, al, lr = np.asarray(p, float)
    s, q = 10.0 ** ls, 10.0 ** lq
    qq = q / (1 + q)
    xc = u0 / np.sin(al) + s * qq                        # where the trajectory crosses the binary axis
    b = 1 - s * s + 2 * s * xc
    sn = (b + np.sqrt(4 * s * s + b * b)) / (2 * s)
    return np.array([t0, (xc - sn * qq) * np.sin(al), tE, np.log10(sn), lq, al, lr])


def fit_partner(data, p, backend=None, rounds=3, maxfev=6000):
    """fit_full from offset_partner(p): (chi2, p') of the partner solution, or of whatever minimum the fit reaches
    (compare its log s with the prediction)."""
    return fit_full(data, offset_partner(p), backend, rounds, maxfev)


def error_factors(data, p, backend=None):
    """Error-bar factor per data set, k = sqrt(chi2_set / N_set) at the solution p (fluxes solved per set);
    data.scale_errors(k) then gives chi2/N = 1 per set, the usual normalisation before a map or MCMC."""
    from . import config
    be = make_backend(backend or config.REFINE_BACKEND) if backend is None or isinstance(backend, dict) else backend
    r = model.residuals(be, np.asarray(p, float), data)
    return {name: float(np.sqrt(np.mean(r[data.idx == k] ** 2))) for k, name in enumerate(data.names)}
