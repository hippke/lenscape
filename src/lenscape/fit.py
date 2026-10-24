"""Grid-point optimisation strategies: s, q fixed; (t0, u0, tE, alpha, log rho) optimised."""
import numpy as np
from scipy.optimize import minimize
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


class Coords:
    """Optimiser coordinates.
    'native': (t0, u0, tE, alpha, log rho)
    'teff'  : (t0, t_eff = u0*tE, ln tE, alpha, log10 t_*), t_* = rho*tE  (what a blended high-mag event measures)
    Steps define a scaled initial simplex (scipy's default is +-5 % of each value, i.e. +-38 d in t0)."""

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
    if step_scale is None:                       # scipy default simplex
        return minimize(f, y0, method="Nelder-Mead", options=dict(maxfev=maxfev, adaptive=True,
                                                                   **({"fatol": fatol} if fatol else {})))
    sim = np.vstack([y0] + [y0 + np.eye(5)[k] * st for k, st in enumerate(f.coords.steps(y0, step_scale))])
    return minimize(f, y0, method="Nelder-Mead", options=dict(maxfev=maxfev, adaptive=True, initial_simplex=sim,
                                                               **({"fatol": fatol} if fatol else {})))


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
    starts = screen_alphas(fx, cfg.get("n_screen", 4), cfg.get("seed"))
    for w in (warm or []):
        if w is not None and np.all(np.isfinite(w)):
            starts.append(np.asarray(w))
    for x0 in starts:
        r = _nm(f, co.from_x(x0), cfg.get("n_explore", 200), cfg.get("explore_scale", sc))
        trials.append((r.fun, r.x))
    trials.sort(key=lambda z: z[0])
    best = trials[0]
    for _, y in trials[:2]:
        r = _nm(f, y, cfg.get("n_refine", 1500), cfg.get("refine_step", None if sc is None else 0.3 * sc), 1e-2)
        if r.fun < best[0]:
            best = (r.fun, r.x)
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
    starts = [np.r_[co.from_x(x0), 0.0, 0.0] for x0 in screen_alphas(fx, cfg.get("n_screen", 4), cfg.get("seed"))]
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
    for _, y in trials[:2]:
        r = nm(y, cfg.get("n_refine", 1500), cfg.get("refine_step", 0.3))
        if r.fun < best[0]:
            best = (r.fun, r.x)
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
