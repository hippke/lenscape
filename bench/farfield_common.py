"""Shared set-up of the point-source speed study (RESEARCH_PLAN_point_source.md): the benchmark event
MOA-2022-BLG-091 (Han's photometry, private: never committed; read from photometry_compare/orig_map), its wings
binned as the maps bin them, and ~20 parameter vectors from the reference lattice map stage1_orig_deep.npz
(18 cells spread over the visible region, Delta chi2 < 100, plus the global minimum (21, 20) and the hard
small-source cell (33, 20)). Everything runs in seconds on one core."""
import os as _os
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
ORIG = os.path.join(ROOT, "..", "photometry_compare", "orig_map")
INPUTS = os.path.join(ORIG, "orig_inputs.npz")
REFMAP = os.path.join(ORIG, "stage1_orig_deep.npz")
OUT = os.path.join(HERE, "farfield_out")                 # figures and numbers (ignored by git)

from lenscape import data as D, fit as F, config as C, model as M, nbmag     # noqa: E402


def load_data(binned=True):
    """The benchmark Data (1728 points); binned=True: wings binned per night as maps.lattice does (K = 1 tE
    around the best fit, runs checked against the best model) -> data.Binned with t_eval."""
    z = np.load(INPUTS)
    data = D.Data(z["t"], z["f"], z["e"], z["idx"], [str(n) for n in z["names"]])
    best = z["best"]
    if not binned:
        return data, best
    be = F.make_backend(C.LATTICE["grid_backend"])
    fit = D.bin_wings(data, best[0], best[2], K=1.0, model=[(be, best)], tol=C.LATTICE.get("bin_tol", 0.05))
    return fit, best


def vectors(n_spread=18):
    """~20 parameter vectors (t0, u0, tE, log s, log q, alpha, log rho) from the reference map: n_spread cells
    spread over the visible region plus (21, 20) and (33, 20). Returns (P (n, 7), cells [(i, j)], dchi2)."""
    m = np.load(REFMAP)
    chi2, pars, logs, logq = m["chi2"], m["pars"], m["logs"], m["logq"]
    d = chi2 - np.nanmin(chi2)
    vis = np.argwhere(d < 100)
    rng = np.random.default_rng(20261026)
    # stratified: sort the visible cells by (i, j) and take evenly spaced ones with a fixed random offset
    order = vis[np.lexsort((vis[:, 1], vis[:, 0]))]
    pick = order[(np.linspace(0, len(order) - 1, n_spread + 2)[1:-1] + rng.integers(-3, 4, n_spread)).clip(0, len(order) - 1).astype(int)]
    cells = [tuple(int(v) for v in c) for c in pick] + [(21, 20), (33, 20)]
    P, dch = [], []
    for i, j in cells:
        t0, u0, tE, al, lr = pars[i, j]
        P.append([t0, u0, tE, logs[i], logq[j], al, lr])
        dch.append(d[i, j])
    return np.array(P), cells, np.array(dch)


def source_positions(p, t):
    """y1, y2 (centre-of-mass frame) at times t for p = (t0, u0, tE, log s, log q, alpha, log rho)."""
    t0, u0, tE, al = p[0], p[1], p[2], p[5]
    tau = (np.asarray(t, float) - t0) / tE
    return u0 * np.sin(al) - tau * np.cos(al), -u0 * np.cos(al) - tau * np.sin(al)


def chi2_of(be, p, data):
    return M.chi2(be, tuple(float(v) for v in p), data)


def ensure_out():
    os.makedirs(OUT, exist_ok=True)
    return OUT
