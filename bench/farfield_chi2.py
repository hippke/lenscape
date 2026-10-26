"""Part 1, step 4d: chi2 shift and refit stability. For the ~20 benchmark vectors: chi2 (binned data, grid backend,
as the maps compute it) and a Levenberg-Marquardt refit at fixed (log s, log q) from the vector (fit.local_fit with
config.LATTICE: short Nelder-Mead exploration then LM to lm_tol = 1e-4), with the far-field tier as compiled in this
process. Run twice (LENSCAPE_FARFIELD=0 / 1 with separate NUMBA_CACHE_DIR), results to farfield_out/chi2_<tag>.npz,
then `python farfield_chi2.py compare` prints the shift distribution.
Usage: python farfield_chi2.py TAG | compare"""
import os, sys, time
import numpy as np
import farfield_common as FC
from lenscape import fit as F, config as C, model as M, nbmag

out = FC.ensure_out()
if sys.argv[1] == "compare":
    a, b = (np.load(os.path.join(out, f"chi2_{t}.npz")) for t in ("off", "on"))
    d = b["chi2"] - a["chi2"]
    print(f"chi2 shift (far-field on - off) at the 20 vectors: max |shift| {np.abs(d).max():.2e}, median {np.median(np.abs(d)):.1e}")
    dr = b["chi2_fit"] - a["chi2_fit"]
    print(f"refit chi2 difference: max |diff| {np.abs(dr).max():.3f}; refit gains (off) median {np.median(a['chi2'] - a['chi2_fit']):.3f}")
    co = F.Coords("teff")
    worst = np.zeros(5)
    for k in range(len(d)):
        ya, yb = co.from_x(a["x_fit"][k]), co.from_x(b["x_fit"][k])
        worst = np.maximum(worst, np.abs(ya - yb) / np.array([1e-4 * 35, 1e-5 * 35, 1e-4, 1e-4, 1e-3]))
    print("refit parameter differences in units of the LM forward-difference steps (t0, t_eff, ln tE, alpha, log t_*):", np.round(worst, 2))
    print("same basin for all vectors (fit.same_start):", all(F.same_start(co.from_x(a["x_fit"][k]), co.from_x(b["x_fit"][k])) for k in range(len(d))))
    print("light curves per refit (off / on):", a["nfev"].sum(), b["nfev"].sum())
    sys.exit()

tag = sys.argv[1]
data, best = FC.load_data()
P, cells, _ = FC.vectors()
be = F.make_backend(C.LATTICE["grid_backend"])
cfg = dict(C.LATTICE)
co = F.Coords(cfg["coords"])
chi2 = np.empty(len(P)); chi2_fit = np.empty(len(P)); x_fit = np.empty((len(P), 5)); nfev = np.zeros(len(P), int)
t_all = time.time()
for k, p in enumerate(P):
    t0, u0, tE, ls, lq, al, lr = p
    chi2[k] = M.chi2(be, tuple(p), data)
    y0 = co.from_x(np.array([t0, u0, tE, al, lr]))
    calls = [0]
    orig = be.lc
    def counted(pp, t, _o=orig):
        calls[0] += 1
        return _o(pp, t)
    be.lc = counted
    c, y = F.local_fit(be, data, ls, lq, y0, cfg, 0.3, 2000)
    be.lc = orig
    chi2_fit[k], x_fit[k], nfev[k] = c, y, calls[0]
    print(f"{k:2d} {cells[k]}: chi2 {chi2[k]:.4f} -> refit {c:.4f} ({calls[0]} light curves)", flush=True)
print(f"total {time.time() - t_all:.0f} s")
np.savez(os.path.join(out, f"chi2_{tag}.npz"), chi2=chi2, chi2_fit=chi2_fit, x_fit=x_fit, nfev=nfev, P=P)
