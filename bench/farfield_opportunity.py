"""Part 1, step 1: how much of a light curve is 'far field'? For the ~20 benchmark vectors: exact point-source
magnification vs (a) the single-lens magnification of the primary, (b) first-order perturbation theory (pert0),
(c) pert0 + 2 Newton steps, per point against the distance to the nearest caustic; fractions of points below 1e-4 /
1e-3 relative deviation; per-point chi2 contributions; chi2 shift of the light curve when the approximation replaces
the exact value at the qualifying points (oracle guard). Writes farfield_out/opportunity.png and prints a table."""
import os, sys
import numpy as np
import farfield_common as FC
import farfield_perturb as FP
from lenscape import caustics as CA, model as M, fit as F, config as C

data, best = FC.load_data()
raw, _ = FC.load_data(binned=False)
P, cells, dch = FC.vectors()
be = F.make_backend(C.LATTICE["grid_backend"])
out = FC.ensure_out()


def caustic_distance(s, q, y1, y2):
    pts = np.concatenate(CA.caustics(s, q, n=1500))
    y = y1 + 1j * y2
    d = np.empty(len(y))
    for k in range(0, len(y), 256):
        d[k:k + 256] = np.min(np.abs(y[k:k + 256, None] - pts[None, :]), axis=1)
    return d


rows = []
allD, allS, allP, allF, allE = [], [], [], [], []
for k, p in enumerate(P):
    t0, u0, tE, ls, lq, al, lr = p
    s, q = 10 ** ls, 10 ** lq
    # unbinned points: per-point chi2 contributions
    y1, y2 = FC.source_positions(p, raw.t)
    Ae, As, Ap, Af, ok, nit, gg = FP.eval_grid(s, q, y1, y2, 2, 8, 1e-24, 1e-6)
    d = caustic_distance(s, q, y1, y2)
    eS, eP, eF = np.abs(As / Ae - 1), np.abs(Ap / Ae - 1), np.abs(Af / Ae - 1)
    fs, fb = M.fluxes(Ae, raw)
    r = (raw.f - fs[raw.idx] * Ae - fb[raw.idx])
    dchi_S = np.abs(raw.w * ((r - fs[raw.idx] * (As - Ae)) ** 2 - r ** 2))
    dchi_P = np.abs(raw.w * ((r - fs[raw.idx] * (Ap - Ae)) ** 2 - r ** 2))
    # binned evaluation points: fractions (what the kernel actually evaluates in a map)
    b1, b2 = FC.source_positions(p, data.t_eval)
    Aeb, Asb, Apb, Afb, okb, nitb, ggb = FP.eval_grid(s, q, b1, b2, 2, 8, 1e-24, 1e-6)
    eSb, ePb, eFb = np.abs(Asb / Aeb - 1), np.abs(Apb / Aeb - 1), np.abs(Afb / Aeb - 1)
    # chi2 shift of the (binned) light curve under an oracle guard at 1e-4 relative error
    Ah, _ = FC.nbmag.lc_hybrid(ls, lq, u0, al, lr, tE, t0, data.t_eval, be.tol, be.hexeps)
    Afull = be.lc(tuple(p), data.t_eval)
    c0 = M.chi2(be, tuple(p), data)

    def shift(Aapp, err):
        m = err < 1e-4
        A2 = Afull.copy(); A2[m] = Afull[m] * (Aapp[m] / Aeb[m])      # same relative perturbation of the hybrid value
        return M.chi2_profiled_binned(A2, data.k_pos, data.k_f, data.k_w, data.k_idx, data.b_m, data.b_p, data.b_h,
                                      data.b_set, data.b_W, data.b_Sf, data.b_Sff, data.b_Stf, data.b_Stt, data.nsets) - c0
    rows.append(dict(cell=cells[k], dchi2=dch[k], ls=ls, lq=lq, lr=lr,
                     fS4=np.mean(eSb < 1e-4), fS3=np.mean(eSb < 1e-3), fP4=np.mean(ePb < 1e-4), fP3=np.mean(ePb < 1e-3),
                     fF4=np.mean(eFb < 1e-4), fF3=np.mean(eFb < 1e-3), fF_ok=np.mean(okb),
                     fF_ok_exact=np.mean(okb & (eFb < 1e-10)), maxF_ok=np.nanmax(np.where(okb, eFb, 0)),
                     dchiS3=np.mean(dchi_S < 1e-3), dchiP3=np.mean(dchi_P < 1e-3),
                     shS=shift(Asb, eSb), shP=shift(Apb, ePb), nit=nitb[okb].mean() if okb.any() else np.nan,
                     ps_frac=np.mean(Afull == Ah)))
    allD.append(d); allS.append(eS); allP.append(eP); allF.append(np.where(ok, eF, np.nan)); allE.append(Ae)
    r_ = rows[-1]
    print(f"{k:2d} {str(cells[k]):9s} dchi2 {dch[k]:6.1f} logrho {lr:5.2f} | single <1e-4 {r_['fS4']:.2f} <1e-3 {r_['fS3']:.2f} | "
          f"pert0 <1e-4 {r_['fP4']:.2f} <1e-3 {r_['fP3']:.2f} | pert2 <1e-4 {r_['fF4']:.2f} | far_roots ok {r_['fF_ok']:.2f} "
          f"exact(1e-10) {r_['fF_ok_exact']:.2f} max err ok {r_['maxF_ok']:.1e} its/pt {r_['nit']:.1f} | "
          f"chi2 shift(oracle 1e-4) single {r_['shS']:+.3f} pert0 {r_['shP']:+.3f} | ps frac {r_['ps_frac']:.2f}", flush=True)

print("\nmeans over vectors: single <1e-4 %.2f <1e-3 %.2f | pert0 <1e-4 %.2f <1e-3 %.2f | pert2 <1e-4 %.2f | far_roots ok %.2f exact %.2f"
      % tuple(np.mean([r[k] for r in rows]) for k in ("fS4", "fS3", "fP4", "fP3", "fF4", "fF_ok", "fF_ok_exact")))
print("per-point |dchi2 contribution| < 1e-3 (unbinned): single %.2f  pert0 %.2f" % (np.mean([r["dchiS3"] for r in rows]), np.mean([r["dchiP3"] for r in rows])))
print("max |chi2 shift| (oracle guard 1e-4): single %.3f pert0 %.3f" % (max(abs(r["shS"]) for r in rows), max(abs(r["shP"]) for r in rows)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D, S, Pp, Ff, E = (np.concatenate(v) for v in (allD, allS, allP, allF, allE))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a_, e, name in ((ax[0], S, "single lens (primary)"), (ax[1], Pp, "first-order perturbation"), (ax[2], Ff, "perturbation + 2 Newton steps")):
    m = np.isfinite(e) & (e > 0)
    a_.scatter(D[m], e[m], s=2, alpha=0.3)
    a_.set_xscale("log"); a_.set_yscale("log"); a_.set_xlabel("distance to nearest caustic [theta_E]"); a_.set_title(name)
    a_.axhline(1e-4, color="k", lw=0.5); a_.axhline(1e-3, color="k", lw=0.5, ls="--")
ax[0].set_ylabel("|A_approx / A_exact - 1|")
plt.tight_layout(); plt.savefig(os.path.join(out, "opportunity.png"), dpi=120)
np.savez(os.path.join(out, "opportunity.npz"), D=D, S=S, P=Pp, F=Ff, E=E)
print("wrote", os.path.join(out, "opportunity.png"))
