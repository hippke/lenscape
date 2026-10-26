"""Part 1, step 2: error maps of the far-field approximations on a source-plane grid at fixed (s, q): pert0 (first
order), pert2, and the acceptance region of far_roots (where it is exact), with the caustics overlaid. Several lens
regimes: the benchmark (resonant, q = 4e-3), wide, close, a small planet, an equal-mass binary. Also the dependence
of far_roots' acceptance and iteration count on the ghost threshold gmin and the Newton tolerance.
Writes farfield_out/errormap_*.png and prints acceptance statistics."""
import os, sys, time
import numpy as np
import farfield_common as FC
import farfield_perturb as FP
from lenscape import caustics as CA, nbmag

out = FC.ensure_out()
CASES = [("benchmark s=1.005 q=4.2e-3", 10 ** 0.00213, 10 ** -2.379), ("wide s=1.8 q=4e-3", 1.8, 4e-3),
         ("close s=0.7 q=4e-3", 0.7, 4e-3), ("small planet s=1.1 q=1e-4", 1.1, 1e-4),
         ("resonant s=1 q=1e-2", 1.0, 1e-2), ("binary s=1 q=0.5", 1.0, 0.5), ("binary s=0.8 q=0.1", 0.8, 0.1)]
L, N = 1.5, 601
x = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, x)
y1, y2 = X.ravel(), Y.ravel()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

summary = []
for name, s, q in CASES:
    Ae, As, Ap, Af, ok, nit, gg = FP.eval_grid(s, q, y1, y2, 2, 8, 1e-24, 1e-6)
    eP = np.abs(Ap / Ae - 1).reshape(N, N)
    eF = np.abs(Af / Ae - 1).reshape(N, N)
    okm = ok.reshape(N, N)
    cau = CA.caustics(s, q, n=1000)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    im = ax[0].imshow(np.log10(np.maximum(eP, 1e-16)), extent=(-L, L, -L, L), origin="lower", vmin=-10, vmax=0, cmap="viridis")
    ax[0].set_title(f"{name}\nlog10 |A_pert0/A - 1|")
    plt.colorbar(im, ax=ax[0])
    ax[0].contour(x, x, eP, levels=[1e-4], colors="w", linewidths=0.8)
    im = ax[1].imshow(np.log10(np.maximum(np.where(okm, eF, np.nan), 1e-16)), extent=(-L, L, -L, L), origin="lower", vmin=-16, vmax=-8, cmap="viridis")
    ax[1].set_title("log10 |A_far_roots/A - 1| where accepted")
    plt.colorbar(im, ax=ax[1])
    ax[2].imshow(okm, extent=(-L, L, -L, L), origin="lower", cmap="Greys_r")
    ax[2].set_title(f"far_roots accepted ({okm.mean() * 100:.1f} % of the grid)")
    for a_ in ax:
        for c in cau:
            a_.plot(c.real, c.imag, "r-", lw=0.6)
        a_.set_xlim(-L, L); a_.set_ylim(-L, L)
    fn = os.path.join(out, "errormap_" + name.split()[0] + "_" + name.split()[1].replace("=", "") + ".png")
    plt.tight_layout(); plt.savefig(fn, dpi=100); plt.close()
    # distances: where accepted, how close to the caustic? where is pert0 < 1e-4?
    pts = np.concatenate(cau)
    yy = (y1 + 1j * y2)
    d = np.empty(len(yy))
    for k in range(0, len(yy), 512):
        d[k:k + 512] = np.min(np.abs(yy[k:k + 512, None] - pts[None, :]), axis=1)
    inside_ok = ok & (Ae > 0)
    worst = float(np.nanmax(np.where(ok, np.abs(Af / Ae - 1), 0)))
    print(f"{name:28s}: far_roots accepts {ok.mean() * 100:5.1f} %, max err {worst:.1e}, min caustic distance of an accepted point "
          f"{d[ok].min():.4f}, max distance of a rejected point {d[~ok].max():.4f}, Newton its/pt {nit[ok].mean():.1f}; "
          f"pert0 < 1e-4 on {np.mean(np.abs(Ap / Ae - 1) < 1e-4) * 100:5.1f} % (min distance {d[np.abs(Ap / Ae - 1) < 1e-4].min():.3f})", flush=True)
    summary.append((name, s, q, ok.mean(), worst, nit[ok].mean()))
    if name.startswith("benchmark"):
        # dependence on gmin and the Newton tolerance
        # (the gmin / Newton-tolerance scan of the prototype: acceptance was independent of gmin in 1e-8 .. 1e-2 and
        # |step| < 1e-7 saved ~1 Newton step per image at the same maximum error; the package uses 1e-7 and 1e-4)
        e2 = np.abs(Ap / Ae - 1)
        print(f"   accepted points: median error {np.median(np.abs(Af / Ae - 1)[ok]):.1e}, 99 % {np.percentile(np.abs(Af / Ae - 1)[ok], 99):.1e}")
        # ghost residual vs caustic distance (for the guard discussion)
        m = ok & np.isfinite(gg)
        b = np.logspace(-3, 0.3, 12)
        idx = np.digitize(d[m], b)
        print("   ghost residual G vs caustic distance (median per bin):")
        for i in range(1, len(b)):
            sel = idx == i
            if sel.sum() > 10:
                print(f"     d in [{b[i-1]:.3f}, {b[i]:.3f}): G median {np.median(gg[m][sel]):.2e}  min {gg[m][sel].min():.2e}")
