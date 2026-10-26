"""Kernel micro-benchmark (port of fastgrid/bench/kernel_bench.py): numba hybrid light curves (nbmag.lc_hybrid and the
full backend light curve) at the ~20 benchmark vectors of farfield_common, best-of-7 timings, branch counts, and the
maximum relative deviation from a reference run stored on the first call (bench/farfield_out/kernel_ref.npz).
Usage: python bench/kernel_bench.py LABEL [--ref]   (--ref: (re)write the reference magnifications)"""
import json, os, sys, time
import numpy as np
import farfield_common as FC
from lenscape import nbmag, fit as F, config as C

label = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "kernel"
data, best = FC.load_data()
P, cells, dch = FC.vectors()
t = data.t_eval
be = F.make_backend(C.LATTICE["grid_backend"])
out = FC.ensure_out()
refpath = os.path.join(out, "kernel_ref.npz")


def best_time(f, reps=7):
    b = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); f(); b = min(b, time.perf_counter() - t0)
    return b


rows, tot_k, tot_full, As = [], 0.0, 0.0, []
for k, p in enumerate(P):
    t0, u0, tE, ls, lq, al, lr = p
    A, flag = nbmag.lc_hybrid(ls, lq, u0, al, lr, tE, t0, t, be.tol, be.hexeps)
    Af = be.lc(tuple(p), t)
    tk = best_time(lambda: nbmag.lc_hybrid(ls, lq, u0, al, lr, tE, t0, t, be.tol, be.hexeps))
    tf = best_time(lambda: be.lc(tuple(p), t))
    bc = np.bincount(flag, minlength=4)
    rows.append((tk * 1e6, tf * 1e6, bc.tolist()))
    tot_k += tk; tot_full += tf; As.append(Af)
As = np.array(As)
maxrel = float("nan")
if "--ref" in sys.argv or not os.path.exists(refpath):
    np.savez(refpath, A=As)
    print("reference magnifications written")
else:
    ref = np.load(refpath)["A"]
    maxrel = float(np.max(np.abs(As / ref - 1)))
for k, (tk, tf, bc) in enumerate(rows):
    print(f"  vec {k:2d} cell {cells[k]}: kernel {tk:7.0f} us  full {tf:7.0f} us  branches ps/quadhex/-/far {bc}")
print(f"{label}: mean kernel {tot_k / len(P) * 1e6:.0f} us, mean full LC {tot_full / len(P) * 1e6:.0f} us, "
      f"max rel dev vs reference {maxrel:.1e}, {len(t)} points")
with open(os.path.join(out, "results.jsonl"), "a") as fh:
    fh.write(json.dumps(dict(label=label, kernel_us=tot_k / len(P) * 1e6, full_us=tot_full / len(P) * 1e6,
                             maxrel=maxrel, rows=rows)) + "\n")
