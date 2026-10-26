"""Cold-start cost: magnifications of independent points (fresh root guesses, nbmag.mags_point / mags_hybrid as used
by tests, caustics.magnification_map, model._repair) with the far-field tier as compiled (LENSCAPE_FARFIELD).
T1 reference set (400 points, various s, q) and a 300x300 point-source map of the benchmark lens."""
import os, time
import numpy as np
import farfield_common as FC
from lenscape import nbmag, caustics as CA
T1 = np.load(os.path.join(FC.ROOT, "tests", "data", "t1_mag.npz"))
s, q, y1, y2, rho = (T1[k] for k in ("s", "q", "y1", "y2", "rho"))
def bt(f, reps=7):
    b = 1e9
    for _ in range(reps):
        t = time.perf_counter(); f(); b = min(b, time.perf_counter() - t)
    return b
nbmag.mags_point(s, q, y1, y2); nbmag.mags_hybrid(s, q, y1, y2, rho, 1e-2, 1e-3)
print(f"FARFIELD={getattr(nbmag, 'FARFIELD', None)}: T1 mags_point {bt(lambda: nbmag.mags_point(s, q, y1, y2)) / len(s) * 1e9:.0f} ns/pt, "
      f"mags_hybrid {bt(lambda: nbmag.mags_hybrid(s, q, y1, y2, rho, 1e-2, 1e-3)) / len(s) * 1e9:.0f} ns/pt", end="; ")
x = np.linspace(-1, 1, 300)
CA.magnification_map(1.005, 4.2e-3, x, x)
print(f"magnification_map 300x300 {bt(lambda: CA.magnification_map(1.005, 4.2e-3, x, x), 3) * 1e3:.0f} ms")
