"""Light-curve speed and accuracy of lenscape against VBBinaryLensing (pip install VBBinaryLensing).

Five binary-lens light curves (1000 points each, +-2 tE around t0): caustic crossings of a resonant planetary
caustic, a central caustic at high magnification, a wide planet, a close binary and a far-field case. Each code runs
single-threaded at two accuracy levels; errors are measured against VBBinaryLensing at Tol = 1e-5.

Usage: python benchmarks/benchmark.py [repeats]
"""
import sys, time
import numpy as np
import lenscape
from lenscape import config

LN10 = np.log(10.0)
CASES = {                                   # (t0, u0, tE, log s, log q, alpha, log rho)
    "resonant caustic crossings": (0.0, -0.0254, 35.5, 0.0020, -2.398, 4.413, -3.28),
    "central caustic, high mag.": (0.0, 0.003, 20.0, 0.05, -3.0, 1.0, -2.7),
    "wide planet": (0.0, 0.1, 25.0, 0.3, -2.5, 0.35, -3.0),
    "close binary": (0.0, 0.05, 30.0, -0.3, -0.5, 2.2, -2.5),
    "far field": (0.0, 0.8, 30.0, 0.1, -3.0, 1.0, -3.0),
}


def vbbl(tol, reltol):
    import VBBinaryLensing
    v = VBBinaryLensing.VBBinaryLensing()
    v.Tol, v.RelTol = tol, reltol

    def lc(p, t):
        t0, u0, tE, ls, lq, al, lr = p
        return np.asarray(v.BinaryLightCurve([ls * LN10, lq * LN10, u0, al, lr * LN10, np.log(tE), t0], list(t))[0])
    return lc


def timed(f, p, t, repeats):
    f(p, t)                                                    # compile / warm up
    best = np.inf
    for _ in range(repeats):
        t1 = time.perf_counter()
        A = f(p, t)
        best = min(best, time.perf_counter() - t1)
    return best, np.asarray(A)


def main(repeats=3):
    codes = {
        "VBBinaryLensing Tol 1e-2, RelTol 3e-3": vbbl(1e-2, 3e-3),
        "lenscape map settings": lenscape.make_backend(config.GRID_BACKEND).lc,
        "VBBinaryLensing Tol 1e-3, RelTol 1e-3": vbbl(1e-3, 1e-3),
        "lenscape tight settings": lenscape.make_backend(config.REFINE_BACKEND).lc,
    }
    ref = vbbl(1e-5, 0.0)
    total = {k: 0.0 for k in codes}
    print(f"{'case':30s}" + "".join(f"{k[:24]:>26s}" for k in codes))
    for name, p in CASES.items():
        p = np.array(p)
        t = np.linspace(p[0] - 2 * p[2], p[0] + 2 * p[2], 1000)
        A0 = ref(p, t)
        row = f"{name:30s}"
        for k, f in codes.items():
            dt, A = timed(f, p, t, repeats)
            total[k] += dt
            row += f"{dt * 1e3:9.2f} ms  err {np.max(np.abs(A / A0 - 1)):7.1e}"
        print(row)
    print(f"{'total':30s}" + "".join(f"{v * 1e3:9.2f} ms{'':15s}" for v in total.values()))
    k = list(total)
    print(f"\nspeed-up, map settings:   {total[k[0]] / total[k[1]]:.1f}x")
    print(f"speed-up, tight settings: {total[k[2]] / total[k[3]]:.1f}x")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
