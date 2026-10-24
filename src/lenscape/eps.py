"""Read plotted data from the vector commands of an EPS figure (e.g. the Supermongo figures of many microlensing
papers): every moveto starts a subpath; filled closed subpaths of marker size are data points, short vertical
coloured segments are error bars, black lines are frames and ticks. Coordinates are device units; mapping them to
data units needs two reference values per axis (frame or tick positions), see calibrate().
"""
import numpy as np


def subpaths(path):
    """All subpaths of an EPS file: list of (points (n, 2) in device units, RGB colour tuple of 0-255 integers)."""
    toks = open(path, errors="ignore").read().split()
    toks = toks[toks.index("gsave"):] if "gsave" in toks else toks
    col = (0, 0, 0)
    out, cur = [], []
    x = y = 0.0
    for i, t in enumerate(toks):
        if t == "C" and i >= 3:
            try:
                col = tuple(int(v) for v in toks[i - 3:i])
            except ValueError:
                pass
            continue
        if t not in ("M", "L", "m", "l") or i < 2:
            continue
        try:
            a, b = float(toks[i - 2]), float(toks[i - 1])
        except ValueError:
            continue
        if t in ("M", "m"):
            if len(cur) >= 2:
                out.append((np.array(cur), col))
            x, y = (a, b) if t == "M" else (x + a, y + b)
            cur = [(x, y)]
        else:
            x, y = (a, b) if t == "L" else (x + a, y + b)
            cur.append((x, y))
    if len(cur) >= 2:
        out.append((np.array(cur), col))
    return out


def markers(subs, size=(5, 30), min_points=8, close=3.0):
    """Closed coloured subpaths of marker size: list of (x centre, y centre, width, height, colour)."""
    out = []
    for p, col in subs:
        if col == (0, 0, 0) or len(p) < min_points or np.hypot(*(p[0] - p[-1])) > close:
            continue
        w, h = np.ptp(p[:, 0]), np.ptp(p[:, 1])
        if size[0] <= w <= size[1] and size[0] <= h <= size[1]:
            out.append((0.5 * (p[:, 0].min() + p[:, 0].max()), 0.5 * (p[:, 1].min() + p[:, 1].max()), w, h, col))
    return out


def error_bars(subs):
    """Coloured vertical two-point segments: list of (x, y_low, y_high, colour)."""
    return [(p[0, 0], p[:, 1].min(), p[:, 1].max(), col) for p, col in subs
            if col != (0, 0, 0) and len(p) == 2 and abs(p[0, 0] - p[1, 0]) < 0.5 and np.ptp(p[:, 1]) > 0]


def calibrate(d1, v1, d2, v2):
    """Linear map device -> data units from two reference points (device d1, d2 with values v1, v2)."""
    scale = (v2 - v1) / (d2 - d1)
    return lambda d: v1 + (np.asarray(d, float) - d1) * scale


def points_with_errors(subs, x_map, y_map, region, colours, size=(5, 30)):
    """Markers inside region = (x0, x1, y0, y1) (device units) whose colour is in `colours` ({colour: name}), with
    the half-length of the matching error bar: list of (name, x, y, error) in data units."""
    bars = error_bars(subs)
    out = []
    for cx, cy, w, h, col in markers(subs, size):
        if col not in colours or not (region[0] < cx < region[1] and region[2] < cy < region[3]):
            continue
        e = [max(cy - lo, hi - cy) for x, lo, hi, c2 in bars if c2 == col and abs(x - cx) < 1 and lo <= cy <= hi]
        if e:
            y = float(y_map(cy))
            out.append((colours[col], float(x_map(cx)), y, abs(float(y_map(cy + max(e))) - y)))
    return out
