# Validation

## Magnifications against VBBinaryLensing

The tests (`tests/`) compare lenscape with stored VBBinaryLensing 3.7 results (tolerance 10⁻⁵) and with
independent calculations. VBBinaryLensing is needed only to regenerate the stored references.

| quantity | reference | agreement |
|---|---|---|
| point-source magnification, 400 points (close, resonant, wide; caustic crossings, cusps, far field) | VBBinaryLensing | < 10⁻⁹ |
| finite source (uniform), same points, map settings | VBBinaryLensing | < 2 × 10⁻³ |
| finite source (uniform), same points, tight settings | VBBinaryLensing | < 5 × 10⁻⁴ |
| finite source, 4000 near-caustic stress points, contour tolerance 10⁻³ | VBBinaryLensing | median 10⁻⁷, max 4 × 10⁻⁶ |
| linear limb darkening (u = 0.3, 0.6), 1400 points | VBBinaryLensing | < 6 × 10⁻⁴ (tight settings) |
| one central-caustic point where the two codes differ by 0.5 % | brute-force ray shooting | lenscape within 2 × 10⁻⁵ |
| caustics | VBBinaryLensing `Caustics` | 10⁻¹⁴ |
| trajectory of the figures vs the light-curve model | lenscape | 10⁻⁴ |
| annual parallax, linear orbital motion | VBBinaryLensing | < 3 × 10⁻⁴ |
| single-lens finite source with limb darkening | 2-D quadrature (10⁻¹¹) | < 10⁻⁵ |

The stress set and the ray-shooting check are part of the research history of the code and are not in the test
suite; the other rows are.

## A published map: MOA-2022-BLG-091

Han et al. (2025, A&A, arXiv:2505.22951) report four degenerate planetary solutions of MOA-2022-BLG-091 and a Δχ²
map on a 50 × 49 lattice. Their photometry is not public, so the comparison used the data points shown in their
light-curve figure (716 points) plus the public KMTNet baseline (2531 points in total).

**Solutions.** Started from the published parameters, lenscape finds all four solutions with the same ranking:

| | Δχ² (lenscape) | Δχ² (paper) | s | q (10⁻³) | α |
|---|---|---|---|---|---|
| Sol B | 0 | 0 | 1.0047 | 4.00 | 4.413 |
| Sol A | +1.1 | +1.1 | 1.0023 | 4.27 | 4.621 |
| Sol D | +5.4 | +5.0 | 1.0187 | 3.90 | 4.749 |
| Sol C | +7.4 | +6.7 | 1.0130 | 3.36 | 4.818 |

For each solution, the source trajectory enters the caustic at HJD′ 9655.27–9655.30 and leaves it at
9659.92–9659.94, the times of the two caustic-crossing features in the light curve.

**Map.** On the paper's lattice, the shape of the allowed region, the best-fit region and the four solutions agree;
66 % of the paper's coloured lattice points lie within one Δχ² band of lenscape's (89 % if the band edges are read
as (kn)² instead of kn²). The outer bands of lenscape's map are steeper, as expected from the longer baseline.
The full map took 98 minutes on a six-core desktop CPU (AMD Ryzen 5 3600, 12 threads).
