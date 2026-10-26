# Method

## Model

A binary lens with separation *s* (Einstein radii) and mass ratio *q* = m₂/m₁ magnifies a source of radius *ρ*
moving on a straight line (impact parameter *u*₀, angle *α*, time of closest approach *t*₀, timescale
*t*<sub>E</sub>). Each data set *k* has its own source and blend flux, *F* = *f*<sub>s,k</sub> *A*(*t*) +
*f*<sub>b,k</sub>; these enter linearly and are solved by weighted least squares in every χ² evaluation.

## Magnifications

All magnification code is lenscape's own, compiled with numba; VBBinaryLensing is used only in the tests, as a
reference.

1. **Point source.** The lens equation is solved as a fifth-order polynomial, as in VBBinaryLensing
   (Bozza 2010; Bozza et al. 2018), with the roots of the previous point as starting values: Newton on each root,
   the five roots stepped together so that the processor overlaps their independent dependency chains (the solve is
   latency-bound: 222 → 151 ns per point on a Ryzen 7950X, 1.23× on a light curve, identical results), and
   Ehrlich–Aberth where a root does not converge. For independent points without a warm start
   (`nbmag.mags_point`, `caustics.magnification_map`) a far-field solve replaces the polynomial iteration: the
   three images of a source outside the caustics start from the two single-lens images of the primary and the
   planet's Chang–Refsdal image and converge by Newton on the lens equation, the two remaining roots follow from
   deflating the quintic, and the set is accepted only when it is complete (three distinct converged images with
   parities +, −, −, two deflated roots that are not images), so it is exact for any (s, q) and falls back
   otherwise (2.5–6× faster than Aberth from fresh starts; `nbmag.FARFIELD` enables it for light curves, where it
   is speed-neutral). Details and measurements: `RESEARCH_point_source_results.md`, scripts in `bench/`.
2. **Near caustics.** The quadrupole test of VBBinaryLensing decides whether the point source is good enough. If
   not, lenscape tries the quadrupole (analytic, from derivatives already computed for the test) and the
   hexadecapole approximation (Gould 2008; Pejcha & Heyrovský 2009; Cassan 2017), evaluated on images tracked from
   the point-source images instead of new polynomial solutions.
3. **On and inside caustics.** Contour integration of the image boundaries (Bozza 2010) with adaptive sampling,
   image tracking between samples and a parity check of every image set. Linear limb darkening is integrated over
   annuli.

Single-lens finite-source magnifications use the one-dimensional integral of Lee et al. (2009) with the
integration split at the kinks of the integrand.

## Fits at fixed (s, q)

The fit parameters are *t*₀, *t*<sub>eff</sub> = *u*₀*t*<sub>E</sub>, ln *t*<sub>E</sub>, *α* and
log *t*<sub>⋆</sub> = log *ρt*<sub>E</sub>: the timescales a light curve measures directly. In these coordinates
the narrow curved valleys of the likelihood become straighter, and Nelder–Mead converges faster and more often.
Each fit evaluates χ² at 20 trajectory angles and refines the best four.

## Maps

`maps.lattice` computes the profile χ² on a (log *s*, log *q*) lattice in stages:

1. **Coarse seed grid.** Quick fits on 2 × 2 merged cells give starting points.
2. **Cell fits.** For every lattice cell, the best fit anywhere inside the cell: *s* and *q* are free within the
   cell (through a sigmoid). A narrow valley that passes between two lattice points still shows up. After the
   fits, every cell is offered its neighbours' solutions: one χ² evaluation decides whether a refit is worthwhile.
3. **Lattice-point fits** at fixed (*s*, *q*), started from the best cell solutions within one cell.
4. **Neighbour refinement.** Every point is refitted from its neighbours' solutions, in sweeps, until no point
   improves by more than 1. After the first sweep only points next to a point that changed are checked: the others
   would repeat a refit with unchanged inputs (identical result, fewer refits).
5. **Outlier recalculation.** A point whose χ² exceeds both neighbours along an axis by more than 8 is a stuck fit
   in practice; it gets a deep search from about 1000 starts plus its neighbours' solutions.

`maps.refine` continues from the cells of stage 2: where a Δχ² band edge runs between neighbouring cells, the cells
are split 2 × 2 and fitted again, `levels` times.

**Why the stages matter.** Fits started cold at every lattice point often end in local minima. On a test map
(KMT-2025-BLG-0121), neighbour starts lowered χ² by more than 1 at 64 % of the points and doubled the area inside
the displayed Δχ² bands. The seed grid together with the one-evaluation refit test is 6 times cheaper than
refitting every cell from all neighbours.

## Parallelism and robustness

Every fit runs in its own forked process, up to `workers` at a time. A fit that crashes or hangs costs one point,
not the map. The compiled kernels are loaded once in the parent before forking.

## Accuracy settings

| setting | maps (`GRID_BACKEND`) | best fits, figures (`REFINE_BACKEND`) |
|---|---|---|
| point source accepted if the quadrupole term < `tol` | 10⁻² | 10⁻³ |
| hexadecapole accepted if its error estimate < `hexeps` | 10⁻³ | 10⁻⁴ |
| contour integration relative tolerance `ctol` | 0.1 | 0.01 |

Magnifications agree with VBBinaryLensing (tolerance 10⁻⁵) to 2 × 10⁻³ with the map settings and 5 × 10⁻⁴ with
the tight settings on 400 test points along caustic-crossing trajectories, and to better than 10⁻² on 4000
near-caustic stress points. See [validation.md](validation.md).

## References

- Bozza, V. 2010, MNRAS, 408, 2188
- Bozza, V., Bachelet, E., Bartolić, F., et al. 2018, MNRAS, 479, 5157
- Cassan, A. 2017, MNRAS, 468, 3993
- Gould, A. 2008, ApJ, 681, 1593
- Goodman, J., & Weare, J. 2010, Comm. App. Math. Comp. Sci., 5, 65
- Lee, C.-H., Riffeser, A., Seitz, S., & Bender, R. 2009, ApJ, 695, 200
- Pejcha, O., & Heyrovský, D. 2009, ApJ, 690, 1772
- Yee, J. C., Shvartzvald, Y., Gal-Yam, A., et al. 2012, ApJ, 755, 102
