<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/hippke/lenscape/main/docs/logo/lenscape-banner-dark.svg">
  <img alt="lenscape" src="https://raw.githubusercontent.com/hippke/lenscape/main/docs/logo/lenscape-banner.svg" width="600">
</picture>

# lenscape

**Fast, high-resolution Δχ² maps of binary-lens microlensing events.**

Planetary microlensing light curves often admit several degenerate binary-lens interpretations. To find them,
analyses compute Δχ² maps over the binary-lens separation *s* and mass ratio *q*: at every point of the map, the
remaining parameters (*t*<sub>0</sub>, *u*<sub>0</sub>, *t*<sub>E</sub>, *α*, *ρ*) are fitted to the light curve,
with finite-source magnifications near the caustics. These maps decide how many solutions an analysis reports and
how wide its mass-ratio constraints are, and they are often the most expensive step of the modelling. Standard
analyses therefore use coarse grids of independent point-by-point fits, which get stuck in local minima.

`lenscape` is an open-source Python package that computes these maps quickly and suppresses local-minimum
artefacts with staged neighbour propagation and adaptive refinement. For the planetary event MOA-2022-BLG-091,
it reproduces the published 50 × 50 map in **14 minutes** and computes a **400 × 400** map in **under two hours**
on a 16-core desktop computer (AMD Ryzen 9 7950X): 14 minutes for the 50 × 50 lattice, then 8, 23 and 74 minutes
for the 100 × 100, 200 × 200 and 400 × 400 refinements. It recovers the event's four published solutions and
their χ² to within 1.

<img alt="Published and lenscape maps of MOA-2022-BLG-091" src="https://raw.githubusercontent.com/hippke/lenscape/main/docs/fig1_grid_8x.png" width="600">

*MOA-2022-BLG-091. (a, c) The published 50 × 50 maps (Han et al. 2025, Fig. 2), in bands of Δχ² < 4, 8, 12, 16
and 20. (b, d) The continuous Δχ² surfaces of `lenscape` at 400 × 400. Top: the (log s, log q) plane. Bottom: the
trajectory angle α against log s, with χ² minimised over all other parameters, including q, at every point.*

## Why it is fast

- **Compiled magnifications.** Pure Python, compiled with numba. Point-source magnifications use the root solver of
  VBMicrolensing (Bozza 2010; Bozza et al. 2018; Skowron & Gould 2012). Near caustics, quadrupole and hexadecapole
  approximations (Gould 2008; Pejcha & Heyrovský 2009; Cassan 2017) are evaluated on images tracked from the
  point-source roots; full contour integration is reserved for sources on or inside a caustic.
- **Fluxes solved, not searched.** The source and blend fluxes of every data set are solved by linear least
  squares at each step, so the fits search only the nonlinear parameters.
- **Fits in the timescales a light curve measures.** The fits use *t*<sub>eff</sub> = *u*<sub>0</sub>*t*<sub>E</sub>
  and *t*<sub>⋆</sub> = *ρt*<sub>E</sub>, which straightens the narrow curved valleys of the likelihood. A short
  Nelder–Mead search finds the valley floor, and Levenberg–Marquardt steps converge to the minimum.
- **Warm starts from the neighbours.** Most fits start from the solutions of neighbouring cells and lattice points,
  near their minimum, instead of from a sweep over trajectory angles.
- **Hierarchical densification.** Fine maps are built from coarser ones: each new point starts from the solutions
  around it, so doubling the resolution costs a fraction of a new map. Refinement can be restricted to the cells
  that a Δχ² contour runs through.

## Why it is complete

Fitting every lattice point independently leaves many fits in local minima and gives speckled maps. `lenscape`
computes a map in stages:

1. **Cell fits.** The best fit anywhere inside each lattice cell (*s* and *q* free within the cell), so that a
   narrow valley between two lattice points is not missed. A coarse seed grid provides starting points, and
   neighbouring cells exchange their solutions.
2. **Lattice-point fits** at fixed (*s*, *q*), started from the cell solutions nearby.
3. **Neighbour refinement.** Solutions propagate from point to point, for both the resolved and the unresolved
   source branch, until no point improves by more than 1.
4. **Outlier recalculation.** Points worse than their neighbours get a deep search with many starts.
5. **Densification** (optional) to finer lattices, or **finer cells** only where a Δχ² band edge runs between
   cells.

The same stages at fixed (log *s*, *α*), with *q* free, give the profile of the trajectory angle. Each stage
prints a numbered header and a progress bar.

## Installation

```bash
pip install lenscape
```

Latest development version:

```bash
pip install git+https://github.com/hippke/lenscape
```

Requirements: Python ≥ 3.10, NumPy, numba, tqdm; matplotlib for figures (`pip install lenscape[plot]`). The first
call compiles the kernels (about a minute); they are cached on disk afterwards.

## Examples

The [example notebooks](https://github.com/hippke/lenscape/tree/main/notebooks) run from public data to finished maps and figures. Start with the
[quickstart](https://github.com/hippke/lenscape/blob/main/notebooks/01_quickstart.ipynb); [tutorial_first_map_research_note](https://github.com/hippke/lenscape/blob/main/notebooks/tutorial_first_map_research_note.ipynb)
computes the map of the research note step by step.

## Documentation

- [Method](https://github.com/hippke/lenscape/blob/main/docs/method.md): magnifications, fits, the stages of a map, accuracy.
- [API reference](https://github.com/hippke/lenscape/blob/main/docs/api.md): all public functions.
- [Validation](https://github.com/hippke/lenscape/blob/main/docs/validation.md): comparisons with VBBinaryLensing and with a published map.

## Contributing

All my repositories embrace collaboration between humans and AI. Contributions are welcome, whether they are written by you, by an AI assistant, or by both together, and they will be handled promptly. Please open an [issue](https://github.com/hippke/lenscape/issues) or a pull request.

## Citing

If you use `lenscape`, please cite the accompanying research note (in preparation). See [`CITATION.cff`](https://github.com/hippke/lenscape/blob/main/CITATION.cff).
