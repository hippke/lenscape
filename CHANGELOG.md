# Changelog

## 0.1.0 (unreleased)

First public version.

- Binary-lens magnifications in numba: point source, quadrupole and hexadecapole approximations on tracked images,
  contour integration with linear limb darkening; single lens and binary source (1L2S).
- Fits at fixed (s, q) with a screen of trajectory angles; cell fits with s and q free inside a cell.
- Delta chi2 maps: `lattice()` (cell fits, lattice-point fits, neighbour refinement, outlier recalculation) and
  `refine()` (finer cells along the Delta chi2 band edges), `densify()` and `polish()` (finer lattices from known
  solutions), `alpha_minima()` (all local solutions in the trajectory angle).
- Caustic-crossing starting points for fits (`cfg["caustic_crossings"]`).
- `khgrid`: the (k, h) search grid of Hall et al. (2024).
- No SciPy dependency: Nelder–Mead copied from SciPy (`lenscape.optimize`, bit-identical results).
- Annual parallax and linear orbital motion, MCMC (affine-invariant ensemble), source radius and Einstein radius.
- Figures: model light curves with residuals and insets, caustics and source trajectories, Δχ² maps.
- Data: KMTNet archive and arXiv downloads (`datasets`), data points from vector figures (`eps`).
- Tutorials: ten notebooks and a reproduction of Han et al. (2025), Figures 1–7.
