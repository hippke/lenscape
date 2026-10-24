# Changelog

## 0.1.0 (unreleased)

First public version.

- Binary-lens magnifications in numba: point source, quadrupole and hexadecapole approximations on tracked images,
  contour integration with linear limb darkening; single lens and binary source (1L2S).
- Fits at fixed (s, q) with a screen of trajectory angles; cell fits with s and q free inside a cell.
- Delta chi2 maps: `lattice()` (cell fits, lattice-point fits, neighbour refinement, outlier recalculation) and
  `refine()` (finer cells along the Delta chi2 band edges).
- Annual parallax and linear orbital motion, MCMC (affine-invariant ensemble), source radius and Einstein radius.
- Figures: model light curves with residuals and insets, caustics and source trajectories.
