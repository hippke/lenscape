# Changelog

## 0.1.0 (unreleased)

First public version.

- Fix: `data.bin_wings` checks every binned run against the best model (maps pass `best`) and keeps runs that a
  linear-in-time model misrepresents; an anomaly farther than one tE from t0 was binned through a straight line.
- Fix: `grid.repair_outliers` flags stuck fits by two rules (above both adjacent values, or above the quadratic
  extrapolation from two neighbours on either side; the second finds stuck fits on slopes that the first missed),
  counts only improvements by more than `min_gain` (1.0) as repairs and does not search a point again unless its
  neighbourhood changed by more than that: ridges of the profile were deep-searched in every pass of every map.
- Fix: refits of rough points (`smooth`) rank the neighbours' solutions by one χ² each and fit the best three
  instead of all eight; gains below the refinement margin are kept but no longer mark the neighbours for another
  check (noise-level gains cascaded through the map until `max_sweeps`); the roughness test interpolates
  quadratically through two neighbours on either side, so the curvature of a coarse map no longer counts as
  roughness. Same for (log s, α) profiles.
- Lattice-point fits (`lattice` stage 3, `densify`) skip starts in the basin of one already fitted (`dedupe`), and
  `densify` first fits the interpolated start by Levenberg–Marquardt alone (`interp_start`, now on) and stops there
  when the result is consistent with the surrounding known solutions (`lm_gate`): MOA-2022-BLG-091, 2× → 4× block,
  5.8× less CPU in the point stage, same Δχ² bands.
- `cfg["lm_tol"]` (0.03) and `cfg["nm_fatol"]` (1e-2): stopping rules of the local fits; LM no longer iterates to
  gains of 1e-3, far below the χ² noise (8 % fewer light curves in the point stage, same bands).
- Cell stage: seeded cells skip the angle screen (`n_screen_seeded=0`) and all cell fits are refined by
  Nelder–Mead + Levenberg–Marquardt in (log s, log q) (`box_lm` now on): MOA-2022-BLG-091 16 × 16 lattice from
  scratch, cell stage 4019 → 1049 CPU-s (the stage was 89 % of the map), 96 % of the points in the same band as
  before. `refine_gap` option (off).
- Binary-lens magnifications in numba: point source, quadrupole and hexadecapole approximations on tracked images,
  contour integration with linear limb darkening; single lens and binary source (1L2S).
- Fits at fixed (s, q) with a screen of trajectory angles; cell fits with s and q free inside a cell.
- Delta chi2 maps: `lattice()` (cell fits, lattice-point fits, neighbour refinement, outlier recalculation) and
  `refine()` (finer cells along the Delta chi2 band edges), `densify()` and `polish()` (finer lattices from known
  solutions), `alpha_minima()` (all local solutions in the trajectory angle).
- Caustic-crossing starting points for fits (`cfg["caustic_crossings"]`).
- Neighbour refinement passes check only points whose neighbourhood changed since their last check (identical results,
  fewer checks).
- `cfg["optimizer"] = "nm_lm"`: short Nelder–Mead then Levenberg–Marquardt (`optimize.least_squares_lm`, residual
  vectors `model.chi2_residuals` also for binned data): 3–4× faster map refinement at the same quality.
- Defaults: `optimizer="nm_lm"` and `smooth=0.2` (faster fits; maps without the speckle of stuck fits). Earlier
  behaviour with `optimizer="nm"`, `smooth=None`.
  (log s, α) profile fits stay Nelder–Mead by default (`profile_optimizer`).
- `cfg["smooth"]`: refinement of rough points (χ² above the interpolation of their neighbours) from all neighbours'
  solutions; removes the speckle of stuck fits in maps. (log s, α) maps wrap α only if the grid covers 2π.
- (log s, α) profiles: `one_sign` option (each solution once, as in published maps). Start assignment in
  `alpha_profile` / `densify` by a bucketed nearest-neighbour search (seconds instead of a full sort per point).
- `fit.offset_partner` / `fit.fit_partner`: start for the offset-degenerate partner of a solution.
- `cfg["interp_start"]` for `densify`; `maps.alpha_profile_refine`; (log s, α) refinement now spreads solutions of
  both signs of u0.
- `khgrid`: the (k, h) search grid of Hall et al. (2024).
- No SciPy dependency: Nelder–Mead copied from SciPy (`lenscape.optimize`, bit-identical results).
- Annual parallax and linear orbital motion, MCMC (affine-invariant ensemble), source radius and Einstein radius.
- Figures: model light curves with residuals and insets, caustics and source trajectories, Δχ² maps.
- Data: KMTNet archive and arXiv downloads (`datasets`), data points from vector figures (`eps`).
- Tutorials: ten notebooks and a reproduction of Han et al. (2025), Figures 1–7.
