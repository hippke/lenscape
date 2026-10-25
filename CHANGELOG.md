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
- `cfg["adaptive"]` (off): `densify` may accept one χ² at the interpolated start where the surrounding solutions
  agree; measured no net gain on MOA-2022-BLG-091 (the refinement repairs the slightly worse values).
- `grid.Pool`: the map stages fork their worker processes once per stage instead of once per task (the per-task
  fork dated from the VBBinaryLensing segfaults); crash tolerance is kept per worker. Large inputs of the refinement
  and outlier tasks moved to module globals. `LENSCAPE_POOL=0` restores the per-task fork.
- Outlier recalculation: the deep search of each outlier runs as one screening task plus one task per exploration
  (`fit.heavy_starts`, `fit.heavy_explore`), so the stage uses all workers (it ran 2–3 searches on 11 workers for
  a minute); `cfg["heavy_optimizer"]` ("nm", or "nm_lm" for a 2.4× cheaper search).
- Neighbour refinement as an asynchronous queue (`grid.refine_queue`, used by `relax` and the (log s, α) profiles):
  improvements spread as fast as the fits finish instead of one lattice step per pass with a barrier; 16 × 16 test
  lattice: refinement 180 → 105 CPU-s, 23 → 10 s wall. `cfg["relax_mode"]="sweeps"` keeps the earlier passes.
- The cell-stage relaxation (`amr._relax`) runs on the queue engine as well: MOA-2022-BLG-091 16 × 16 lattice from
  scratch 237 → 148 s wall, same bands (its passes ran 1–17 refits behind a barrier at ~10–25 % CPU use).
- `dchi2_cut` (5 × the last band edge) in `lattice`, `densify` and `polish`: points far above the displayed bands
  get one start and no refinement or outlier search unless a neighbour is below the cut; no change on maps that lie
  inside the bands, large savings on wide maps.
- Outlier rule at the edges of a map: an edge point above both its adjacent value and the one-sided quadratic
  extrapolation from the interior by more than the threshold is searched (one such point had stayed 16 above).
- `cfg["bin_K"]`, `cfg["bin_tol"]`: the wing-binning window and the per-night check tolerance as settings
  (`bin_K=0.5`: −10 % CPU on the test block, same bands; default 1.0).
- `maps.multilevel`: lattice on a coarse sub-lattice, then densify level by level (the route for fine maps; the
  coarse lattice must resolve the valleys).
- `alpha_profile` stage 1 options `profile_interp` and `profile_dedupe` (gated LM start and start deduplication as
  in `densify`): 1.4× less CPU on a profile densified from a coarser one, with ~8 % more points worse by > 1, so off.
- `cfg["lm_jac"]="analytic"` (option): analytic point-source gradient for the Levenberg–Marquardt Jacobian
  (`nbmag._point_grad`, `contour.lc_full_grad`, `model.residual_jacobian`, `fit.lm_jacobian`); fewer evaluations
  per fit, but slower on caustic-crossing data where the finite-source points dominate, so the default stays
  forward differences.
- Outlier deep search with Nelder–Mead + Levenberg–Marquardt and the analytic Jacobian by default
  (`heavy_optimizer="nm_lm"`, `heavy_jac="analytic"`): 2.5× less CPU in the stage, same map; the exact Jacobian
  matters there because the starts are far from the minimum.
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
