# Changelog

## 0.1.0 (unreleased)

First public version.

- `densify()` and `polish()` default to `config.DENSIFY`: the `LATTICE` settings (precise fits, deep deduplicated
  cheap-then-polish refinement, clipped exploration) without the source-size starts. Han's photometry of
  MOA-2022-BLG-091, 2x densify on a 7950X: `DEFAULT` 168 s (20 % of visible points improvable by > 1, 14 points
  worse than the full-settings map by > 5), `DENSIFY` 488 s (2 %, 5 points), `LATTICE` 1520 s (1 %).
- Fix: `fit.Coords.from_x` added 0.001 d to tE (a guard against tE <= 0), so every start was displaced by 0.001 d
  on each coordinate round trip and a solution in a narrow minimum was lost (2x densify of MOA-2022-BLG-091:
  10 % of the points coincident with the 1x lattice came out 1-3 above it); now a floor.
- Stage-1 refinement (`config.LATTICE`): every neighbour's solution and the source-size variants of the two best
  neighbours are fitted (`relax_deep`; the one-chi2 pre-check rejected the better family's solution at family
  boundaries, where it evaluates far worse but its fit converges), starts in one basin are fitted once
  (`relax_dedupe`), every start gets a short fit and only the best result the precise one (`relax_cheap`), and
  the neighbours' solutions are also tried at the other source sizes (`rho_refine`). Han's photometry of
  MOA-2022-BLG-091, 50 x 49 from scratch: the map of the full fits at 0.48x the CPU (57k -> 27k CPU-s; 2.0 % of
  visible points differ by > 1, none by > 10); against the earlier refinement, roughness 0.97 -> 0.81 at 1.0x.
- `cfg["explore_clip"]` (10 in `config.LATTICE`): the Nelder-Mead exploration of a local fit minimises the chi2
  with every residual capped at this many sigma; LM and the returned value use the exact chi2. A point-like
  source makes needles in the chi2 surface wherever a data point sits on a fold caustic (3e-5 wide in u0,
  10x the surroundings), and the simplex contracts around them: fits from a neighbour's solution reach the map
  value 73 % instead of 67 % of the time (231 starts, equal cost). `explore_rho_floor` (log rho floored in the
  exploration) was worse (41 %) and is off.
- Options measured without gain, kept off: `branches` (k distinct solutions kept and propagated per point:
  same map at 1.9x `relax_cheap`'s cost), `point_screen` (angle x source-size screen at every lattice point:
  alone it loses the second valley along u0 tE; with `branches=3` the best map of the test at 1.9x),
  `relax_deep="demand"` (deep fits only where a neighbour is lower: no saving on a steep surface),
  `relax_cheap_top`, `relax_deep_rho=1` (misses), `precise_top=2` (cost-neutral with the full fits).
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
- `cfg["bin_K"]` (0.5), `cfg["bin_tol"]`: the wing-binning window and the per-night check tolerance as settings;
  `densify`, `polish` and the profiles check each night against the best fit and six known solutions spread over
  the map, `lattice()` from scratch uses max(bin_K, 1). −10 % CPU on the test block, same bands.
- `maps.multilevel`: lattice on a coarse sub-lattice, then densify level by level (the route for fine maps; the
  coarse lattice must resolve the valleys).
- `alpha_profile` stage 1: start deduplication (`profile_dedupe`, on: −9 % CPU, same quality) and a gated LM start
  from the interpolated solutions (`profile_interp`, off: it settles points too early at fixed α).
- `cfg["lm_jac"]="analytic"` (option): analytic point-source gradient for the Levenberg–Marquardt Jacobian
  (`nbmag._point_grad`, `contour.lc_full_grad`, `model.residual_jacobian`, `fit.lm_jacobian`); fewer evaluations
  per fit, but slower on caustic-crossing data where the finite-source points dominate, so the default stays
  forward differences.
- Outlier deep search with Nelder–Mead + Levenberg–Marquardt and the analytic Jacobian by default
  (`heavy_optimizer="nm_lm"`, `heavy_jac="analytic"`): 2.5× less CPU in the stage, same map; the exact Jacobian
  matters there because the starts are far from the minimum.
- Cell fits explore each start with 150 Nelder–Mead evaluations instead of 300 (`n_explore_box`): −8 % CPU on a
  lattice from scratch with fewer points worse than the reference; 100 lost a point.
- Levenberg–Marquardt normal equations in numba instead of BLAS (a matrix product woke OpenBLAS's thread pool when
  numpy had been imported before lenscape, and its threads then spun), and the worker warm-up covers the LM
  kernels.
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
- `cfg["rho_starts"]`, `cfg["rho_fit"]`: extra source-size starts for lattice-point fits.
- `lattice()` defaults to `config.LATTICE` (source-size starts, precise local fits: `explore_nfev=300`, `lm_tol=1e-4`,
  `lm_maxit` 200) and no `dchi2_cut`: stage-1 maps much cleaner (42 % → 12.5 % of visible points worse by > 1).
- `fit.offset_partner` / `fit.fit_partner`: start for the offset-degenerate partner of a solution.
- `cfg["interp_start"]` for `densify`; `maps.alpha_profile_refine`; (log s, α) refinement now spreads solutions of
  both signs of u0.
- `khgrid`: the (k, h) search grid of Hall et al. (2024).
- No SciPy dependency: Nelder–Mead copied from SciPy (`lenscape.optimize`, bit-identical results).
- Annual parallax and linear orbital motion, MCMC (affine-invariant ensemble), source radius and Einstein radius.
- Figures: model light curves with residuals and insets, caustics and source trajectories, Δχ² maps.
- Data: KMTNet archive and arXiv downloads (`datasets`), data points from vector figures (`eps`).
- Tutorials: ten notebooks and a reproduction of Han et al. (2025), Figures 1–7.
- Point-source solve (`nbmag`): the per-root Newton iteration from the previous point's roots steps the five
  roots together instead of one after another; the processor overlaps the independent chains (the solve is
  latency-bound): 222 → 151 ns per point on a Ryzen 7950X, kernel light curve of MOA-2022-BLG-091 (1 090 points)
  396 → 321 µs, full light curve 654 → 612 µs, bit-identical results. New far-field solve `nbmag._roots_far`
  (images from perturbation theory and Newton on the lens equation, ghosts by deflating the quintic, accepted only
  when the root set is verified complete, so exact for any s, q): used for independent points without a warm
  start (`mags_point` 920 → 441 ns per point, `caustics.magnification_map` 5×); for light curves it is
  speed-neutral and off (`nbmag.FARFIELD`, `LENSCAPE_FARFIELD=1`). A branch-free block solve vectorised over 8–32
  points (AVX-512) was prototyped and is slower end to end (fixed step count, 7–9 % fallbacks). Scripts and
  numbers: `bench/farfield_*.py`, `bench/part2_*.py`, `RESEARCH_point_source_results.md` (project directory).
