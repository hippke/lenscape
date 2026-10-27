# API reference

`import lenscape` gives the submodules below and, at top level, the most common functions:
`Data`, `read_pysis`, `read_text`, `lattice`, `refine`, `make_backend`, `seed_from_data`, `fit_full`,
`error_factors`.

## Conventions

**Parameter vector.** A binary-lens solution is a 7-vector

```
p = (t0, u0, tE, log10 s, log10 q, alpha, log10 rho)
```

on the time axis of the data (for example HJD − 2450000). `s` is the projected separation in units of the
Einstein radius, `q = m2 / m1` the mass ratio, `alpha` the trajectory angle in radians, `rho` the source radius in
Einstein radii. Fits at fixed (s, q) work on the 5-vector `(t0, u0, tE, alpha, log10 rho)`.

**Frame.** Centre of mass, primary on the left at x = −s q/(1 + q), secondary on the right at x = s/(1 + q).
The source moves as

```
y1 =  u0 sin(alpha) − tau cos(alpha)
y2 = −u0 cos(alpha) − tau sin(alpha),     tau = (t − t0) / tE
```

The mirror image (u0, alpha) → (−u0, −alpha) gives the same light curve. Some papers draw the configuration
mirrored in y; `figures.plot_geometry(..., mirror=True)` reproduces that.

**Fluxes.** Each data set k has its own source and blend flux, F = fs_k A(t) + fb_k. They enter linearly and are
solved for in every χ² evaluation (weighted least squares), so they never appear in the parameter vector.

**Backends.** A backend computes magnifications: `backend.lc(p, t)` returns A(t). `make_backend(config)` builds one
from a settings dict; `config.GRID_BACKEND` (loose, for the many fits of a map) and `config.REFINE_BACKEND` (tight,
for best fits, MCMC and figures) are the defaults.

---

## `lenscape.data` — photometry

### `Data(t, f, e, idx, names)`
Container for several data sets: concatenated times `t`, fluxes `f`, flux errors `e`, data-set index `idx`
(0 … nsets−1) and the list of `names`. Attributes `w = 1/e²`, `nsets`, `slices`.

- `Data.from_sets({name: (t, flux, flux_err), ...})` — build from arrays; non-finite points and points with
  `e ≤ 0` are dropped.
- `data.select(mask)`, `data.window(t1, t2)` — new `Data` with a subset of the points.
- `data.scale_errors(k)` — new `Data` with the errors of each set multiplied by `k[name]` (dict) or by one factor.

### `read_pysis(paths, t_offset=2450000.0, window=None)`
KMTNet pySIS files (columns HJD, flux, flux error, …) → `Data`. `paths` is a list or a glob pattern; one data set
per file, named after the file up to the first `_` (`KMTC22_I.pysis` → `KMTC22`). `t_offset` is subtracted from
HJD; `window = (t1, t2)` on the shifted axis.

### `read_text(path, cols=(0, 1, 2), t_offset=0.0, mag=False, zp=22.0, comments="#", **kw)`
`(t, flux, flux_err)` from a text file with columns `cols = (time, value, error)`. With `mag=True`, value and error
are magnitudes and are converted with flux = 10^(−0.4 (mag − zp)). Combine several files with `Data.from_sets`.

### `rescale(fe, emag, k, smin)`
Error rescaling of Yee et al. (2012): σ′ = k √(σ² + σ_min²) in magnitudes, applied to the flux errors `fe`.

### `bin_wings(data, t0, tE, K=1.0, gap=0.3, model=None, tol=0.05)`
`model` may also be a list of `(backend, p)` pairs; a run is kept unbinned if it fails the check for any of them.
Speed option for maps: points with |t − t0| > K tE are grouped per night (runs without gaps > `gap` days) and
replaced by their sufficient statistics. χ² stays exact for a model that is linear in time within a night. With
`model = (backend, p)` every run is checked against the model `p` (the best fit): a run whose binned χ² differs
from its exact χ² by more than `tol` is kept at full cadence, so an anomaly farther than K tE from t0 (a planetary
caustic crossed at τ > 1) is never binned through. The result reports `n_unbinned` and `max_bin_error`. The map
functions pass their `best` as the model (`bin_wings=True`).

---

## `lenscape.maps` — Δχ² maps

### `lattice(data, logs, logq, best=None, cfg=None, workers=None, bands=(4, 16, 36, 64, 100), seed_grid=2, bin_wings=True, max_sweeps=30, dchi2_cut=None, log=print)`
Profile χ² at every point of the lattice `logs × logq` (1-D arrays with uniform steps): s and q fixed, the other
parameters fitted. Runs in stages (cell fits from a coarse seed grid, lattice-point fits, neighbour refinement,
outlier recalculation; see [method.md](method.md)), each with a numbered header and a progress bar.

- `best`: a best fit of the event (7-vector). It is the start of the fits; without it, `fit.seed_from_data` is used.
- `cfg`: settings dict (default `config.LATTICE`: `config.DEFAULT` with source-size starts `rho_starts`, `rho_fit`
  and precise local fits `explore_nfev=300`, `lm_tol=1e-4`, `lm_maxit=200`; a map from scratch starts its fits far from
  the solution). `dchi2_cut=None` by default here (every point gets all starts and the refinement). MOA-2022-BLG-091,
  50 × 49 from scratch against a converged 8× map: visible points worse by > 1 42 % → 12.5 % (Δχ² < 16: 39 % → 7 %),
  stage time 2.1×.
- `workers`: parallel processes (default: number of cores − 1).
- `bands`: Δχ² levels; cells crossed by a band edge get extra attention.
- `dchi2_cut`: in units of `bands[-1]`; points more than `dchi2_cut * bands[-1]` above the minimum get one start,
  no neighbour refinement and no outlier search unless a neighbour is below the cut (`None`: everything everywhere).
  No effect where the whole lattice lies inside the bands (MOA-2022-BLG-091); on wide maps most cells are above.

Returns a dict:

| key | content |
|---|---|
| `logs`, `logq` | the lattice |
| `chi2` | (ns, nq) χ² at the lattice points |
| `pars` | (ns, nq, 5) best fit at each point: t0, u0, tE, alpha, log rho |
| `chi2_ref` | lowest χ² found (use `chi2 − chi2_ref` as Δχ²) |
| `cells` | stage-1 cell fits (see `refine`) |
| `chi2_first` | χ² before the neighbour refinement (for diagnostics) |
| `wall`, `wall_cells`, `wall_points`, `wall_refinement` | wall-clock times (s) |

### `refine(result, data, levels=2, best=None, cfg=None, workers=None, bands=(4, 16, 36, 64, 100), bin_wings=True, log=print)`
Finer cells where a Δχ² band edge runs between neighbouring cells, `levels` times (each level halves the cell
size). Starts from `result = lattice(...)`. Each cell holds the best fit anywhere inside it (s, q free within the
cell). Returns the leaves as a dict of arrays: `logs`, `logq` (cell centres), `hs`, `hq` (half widths), `chi2`,
`x` (best fit: t0, u0, tE, alpha, log rho, log s, log q), `level`, `chi2_ref`, `nfit`, `wall`.

### `densify(previous, data, logs, logq, best=None, cfg=None, workers=None, n_starts=3, bin_wings=True, max_sweeps=30, dchi2_cut=5.0, log=print)`

`cfg` defaults to `config.DENSIFY` (the `LATTICE` settings without the source-size starts; `polish` likewise).
A (finer) lattice started from known solutions: every point is fitted from the nearest solutions of `previous`
(a list of `lattice`, `refine` or `densify` results; the nearest one is always included), then neighbour refinement
and outlier recalculation. No seed grid and no cell fits: for MOA-2022-BLG-091 (research note), the 2×, 4× and 8× densifications of
the 50 × 50 map took 8, 23 and 74 minutes on a 16-core desktop CPU.

### `multilevel(data, logs, logq, best=None, cfg=None, workers=None, levels=1, bands=(4, 16, 36, 64, 100), bin_wings=True, log=print)`
`lattice()` on every 2^levels-th point of the lattice, then `densify()` level by level to the full lattice: the
route for fine maps (cell fits cost about ten lattice points each). The coarse lattice must still resolve the
valleys of the map; from a 5 × 5 coarse lattice a 17 × 17 test block ended worse at 13 % of its points, from the
50 × 49 paper lattice the 8× map of the research note is the same route by hand. `len(logs) − 1` and
`len(logq) − 1` must be multiples of 2^levels.

### `polish(result, data, previous=(), best=None, cfg=None, workers=None, max_sweeps=30, bin_wings=True, dchi2_cut=5.0, log=print)`
Improves a result: at points shared with `previous` lattices it keeps the better solution, then runs neighbour
refinement and outlier recalculation again.

### `alpha_minima(result, data, best=None, cfg=None, workers=None, n_alpha=20, tol_alpha=0.02, bin_wings=True, log=print)`
All local minima in the trajectory angle at every lattice point (fits from `n_alpha` angles, distinct minima kept):
rows of (log s, log q, χ², t0, u0, tE, α, log ρ). This is how (log s, α) maps are usually drawn.

### `alpha_profile(previous, data, logs, alphas, best=None, cfg=None, workers=None, n_starts=3, margin=1.0, max_sweeps=30, bin_wings=True, log=print, changed_only=True, u0_ref=None, smooth=None, one_sign=False)`
Profile χ² over (log s, α): at every grid point χ² minimised over all other parameters including log q, so the map
has a value everywhere. Started from the nearest known solutions of `previous` (lattices, cells, `alpha_minima`
rows), then neighbour refinement (α periodic). Solutions with the opposite sign of u0 to `u0_ref` are counted at
α + π. `u0_ref`: default that of an `alpha_profile` result in `previous`, else of the best known solution (pass it
when `previous` holds solutions of both signs). `smooth`: as `cfg["smooth"]` (the default). Returns
`dict(logs, alphas, chi2 (ns, na), pars (ns, na, 7), chi2_ref, wall, u0_ref, one_sign)`.

**Neighbour refinement** (in `lattice`, `densify`, `polish`, `alpha_profile`): every point whose neighbourhood
changed is a task; a result that improves a point by more than the margin (1) makes its eight neighbours stale, and
stale points are queued, the largest gap to the best neighbour first. Workers (`grid.Pool`) pull from the queue as
they finish, so an improvement spreads as fast as the fits complete, without the barrier between passes
(`grid.refine_queue`; MOA-2022-BLG-091 16 × 16 lattice: refinement 180 → 105 CPU-s and 23 → 10 s wall, same
bands). The end state satisfies the same rule as the earlier passes (no point has a neighbour solution that beats
it by more than the margin after a refit), reached in a timing-dependent order, so single points can end in another
local minimum from run to run. `cfg["relax_mode"] = "sweeps"` restores the earlier Jacobi passes (a front moves
one lattice step per pass; `changed_only=False` then checks every point in every pass); `max_sweeps` caps the
passes, or the refits of one point in queue mode.

`one_sign=True`: only solutions with the sign of u0 of the reference (known solutions of the other sign are mirrored,
(u0, α) → (−u0, −α) being the same light curve). Each solution then appears once, as in maps of codes that keep u0 > 0
(e.g. Han et al. 2025). Without it every solution also appears mirrored about α = 3π/2; that map equals
min(f(α), f(3π − α)) of the one-sign map, so it never needs its own computation.

### `alpha_profile_refine(result, data, best=None, cfg=None, workers=None, margin=1.0, max_sweeps=30, bin_wings=True, log=print, changed_only=True, smooth=None, one_sign=None)`
More neighbour-refinement passes for an `alpha_profile` result, e.g. one stopped at `max_sweeps` (`one_sign`: default that of
the result). Neighbours'
solutions of either sign of u0 are tried, each at its own model α. The lowest-χ² neighbour of each sign is always fitted at rough
points and where it beats the point by more than `cfg["profile_step"]` (10): across a step its solution evaluates
far worse before the refit, and a one-χ² pre-check would never fit it (the staircase edges of earlier maps). Each
such neighbour is also tried at its tangent prediction (its branch extrapolated linearly from the next point beyond
it), and if no fit beats the point, its solution is continued to the point in `cfg["profile_continuation"]` (16)
sub-steps of short fits (0 turns this off).

---

## `lenscape.fit` — fits

### `fit_point(ls, lq, data, cfg, warm=None)`
Best fit at fixed `log s = ls`, `log q = lq`. Screens 20 trajectory angles at the start vector `cfg["seed"]`,
optimises the best `cfg["n_screen"]` with Nelder–Mead in (t0, t_eff, ln tE, alpha, log t⋆). Returns `(chi2, x)`
with `x = (t0, u0, tE, alpha, log rho)`. `warm`: extra start vectors (e.g. neighbours' solutions) when
`cfg["warm"]` is true.

**Caustic-crossing starts.** For events with resolved caustic crossings, `cfg["caustic_crossings"] = (t_in, t_out)`
adds starting points built from the crossing times (the caustic-crossing parameterisation of Cassan 2008): every
pair of points on the caustics of (s, q) fixes a trajectory that is on the caustic at both times. The best
candidates are refined by moving their entry and exit points along the caustic (`cfg["caustic_refine"]`, default
on), and the best `cfg["n_caustic"]` (default 4) are fitted along with the angle screen (`cfg["caustic_only"] = True`
drops the screen). From a rough start this finds solutions the angle screen misses; from the event's best fit (as in maps) it
still finds the best solution at a lattice point more often (MOA-2022-BLG-091: 26/40 instead of 18/40 points within
Δχ² < 1 of the best known), at 2.7× the cost of a fit, so it is off by default. See notebook 10.
`fit.caustic_starts(ls, lq, crossings, logrho, n=24)` returns the candidate vectors.

### `fit_point_box(box, data, cfg, warm=None)`
Best fit anywhere in the cell `box = (logs_lo, logs_hi, logq_lo, logq_hi)`: s and q are free inside the cell.
Returns `(chi2, x)` with the 7-vector `x = (t0, u0, tE, alpha, log rho, log s, log q)`.

### `fit_point_heavy(ls, lq, data, cfg, warm=None)`
Deep search at fixed (s, q) from about 1000 diverse starts (used for the outlier recalculation): `heavy_starts`
screens them and returns the best `cfg["n_heavy"]` (30), `heavy_explore` fits one of them: Nelder–Mead 300, then
with `cfg["heavy_optimizer"] = "nm_lm"` (default) a short Nelder–Mead and Levenberg–Marquardt with the Jacobian of
`cfg["heavy_jac"]` (`"analytic"` by default, see `lm_jac`), or with `"nm"` 1500 Nelder–Mead evaluations. 17 × 17 test
lattice: outlier stage 1172 → 471 CPU-s with the same map as Nelder–Mead (finite-difference LM: 529 CPU-s and
slightly more points worse by > 1). The map stages run the 30 explorations of each outlier as separate
tasks, so a few outliers no longer occupy a few workers for a minute while the others idle (16 × 16 test lattice:
outlier stage 57 s at 17 % CPU use before).

### `fit_full(data, p0, backend=None, rounds=3, maxfev=6000)`
Best fit of all seven parameters from the start `p0` (Nelder–Mead, restarted `rounds` times; default backend
`config.REFINE_BACKEND`). Returns `(chi2, p)`.

### `offset_partner(p)`, `fit_partner(data, p, backend=None)`
Start vector for the offset-degenerate partner of a solution (Zhang et al. 2022; the reflection RTModel uses): the
separation at which the trajectory crosses the binary axis at the same place relative to the other caustic. Its own
inverse. `fit_partner` runs `fit_full` from it. For KMT-2024-BLG-1209 and MOA-2022-BLG-091 it predicted the
partner's log s to about 0.001 and the fit found the partner in 3 of 4 cases; try it after finding a first solution
of a new event.

### `error_factors(data, p, backend=None)`
Error-bar factor per data set, k = √(χ²_set / N_set) at the solution `p`: `data.scale_errors(error_factors(data, p))`
gives χ²/N = 1 per set.

### `seed_from_data(data, u0=0.05, tE=20.0, logrho=-3.0)`
Rough start vector `dict(t0, u0, tE, logrho)` with t0 at the brightest point. Fits screen the trajectory angle,
but they need t0, u0 and tE in the right neighbourhood; for strong anomalies, start from a proper best fit.

### `make_backend(cfg)`
Backend from a settings dict, e.g. `make_backend(config.REFINE_BACKEND)`. Other codes can be registered with
`model.register_backend(kind, cls)` (the tests register VBBinaryLensing this way).

### `Coords(kind)`
Optimiser coordinates: `"teff"` (t0, t_eff = u0 tE, ln tE, alpha, log t⋆ = log ρ tE) or `"native"`.
Methods `to_x(y)`, `from_x(x)`, `steps(y, scale)`.

---

## `lenscape.model` — χ² and magnifications

- `chi2(backend, p, data)` — χ² of the 7-vector `p` with fluxes solved per data set. Parameters outside the
  physical box (log ρ ∈ [−5, −0.5], tE ∈ [0.5, 1000], |u0| ≤ 5) are clamped and penalised.
- `fluxes(A, data)` — `(fs, fb)` per data set for magnifications `A` at the data times.
- `residuals(backend, p, data)` — residuals in units of the errors, (f − model)/e.
- `chi2_residuals(backend, p, data)` — a residual vector whose sum of squares is `chi2` (for binned data up to a
  constant; two entries per binned night); input of least-squares optimisers.
- `NumbaHybrid(tol, hexeps, ctol, ld=0.0, reltol=None)` — the backend class: `lc(p, t)` returns A(t).
  `ld` is the linear limb-darkening coefficient.
- `register_backend(kind, cls)` — make a backend class available to `make_backend`.

## `lenscape.optimize` — optimisers

- `minimize(fun, x0, method="Nelder-Mead", options=None)` — SciPy's Nelder–Mead (copied, bit-identical).
- `least_squares_lm(fun, x0, h, cap, maxit=50, tol=1e-3)` — Levenberg–Marquardt for `sum(fun(x)**2)` with
  forward-difference steps `h` (choose them above the numerical noise of `fun`) and a largest step `cap` per
  parameter; returns `OptimizeResult(x, fun, nfev, nit)`.
- `bisect(f, a, b)` — root by bisection.

---

## `lenscape.config` — settings

`DEFAULT` (settings of fits and maps), `GRID_BACKEND`, `REFINE_BACKEND`. Copy and modify:
`cfg = dict(lenscape.config.DEFAULT, n_screen=6, seed=dict(t0=..., u0=..., tE=..., logrho=...))`.

Options of the maps. `optimizer="nm_lm"`, `smooth=0.2`, `interp_start`, `lm_gate=1.0` and `dedupe` are the defaults; set `optimizer="nm"`, `smooth=None`, `interp_start=False`, `lm_gate=None`, `dedupe=False`, `box_lm=False`, `n_screen_seeded=4` for the earlier behaviour (Nelder–Mead only, no refits of rough points, every start fitted in full). (log s, α) profile fits use `profile_optimizer` (default `"nm"`: `nm_lm` was 4× faster there but worse from distant starts):

| key | effect |
|---|---|
| `optimizer="nm_lm"` | local fits as a short Nelder–Mead exploration (`explore_nfev`, default 100 evaluations) followed by Levenberg–Marquardt on the residual vector (`optimize.least_squares_lm`), instead of Nelder–Mead alone. Used by the lattice-point fits, neighbour refinement, `densify`, `polish`, `alpha_profile` and the refinement step of `alpha_minima` and `fit_point`; cell fits stay Nelder–Mead. MOA-2022-BLG-091: densified block 3.1× less CPU (99.8 % of points in the same Δχ² band as Nelder–Mead), (log s, α) profile 4.0× |
| `smooth=0.2` | neighbour refinement also refits every point whose χ² exceeds the interpolation through its opposite neighbours by more than this, from all 8 neighbours' solutions (no one-χ² pre-check, no margin). Removes the speckle of fits stuck 1–3 above their neighbours, which the normal trigger (neighbour better by > 1 at once) misses; MOA-2022-BLG-091 8× lattice block: roughness 0.72 → 0.09, half of the points 2.3 lower (median), a whole corner > 4 lower. Recommended for figures |
| `box_lm=True` (default) | cell fits (`fit_point_box`) refined by a short Nelder–Mead and Levenberg–Marquardt in (log s, log q) clipped to the cell instead of 1500 Nelder–Mead evaluations. MOA-2022-BLG-091, 16 × 16 lattice from scratch: cell stage 3432 → 1049 CPU-s with `n_screen_seeded=0`; final map 96.1 % of the points in the same band as the stored 50 × 49 map (Nelder–Mead cells: 95.7 %), 2 of 256 points worse by > 1 |
| `n_explore_box=150` | cell fits: Nelder–Mead evaluations per start before the refinement. 17 × 17 lattice from scratch, matched runs: 300 → 1986 CPU-s with 1.73 % of the points worse by > 1 against the stored map, 150 → 1825 CPU-s with 0.35 %, 100 → one point 25 worse |
| `n_screen_seeded=0` (default) | cells that receive warm starts (seeded level-0 cells, children of split cells) run no 20-angle screen and explorations of their own; the seed grid has screened the angles already. Cell stage 4019 → 3414 CPU-s, same map. `n_screen` still applies to the seed grid and to cells without seeds |
| `refine_gap=None` | `fit_point_box` / `fit_point_screen_d`: refine the second-best explored trial only if it lies within this gap of the best. Measured no gain (the second trial rarely costs much), so off |
| `interp_start=True`, `lm_gate=1.0` | `densify`: a start interpolated from the equally nearest known solutions (the bilinear interpolation on a lattice refined by 2) is fitted by Levenberg–Marquardt alone first (~20 light curves). If its χ² lies within `lm_gate` of the interpolation of those solutions' χ² and below every nearest solution's own χ² at the point, the point is settled; otherwise the `n_starts` nearest solutions are fitted as before. MOA-2022-BLG-091, 2× → 4× block: 5.8× less CPU in the point stage, same bands (with `dedupe`) |
| `lm_tol=0.03`, `nm_fatol=1e-2` | stopping rules of the local fits: Levenberg–Marquardt stops when a step gains less than `lm_tol` in χ² (the χ² noise of the map backend is ~0.07; 1e-3 chased it: 8 % more light curves), Nelder–Mead at `fatol = nm_fatol` (0.05 changed nothing measurable) |
| `adaptive=False` | `densify`: where the equally nearest known solutions agree (same band, χ² spread < 0.1 dex), accept one light curve at the interpolated start as the point's value (a true profile value, an upper bound) before trying Levenberg–Marquardt. Measured on the 4× block: point stage unchanged (140 vs 135 CPU-s), refinement doubled (83 vs 39) because the sweeps then repair the slightly worse values; off |
| `bin_K=0.5`, `bin_tol=0.05` | maps: nights beyond `bin_K` tE from t0 are binned per night (see `bin_wings`), each checked with tolerance `bin_tol` in χ² against the best model and, in `densify`, `polish` and the profiles, against up to six known solutions spread over the map's χ² range, so nights where other solutions bend stay unbinned too. −10 % CPU on the 4× block, same bands. `lattice()` from scratch has only the best fit to check against and uses max(`bin_K`, 1) |
| `profile_dedupe=True`, `profile_interp=False` | the two devices of `densify` for the first stage of `alpha_profile` (`maps._sa_point`). Measured on a 10 × 20 (log s, α) block densified from a coarse profile (no gate, no dedupe: 2716 CPU-s, 20 % of the points worse by > 1 against the reference block): dedupe alone 2474 CPU-s with the same quality (on by default); the gated LM start from the interpolated solutions settles points too early at fixed α (both: 1924 CPU-s but 28.5 % worse by > 1; gate alone: slower, 3689 CPU-s), so it stays off |
| `rho_starts=None`, `rho_fit=False` | lattice-point fits (`lattice`, `densify`, `polish`): every start also at these log ρ values, e.g. `(-3.5, -4, -4.5, -5)`; with `rho_fit` each variant is fitted, otherwise only one that beats its start at once. Separate small- and large-source minima exist; MOA-2022-BLG-091 from scratch, right arm of the 1× lattice: median deficit against the converged map 3.5 → 0.4 with `rho_fit` (screen only: 3.0), at ~2.3× CPU per lattice point. Useful in `lattice()`, whose solutions the finer `densify` stages inherit |
| `rho_refine=False` (`LATTICE`: on) | neighbour refinement: the neighbours' solutions at the `rho_starts` source sizes too, the best by one χ² fitted |
| `relax_deep=False` (`LATTICE`: on) | neighbour refinement fits every neighbour's solution plus the source-size variants of the `relax_deep_rho` (2) best neighbours, without the one-χ² pre-check; `"demand"`: only where a neighbour's own χ² is lower (measured no saving) |
| `relax_dedupe=False` (`LATTICE`: on) | ... skipping starts in the basin of one already fitted (`fit.same_start`) |
| `relax_cheap=False` (`LATTICE`: on) | ... every start a short fit (`explore_nfev=100`, `lm_tol=0.03`), the best `relax_cheap_top` (1) polished with the full settings: 0.5× the CPU, same map |
| `explore_clip=None` (`LATTICE`: 10) | local fits: Nelder–Mead exploration on the χ² with residuals capped at this many σ (needles of an unresolved source on fold caustics); LM and results exact |
| `explore_rho_floor=None` | local fits: log ρ floored in the exploration (measured worse; off) |
| `branches=None` | lattice point fits and refinement keep the k best distinct solutions per point and propagate all of them (a branch losing here seeds the points where it wins); same map at 1.9× the cost on the test event |
| `point_screen=None` | lattice point fits: this many trajectory angles × the source sizes screened at the best start's (t0, u0, tE), the best `point_screen_keep` (3) new basins fitted too |
| `heavy_optimizer="nm_lm"`, `heavy_jac="analytic"` | the outlier deep search's refinement: Nelder–Mead + Levenberg–Marquardt with the analytic Jacobian. Its starts are far from the minimum, which is where the exact Jacobian pays: 2.5× less CPU in the stage than Nelder–Mead, identical map; with finite differences slightly more points worse by > 1 |
| `lm_jac="fd"` | Jacobian of the Levenberg–Marquardt fits: forward differences (6 light curves per iteration), or `"analytic"`: the gradient of the point-source magnification through the source position (`nbmag._point_grad`, exact at point-source points, with the quadrupole term's ρ-derivative at finite-source points) and differences only at the contour points, projected per data set as in variable projection (`model.residual_jacobian`, Kaufman's form). Fewer evaluations per fit (synthetic event: 14 vs 43), but on MOA-2022-BLG-091, where 30 % of the evaluated points are finite-source points with an approximate gradient, the 4× block took 183 instead of 117 CPU-s in the point stage; for point-source-dominated events only |
| `dedupe=True` | lattice-point fits (`lattice` stage 3, `densify`): the starts are fitted in the order of their χ² at the point, and a start in the basin of one already fitted (same t0 to 1 % tE, t_eff to 2 %, ln tE to 0.02, α to 0.02, log t⋆ to 0.05) is skipped; the nearest solutions of a converged coarse map are usually the same solution. Alone 2.6× less CPU in the point stage |

---

## `lenscape.caustics` — geometry

- `lens_positions(s, q)` — x of the primary and of the secondary (centre-of-mass frame).
- `critical_curves(s, q, n=1000)` — `(critical, caustics)`: lists of closed complex arrays.
- `caustics(s, q, n=1000)` — caustics only.
- `trajectory(p, t)` — source positions (complex) at times `t`.
- `magnification_map(s, q, x, y)` — point-source magnification on the grid `x × y`; shape (ny, nx).

## `lenscape.figures` — figures (matplotlib)

### `plot_lightcurve(backend, solutions, data, labels=None, window=None, insets=(), ref=0, zp=22.0, t_offset=0.0, colors=..., n_model=4000, figsize=(7.0, 7.5), path=None, xlabel="HJD - 2450000", res_lim=0.3)`
Data and model light curves of several solutions in magnitudes, zoom `insets = [(t1, t2), ...]`, and one residual
panel per solution. All data sets are put on the magnitude scale of set `ref` with each solution's own fluxes.
`colors`: a sequence by data-set index or `{name: colour}`. `n_model`: points of the model curves. Returns the matplotlib figure; saves it to `path`.

### `plot_geometry(solutions, labels=None, extent=(-0.3, 0.5, -0.2, 0.2), ncols=2, mirror=False, centre="primary", levels=None, figsize=None, path=None, t_arrow=None)`
One panel per solution: caustics, source trajectory with direction arrow, lens positions and equal-magnification
contours. `centre="primary"` puts the origin at the primary, `"com"` at the centre of mass. `mirror=True` reflects
y (the (u0, α) → (−u0, −α) symmetry; some papers draw the source above the axis). `t_arrow`: time of the direction
arrow (default t0).

### `plot_map(result, bands=None, ax=None, vmax=100.0, cmap="jet_r", colorbar=True, path=None, marks=None, chi2_ref=None)`
Δχ² map of a `lattice()` result, of `refine()` cells or of an `alpha_profile()` result: continuous (log scale up to
`vmax`) or, with `bands=(4, 16, 36, ...)`, in colour bands (orange = best, white above the last edge).
`marks={label: (x, y)}` draws labelled crosses (e.g. published solutions). `chi2_ref`: the reference χ² (default
the result's own; pass a common one to compare maps).

### `aligned(backend, p, data, ref=0, zp=22.0)`, `model_mag(backend, p, t, fs_ref, fb_ref, zp=22.0)`
Data magnitudes on a common scale and model magnitudes, for your own plots: `aligned` returns
`(mag, err, fs_ref, fb_ref)`. Difference-imaging data with a negative source flux are shown as the magnified
source alone (blend set to zero).

---

## `lenscape.lightcurve` — parallax and orbital motion

- `Trajectory(t, tref, ra, dec, t0par)` — precomputed annual-parallax geometry for times `t` (HJD − tref) and
  event coordinates (`"hh:mm:ss.s"`, `"±dd:mm:ss.s"` or radians).
- `HighOrder(traj, tol, hexeps, ctol, ld=0.0)` — `ho.mag(p)` gives the magnifications at the trajectory's times
  for the 7-, 9- or 11-vector `(t0, u0, tE, log s, log q, alpha, log rho[, piEN, piEE[, ds/dt, dalpha/dt]])`:
  parallax components (north, east) and linear orbital motion (s in Einstein radii per year, alpha in rad per year).
- `chi2(ho, p, data)` — χ² with fluxes solved per data set.

## `lenscape.single` — single lens and binary source

- `pspl(u)`, `espl(u, rho, ld=0.0)` — point-source and finite-source (linear limb darkening) magnification.
- `lc_1l1s(t0, u0, tE, rho, t, ld)` — single-lens light curve.
- `lc_1l2s(t0a, u0a, t0b, u0b, tE, rhoa, rhob, qf, t, ld)` — binary-source light curve, flux ratio `qf`.
- `chi2_1l2s(p, data, ld=0.0)` — χ² of the binary-source model,
  `p = (t0_1, u0_1, t0_2, u0_2, tE, log rho_1, log rho_2, log qF)`.

## `lenscape.mcmc` — sampling

- `run(lp, p0, scale, nwalkers=32, nsteps=2000, a=2.0, seed=0, workers=1, init=None)` — affine-invariant ensemble
  sampler (Goodman & Weare 2010) for the log-probability `lp(p)`. Returns `dict(chain, lnp, acc)`
  (chain: (nwalkers, nsteps, ndim); acc: acceptance fraction).
- `run_ensembles(lp, p0, scale, nwalkers=16, nsteps=2000, workers=4)` — independent ensembles, one per process.
- `autocorr_time(chain)`, `summary(chain, names=None)` — autocorrelation time; median and 1σ intervals.

## `lenscape.source` — Einstein radius and proper motion

- `source_chain(vi, i, vi_rgc, i_rgc, i_rgc0, rho, tE)` — from the source colour and magnitude, the red-clump
  calibration ((V − I)_RC,0 = 1.06, Bensby et al. 2013; I_RC,0 of the field, e.g. Nataf et al. 2013, is an input),
  V − K (Pecaut & Mamajek 2013) and the
  surface-brightness relation (Kervella et al. 2004): returns (V − I)₀, I₀, θ⋆ [μas], θ_E [mas], μ_rel [mas/yr].
- `sample_chain(samples_rho, samples_tE, vi0, i0, ...)` — propagates posterior samples and colour errors.
- `deredden`, `theta_star`, `einstein` — the individual steps.

---

## `lenscape.datasets` — public data

- `kmtnet(event, band="I", window=None, t_offset=2450000.0)` — KMTNet pySIS photometry of an event from the KMTNet
  archive, e.g. `kmtnet("KMT-2022-BLG-0114")`; downloaded once and cached (`$LENSCAPE_DATA` or
  `~/.cache/lenscape/data`). Please acknowledge KMTNet when you use these data.
- `arxiv_source(arxiv_id)` — download and extract a paper's arXiv source; returns its directory.

## `lenscape.eps` — data from vector figures

Many microlensing papers publish their figures as EPS vector graphics, which store every plotted point.
- `subpaths(path)` — all subpaths with their colours.
- `markers(subs, size=(5, 30))`, `error_bars(subs)` — filled markers and vertical error-bar segments.
- `calibrate(d1, v1, d2, v2)` — linear map from device units to data units (from two frame or tick positions).
- `points_with_errors(subs, x_map, y_map, region, colours)` — `(name, x, y, error)` for every marker of the given
  colours inside a panel.

See `notebooks/han2025.py` for a complete example.

## Lower level

- `contour.mag_disk(s, q, y1, y2, rho, reltol)`, `contour.mag_disk_ld(..., ld)` — finite-source magnification by
  contour integration (uniform disk; linear limb darkening).
- `contour.mags_full(s, q, y1, y2, rho, tol, hexeps, ctol, ld=0.0)` — magnifications for arrays of samples.
- `nbmag.mags_point(s, q, y1, y2)` — point-source magnifications for arrays.
- `grid.run_isolated(fn, tasks, workers)` — run `fn(*args)` for each task in forked worker processes (`grid.Pool`:
  the workers are forked once per call and each runs many tasks; a crashing task is retried once in a fresh worker
  and then returns `None` instead of stopping the run). `LENSCAPE_POOL=0` forks one process per task instead.
- `amr.run_contour(...)` — the cell-refinement engine behind `maps.lattice` and `maps.refine`.

## Environment variables

| variable | effect |
|---|---|
| `LENSCAPE_PROGRESS=0/1` | progress bars off / on (default: on when writing to a terminal) |
| `NUMBA_CACHE_DIR` | location of the compiled kernels (default: `~/.cache/lenscape/numba-<hash of the sources>`) |
