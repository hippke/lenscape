# API reference

`import lenscape` gives the submodules below and, at top level, the most common functions:
`Data`, `read_pysis`, `read_text`, `lattice`, `refine`, `make_backend`, `seed_from_data`.

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

### `read_text(path, cols=(0, 1, 2), t_offset=0.0, mag=False, zp=22.0, comments="#")`
`(t, flux, flux_err)` from a text file with columns `cols = (time, value, error)`. With `mag=True`, value and error
are magnitudes and are converted with flux = 10^(−0.4 (mag − zp)). Combine several files with `Data.from_sets`.

### `rescale(fe, emag, k, smin)`
Error rescaling of Yee et al. (2012): σ′ = k √(σ² + σ_min²) in magnitudes, applied to the flux errors `fe`.

### `bin_wings(data, t0, tE, K=1.0, gap=0.3)`
Speed option for maps: points with |t − t0| > K tE are grouped per night (runs without gaps > `gap` days) and
replaced by their sufficient statistics. χ² stays exact for a model that is linear in time within a night. Used by
`maps.lattice` and `maps.refine` when `best` is given (`bin_wings=True`).

---

## `lenscape.maps` — Δχ² maps

### `lattice(data, logs, logq, best=None, cfg=None, workers=None, bands=(4, 16, 36, 64, 100), seed_grid=2, bin_wings=True, log=print)`
Profile χ² at every point of the lattice `logs × logq` (1-D arrays with uniform steps): s and q fixed, the other
parameters fitted. Runs in stages (cell fits from a coarse seed grid, lattice-point fits, neighbour refinement,
outlier recalculation; see [method.md](method.md)), each with a numbered header and a progress bar.

- `best`: a best fit of the event (7-vector). It is the start of the fits; without it, `fit.seed_from_data` is used.
- `cfg`: settings dict (default `config.DEFAULT`).
- `workers`: parallel processes (default: number of cores − 1).
- `bands`: Δχ² levels; cells crossed by a band edge get extra attention.

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

---

## `lenscape.fit` — fits

### `fit_point(ls, lq, data, cfg, warm=None)`
Best fit at fixed `log s = ls`, `log q = lq`. Screens 20 trajectory angles at the start vector `cfg["seed"]`,
optimises the best `cfg["n_screen"]` with Nelder–Mead in (t0, t_eff, ln tE, alpha, log t⋆). Returns `(chi2, x)`
with `x = (t0, u0, tE, alpha, log rho)`. `warm`: extra start vectors (e.g. neighbours' solutions) when
`cfg["warm"]` is true.

### `fit_point_box(box, data, cfg, warm=None)`
Best fit anywhere in the cell `box = (logs_lo, logs_hi, logq_lo, logq_hi)`: s and q are free inside the cell.
Returns `(chi2, x)` with the 7-vector `x = (t0, u0, tE, alpha, log rho, log s, log q)`.

### `fit_point_heavy(ls, lq, data, cfg, warm=None)`
Deep search at fixed (s, q) from about 1000 diverse starts (used for the outlier recalculation).

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
- `NumbaHybrid(tol, hexeps, ctol, ld=0.0, reltol=None)` — the backend class: `lc(p, t)` returns A(t).
  `ld` is the linear limb-darkening coefficient.
- `register_backend(kind, cls)` — make a backend class available to `make_backend`.

## `lenscape.config` — settings

`DEFAULT` (settings of fits and maps), `GRID_BACKEND`, `REFINE_BACKEND`. Copy and modify:
`cfg = dict(lenscape.config.DEFAULT, n_screen=6, seed=dict(t0=..., u0=..., tE=..., logrho=...))`.

---

## `lenscape.caustics` — geometry

- `lens_positions(s, q)` — x of the primary and of the secondary (centre-of-mass frame).
- `critical_curves(s, q, n=1000)` — `(critical, caustics)`: lists of closed complex arrays.
- `caustics(s, q, n=1000)` — caustics only.
- `trajectory(p, t)` — source positions (complex) at times `t`.
- `magnification_map(s, q, x, y)` — point-source magnification on the grid `x × y`; shape (ny, nx).

## `lenscape.figures` — figures (matplotlib)

### `plot_lightcurve(backend, solutions, data, labels=None, window=None, insets=(), ref=0, zp=22.0, t_offset=0.0, colors=..., path=None, xlabel="HJD - 2450000", res_lim=0.3)`
Data and model light curves of several solutions in magnitudes, zoom `insets = [(t1, t2), ...]`, and one residual
panel per solution. All data sets are put on the magnitude scale of set `ref` with each solution's own fluxes.
`colors`: a sequence by data-set index or `{name: colour}`. Returns the matplotlib figure; saves it to `path`.

### `plot_geometry(solutions, labels=None, extent=(-0.3, 0.5, -0.2, 0.2), ncols=2, mirror=False, centre="primary", levels=None, path=None)`
One panel per solution: caustics, source trajectory with direction arrow, lens positions and equal-magnification
contours. `centre="primary"` puts the origin at the primary, `"com"` at the centre of mass.

### `aligned(backend, p, data, ref=0, zp=22.0)`, `model_mag(backend, p, t, fs_ref, fb_ref, zp=22.0)`
Data magnitudes on a common scale, and model magnitudes, for your own plots.

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

## Lower level

- `contour.mag_disk(s, q, y1, y2, rho, reltol)`, `contour.mag_disk_ld(..., ld)` — finite-source magnification by
  contour integration (uniform disk; linear limb darkening).
- `contour.mags_full(s, q, y1, y2, rho, tol, hexeps, ctol, ld=0.0)` — magnifications for arrays of samples.
- `nbmag.mags_point(s, q, y1, y2)` — point-source magnifications for arrays.
- `grid.run_isolated(fn, tasks, workers)` — run `fn(*args)` for each task in its own forked process (a crashing
  task returns `None` instead of stopping the run).
- `amr.run_contour(...)` — the cell-refinement engine behind `maps.lattice` and `maps.refine`.

## Environment variables

| variable | effect |
|---|---|
| `LENSCAPE_PROGRESS=0/1` | progress bars off / on (default: on when writing to a terminal) |
| `NUMBA_CACHE_DIR` | location of the compiled kernels (default: `~/.cache/lenscape/numba-<hash of the sources>`) |
