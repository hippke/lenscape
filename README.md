<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/logo/lenscape-banner-dark.svg">
  <img alt="lenscape" src="docs/logo/lenscape-banner.svg" width="600">
</picture>

# lenscape

**Fast and complete Δχ² maps of binary-lens microlensing events.**

Planetary microlensing papers find and present degenerate solutions with Δχ² maps over the binary-lens
separation *s* and mass ratio *q*. At every point of such a map, the remaining parameters are fitted to the light
curve, with finite-source magnifications near the caustics. These maps are among the most expensive steps of an
analysis.

`lenscape` computes them about **20 times faster** than a pipeline built on VBBinaryLensing, and uses the speed
for a more **complete** search: fits start from their neighbours' solutions, cells are fitted with *s* and *q* free
inside them, and the map is refined along the Δχ² contours. A full map of a published planetary event
(50 × 49 points, 2531 data points) takes 98 minutes on a six-core desktop computer.

## Why it is fast

- **Own magnification code, compiled with numba.** Point-source magnifications follow VBBinaryLensing
  (Bozza 2010; Bozza et al. 2018). Near caustics, `lenscape` uses quadrupole and hexadecapole approximations
  (Gould 2008; Pejcha & Heyrovský 2009; Cassan 2017) on images tracked from point to point, and its own contour
  integration only where these fail. Light curves are about 5 times faster than with VBBinaryLensing, at
  better accuracy (table below).
- **Fits in the timescales a light curve measures.** The fits use *t*<sub>eff</sub> = *u*<sub>0</sub>*t*<sub>E</sub>
  and *t*<sub>⋆</sub> = *ρt*<sub>E</sub> instead of *u*<sub>0</sub> and *ρ*, which straightens the narrow curved
  valleys of the likelihood. Each fit screens 20 trajectory angles and refines the best few.
- **One process per fit, on all cores.** A fit that fails cannot take a map down with it.

Light curves, 1000 points, five binary-lens configurations, one thread (`benchmarks/benchmark.py`; errors against
VBBinaryLensing at Tol = 10⁻⁵; Intel Core i5-13500 under load):

| | VBBinaryLensing (Tol 10⁻³) | lenscape (tight settings) | speed-up |
|---|---|---|---|
| total time, 5 light curves | 40.5 ms | 8.6 ms | **4.7×** |
| largest relative error | 1.4 × 10⁻⁴ | 1.0 × 10⁻⁴ | |

Maps (per lattice point, including the fits):

| | VBBinaryLensing + Nelder–Mead | lenscape |
|---|---|---|
| CPU time per lattice point | 42 s | **≈ 2 s** |
| full 50 × 49 map (MOA-2022-BLG-091, 2531 data points) | — | **98 min on 6 cores** |

## Why it is complete

Fitting every lattice point independently leaves many fits in local minima. In tests, starting from the
neighbours' solutions **doubled the allowed region** of a map. `lenscape` computes a map in stages:

1. **Cell fits.** The best fit anywhere inside each lattice cell (*s* and *q* free within the cell), so that a
   narrow valley between two lattice points is not missed. A coarse seed grid provides starting points.
2. **Lattice-point fits** at fixed (*s*, *q*), started from the cell solutions nearby.
3. **Neighbour refinement.** Points are refitted from their neighbours' solutions until none improves.
4. **Outlier recalculation.** Points worse than their neighbours get a deep search with many starts.
5. **Finer cells** (optional) where a Δχ² band edge runs between cells, so contours are resolved without refining
   the whole map.

Each stage prints a numbered header and a progress bar.

## Installation

```bash
pip install git+https://github.com/hippke/lenscape
```

Requirements: Python ≥ 3.10, NumPy, SciPy, numba, tqdm; matplotlib for figures. The first call compiles the
kernels (about a minute); they are cached on disk afterwards.

## Usage

```python
import numpy as np
import lenscape

# KMTNet pySIS files (HJD, flux, flux error, ...); one data set per file, time axis HJD - 2450000
data = lenscape.read_pysis("KMT-2022-BLG-0114/*_I.pysis", window=(9600, 9720))

# a best fit of the event: (t0, u0, tE, log s, log q, alpha, log rho)
best = np.array([9657.77, -0.0254, 35.5, 0.0020, -2.398, 4.41, -3.3])

# Delta chi2 map on a (log s, log q) lattice, then finer cells along the band edges
logs = np.linspace(-0.010, 0.016, 50)
logq = np.linspace(-2.6, -2.0, 49)
result = lenscape.lattice(data, logs, logq, best=best)
cells = lenscape.refine(result, data, levels=2, best=best)

dchi2 = result["chi2"] - result["chi2_ref"]          # (50, 49), Delta chi2 at every lattice point
```

Figures of the solutions:

```python
from lenscape import figures, make_backend, config

backend = make_backend(config.REFINE_BACKEND)
figures.plot_lightcurve(backend, [best], data, window=(9652, 9663), insets=[(9655.0, 9655.5)],
                        path="lightcurve.png")
figures.plot_geometry([best], path="geometry.png")    # caustics, trajectory, magnification contours
```

## Documentation

- [Method](docs/method.md): magnifications, fits, the stages of a map, accuracy.
- [API reference](docs/api.md): all public functions.
- [Validation](docs/validation.md): comparisons with VBBinaryLensing and with a published map.

## Citing

If you use `lenscape`, please cite the accompanying research note (in preparation) and the papers of the methods
it builds on (see [docs/method.md](docs/method.md)). See `CITATION.cff`.

## License

GPL-3.0-or-later.
