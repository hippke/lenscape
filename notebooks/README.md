# Tutorials

Run them with Jupyter after `pip install lenscape[plot] jupyter` (or from a clone with `pip install -e ".[plot]"`).
Data are downloaded on first use from the public KMTNet archive and cached in `~/.cache/lenscape/data`.

| notebook | content | runtime* |
|---|---|---|
| `01_quickstart` | public data → fit → map → figures | 5 min (incl. compiling) |
| `02_data` | loading, cuts, error bars, outliers, binning | 1 s |
| `03_magnification` | backends, finite source, limb darkening, accuracy, VBBinaryLensing comparison | 9 s |
| `04_caustics` | caustic topologies, magnification maps, geometry plots | 2 s |
| `05_fitting` | fits at fixed (s, q), full fits, cell fits, the four solutions of MOA-2022-BLG-091 | 31 s |
| `06_maps` | Δχ² maps step by step, refinement, the trajectory angle | 4 min |
| `07_higher_order` | parallax, ecliptic degeneracy, orbital motion, binary source | 23 s |
| `08_mcmc` | posteriors with the ensemble sampler | 12 s |
| `09_physical_parameters` | source radius, Einstein radius, proper motion | 1 s |
| `10_caustic_crossings` | fits from rough starts with caustic-crossing starting points (simulated event, MOA-2022-BLG-091) | 1 min |
| `han2025_figures` | Han et al. (2025) light curves and lens configurations (Figs. 1, 3, 4, 5, 7) from the CSV photometry (`han2025.py` rebuilt it from the paper's figures) | 10 s |
| `tutorial_first_map_research_note` | for first-time users: from the photometry file `data/moa091_photometry.csv` to the research note's map (50 × 49 lattice, 2× and 4× densify) with all defaults | 45 min |
| `tutorial_alpha_map` | the trajectory-angle map: all local minima in α and the filled (log *s*, α) profile, on the 1× map of the first tutorial | 60 min |

\* Measured on a 16-core AMD Ryzen 9 7950X, data already downloaded; the first notebook also compiles the
kernels (about 2 minutes, cached afterwards).

The maps in `06_maps` and `han2025_figures` take longer at finer resolution; each notebook says how the runtime
scales. `data/` holds the photometry CSVs, the stored solutions and the 1× map used by the tutorials.
