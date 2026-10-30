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
| `example_kb200202` | example event KMT-2020-BLG-0202 (Han et al. 2026, one telescope, resonant caustic): the paper's 43 × 54 lattice, 2× and 4× densify, alternative solutions, the limits (maps not converged on a wide lattice, unrefined floor region), and a zoom on the central region (8× and 16× boxes, then strips along a narrow branch) that finds a minimum 2.3 below the published-start fit; stored maps, `RECOMPUTE = True` to rerun | 15 s (rerun: 10.4 h on a 20-thread i5-13500) |
| `example_kb230466` | example event KMT-2023-BLG-0466 (Han et al. 2026, three telescopes, two caustic grazes): the paper's 24 × 52 lattice, 2×, 4× and 8× densify, a zoom on the central valley, comparison with the published map, alternative solutions, and the single data point that excludes the close one; stored maps, `RECOMPUTE = True` to rerun | 1 min (rerun: 6.3 h on a 20-thread i5-13500) |
| `example_kb251314` | example event KMT-2025-BLG-1314 (Ren, Zang, Zhu et al. 2026, planet/stellar-binary degeneracy, public KMTNet data only): best fit from the ten published solutions, 1× map over the paper's panel, the q ↔ 1/q symmetry and folding, accuracy of the map values, comparison with the published map, and why a densify does not pay here; stored maps, `RECOMPUTE = True` to rerun | 10 s (rerun: 6.6 h on a 20-thread i5-13500) |
| `example_kb250811` | example event KMT-2025-BLG-0811 (Tang, Zang et al. 2026, four degenerate low-*q* solutions, public KMTNet data only): best fit from the published solutions, coarse 1× map on the paper's 61 × 61 lattice, accuracy of the map values, comparison with the published map, the *t*ₑ degeneracy (*t*ₑ profile), and why a densify does not pay here; stored maps, `RECOMPUTE = True` to rerun | 30 s (rerun: 13.5 h on a 20-thread i5-13500) |
| `example_kb221551` | example event KMT-2022-BLG-1551 (Han et al. 2026, high magnification, central and planetary caustics): 1× map, 2× densify everywhere and 4× in the central region, comparison with the published map, zoom on the close valley (8× and 16× over a box, then strips that follow the valley at 64× and 128× in log s), which finds a second minimum at q = 2.1 × 10⁻³ as good as the published one; what the alternatives look like; stored maps, `RECOMPUTE = True` to rerun | 7 s (rerun: 10.2 h on a 20-thread i5-13500) |
| `example_kb250121` | example event KMT-2025-BLG-0121 (Han et al. 2026, a minimum sharper than the lattice; photometry read from the paper's Fig. 6 plus public pySIS): 1× map, accuracy of the map values, comparison with the published map, local minima, a zoom on the central region at 2×, 4× and 8×, and the valley itself at 16× to 256× (slanted, about 0.0005 dex wide in log *s*, resolved to below 0.1 in Δχ²); stored maps, `RECOMPUTE = True` to rerun | 5 s (rerun: 11 h on a 12-thread Ryzen 5 3600 for the 1× map, plus 5 h on a 20-thread i5-13500 for the zooms) |
| `example_kb251616` | example event KMT-2025-BLG-1616 (Yang, Zang et al. 2026, four degenerate solutions that only the non-public DREAMS data break, public KMTNet data only): best fit from the published solutions, single-lens floor, coarse 1× map on the paper's 61 × 61 lattice, accuracy of the map values, comparison with the published map, local minima; zoom on the paper's dense box at 2× to 16× of the coarse step, which resolves the four valleys (the narrow resonant ones only at 16×); stored maps, `RECOMPUTE = True` to rerun | 8 s (rerun: 6.2 h on a 20-thread i5-13500, plus 5.3 h for the zooms) |

\* Measured on a 16-core AMD Ryzen 9 7950X, data already downloaded; the first notebook also compiles the
kernels (about 2 minutes, cached afterwards).

The maps in `06_maps` and `han2025_figures` take longer at finer resolution; each notebook says how the runtime
scales. `data/` holds the photometry CSVs, the stored solutions and the 1× map used by the tutorials.
