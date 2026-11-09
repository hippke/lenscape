# Notebooks

Run them with Jupyter after `pip install lenscape[plot] jupyter` (or from a clone with `pip install -e ".[plot]"`),
from inside their folder: each folder has a `data/` subfolder with the photometry, the stored solutions and the maps
its notebooks load. KMTNet data are downloaded on first use from the public archive and cached in
`~/.cache/lenscape/data`.

## [`tutorials/`](tutorials)

The package step by step, from loading data to maps, MCMC and physical parameters.

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
| `tutorial_first_map_research_note` | for first-time users: from the photometry file `data/moa091_photometry.csv` to the research note's map (50 × 49 lattice, 2× and 4× densify) with all defaults | 45 min |
| `tutorial_alpha_map` | the trajectory-angle map: all local minima in α and the filled (log *s*, α) profile, on the 1× map of the first tutorial | 60 min |

## [`real_events/`](real_events)

Published events (KMTNet, MOA), reanalysed with lenscape from public data and compared with the published solutions
and maps.

| notebook | content | runtime* |
|---|---|---|
| `han2025_figures` | Han et al. (2025) light curves and lens configurations (Figs. 1, 3, 4, 5, 7) from the CSV photometry (`han2025.py` rebuilt it from the paper's figures) | 10 s |
| `example_kb200202` | example event KMT-2020-BLG-0202 (Han et al. 2026, one telescope, resonant caustic): the paper's 43 × 54 lattice, 2× and 4× densify, alternative solutions, the limits (maps not converged on a wide lattice, unrefined floor region), and a zoom on the central region (8× and 16× boxes, then strips along a narrow branch) that finds a minimum 2.3 below the published-start fit; stored maps, `RECOMPUTE = True` to rerun | 15 s (rerun: 10.4 h on a 20-thread i5-13500) |
| `example_kb230466` | example event KMT-2023-BLG-0466 (Han et al. 2026, three telescopes, two caustic grazes): the paper's 24 × 52 lattice, 2×, 4× and 8× densify, a zoom on the central valley, comparison with the published map, alternative solutions, and the single data point that excludes the close one; stored maps, `RECOMPUTE = True` to rerun | 1 min (rerun: 6.3 h on a 20-thread i5-13500) |
| `example_kb251314` | example event KMT-2025-BLG-1314 (Ren, Zang, Zhu et al. 2026, planet/stellar-binary degeneracy, public KMTNet data only): best fit from the ten published solutions, 1× map over the paper's panel, the q ↔ 1/q symmetry and folding, accuracy of the map values, comparison with the published map, and why a densify does not pay here; stored maps, `RECOMPUTE = True` to rerun | 10 s (rerun: 6.6 h on a 20-thread i5-13500) |
| `example_kb250811` | example event KMT-2025-BLG-0811 (Tang, Zang et al. 2026, four degenerate low-*q* solutions, public KMTNet data only): best fit from the published solutions, coarse 1× map on the paper's 61 × 61 lattice, accuracy of the map values, comparison with the published map, the *t*ₑ degeneracy (*t*ₑ profile), and why a densify does not pay here; stored maps, `RECOMPUTE = True` to rerun | 30 s (rerun: 13.5 h on a 20-thread i5-13500) |
| `example_kb221551` | example event KMT-2022-BLG-1551 (Han et al. 2026, high magnification, central and planetary caustics): 1× map, 2× densify everywhere and 4× in the central region, comparison with the published map, zoom on the close valley (8× and 16× over a box, then strips that follow the valley at 64× and 128× in log s, and the full region around them at 64×), which finds a second minimum at q = 2.1 × 10⁻³ as good as the published one; what the alternatives look like; stored maps, `RECOMPUTE = True` to rerun | 7 s (rerun: 11.6 h on a 20-thread i5-13500) |
| `example_kb250121` | example event KMT-2025-BLG-0121 (Han et al. 2026, a minimum sharper than the lattice; photometry read from the paper's Fig. 6 plus public pySIS): 1× map, accuracy of the map values, comparison with the published map, local minima, a zoom on the central region at 2×, 4× and 8×, and the valley itself at 16× to 256× (slanted, about 0.0005 dex wide in log *s*, resolved to below 0.1 in Δχ²); stored maps, `RECOMPUTE = True` to rerun | 5 s (rerun: 11 h on a 12-thread Ryzen 5 3600 for the 1× map, plus 5 h on a 20-thread i5-13500 for the zooms) |
| `example_kb251616` | example event KMT-2025-BLG-1616 (Yang, Zang et al. 2026, four degenerate solutions that the DREAMS data break). Public KMTNet data: best fit from the published solutions, single-lens floor, coarse 1× map on the paper's 61 × 61 lattice, accuracy of the map values, comparison with the published map, local minima; zoom on the paper's dense box at 2× to 16× of the coarse step, which resolves the four valleys (the narrow resonant ones only at 16×). Section 6, all data (KMTNet, DREAMS, PRIME; photometry provided by W. Zang): best fit, 1× to 16× maps over the dense box (central valleys ≈ 200 above), and the two resonant valleys resolved at 256× in log *s* against s-free floor profiles (wide resonant preferred by Δχ² 1.88); stored maps, `RECOMPUTE = True` to rerun | 10 s (rerun: 6.2 h on a 20-thread i5-13500, plus 5.3 h for the zooms; all data: ~41 h on 10 cores) |

## [`blind_roman_data_challenge/`](blind_roman_data_challenge)

Blind analyses of simulated Roman events from the Roman Microlensing Data Challenge 2026, organised by the Roman
Galactic Exoplanet Survey Project Infrastructure Team (RGES-PIT; data public domain,
[rges-pit.org/data-challenge](https://rges-pit.org/data-challenge/)). The true parameters are not published.

| notebook | content | runtime* |
|---|---|---|
| `example_rmdc26_000026` | blind analysis of a simulated Roman event (Roman Microlensing Data Challenge 2026, Beginner tier, public domain; true parameters not published): reducing 49 096 points in three filters to 1773 for the maps (event season only, time bins outside the anomaly, checked against the unbinned data), guided close/wide fits, coarse map, zooms at 8× and 64× that resolve a single sharp minimum (close planet, q = 1.16 × 10⁻⁴, s = 0.697), light curve, caustics and source trajectory, rough physical scale; stored maps, `RECOMPUTE = True` to rerun | 5 s (rerun: 30 min on a 20-thread i5-13500) |
| `example_rmdc26_001422` | blind analysis of a simulated Roman event with a central-caustic anomaly (Roman Microlensing Data Challenge 2026, Experienced tier, public domain): data reduction as in `example_rmdc26_000026`, guided fits, coarse map over 0.1 < s < 10 and 10⁻⁵ < q < 1 (the V of central-caustic events), zooms at 4×, 32× and 64× that resolve a close (s = 0.918) and a wide (s = 1.045) solution with the same q = 3.28 × 10⁻⁴, only Δχ² = 2.9 apart; light curves, caustics, rough physical scale; stored maps, `RECOMPUTE = True` to rerun | 4 s (rerun: about 1 h on a 20-thread i5-13500) |
| `example_rmdc26_001576` | blind analysis of a simulated Roman event with a resonant caustic (Roman Microlensing Data Challenge 2026, Experienced tier, public domain; 12 mag of extinction in F087): same steps as `example_rmdc26_001422`; coarse map over 0.1 < s < 10 and 10⁻⁵ < q < 1, zooms at 4× and 64× that resolve one very sharp minimum (s = 0.969, q = 9.9 × 10⁻⁴, a Jupiter-like mass ratio); light curve, resonant caustic and trajectory, rough physical scale; stored maps, `RECOMPUTE = True` to rerun | 3 s (rerun: about 45 min on a 20-thread i5-13500) |
| `example_rmdc26_001279` | blind analysis of a simulated high-magnification Roman event (Roman Microlensing Data Challenge 2026, Experienced tier, public domain): the angle-screen guided starts fail, caustic-crossing starts (Cassan 2008) find the solution; coarse map with two valleys thinner than the lattice step, zooms at 8× and 64× that resolve a wide (s = 1.284) and a close (s = 0.769) solution with q = 5.5 × 10⁻³, Δχ² = 27 apart; light curves, central caustic with a cusp approach at magnification 120, rough physical scale (brown-dwarf host); stored maps, `RECOMPUTE = True` to rerun | 6 s (rerun: about 2.5 h on a 20-thread i5-13500) |
| `example_rmdc26_001532` | blind analysis of a simulated Roman event with a double peak (Roman Microlensing Data Challenge 2026, Experienced tier, public domain): caustic-crossing starts find the solution where the angle screen fails; coarse map with the textbook close/wide pair of tilted valleys, zooms at 4× and 64×; close solution s = 0.771, q = 3.9 × 10⁻³, preferred by Δχ² = 20 over the wide one (s = 1.376); light curves, central caustic with a cusp approach, rough physical scale; stored maps, `RECOMPUTE = True` to rerun | 4 s (rerun: about 2.5 h on a 20-thread i5-13500) |
| `example_rmdc26_000209` | blind analysis of a simulated low-magnification Roman event with a 1.5-hour planetary-caustic spike (Roman Microlensing Data Challenge 2026, Experienced tier, public domain): the spike time alone points to *s* ≈ 1.9; the angle-screen guided fits find the solution, the coarse map shows one tilted valley, an 8× zoom resolves it and a 64× zoom confirms a single minimum (wide planet, s = 1.90, q = 1.4 × 10⁻³; the best close solution is Δχ² = 216 worse); light curve, the source crossing a cusp of the planetary caustic, rough physical scale; stored maps, `RECOMPUTE = True` to rerun | 10 s (rerun: about 2 h on a 20-thread i5-13500) |
| `example_rmdc26_000566` | blind analysis of a simulated high-magnification Roman event with a spike next to the peak (Roman Microlensing Data Challenge 2026, Experienced tier, public domain): the angle-screen guided fits find a close/wide pair at the central caustic; coarse map with two valleys far thinner than the lattice step (missed by 175 and 440), still missed by 13–18 at 8×, resolved by 64× zooms; close solution s = 0.857, q = 5.4 × 10⁻⁴, preferred by Δχ² = 12 over the wide one (s = 1.200); light curves, the source passing the front cusp of the central caustic, rough physical scale (Saturn-mass planet); stored maps, `RECOMPUTE = True` to rerun | 10 s (rerun: about 1 h on a 20-thread i5-13500) |

\* Measured on a 16-core AMD Ryzen 9 7950X, data already downloaded; the first notebook also compiles the
kernels (about 2 minutes, cached afterwards).

The maps in `06_maps` and `han2025_figures` take longer at finer resolution; each notebook says how the runtime
scales.
