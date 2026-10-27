"""Default settings of the fits and maps. Copy and modify a dict (dict(config.DEFAULT, n_screen=6)) rather than
changing the module values.

Backends compute light curves: kind "nb" is lenscape's own numba code. Tolerances: tol (point-source / finite-source
switch), reltol (relative accuracy of the finite-source magnification), hexeps (acceptance of the quadrupole and
hexadecapole approximations), ctol (contour integration).
"""

# loose settings for the many fits of a map; tight settings for best fits, MCMC and figures
GRID_BACKEND = dict(kind="nb", tol=1e-2, reltol=3e-3, hexeps=1e-3, ctol=0.1)
REFINE_BACKEND = dict(kind="nb", tol=1e-3, hexeps=1e-4, ctol=0.01)

DEFAULT = dict(
    strategy="screen_d",           # 20 trajectory angles screened, the best n_screen refined (fit.fit_point)
    n_screen=4,
    coords="teff",                 # fit in (t0, t_eff = u0 tE, ln tE, alpha, log t_* = log rho tE)
    explore_scale=None,
    refine_step=0.3,
    grid_backend=GRID_BACKEND,
    refine_backend=REFINE_BACKEND,
    seed=None,                     # start vector dict(t0, u0, tE, logrho); fit.seed_from_data(data) estimates one
    optimizer="nm_lm",             # local fits: short Nelder-Mead, then Levenberg-Marquardt ("nm": Nelder-Mead only)
    heavy_optimizer="nm_lm",       # deep search of outliers: Nelder-Mead + Levenberg-Marquardt refinements (2.5x cheaper than "nm")
    heavy_jac="analytic",          # ... with the analytic Jacobian (same map as Nelder-Mead; "fd" was slightly worse)
    adaptive=False,                # densify: accept one chi2 at the interpolated start where the surroundings agree (no gain measured)
    lm_jac="fd",                   # LM Jacobian: forward differences, or "analytic" (point-source gradient; slower on caustic-rich data)
    lm_tol=0.03,                   # Levenberg-Marquardt stops when a step gains less than this in chi2 (noise ~0.07)
    nm_fatol=1e-2,                 # Nelder-Mead fatol of the local fits
    bin_K=0.5,                     # maps: nights beyond bin_K tE from t0 are binned (1.0 when only the best fit can be checked)
    bin_tol=0.05,                  # ... unless their binned chi2 differs from the exact one by > this for any checked model
    smooth=0.2,                    # refinement also refits points > 0.2 above the interpolation of their neighbours
    n_explore_box=150,             # cell fits: Nelder-Mead evaluations per start before the refinement (300 cost more, 100 lost a point)
    n_screen_seeded=0,             # cell fits with warm starts (seeded cells, children): no angle screen of their own
    box_lm=True,                   # cell fits refined by short Nelder-Mead + Levenberg-Marquardt in (log s, log q)
    refine_gap=None,               # refine the second-best explored trial only within this gap of the best (None: always)
    rho_starts=None,               # lattice/densify point fits: extra starts at these log rho values when one beats its start
    rho_fit=False,                 # ... fit every rho_starts variant instead of only the one beating its start (x2.3 CPU per point)
    profile_optimizer="nm",        # (log s, alpha) profile fits: Nelder-Mead ("nm_lm" is faster but weaker there)
    profile_interp=False,          # profile stage 1: LM alone from an interpolated start first (settles points too early at fixed alpha)
    profile_dedupe=True,           # profile stage 1: skip starts in the basin of one already fitted (-9 % CPU, same quality)
    profile_step=10.0,             # profile refinement: always fit the best neighbour (per u0 sign) when it beats a point by more
)

# Stage 1 (maps.lattice() from scratch): the fits there start far from the solution, so they get ρ starts and more
# precise local fits. MOA-2022-BLG-091, 50 x 49 from scratch, against a converged 8x map: visible points (reference
# Delta chi2 < 100) worse by > 1 42 % -> 12.5 % (Delta chi2 < 16: 39 % -> 7 %), 2.1x the stage-1 time.
# The refinement fits every neighbour's solution and the source-size variants of the two best ones (relax_deep:
# at the boundary of two solution families the better family's solution evaluated at the point is far worse, but
# its fit converges), deduplicated and with short fits first (relax_dedupe, relax_cheap: 0.5x the CPU of the
# full fits, same map on Han's photometry of MOA-2022-BLG-091: 57k -> 27k CPU-s, 2.0 % of visible points differ
# by > 1 either way, none by > 10); residuals capped at 10 sigma in the Nelder-Mead exploration (explore_clip:
# fits from a neighbour's solution reach the map value 73 % instead of 67 % of the time at equal cost)
LATTICE = dict(DEFAULT,
               rho_starts=(-3.5, -4.0, -4.5, -5.0),   # extra starts at these log rho values, each fitted (rho_fit)
               rho_fit=True,
               rho_refine=True,                       # refinement: the neighbours' solutions at the other source sizes too
               relax_deep=True,                       # refinement: fit every neighbour's solution (no one-chi2 pre-check)
               relax_dedupe=True,                     # ... skipping starts in the basin of one already fitted
               relax_cheap=True,                      # ... short fits, only the best result polished precisely
               explore_clip=10.0,                     # Nelder-Mead exploration on residuals capped at 10 sigma (needles)
               explore_nfev=300,                      # Nelder-Mead exploration before Levenberg-Marquardt (100 stopped early)
               lm_tol=1e-4,                           # Levenberg-Marquardt to convergence (0.03 left points 1-3 too high)
               lm_maxit=200)

# densify() and polish() (points started from known solutions): the LATTICE settings without the source-size starts,
# which cost 3x there for the last few points. Han's photometry of MOA-2022-BLG-091, 2x densify of the 50 x 49 map on a
# 7950X: DEFAULT 168 s (20 % of visible points improvable by > 1, 14 worse than the LATTICE map by > 5), DENSIFY 488 s
# (2 %, 5 points), LATTICE 1520 s (1 %)
DENSIFY = dict(LATTICE, rho_starts=None, rho_fit=False)

# backwards-compatible name
CURRENT = DEFAULT
