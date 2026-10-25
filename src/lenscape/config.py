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
    profile_optimizer="nm",        # (log s, alpha) profile fits: Nelder-Mead ("nm_lm" is faster but weaker there)
    profile_interp=False,          # profile stage 1: LM alone from an interpolated start first (settles points too early at fixed alpha)
    profile_dedupe=True,           # profile stage 1: skip starts in the basin of one already fitted (-9 % CPU, same quality)
)

# backwards-compatible name
CURRENT = DEFAULT
