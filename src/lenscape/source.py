"""Source angular radius, Einstein radius and lens-source proper motion from the source colour and magnitude.

Standard KMTNet procedure (e.g. Han et al. 2026, Sect. 5):
  1. calibrate against the red giant clump (RGC): (V-I, I)_0 = (V-I, I)_s - (V-I, I)_RGC + (V-I, I)_RGC,0,
     with (V-I)_RGC,0 = 1.06 (Bensby et al. 2013) and I_RGC,0 from Nataf et al. (2013) (field dependent);
  2. convert V-I to V-K with a dwarf colour sequence;
  3. surface-brightness relation (Kervella et al. 2004, dwarfs): log10(2 theta_*/mas) = 0.5175 + 0.2753 (V-K)_0 - 0.2 V_0;
  4. theta_E = theta_* / rho, mu_rel = theta_E / t_E.

The paper converts V-I to V-K with Bessell & Brett (1988); we use the dwarf sequence of Pecaut & Mamajek (2013,
ApJS 208, 9; table version 2022.04.16), which agrees with the paper's Table 5 theta_* to ~1-3 % (see tests). A different
sequence can be passed as (vi, vk) arrays.
"""
import numpy as np

VI_RGC0 = 1.06                    # Bensby et al. (2013)
KERVELLA_A, KERVELLA_B = 0.5175, 0.2753
DAY_PER_YR = 365.25

# Pecaut & Mamajek (2013) mean dwarf sequence, F0V .. M5V: (V-Ic, V-Ks); G1V dropped (V-Ic not monotonic)
PM13 = np.array([
    (0.339, 0.734), (0.385, 0.819), (0.432, 0.925), (0.449, 0.964), (0.476, 1.012), (0.506, 1.079), (0.553, 1.190),
    (0.579, 1.221), (0.599, 1.290), (0.620, 1.335), (0.640, 1.403), (0.673, 1.437), (0.713, 1.564), (0.722, 1.588),
    (0.733, 1.611), (0.738, 1.635), (0.758, 1.691), (0.766, 1.705), (0.786, 1.768), (0.820, 1.857), (0.853, 1.953),
    (0.879, 2.06), (0.920, 2.13), (1.013, 2.40), (1.182, 2.70), (1.272, 2.88), (1.420, 3.08), (1.519, 3.35),
    (1.632, 3.48), (1.699, 3.55), (1.766, 3.65), (1.886, 3.84), (2.005, 4.00), (2.089, 4.10), (2.173, 4.23),
    (2.306, 4.43), (2.420, 4.60), (2.680, 5.00), (2.831, 5.25), (3.073, 5.65), (3.278, 5.95)])


def deredden(vi, i, vi_rgc, i_rgc, i_rgc0, vi_rgc0=VI_RGC0):
    """(V-I)_0, I_0 of the source from its offset to the red giant clump centroid."""
    return vi - vi_rgc + vi_rgc0, i - i_rgc + i_rgc0


def vi_to_vk(vi0, table=PM13):
    if not (table[0, 0] <= vi0 <= table[-1, 0]):
        raise ValueError(f"(V-I)_0 = {vi0} outside the colour table ({table[0, 0]} .. {table[-1, 0]})")
    return float(np.interp(vi0, table[:, 0], table[:, 1]))


def theta_star(vi0, i0, table=PM13):
    """Angular source radius in micro-arcsec from the de-reddened (V-I)_0, I_0 (Kervella et al. 2004, V-K)."""
    vk, v0 = vi_to_vk(vi0, table), i0 + vi0
    return 0.5e3 * 10.0 ** (KERVELLA_A + KERVELLA_B * vk - 0.2 * v0)


def einstein(theta_s_uas, rho, tE):
    """Einstein radius theta_E [mas] and geocentric mu_rel [mas/yr] from theta_* [uas], rho and t_E [days]."""
    thE = 1e-3 * theta_s_uas / rho
    return thE, thE / tE * DAY_PER_YR


def source_chain(vi, i, vi_rgc, i_rgc, i_rgc0, rho, tE, vi_rgc0=VI_RGC0, table=PM13):
    """Full chain; returns dict with (V-I)_0, I_0, V-K, theta_* [uas], theta_E [mas], mu [mas/yr]."""
    vi0, i0 = deredden(vi, i, vi_rgc, i_rgc, i_rgc0, vi_rgc0)
    th = theta_star(vi0, i0, table)
    thE, mu = einstein(th, rho, tE)
    return dict(vi0=vi0, i0=i0, vk0=vi_to_vk(vi0, table), theta_star=th, theta_E=thE, mu=mu)


def sample_chain(samples_rho, samples_tE, vi0, i0, sig_vi0=0.0, sig_i0=0.0, frac_theta=0.0, rng=None, table=PM13):
    """Propagate posterior samples of (rho, tE) plus Gaussian colour/magnitude errors and an intrinsic fractional
    scatter of the surface-brightness relation into samples of theta_E and mu (arrays)."""
    rng = np.random.default_rng(rng)
    n = len(samples_rho)
    vi = vi0 + sig_vi0 * rng.standard_normal(n)
    ii = i0 + sig_i0 * rng.standard_normal(n)
    th = np.array([theta_star(a, b, table) for a, b in zip(vi, ii)]) * (1 + frac_theta * rng.standard_normal(n))
    thE = 1e-3 * th / np.asarray(samples_rho)
    return thE, thE / np.asarray(samples_tE) * DAY_PER_YR
