"""lenscape: fast and complete (log s, log q) Delta chi2 maps and binary-lens models for gravitational microlensing."""
import os as _os

# one BLAS thread per process: grids run one process per core; idle OpenBLAS threads spin (perf: ~10 % of CPU)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

try:                                    # numpy may already be imported (env vars then come too late): cap at runtime
    from threadpoolctl import threadpool_limits as _tpl
    _tpl(1)
except ImportError:
    pass


def _numba_cache_dir():
    """Numba invalidates a cached function only when its *own* file changes; cached callers in other modules keep
    stale inlined callees (contour.lc_full inlines nbmag._one: an edit of nbmag.py went unnoticed). So the cache
    lives in a directory keyed by a hash of all package sources. NUMBA_CACHE_DIR set by the user wins."""
    import glob, hashlib, shutil, time
    import numba
    pkg = _os.path.dirname(_os.path.abspath(__file__))
    h = hashlib.sha1(numba.__version__.encode())
    for f in sorted(glob.glob(_os.path.join(pkg, "**", "*.py"), recursive=True)):
        h.update(_os.path.relpath(f, pkg).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    base = _os.path.join(_os.environ.get("XDG_CACHE_HOME", _os.path.expanduser("~/.cache")), "lenscape")
    d = _os.path.join(base, "numba-" + h.hexdigest()[:12])
    try:
        _os.makedirs(d, exist_ok=True)
        _os.utime(d)
        old = sorted(glob.glob(_os.path.join(base, "numba-*")), key=_os.path.getmtime)[:-8]   # keep the 8 newest
        for o in old:
            if time.time() - _os.path.getmtime(o) > 3600:
                shutil.rmtree(o, ignore_errors=True)
    except OSError:
        return
    numba.config.CACHE_DIR = d


if not _os.environ.get("NUMBA_CACHE_DIR"):
    _numba_cache_dir()

__version__ = "0.1.0"

from . import config, data, model, fit, grid, amr, maps, caustics, figures, single, lightcurve, mcmc, source, datasets, eps, optimize, khgrid  # noqa: E402
from .data import Data, read_pysis, read_text                                                              # noqa: E402
from .maps import lattice, refine                                                                          # noqa: E402
from .fit import make_backend, seed_from_data, fit_full, error_factors                                       # noqa: E402
