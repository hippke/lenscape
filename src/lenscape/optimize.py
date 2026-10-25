"""Nelder-Mead minimisation (copied from SciPy, so that lenscape needs no SciPy) and a bisection root finder.

minimize() is SciPy's ``scipy.optimize.minimize(method="Nelder-Mead")`` as of SciPy 1.18.1
(``scipy/optimize/_optimize.py``: ``_minimize_neldermead`` and ``_wrap_scalar_function_maxfun_validation``), with
the code paths lenscape does not use removed (bounds, callback, return_all, disp). The algorithm, its default
simplex, adaptive coefficients (Gao & Han 2012) and stopping rules are unchanged, so results are bit-identical to
SciPy's (tests/test_optimize.py checks this when SciPy is installed).

The copied code is under the SciPy licence:

Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
   following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
   following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
   products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
from numba import njit


class OptimizeResult(dict):
    """Result of minimize(): x, fun, nit, nfev, status, success, message, final_simplex (also as attributes)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class _MaxFuncCallError(RuntimeError):
    pass


_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has been exceeded.',
                   'maxiter': 'Maximum number of iterations has been exceeded.'}


def _wrap_scalar_function_maxfun_validation(function, args, maxfun):
    # wraps a minimizer function to count number of evaluations
    # and to easily provide an args kwd.
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(x, *wrapper_args):
        if ncalls[0] >= maxfun:
            raise _MaxFuncCallError("Too many function calls")
        ncalls[0] += 1
        # A copy of x is sent to the user function (gh13740)
        fx = function(np.copy(x), *(wrapper_args + args))
        # Ideally, we'd like to a have a true scalar returned from f(x). For
        # backwards-compatibility, also allow np.array([1.3]),
        # np.array([[1.3]]) etc.
        if not np.isscalar(fx):
            _dt = getattr(fx, "dtype", np.dtype(np.float64))
            try:
                fx = _dt.type(np.asarray(fx).item())
            except (TypeError, ValueError) as e:
                raise ValueError("The user-provided objective function "
                                 "must return a scalar value.") from e
        return fx

    return ncalls, function_wrapper


def _minimize_neldermead(func, x0, args=(), maxiter=None, maxfev=None, initial_simplex=None, xatol=1e-4, fatol=1e-4,
                         adaptive=False):
    """Minimization of scalar function of one or more variables using the Nelder-Mead algorithm (SciPy).
    maxiter, maxfev: maximum iterations / function evaluations (default N*200 if neither is set);
    initial_simplex: array (N + 1, N), overrides x0; xatol, fatol: absolute tolerances in x and f;
    adaptive: Gao & Han (2012) parameters."""
    maxfun = maxfev

    x0 = np.atleast_1d(x0).flatten()
    dtype = x0.dtype if np.issubdtype(x0.dtype, np.inexact) else np.float64
    x0 = np.asarray(x0, dtype=dtype)

    if adaptive:
        dim = float(len(x0))
        rho = 1
        chi = 1 + 2/dim
        psi = 0.75 - 1/(2*dim)
        sigma = 1 - 1/dim
    else:
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5

    nonzdelt = 0.05
    zdelt = 0.00025

    if initial_simplex is None:
        N = len(x0)

        sim = np.empty((N + 1, N), dtype=x0.dtype)
        sim[0] = x0
        for k in range(N):
            y = np.array(x0, copy=True)
            if y[k] != 0:
                y[k] = (1 + nonzdelt)*y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y
    else:
        sim = np.atleast_2d(initial_simplex).copy()
        dtype = sim.dtype if np.issubdtype(sim.dtype, np.inexact) else np.float64
        sim = np.asarray(sim, dtype=dtype)
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
            raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
        if len(x0) != sim.shape[1]:
            raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
        N = sim.shape[1]

    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N * 200
        maxfun = N * 200
    elif maxiter is None:
        # Convert remaining Nones, to np.inf, unless the other is np.inf, in
        # which case use the default to avoid unbounded iteration
        if maxfun == np.inf:
            maxiter = N * 200
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 200
        else:
            maxfun = np.inf

    one2np1 = list(range(1, N + 1))
    fsim = np.full((N + 1,), np.inf, dtype=float)

    fcalls, func = _wrap_scalar_function_maxfun_validation(func, args, maxfun)

    try:
        for k in range(N + 1):
            fsim[k] = func(sim[k])
    except _MaxFuncCallError:
        pass
    finally:
        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)

    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = np.take(sim, ind, 0)

    iterations = 1

    while (fcalls[0] < maxfun and iterations < maxiter):
        try:
            if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and
                    np.max(np.abs(fsim[0] - fsim[1:])) <= fatol):
                break

            xbar = np.add.reduce(sim[:-1], 0) / N
            xr = (1 + rho) * xbar - rho * sim[-1]
            fxr = func(xr)
            doshrink = 0

            if fxr < fsim[0]:
                xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
                fxe = func(xe)

                if fxe < fxr:
                    sim[-1] = xe
                    fsim[-1] = fxe
                else:
                    sim[-1] = xr
                    fsim[-1] = fxr
            else:  # fsim[0] <= fxr
                if fxr < fsim[-2]:
                    sim[-1] = xr
                    fsim[-1] = fxr
                else:  # fxr >= fsim[-2]
                    # Perform contraction
                    if fxr < fsim[-1]:
                        xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                        fxc = func(xc)

                        if fxc <= fxr:
                            sim[-1] = xc
                            fsim[-1] = fxc
                        else:
                            doshrink = 1
                    else:
                        # Perform an inside contraction
                        xcc = (1 - psi) * xbar + psi * sim[-1]
                        fxcc = func(xcc)

                        if fxcc < fsim[-1]:
                            sim[-1] = xcc
                            fsim[-1] = fxcc
                        else:
                            doshrink = 1

                    if doshrink:
                        for j in one2np1:
                            sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                            fsim[j] = func(sim[j])
            iterations += 1
        except _MaxFuncCallError:
            pass
        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)

    x = sim[0]
    fval = np.min(fsim)
    warnflag = 0

    if fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
    elif iterations >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
    else:
        msg = _status_message['success']

    return OptimizeResult(fun=fval, nit=iterations, nfev=fcalls[0],
                          status=warnflag, success=(warnflag == 0),
                          message=msg, x=x, final_simplex=(sim, fsim))


def minimize(fun, x0, args=(), method="Nelder-Mead", options=None):
    """Drop-in for ``scipy.optimize.minimize(fun, x0, method="Nelder-Mead", options=...)``; options: maxiter,
    maxfev, initial_simplex, xatol, fatol, adaptive."""
    if method.lower() != "nelder-mead":
        raise ValueError("lenscape.optimize.minimize implements only method='Nelder-Mead'")
    x0 = np.atleast_1d(np.asarray(x0))
    if x0.dtype.kind in np.typecodes["AllInteger"]:
        x0 = np.asarray(x0, dtype=float)
    return _minimize_neldermead(fun, x0, args, **(options or {}))


def bisect(f, a, b, xtol=1e-10, maxiter=200):
    """Root of f in [a, b] (f(a) and f(b) of opposite sign) by bisection, to |interval| <= xtol."""
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if np.sign(fa) == np.sign(fb):
        raise ValueError("f(a) and f(b) must have opposite signs")
    for _ in range(maxiter):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0 or (b - a) / 2 < xtol:
            return m
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------------------------------------------------
# The code below is lenscape's own (GPL-3.0-or-later, like the rest of the package), not copied from SciPy.

@njit(cache=True)
def _normal_equations(J, r):
    """J^T J and J^T r without BLAS: a (m x 5)-sized matrix product through OpenBLAS wakes its thread pool, whose
    threads then spin (4 % of the CPU of a map stage in a perf profile when numpy was imported before lenscape)."""
    m, n = J.shape
    JTJ = np.zeros((n, n))
    g = np.zeros(n)
    for i in range(m):
        for a in range(n):
            v = J[i, a]
            g[a] += v * r[i]
            for b in range(a, n):
                JTJ[a, b] += v * J[i, b]
    for a in range(n):
        for b in range(a):
            JTJ[a, b] = JTJ[b, a]
    return JTJ, g


def least_squares_lm(fun, x0, h, cap, maxit=50, tol=1e-3, lam0=1e-3, jac=None):
    """Levenberg-Marquardt for sum(fun(x)**2) (RTModel-style, Bozza 2024): forward-difference Jacobian with steps h
    (larger than the numerical noise of fun, not ~1e-8), damping lam * diag(J^T J), steps scaled down so that no
    |dx_k| exceeds cap_k. Stops when a step gains less than tol or no damping gives a decrease. jac(x) -> (r, J):
    the residual vector and its Jacobian in one call (used instead of the n + 1 evaluations of fun per iteration).
    Returns OptimizeResult(x, fun = sum of squares, nfev, nit)."""
    x = np.array(x0, dtype=float)
    h, cap = np.asarray(h, float), np.asarray(cap, float)
    n = len(x)
    if jac is None:
        r = fun(x)
        J = None
    else:
        r, J = jac(x)
    c = float(r @ r)
    nfev, lam, it = 1, lam0, 0
    for it in range(1, maxit + 1):
        if J is None:
            J = np.empty((len(r), n))
            for k in range(n):
                xk = x.copy()
                xk[k] += h[k]
                J[:, k] = (fun(xk) - r) / h[k]
            nfev += n
        JTJ, g = _normal_equations(np.ascontiguousarray(J), np.ascontiguousarray(r))
        D = np.diag(np.diag(JTJ) + 1e-12)
        gain = None
        while lam < 1e8:
            d = -np.linalg.solve(JTJ + lam * D, g)
            sc = np.max(np.abs(d) / cap)
            if sc > 1:
                d /= sc
            rn = fun(x + d)
            nfev += 1
            cn = float(rn @ rn)
            if cn < c:
                gain = c - cn
                x, r, c = x + d, rn, cn
                lam = max(lam / 5, 1e-7)
                break
            lam *= 4
        if gain is None or gain < tol:
            break
        if jac is None:
            J = None
        else:
            r, J = jac(x)                   # the accepted point's residuals again, with the Jacobian
            c = float(r @ r)
            nfev += 1
    return OptimizeResult(x=x, fun=c, nfev=nfev, nit=it, success=gain is not None, status=0, message="")
