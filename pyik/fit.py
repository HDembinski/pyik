# -*- coding: utf-8 -*-
"""Functions related to fitting and covariance calculation. Uses nlopt library."""
from __future__ import print_function
from six.moves import range
import numpy as np


def covariance(function, vmin, up, fast=False, bounds=None):
    """
    Numerically compute the covariance matrix from a chi^2 or -logLikelihood function.

    Parameters
    ----------
    function: function-like
      The function may accept only a vector argument and has to return a scalar.
    vmin: array of floats
      Position of the minimum.
    up: float
      Threshold value to pass when climbing uphill.
      up = 1   for a chi^2 function
      up = 0.5 for a -logLikelihood function
    fast: boolean
      If true invert hesse matrix at the minimum, use this if computing function is expensive.

    Examples
    --------
    >>> cov = ((2.0,0.2),(0.2,2.0))
    >>> invcov = np.linalg.inv(cov)
    >>> xs = np.array((1.0,-1.0))
    >>> def ChiSquare(pars, grad = None): return np.dot(xs-pars,np.dot(invcov,xs-pars))
    >>> def NegLogLike(pars, grad = None): return 0.5*ChiSquare(pars)
    >>> covariance(ChiSquare, xs, 1.0)
    array([[2. , 0.2],
           [0.2, 2. ]])
    >>> covariance(ChiSquare, xs, 1.0, fast=True)
    array([[2. , 0.2],
           [0.2, 2. ]])
    >>> covariance(NegLogLike, xs, 0.5)
    array([[2. , 0.2],
           [0.2, 2. ]])
    >>> covariance(NegLogLike, xs, 0.5, fast=True)
    array([[2. , 0.2],
           [0.2, 2. ]])

    Notes
    -----
    The algorithm is slow (it needs many function evaluations), but robust.
    The covariance matrix is derived by explicitly following the chi^2
    or -logLikelihood function uphill until it crosses the 1-sigma contour.

    The fast alternative is to invert the hessian matrix at the minimum.
    """

    from scipy.optimize import brentq

    class Func:

        def __init__(self, function, vmin, up):
            self.dir = np.zeros_like(vmin)
            self.up = up
            self.vmin = vmin
            self.fmin = function(vmin)
            self.func = function

        def __call__(self, x):
            return self.func(self.vmin + x * self.dir) - self.fmin - self.up

        def SetDirection(self, i, j):
            self.dir *= 0
            self.dir[abs(i)] = 1 if i >= 0 else -1
            self.dir[abs(j)] = 1 if j >= 0 else -1

        def GetBoundary(self, sign):
            eps = np.sqrt(np.finfo(np.double).eps)
            h = eps
            x0 = abs(np.dot(self.vmin, self.dir))

            def IsNonsense(x):
                return np.isnan(x) or np.isinf(x)

            def x(h):
                return sign * (h * x0 if x0 != 0 else h)

            while True:
                # (1) do smallest possible step first,
                #     then grow exponentially until zero+ is crossed,

                if IsNonsense(x(h)):
                    raise Exception("profile does not cross fmin + up")

                t = self(x(h))

                if IsNonsense(t):
                    # (2) if stepped into nonsense region (infinite, nan, ...),
                    #     do bisection search towards last valid step
                    a = h / 8.0
                    b = h
                    while True:
                        if 2 * (b - a) < eps * (b + a):
                            raise Exception(
                                "profile does not cross fmin + up")
                        h = (a + b) / 2.0
                        t = self(x(h))

                        if IsNonsense(t):
                            b = h
                            continue

                        if t < 0:
                            a = h
                            continue

                        return x(h)

                if t > 0:
                    return x(h)

                h *= 16

    n = len(vmin)

    if fast:
        from pyik.numpyext import hessian
        releps = 1e-3
        dvmin = np.abs(vmin) * releps
        dvmin = vmin * releps
        dvmin[dvmin == 0] = releps
        if bounds is not None:
            m = (vmin - dvmin) < bounds[:, 0]
            dvmin[m] = vmin[m] - bounds[m, 0]
            m = (vmin + dvmin) > bounds[:, 1]
            dvmin[m] = bounds[m, 1] - vmin[m]
        a = hessian(function, vmin, dvmin) / up
    else:
        # Ansatz: (f(r) - fmin)/up = 1/2 r^T C r == 1
        # Diagonal elements:
        # 1 != 1/2 sum_{ij} delta_ik x delta_jk x C_ij
        #    = x^2/2 C_kk
        # => C_kk = 2/x^2
        # Off-diagonal elements:
        # 1 != 1/2 x (delta_ik + delta_il) C_ij x (delta_jk + delta_jl)
        #    = x^2/2 (C_kk + C_kl + C_lk + C_ll) = x^2/2 (2 C_kl + C_kk + C_ll)
        # => C_kl = 0.5 * (2/x^2 - C_kk - C_ll)

        func = Func(function, vmin, up)
        d = np.empty((n, n))
        for i in range(n):
            func.SetDirection(i, i)

            if bounds is None:
                xu = func.GetBoundary(+1)
                t = func(-xu)
                xd = -xu if (t > 0.0 and not np.isinf(t)) \
                    else func.GetBoundary(-1)
            else:
                xd = bounds[i][0]
                xu = bounds[i][1]
                if xd == xu:
                    d[i, i] = 0.0
                    continue

            x1 = +brentq(func, 0, xu)
            x2 = -brentq(func, xd, 0)
            x = 0.5 * (x1 + x2)

            if x < 0:
                raise Exception("x may not be negative")

            d[i, i] = x

        for i in range(n - 1):
            for j in range(i + 1, n):
                func.SetDirection(i, j)

                if (bounds is not None and
                    (bounds[i][0] == bounds[i][1] or
                     bounds[j][0] == bounds[j][1])):
                    d[i, j] = d[j, i] = 0.0
                    continue

                xu = func.GetBoundary(+1)
                t = func(-xu)
                xd = -xu if (t > 0.0 and not np.isinf(t)) \
                    else func.GetBoundary(-1)

                x1 = +brentq(func, 0, xu)
                x2 = -brentq(func, xd, 0)
                x = 0.5 * (x1 + x2)

                if x < 0:
                    raise Exception("x may not be negative")

                # check whether x is in possible range
                a = d[i, i]
                b = d[j, j]
                xmax = np.inf if a <= b else 1.0 / (1.0 / b - 1.0 / a)
                xmin = 1.0 / (1.0 / b + 1.0 / a)

                if x <= xmin:
                    print("covariance(...):", xmin, "<", x, "<", xmax, "violated")
                    x = xmin * 1.01
                if x >= xmax:
                    print("covariance(...):", xmin, "<", x, "<", xmax, "violated")
                    x = xmax * 0.99

                d[i, j] = d[j, i] = x

        a = 2.0 / d ** 2

        for i in range(n - 1):
            for j in range(i + 1, n):
                a[i, j] = a[j, i] = 0.5 * (a[i, j] - a[i, i] - a[j, j])

    # Beware: in case of a chi^2 we calculated
    # t^2 = (d^2 chi^2 / d par^2)^{-1},
    # while s^2 = (1/2 d^2 chi^2 / d par^2)^{-1} is correct,
    # thus s^2 = 2 t^2

    m = np.diag(a) > 0.0
    if np.all(m):
        cov = 2.0 * np.linalg.inv(a)
    else:
        k = a.shape[0]
        n = np.sum(m)
        mm = np.outer(m, m).flatten()
        a1 = a.flatten()[mm]
        cov1 = 2.0 * np.linalg.inv(a1.reshape(n, n))
        cov = np.nan * np.empty(k * k)
        cov[mm] = cov1
        cov = cov.reshape(k, k)

    # first aid, if 1-sigma contour does not look like hyper-ellipsoid
    for i in range(n):
        if cov[i, i] < 0:
            print("covariance(...): error, cov[%i,%i] < 0, returning zero" % (i, i))
            for j in range(n):
                cov[i, j] = 0

    return cov


class Minimizer(object):
    """
    Convenience wrapper for nlopt.

    Notes
    -----
    Implemented methods:
    BOBYQA, SBPLX, PRAXIS, COBYLA, MLSL, DIRECT, DIRECT-L
    """

    @staticmethod
    def GetMethods():
        # first entry is default
        return ("BOBYQA", "SBPLX", "PRAXIS", "COBYLA", "MLSL", "DIRECT", "DIRECT-L")

    def __init__(self):
        self.upper_bounds = None
        self.lower_bounds = None
        self.maxeval = None
        self.ftolabs = 1e-6
        self.ftolrel = 0.0
        self.method = None
        self.stochastic_population = 0
        self.max_time = 0
        self.neval = 0

    def SetMethod(self, method):
        if method not in self.GetMethods():
            raise ValueError("method %s is not recognized" % method)
        self.method = method

    def SetUpperBounds(self, bounds):
        self.upper_bounds = bounds

    def SetLowerBounds(self, bounds):
        self.lower_bounds = bounds

    def SetMaximumEvaluations(self, maxeval):
        self.maxeval = maxeval

    def SetAbsoluteTolerance(self, ftolabs):
        self.ftolabs = ftolabs

    def SetRelativeTolerance(self, ftolrel):
        self.ftolrel = ftolrel

    def SetStochasticPopulation(self, pop):
        self.stochastic_population = pop

    def SetMaximumTime(self, max_time):
        self.max_time = max_time

    def GetNumberOfEvaluations(self):
        return self.neval

    def GetNumberOfFittedParameters(self):
        k = 0
        for i in range(len(self.lower_bounds)):
            if self.lower_bounds[i] < self.upper_bounds[i]:
                k += 1
        return k

    def __call__(self, function, starts):
        import nlopt

        nExt = len(starts)

        fix = []
        iStarts = []
        iLower = []
        iUpper = []
        for ipar in range(nExt):
            s = starts[ipar]
            l = - \
                np.inf if self.lower_bounds is None else self.lower_bounds[ipar]
            u = np.inf if self.upper_bounds is None else self.upper_bounds[ipar]
            if s < l or u < s:
                raise ValueError(
                    "Par %i: starting value %g not in range [%g,%g]" % (ipar, s, l, u))
            if l == u:
                fix.append(ipar)
            else:
                iStarts.append(s)
                iLower.append(l)
                iUpper.append(u)

        iStarts = np.asarray(iStarts)
        iLower = np.array(iLower)
        iUpper = np.array(iUpper)

        def WrappedFunction(iPars, grad=None):
            self.neval += 1
            ePars = np.empty(nExt)
            ipar = 0
            for k in range(nExt):
                if k in fix:
                    ePars[k] = starts[k]
                else:
                    ePars[k] = iPars[ipar]
                    ipar += 1
            return function(ePars)

        def MinimizeWithMethod(method):
            if method == "PRAXIS":
                method = nlopt.LN_PRAXIS
            elif method == "BOBYQA":
                method = nlopt.LN_BOBYQA
            elif method == "COBYLA":
                method = nlopt.LN_COBYLA
            elif method == "SBPLX":
                method = nlopt.LN_SBPLX
            elif method == "MLSL":
                method = nlopt.G_MLSL_LDS, nlopt.LN_BOBYQA
            elif method == "DIRECT":
                method = nlopt.GN_DIRECT
            elif method == "DIRECT-L":
                method = nlopt.GN_DIRECT_L
            npar = len(iStarts)
            if type(method) == tuple:
                opt = nlopt.opt(method[0], npar)
                local_opt = nlopt.opt(method[1], npar)
                local_opt.set_lower_bounds(iLower)
                local_opt.set_upper_bounds(iUpper)
                opt.set_local_optimizer(local_opt)
                opt.set_population(self.stochastic_population)
            else:
                opt = nlopt.opt(method, npar)
            opt.set_min_objective(WrappedFunction)
            opt.set_ftol_abs(self.ftolabs)
            opt.set_ftol_rel(self.ftolrel)
            opt.set_maxtime(self.max_time)
            if self.maxeval is None:
                opt.set_maxeval(1000 + 100 * npar ** 2)
            else:
                opt.set_maxeval(self.maxeval)
            opt.set_lower_bounds(iLower)
            opt.set_upper_bounds(iUpper)
            sqrtdbleps = np.sqrt(np.finfo(np.double).eps)
            opt.set_xtol_rel(sqrtdbleps)
            iresult = opt.optimize(iStarts)
            if np.isinf(opt.last_optimum_value()):
                raise ValueError("got inf")
            if np.isnan(opt.last_optimum_value()):
                raise ValueError("got nan")
            return iresult

        iResult = None
        if self.method is None:
            # try in order...
            for method in ("BOBYQA", "SBPLX", "PRAXIS"):
                try:
                    iResult = MinimizeWithMethod(method)
                    break
                except:
                    import sys
                    etype, e = sys.exc_info()[:2]
                    if etype is KeyboardInterrupt:
                        raise SystemExit("KeyboardInterrupt")
                    else:
                        print("Caught exception %s during method %s: %s, trying next" % (etype, method, e))
            if iResult is None:
                raise
        else:
            iResult = MinimizeWithMethod(self.method)

        result = np.empty(nExt)
        ipar = 0
        for k in range(nExt):
            if k in fix:
                result[k] = starts[k]
            else:
                result[k] = iResult[ipar]
                ipar += 1
        return result


class ChiSquareFunction(object):
    """
    Chi^2 function for fitting a model to data with the least-squares method.

    Parameters
    ----------
    model: function-like
      Model function. Has to be callable with two arguments:
        x, the abscissa value (may be a vector)
        par, the parameter vector
      The function has to be vectorized, see numpy.frompyfunc and numpy.vectorize.
    xs: array of floats
      Array of abscissa values (may be an array of vectors).
    ys: array of floats
      Data array.
    yerrs: array of floats (optional)
      Array with the expected uncertainties of the data values.
      If this is not set, the computed covariance of your fit may be very wrong!
    xerrs: array of floats (optional)
      Array with the expected uncertainties of the abscissa values.
      Be careful when using this. The computation becomes very slow if these
      are defined. If your abscissa values have uncertainties,
      it is better to define a -logLikelihood function for your problem.

    Examples
    --------
    >>> def model(x, par): return par[0]*x
    >>> xs = np.linspace(0,1,5)
    >>> ys = model(xs,[2]) + np.array([ 1.1446054 , -1.84184869,  0.43702669,  0.0513386 , -0.79315476])
    >>> yerrs = np.ones(5)
    >>> par, cov, chi2, ndof = ChiSquareFunction(model, xs, ys, yerrs).Minimize(1)
    >>> print("%.3f +/- %.3f" % (par,np.sqrt(cov)))
    1.468 +/- 0.730
    >>> print("chi2/ndof = %.1f/%i = %.1f" % (chi2, ndof, chi2/ndof))
    chi2/ndof = 5.0/4 = 1.2
    """

    def __init__(self, model, xs, ys, yerrs=None, xerrs=None):
        self.model = model

        xs = np.atleast_1d(xs)
        ys = np.atleast_1d(ys)

        if yerrs is None:
            yerrs = np.ones_like(ys)
            self.guessCovariance = True
        else:
            self.guessCovariance = False
            yerrs = np.atleast_1d(yerrs)
            if yerrs.shape != ys.shape:
                raise ValueError("shapes of ys and yerrs differ")

        # check for invalid yerrs
        mask = yerrs > 0
        nBad = len(mask) - np.sum(mask)
        if nBad > 0:
            print("Warning: %i zeros found in yerrs, corresponding data points will be ignored" % nBad)

        self.ys = ys[mask].flatten()
        self.yerrs2 = yerrs[mask].flatten() ** 2

        if ys.ndim != 1:
            self.xs = np.array([t[mask].flatten() for t in xs])
            self.xerrs = None if xerrs is None else np.array(
                [t[mask].flatten() for t in xerrs])
        else:
            self.xs = xs[mask]
            self.xerrs = None if xerrs is None else xerrs[mask]

    def __call__(self, par, grad=None):

        from pyik.numpyext import derivative

        def parmodel(x):
            return self.model(x, par)

        yerrs2 = self.yerrs2
        if self.xerrs is not None:
            eys = np.frompyfunc(lambda x, xerr: derivative(
                parmodel, x) * xerr, 2, 1)(self.xs, self.xerrs)
            # yerrs2 += eys * eys
            yerrs2 = yerrs2 + eys * eys

        try:
            yms = parmodel(self.xs)
        except TypeError:
            raise TypeError(
                "model is not vectorized, see numpy.frompyfunc and numpy.vectorize")

        result = np.sum((self.ys - yms) ** 2 / yerrs2)

        # workaround for strange bug in sum
        if result.shape:
            return result[0]
        else:
            return result

    def Minimize(self, starts, lower_bounds=None, upper_bounds=None, method=None,
                 absolute_tolerance=1e-6, relative_tolerance=0.0,
                 stochastic_population=0, max_evaluations=None,
                 max_time=0, covarianceMethod="fast"):
        """
        Minimize the chi^2 function.

        Parameters
        ----------
        starts: array of floats or single float
          Array with starting values for the numerical minimizer.
        lower_bounds: array of floats or single float (optional)
          Array with lower bounds on the parameters.
        upper_bounds: array of floats or single float (optional)
          Array with upper bounds on the parameters.
        method: string (optional)
          Minimization method to use, see Minimizer.GetMethods()
          for the available selection.
        absolute_tolerance : float (optional)
          Absolute tolerance used as stopping criterion. If set
          to zero, this criterion will be disabled.
        relative_tolerance : float (optional)
          Relative tolerance used as stopping criterion. If set
          to zero, this criterion will be disabled.
        stochastic_population : int (optional)
          Size of the stochastic population for stochastic search
          algorithms. If set to zero, the internal heuristic
          will be used.
        max_evaluations : int (optional)
          Maximum number of function evaluations used as stopping
          criterion. If set to None, this criterion will be
          disabled.
        covarianceMethod: string
          Either "fast" or "slow". Fast algorithm computes inverse
          of Hesse matrix from numerical derivatives, slow follows
          the chi2 function until it crosses +1.

        Returns
        -------
        par: array of floats
          Best fit of the parameter vector.
        cov: matrix of floats
          Covariance matrix of the solution.
        chi2: float
          chi^2 value at the minimum.
        ndof: float
          Statistical degrees of freedom of the fit.
        """

        # to decide on format of return values in case of 1d fits
        do_squeeze = hasattr(starts, "__len__")

        starts = np.atleast_1d(starts)

        m = Minimizer()

        if method is not None:
            m.SetMethod(method)

        if lower_bounds is not None:
            if np.any(starts < lower_bounds):
                raise ValueError(
                    "A start value is smaller than its lower bound.")
            m.SetLowerBounds(lower_bounds)

        if upper_bounds is not None:
            if np.any(starts > upper_bounds):
                raise ValueError(
                    "A start value is larger than its upper bound.")
            m.SetUpperBounds(upper_bounds)

        m.SetAbsoluteTolerance(absolute_tolerance)
        m.SetRelativeTolerance(relative_tolerance)
        m.SetStochasticPopulation(stochastic_population)
        m.SetMaximumEvaluations(max_evaluations)
        m.SetMaximumTime(max_time)

        pars = m(self, starts)
        chi2 = self(pars)
        ndof = len(self.ys) - len(starts)

        if covarianceMethod is None:

            if do_squeeze:
                pars = np.squeeze(pars)
            return pars, chi2, ndof

        else:
            useFastMethod = covarianceMethod == "fast"

            up = 1.0
            if self.guessCovariance:
                # this assumes equal errors on all data points and chi2 = ndof
                up = chi2 / ndof
                chi2 = ndof
            cov = covariance(self, pars, up, fast=useFastMethod)
            if do_squeeze:
                pars, cov = np.squeeze(pars), np.squeeze(cov)
            return pars, cov, chi2, ndof
