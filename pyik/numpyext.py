# -*- coding: utf-8 -*-
"""Contains extensions to numpy."""
from six.moves import range
import numpy as np


def linear_least_squares_fit(model, npar, x, y, yerr=None):
    """
    Fits a model that is linear in the parameters.

    Parameters
    ----------
    model: vectorized function, args=(x, par)
    npar: number of parameters for model (length of par vector)
    x, y, yerr: coordinates and errors of data points

    Returns
    -------
    x: best-fit vector of parameters
    cov: covariance matrix of parameters
    chi2: chi2 at minimum
    ndof: statistical degrees of freedom
    """

    if yerr is None:
        b = np.atleast_1d(y)
        X = np.transpose([model(x, u) for u in np.identity(npar)])
    else:
        ye = np.atleast_1d(yerr)
        b = np.atleast_1d(y) / ye
        X = np.transpose([model(x, u) / ye for u in np.identity(npar)])

    XTX_inv = np.linalg.inv(np.dot(X.T, X))
    x = np.dot(np.dot(XTX_inv, X.T), b)
    chi2 = np.sum((b - np.dot(X, x))**2)
    ndof = len(y) - npar
    return x, XTX_inv, chi2, ndof


def rebin(factor, w, edges=None, axis=0):
    """
    Re-bins a N-dimensional histogram along a chosen axis.

    Parameters
    ----------
    factor: integer
      Number of neighboring bins to merge. Number of original
      bins must be divisible by factor.
    w: array-like
      Number field that represents the histogram content.
    edges: array-like (optional)
      Bin edges of the axis to re-bin.
    axis: integer (optional)
      Axis to re-bin, defaults to first axis.

    Returns
    -------
    w: array
      Number field that represents the re-binned histogram
      content.
    edges: array (only if edges were supplied)
      Bin edges after re-binning.
    """

    w = np.atleast_1d(w)
    nbin = w.shape[axis]
    if nbin % factor != 0:
        raise ValueError("factor %i is not a divider of %i bins" % (factor, nbin))
    n = nbin / factor
    shape = np.array(w.shape)
    shape[axis] = n
    w2 = np.zeros(shape, dtype=w.dtype)
    for i in range(factor):
        mask = [slice(x) for x in shape]
        mask[axis] = slice(i, nbin, factor)
        w2 += w[mask]

    if edges is not None:
        edges2 = [edges[factor * i] for i in range(n)] + [edges[-1]]
        return w2, edges2
    else:
        return w2


def bin(x, y, bins=10, range=None):
    """
    Bin x and returns lists of the y-values inside each bin.

    Parameters
    ----------
    x: array-like
      Variable that is binned.
    y: array-like
      Variable that is sorted according to the binning of x.
    bins: integer or array-like
      Number of bins or array of lower bin edges + last high bin edge.
    range: tuple, lenght of 2 (optional)
      If range is set, only (x,y) pairs are used where x is inside the range.
      Ignored, if bins is an array.

    Returns
    -------
    yBins: list of lists
      List of y-values which correspond to the x-bins.
    xegdes: array of floats
      Lower bin edges. Has length len(yBins)+1.
    """

    ys = np.atleast_1d(y)
    xs = np.atleast_1d(x)

    if type(bins) is int:
        if range is None:
            range = (min(x), max(x) + np.finfo(float).eps)
        else:
            mask = (range[0] <= xs) & (xs < range[1])
            xs = xs[mask]
            ys = ys[mask]
        xedges = np.linspace(range[0], range[1], bins + 1)
    else:
        xedges = bins
        bins = len(xedges) - 1

    binnedys = []
    for i in range(bins):

        if i == bins - 1:
            binnedys.append(ys[(xedges[i] <= xs) & (xs <= xedges[i + 1])])
        else:
            binnedys.append(ys[(xedges[i] <= xs) & (xs < xedges[i + 1])])

    return binnedys, xedges


def profile(x, y, bins=10, range=None, sigma_cut=None):
    """
    Compute the (robust) profile of a set of data points.

    Parameters
    ----------
    x,y : array-like
      Input data. The (x,y) pairs are binned according to the x-array,
      while the averages are computed from the y-values inside a x-bin.
    bins : int or array-like, optional
      Defines the number of equal width bins in the given range (10,
      by default). If bins is an array, it is used for the bin edges
      of the profile.
    range : (float,float), optional
      The lower and upper range of the bins. If not provided, range
      is simply ``(a.min(), a.max())``. Values outside the range are
      ignored.
    sigma_cut : float, optional
      If sigma_cut is set, outliers in the data are rejected before
      computing the profile. Outliers are detected based on the scaled
      MAD and the median of the distribution of the y's in each bin.
      All data points with |y - median| > sigma_cut x MAD are ignored
      in the computation.

    Returns
    -------
    yavg : array of dtype float
      Returns the averages of the y-values in each bin.
    ystd : array of dtype float
      Returns the standard deviation in each bin. If you want the
      uncertainty of ymean, calculate: yunc = ystd/numpy.sqrt(n-1).
    n : array of dtype int
      Returns the number of events in each bin.
    xedge : array of dtype float
      Returns the bin edges. Beware: it has length(yavg)+1.

    Examples
    --------
    >>> yavg, ystd, n, xedge = profile(np.array([0.,1.,2.,3.]), np.array([0.,1.,2.,3.]), 2)
    >>> yavg
    array([0.5, 2.5])
    >>> ystd
    array([0.5, 0.5])
    >>> n
    array([2, 2])
    >>> xedge
    array([0. , 1.5, 3. ])
    """

    y = np.asfarray(np.atleast_1d(y))

    n, xedge = np.histogram(x, bins=bins, range=range)

    if sigma_cut is None:
        ysum = np.histogram(x, bins=bins, range=range, weights=y)[0]
        yysum = np.histogram(x, bins=bins, range=range, weights=y * y)[0]
    else:
        if sigma_cut <= 0:
            raise ValueError("sigma_cut <= 0 detected, has to be positive")
        # sort y into bins
        ybin = bin(x, y, bins, range)[0]

        if type(bins) is int:
            nbins = bins
        else:
            nbins = len(bins) - 1

        # reject outliers in calculation of avg, std
        ysum = np.zeros(nbins)
        yysum = np.zeros(nbins)
        for i in range(nbins):
            ymed = np.median(ybin[i])
            ymad = mad(ybin[i])
            for y in ybin[i]:
                if ymad == 0 or abs(y - ymed) < sigma_cut * ymad:
                    ysum[i] += y
                    yysum[i] += y * y
                else:
                    n[i] -= 1

    mask = n == 0
    n[mask] = 1
    yavg = ysum / n
    ystd = np.sqrt(yysum / n - yavg * yavg)
    yavg[mask] = np.nan
    ystd[mask] = np.nan

    return yavg, ystd, n, xedge


def profile2d(x, y, z, bins=(10, 10), range=None):
    """
    Compute the profile of a set of data points in 2d.
    """

    if not isinstance(z, np.ndarray):
        z = np.array(z)

    ws, xedges, yedges = np.histogram2d(x, y, bins, range)

    zsums = np.histogram2d(x, y, bins, range, weights=z)[0]
    zzsums = np.histogram2d(x, y, bins, range, weights=z * z)[0]

    zavgs = zsums / ws
    zstds = np.sqrt(zzsums / ws - zavgs * zavgs)

    return zavgs, zstds, ws, xedges, yedges


def centers(x):
    """
    Compute the centers of an array of bin edges.

    Parameters
    ----------
    x: array-like
      A 1-d array containing lower bin edges.

    Returns
    -------
    c: array of dtype float
      Returns the centers of the bins.
    hw: array of dtype float
      Returns the half-width of the bins.

    Examples
    --------
    >>> centers([0.0, 1.0, 2.0])
    (array([0.5, 1.5]), array([0.5, 0.5]))
    """

    x = np.atleast_1d(x)
    assert len(x) > 1, "Array should have size > 1 to make call to centers() reasonable!"
    hw = 0.5 * (x[1:] - x[:-1])
    return x[:-1] + hw, hw


def derivative(f, x, step=None, order=1):
    """
    Numerically calculate the first or second derivative of a function.

    Parameters
    ----------
    f: function-like
      Function of which to calculate the derivative.
      It has to accept a single float argument and may return a vector or a float.
    x: float
      Where to evaluate the derivative.
    step: float (optional)
      By default, the step size for the numerical derivative is calculated
      automatically. This may take many more evaluations of f(x) than necessary.
      The calculation can be speed up by setting the step size.
    order: integer (optional)
      Order of the derivative. May be 1 or 2 for the first or second derivative.

    Returns
    -------
    The first or second derivative of f(x).

    Notes
    -----
    Numerically calculated derivatives are not exact and we do not give an error
    estimate.

    Examples
    --------
    >>> def f(x) : return 2 + x + 2*x*x + x*x*x
    >>> round(derivative(f, 1.0), 3)
    8
    >>> round(derivative(f, 1.0, step=1e-3), 3)
    8
    >>> round(derivative(f, 1.0, order=2), 3)
    10
    >>> np.round(derivative(f, np.ones(2)), 3)
    array([8., 8.])
    >>> np.round(derivative(f, np.ones(2), order=2), 3)
    array([10., 10.])

    Notes
    -----
    The first derivative is calculated with the five point stencil,
    see e.g. Wikipedia. The code to determine the step size was taken
    from the GNU scientific library.
    """

    eps = np.finfo(float).eps
    # the correct power is 1/order of h in the
    # error term of the numerical formula
    h0 = h = eps ** 0.33 if order == 1 else eps ** 0.25

    userStep = step is not None
    for i in range(10):
        dx = step if userStep else (h * x if np.all(x) else h)
        tmp = x + dx
        dx = tmp - x

        fpp = f(x + 2.0 * dx)
        fp = f(x + dx)
        fm = f(x - dx)
        fmm = f(x - 2.0 * dx)

        if userStep:
            break

        if order == 1:
            a = np.abs(fpp - fp)
            b = np.abs(fpp + fp)
            if np.all(a > 0.5 * b * h0):
                break
        else:
            a = np.abs(fpp + fmm - fp - fm)
            b = np.abs(fpp + fmm + fp + fm)
            if np.all(a > 0.5 * b * h0):
                break

        h *= 10

    if order == 1:
        return (fmm - fpp + 8.0 * (fp - fm)) / (12.0 * dx)
    else:
        return (fpp + fmm - fp - fm) / (3.0 * dx * dx)


def derivativeND(f, xs, step=1e-8):
    """
    Numerically calculates the first derivatives of an R^n -> R function.

    The derivatives can be calculated at several points at once.

    Parameters
    ----------
    f : callable
      An R^n -> R function to differentiate. Has to be callable with
      f(xs), where xs is a 2-d array of shape n_points x n_variables.
    xs : array-like
      A 2-d array of function values of shape n_points x n_variables.
    step : float
      Step size for the differentiation.

    Notes
    -----
    The derivatives are calculated using the central finite difference
    method with 2nd order accuracy (i.e., a two point stencil) for each
    dimension.

    Returns
    -------
    A 2-d array of the derivatives for each point and dimension. The
    shape is n_points x n_variables.

    Examples
    --------
    >>> def f(xy):
    ...     x, y = xy.T
    ...     return x ** 2 + y ** 2
    ...
    >>> derivativeND(f, ([0., 0.], [1., 0.], [0., 1.]))
    array([[0., 0.],
           [2., 0.],
           [0., 2.]])
    """
    xs = np.atleast_2d(xs)

    n_rows, n_vars = xs.shape

    bloated_xs = np.repeat(xs, n_vars, 0)
    epsilons = np.tile(np.eye(n_vars) * step, [n_rows, 1])

    return (f(bloated_xs + epsilons) -
            f(bloated_xs - epsilons)).reshape(-1, n_vars) / (2 * step)


def jacobian(f, x, steps=None):
    """
    Numerically calculate the matrix of first derivatives.

    Parameters
    ----------
    f: function-like
      Has to be callable as f(x).
    x: array-like
      Vector of parameters.
    steps: array-like (optional)
      Vector of deltas to use in the numerical approximation,
      see derivative(...). Has to have the same length as x.

    Returns
    -------
    The Jacobi matrix of the first derivatives.

    Examples
    --------
    >>> def f(v): return 0.5*np.dot(v,v)
    >>> jacobian(f,np.ones(2))
    array([[1., 1.]])
    >>> def f(v): return np.dot(v,v)*v
    >>> jacobian(f,np.ones(2))
    array([[4., 2.],
           [2., 4.]])
    """

    nx = len(x)

    # cheap way to determine dimension of f's output
    y = f(x)
    ny = len(y) if hasattr(y, "__len__") else 1

    jacobi = np.zeros((ny, nx))

    e = np.zeros(nx)

    for ix in range(nx):
        e *= 0
        e[ix] = 1

        der = derivative(lambda z: f(x + z * e), 0,
                         step=None if steps is None else steps[ix])

        for iy in range(ny):
            jacobi[iy, ix] = der[iy] if ny > 1 else der

    return jacobi


def hessian(f, x, steps):
    """
    Numerically calculate the matrix of second derivatives.

    Parameters
    ----------
    f: function-like
      Has to be callable as f(x).
    x: array-like
      Vector of parameters.
    steps: array-like
      Vector of deltas to use in the numerical approximation.
      Has to have the same length as x.

    Returns
    -------
    The symmetric Hesse matrix of the second derivatives.
    """

    xx = np.array(x, dtype=np.float)

    n = len(x)
    hesse = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            xpp = xx.copy()
            xpp[i] += steps[i]
            xpp[j] += steps[j]

            xmm = xx.copy()
            xmm[i] -= steps[i]
            xmm[j] -= steps[j]

            if i == j:
                xm = xx.copy()
                xm[i] -= steps[i]

                xp = xx.copy()
                xp[i] += steps[i]

                hesse[i, i] = ((f(xmm) + f(xpp) - f(xp) - f(xm))
                               / (3.0 * steps[i] * steps[i]))

            else:
                xpm = xx.copy()
                xpm[i] += steps[i]
                xpm[j] -= steps[j]

                xmp = xx.copy()
                xmp[i] -= steps[i]
                xmp[j] += steps[j]

                hesse[i, j] = hesse[j, i] = (
                    f(xpp) + f(xmm) - f(xpm) - f(xmp)) / (4.0 * steps[i] * steps[j])

    return hesse


def propagate_covariance(f, x, cov):
    """
    Compute the covariance matrix of y for the transformation y = f(x), given x with covariance matrix cov.

    Parameters
    ----------
    f: function-like
      Has to be callable as f(x).
    x: array-like
      Vector of parameters.
    cov: 2-d array of floats
      Covariance matrix of x.

    Returns
    -------
    fcov: matrix of floats
      The covariance matrix of the output of f.

    Examples
    --------
    >>> v = np.ones(2)
    >>> cov = np.ones((2,2))
    >>> def f(r):return np.dot(r,r)
    >>> "%.3g" % propagate_covariance(f,v,cov)
    '16'
    >>> def f(r):return 2*r
    >>> propagate_covariance(f,v,cov)
    array([[4., 4.],
           [4., 4.]])
    """

    ncol = len(x)
    dx = np.empty(ncol)
    for icol in range(ncol):
        dx[icol] = (np.sqrt(cov[icol][icol]) if cov[icol][icol] > 0.0 else 1.0) * 1e-3

    jacobi = jacobian(f, x, dx)

    return np.dot(jacobi, np.dot(cov, jacobi.T))


def uncertainty(f, x, cov):
    """
    Compute the standard deviation of f(v), given v with covariance matrix cov.

    This is a convenience function that wraps propagate_covariance(...).

    Parameters
    ----------
    f: function-like
      Has to be callable as f(x).
    x: array-like or single float
      Vector of parameters.
    cov: 2-d array of floats or single float
      Covariance matrix of x.

    Returns
    -------
    The standard deviation of f(x).

    Examples
    --------
    >>> def f(r):return np.dot(r,r)
    >>> v = np.ones(2)
    >>> cov = np.ones((2,2))
    >>> "%.3g" % uncertainty(f,v,cov)
    '4'
    """

    prop_cov = propagate_covariance(f, np.atleast_1d(x), np.atleast_2d(cov))
    return np.sqrt(prop_cov[0, 0])


def quantiles(ds, qs, weights=None):
    """
    Compute the quantiles qs of 1-d ds with possible weights.

    Parameters
    ----------
    ds : ds to calculate quantiles from
      1-d array of numbers

    qs : 1-d array of quantiles

    weights : 1-d array of weights, optional (default: None)
      Is expected to correspond point-to-point to values in ds

    Returns
    -------
    quantiles of ds corresponding to qs
      1-d array of equal length to qs
    """

    if weights is None:
        from scipy.stats.mstats import mquantiles
        return mquantiles(ds, qs)
    else:
        ds = np.atleast_1d(ds)
        qs = np.atleast_1d(qs)
        weights = np.atleast_1d(weights)

        assert len(ds) == len(
            weights), "Data and weights arrays need to have equal length!"
        assert np.all((qs >= 0) & (qs <= 1)
                      ), "Quantiles need to be within 0 and 1!"
        assert np.all(weights > 0), "Each weight must be > 0!"

        m_sort = np.argsort(ds)
        ds_sort = ds[m_sort]
        ws_sort = weights[m_sort]

        ps = (np.cumsum(ws_sort) - 0.5 * ws_sort) / np.sum(ws_sort)
        return np.interp(qs, ps, ds_sort)


def median(a, weights=None, axis=0):
    """
    Compute the median of data in a with optional weights.

    Parameters
    ----------
    a : data to calculate median from
      n-d array of numbers

    weights : weights of equal shape to a
      n-d array of numbers

    axis : axis to calculate median over (optional, default: 0)
      To note, weighted calculation does currently only support 1-d arrays

    Returns
    -------
    Median: float or 1-d array of floats
    """

    a = np.atleast_1d(a)
    if weights is None:
        return np.median(a, axis=axis)
    else:
        assert a.ndim == 1, "Only 1-d calculation of weighted median is currently supported!"
        return quantiles(a, 0.5, weights)[0]


def mad(a, weights=None, axis=0):
    """
    Calculate the scaled median absolute deviation of a random distribution.

    Parameters
    ----------
    a : array-like
      1-d or 2-d array of random numbers.

    weights : array-like
      Weights corresponding to data in a.
      Calculation with weights is currently only supported for 1-d data.

    Returns
    -------
    mad : float or 1-d array of floats
      Scaled median absolute deviation of input sample. The scaling factor
      is chosen such that the MAD estimates the standard deviation of a
      normal distribution.

    Notes
    -----
    The MAD is a robust estimate of the true standard deviation of a random
    sample. It is robust in the sense that its output is not sensitive to
    outliers.

    The standard deviation is usually estimated by the square root of
    the sample variance. Note, that just one value in the sample has to be
    infinite for the sample variance to be also infinite. The MAD still
    provides the desired answer in such a case.

    In general, the sample variance is very sensitive to the tails of the
    distribution and will give undesired results if the sample distribution
    deviates even slightly from a true normal distribution. Many real world
    distributions are not exactly normal, so this is a serious issue.
    Fortunately, this is not the case for the MAD.

    Of course there is a price to pay for these nice features. If the sample is
    drawn from a normal distribution, the sample variance is the more
    efficient estimate of the true width of the Gaussian, i.e. its
    statistical uncertainty is smaller than that of the MAD.

    Examples
    --------
    >>> a = [1.,0.,5.,4.,2.,3.,1e99]
    >>> round(mad(a), 3)
    2.965
    """

    const = 1.482602218505602  # 1.0/inverse_cdf(3/4) of normal distribution
    med = median(a, weights=weights, axis=axis)

    if axis == 0:
        absdevs = np.absolute(a - med)
    elif axis == 1:
        absdevs = np.absolute(a.T - med).T

    return const * median(absdevs, weights=weights, axis=axis)


class ConvexHull:
    """
    Calculate the (fractional) convex hull of a point cloud in 2-d.

    Parameters
    ----------
    x: 1-d array
      vector of parameters

    y: 1-d array
      vector of parameters

    frac: int
      fraction of points contained in convex hull, default is 1.0

    byprob: boolean
      if false and frac < 1.0, will remove points contained in hull shape
      if true and frac < 1.0, will remove least probable point based on kde estimate

    Returns
    -------
    points: 2-d array of floats
      remaining points to analyses hull object

    hull: object generated by scipy.spatial.qhull.ConvexHull
      contains information of ConvexHull


    Notes
    -----
    A convex hull can be thought of as a rubber band put around the point cloud.
    To plot a closed object, use the simplices contained in "hull".

    Examples
    --------
    >>> m1 = [-0.9, -0.1, -0.0, 0.7, 1.3, 0.4, 0.6, -1.9, 0.2, -1.1]
    >>> m2 = [ 0.1, 0.7, -0.9, -0.1, -0.5, -0.7, -0.9, -0.2, -0.2, -0.5]
    >>> hull = ConvexHull(m1, m2)
    >>> points, hull = hull()

    Plot the hull:
    for simplex in hull.simplices:
       plt.plot(points[simplex, 0], points[simplex, 1], 'k--')
    """

    def __init__(self, x, y, frac=1.0, byprob=True):
        from scipy.stats import gaussian_kde

        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.frac = frac
        self.remove = byprob

        data = np.vstack([self.x, self.y])
        self.kernel = gaussian_kde(data)

    def __call__(self):
        return self.fractionalHull()

    def convexHull(self, pos):
        from scipy.spatial import ConvexHull
        return ConvexHull(pos)

    def removal(self, pos, bound):
        x = np.array([p[0] for p in pos])
        y = np.array([p[1] for p in pos])
        for b in range(len(bound)):
            px = np.where(x == bound[b][0])
            py = np.where(y == bound[b][1])
            if px == py:
                x = np.delete(x, px)
                y = np.delete(y, px)
        return x, y

    def removeByProb(self, pos, bound):
        boundary = np.vstack([bound[:, 0], bound[:, 1]])
        prob = self.kernel(boundary)

        index = prob.argsort()
        prob = prob[index]
        boundary = bound[index]

        return self.removal(pos, [boundary[0]])

    def removePoints(self, pos):
        hull = self.convexHull(pos)
        boundary = np.dstack((pos[hull.vertices, 0], pos[hull.vertices, 1]))[0]

        if not self.remove:
            x, y = self.removal(pos, boundary)
        if self.remove:
            x, y = self.removeByProb(pos, boundary)

        points = np.dstack((x, y))[0]
        hull = self.convexHull(points)
        return points, hull

    def fractionalHull(self):
        points = np.dstack((self.x.copy(), self.y.copy()))[0]
        n = self.frac * len(points)
        if self.frac == 1:
            hull = self.convexHull(points)
        else:
            while len(points) > n:
                points, hull = self.removePoints(points)
        # boundary = np.dstack((points[hull.vertices,0], points[hull.vertices,1]))[0]
        return points, hull


def bootstrap(function, x, r=1000):
    """
    Generate r balanced bootstrap replicas of x and returns the results of a statistical function on them.

    Notes
    -----
    The bootstrap is a non-parametric method to obtain the statistical bias
    and variance of a statistical estimate. In general, the result is
    approximate. You should only use this if you have no idea of the
    theoretical form of the underlying p.d.f. from which the data are drawn.
    Otherwise you should draw samples from that p.d.f., which may be fitted to
    the data.

    To obtain good results, r has to be in the range of 200 to 1000. As with
    every simulation technique, the precision of the result is proportional to
    r^(-1/2).

    Parameters
    ----------
    function: callable
      The statistical function. It has to accept an array of the type of x and may
      return a float or another array.
    x: array-like
      The original input data for the statistical function.
    r: int
      Number of bootstrap replicas.

    Returns
    -------
    Array of results of statFunc.
    """

    n = np.alen(x)
    xx = np.array(x)
    iB = np.array(np.random.permutation(n * r) % n)
    xbGen = (xx[iB[ir * n:(ir + 1) * n]] for ir in range(r))
    ybs = map(function, xbGen)
    return np.array(ybs)


def bootstrap_confidence_interval(statfunc, x, coverage=0.68, replicas=1000):
    """
    Calculate the bootstrap confidence interval of the result of a statistical function.

    Notes
    -----
    See remarks of BootstrapReplicas.

    Parameters
    ----------
    statfunc: callable
      The statistical function. It has to accept an array of the type of x and may
      return a float or another array.
    x: array-like
      The original input data for the statistical function.
    coverage: float
      Fraction of bootstrap replicas inside the interval.
    replicas: integer
      Number of bootstrap replicas (defines accuracy of interval)

    Returns
    -------
    v,dv-,dv+ : floats or arrays of floats
      statfunc(x), downward uncertainty interval, upward uncertainty interval
    """

    if len(x) == 0:
        return 0, 0, 0

    r = int(round(replicas / 200.0)) * 200  # has to be multiple of 200
    q = int(round(r * coverage))
    qA = (r - q) / 2
    qB = r - qA

    t = statfunc(x)
    tB = np.sort(bootstrap(statfunc, x, r), axis=0)

    return t, t - tB[qA], tB[qB] - t


def bootstrap_covariance(statfunc, x, r=1000):
    """
    Calculate the uncertainty of statfunc over data set x with a balanced bootstrap.

    Notes
    -----
    See remarks of BootstrapReplicas.

    Parameters
    ----------
    statfunc: callable
      The statistical function. It has to be callable as statfunc(x)
      and may return a float or another array.
    x: array-like
      The original input data for the statistical function.

    Returns
    -------
    The covariance matrix of the result of statfunc.
    """

    return np.cov(bootstrap(statfunc, x, r))


def binomial_proportion(nsel, ntot, coverage=0.68):
    """
    Calculate a binomial proportion (e.g. efficiency of a selection) and its confidence interval.

    Parameters
    ----------
    nsel: array-like
      Number of selected events.
    ntot: array-like
      Total number of events.
    coverage: float (optional)
      Requested fractional coverage of interval (default: 0.68).

    Returns
    -------
    p: array of dtype float
      Binomial fraction.
    dpl: array of dtype float
      Lower uncertainty delta (p - pLow).
    dpu: array of dtype float
      Upper uncertainty delta (pUp - p).

    Examples
    --------
    >>> p, dpl, dpu = binomial_proportion(50,100,0.68)
    >>> round(p, 3)
    0.5
    >>> round(dpl, 3)
    0.049
    >>> round(dpu, 3)
    0.049
    >>> abs(np.sqrt(0.5*(1.0-0.5)/100.0)-0.5*(dpl+dpu)) < 1e-3
    True

    Notes
    -----
    The confidence interval is approximate and uses the score method
    of Wilson. It is based on the log-likelihood profile and can
    undercover the true interval, but the coverage is on average
    closer to the nominal coverage than the exact Clopper-Pearson
    interval. It is impossible to achieve perfect nominal coverage
    as a consequence of the discreteness of the data.
    """

    from scipy.stats import norm

    z = norm().ppf(0.5 + 0.5 * coverage)
    z2 = z * z
    p = np.asarray(nsel, dtype=np.float) / ntot
    div = 1.0 + z2 / ntot
    pm = (p + z2 / (2 * ntot))
    dp = z * np.sqrt(p * (1.0 - p) / ntot + z2 / (4 * ntot * ntot))
    pl = (pm - dp) / div
    pu = (pm + dp) / div

    return p, p - pl, pu - p


def poisson_uncertainty(x):
    """
    Return "exact" confidence intervals, assuming a Poisson distribution for k.

    Notes
    -----
    Exact confidence intervals from the Neyman construction tend to overcover
    discrete distributions like the Poisson and Binomial distributions. This
    is due to the discreteness of the observable and cannot be avoided.

    Parameters
    ----------
    x: array-like or single integer
      Observed number of events.

    Returns
    -------
    A tuple containing the uncertainty deltas or an array of such tuples.
    Order: (low, up).
    """

    from scipy.stats import chi2

    x = np.atleast_1d(x)
    r = np.empty((2, len(x)))
    r[0] = x - chi2.ppf(0.16, 2 * x) / 2
    r[1] = chi2.ppf(0.84, 2 * (x + 1)) / 2 - x
    return r


def azip(*args):
    """Convenience wrapper for numpy.column_stack."""
    return np.column_stack(args)


def IsInUncertaintyEllipse(point, center, covariance, alpha=0.68):
    """Test whether a point is inside the hypervolume defined by a covariance matrix.

    Parameters
    ----------
    point: array of floats
      Point to test.
    center: array of floats
      Center of the hypervolume.
    covariance: 2d array of floats
      Covariance matrix that defines the confidence interval.
    alpha: float (optional)
      Requested coverage of the hypervolume.

    Returns
    -------
    True if point is covered and False otherwise.
    """

    from scipy.stats import chi2
    w, u = np.linalg.eig(covariance)
    x = np.dot(u.T, point - center) / np.sqrt(w)
    return np.sum(x * x) <= chi2(len(center)).ppf(alpha)


def LOOCV(function, xs, ys, estimates, xerrs=None, yerrs=None):
    """
    Performs a Leave-one-out cross-validation of the prediction power of function assuming normally distributed values.

    Parameters
    ----------
    function: callable function f(xs,pars)
        Function which is evaluated at xs to predict ys.
        Fit parameters are expected as a second argument.
    xs: array-like
        Function variable
    ys: array-like
        ys = function(xs,pars)
    estimates: array-like
        Estimates of the optimized parameters of function(xs,pars).
        At least provide np.ones(n) or np.zeros(n) according to the number of parameters.

    Returns
    -------
    LOOCV: The LOOCV value being proportional to bias^2 + variance.
        The prediction power of function is proportional to -LOOCV.
    """

    xs = np.atleast_1d(xs)
    ys = np.atleast_1d(ys)

    from pyik.fit import ChiSquareFunction

    # wrapper to np.delete if arr might be None
    def DelIfNotNone(arr, i):
        return None if arr is None else np.delete(arr, i)

    loocv = 0.
    for i in range(len(xs)):

        # fitting function to all points except for the (i+1)th
        pars_i = ChiSquareFunction(function, np.delete(xs, i), np.delete(ys, i),
                                   xerrs=DelIfNotNone(xerrs, i),
                                   yerrs=DelIfNotNone(yerrs, i)).Minimize(starts=estimates)[0]

        # estimating residual at left-out point
        loocv += (ys[i] - function(xs[i], pars_i)) ** 2

    return loocv


class FeldmanCousins(object):
    """
    A convenience class to calculate confidence intervals using a unified frequentistic approach developed by Feldman & Cousins.

    In particular, the method yields valid results when there are (physical) constraints on the parameters of the assumed pdf.
    Application example: Estimation of upper limits for empty bins of an energy spectrum measurement.

    Notes
    -----
    The method is described in detail in arXiv:physics/9711021v2.
    The confidence intervals are created with the Neyman construction using a ranking according to likelihood-ratios.
    Undercoverage resulting from decision-biased choices of the confidence interval after looking at data,
    known as flip-flopping, or empty confidence intervals in forbidden parameter regions are avoided.

    The standard constructor declares a Poisson distribution. To manually change the distribution, use SetCustomPdf().
    SetNormalPdf() and SetLogNormalPdf() can be used to use the respective pdfs.
    Note that the parameter and variable ranges need to be carefully adjusted in these cases.

    In the case of discrete distributions e.g. Poisson, the confidence intervals will
    overcover by construction due to the discreteness of the random variable.

    Parameters
    ----------
    cl: float between 0 and 1
        Desired coverage of the constructed confidence intervals.

    Optional parameters
    -------------------
    nbg: float
        Mean expectation of background events for the Poisson distribution.
    murange: array-like
        Lower and upper limits of the parameter range.
        In any case, the upper parameter limit
        needs to be well above the observation x for which the interval is evaluated.
    xvrange:
        Lower und upper limits of the variable range.
    mustep:
        Step size in the constructed (true) parameter space.
        Smaller values will result in more accurate results but also in a significant increase in computing time.

    Example
    -------
    >>> fc=FeldmanCousins(0.95)
    >>> np.round(fc.FCLimits(0.), 3)
    array([0.   , 3.095])
    >>> np.round(fc.FCLimits(1.), 3)
    array([0.055, 5.145])
    """

    def __init__(self, cl, nbg=0., murange=None, xvrange=None, mustep=0.005):
        from scipy.stats import poisson
        from pyik.fit import Minimizer

        self._m = Minimizer()
        self._cl = cl
        # strictly, the poisson pmf doesn't exist for mu+bg=0, but the confidence
        # interval still exists in the limit bg -> 0.
        self._bg = max(nbg, 1e-10)
        self._pdf = lambda k, mu: poisson.pmf(k, mu + self._bg)
        self._murange = [0., 100.]
        self._xrange = self._murange
        self._mustep = mustep
        self._discrete_pdf = True
        self._pdftypes = ["poisson", "normal", "lognormal", "custom"]
        self._pdftype = self._pdftypes[0]
        if murange is not None:
            self.SetParameterBounds(murange)
        if xvrange is not None:
            self.SetVariableBounds(xvrange)

    def SetParameterBounds(self, bounds):
        """Define the parameter limits."""
        self._murange = bounds

    def SetVariableBounds(self, bounds):
        """Define the variable limits."""
        self._xrange = bounds

    def SetCustomPDF(self, pdf, murange, xvrange, discrete=False):
        """
        Declare a custom probability distribution function.

        Parameters
        ----------
        pdf: function-like
            The custom pdf. Is supposed to accept variable (observable) as first argument
            and parameter as second argument.
        murange: array-like
            The parameter range.
        xvrange: array-like
            The observable range.
        discrete (optional): boolean
            Declare whether discrete or continuous variables are expected.
        """
        self._pdf = pdf
        self.SetParameterBounds(murange)
        self.SetVariableBounds(xvrange)
        self._discrete_pdf = discrete
        self._pdftype = self._pdftypes[-1]

    def SetNormalPdf(self, sigma, murange=[0, 100]):
        """Prepare a normal pdf with s.d. sigma and (constrained) parameter range murange."""
        from scipy.stats import norm

        self._pdf = lambda x, mu: norm.pdf(x, loc=mu, scale=sigma)

        self.SetParameterBounds(murange)
        self.SetVariableBounds([-murange[-1], murange[-1]])
        self._discrete_pdf = False
        self._pdftype = self._pdftypes[1]

    def SetLogNormalPdf(self, sigma, murange=[0, 100]):
        """Prepare a log-normal pdf with parameter (not s.d.) sigma and (constrained) parameter range murange."""
        # the scipy.stats implementation of the log-normal pdf differs from the
        # common mathematical def., therefore the pdf will be defined by hand
        # here.
        from scipy.stats import norm

        self._pdf = lambda x, mu: norm.pdf(np.log(x), loc=mu, scale=sigma) / x

        self.SetParameterBounds(murange)
        self.SetVariableBounds(murange)
        self._discrete_pdf = False
        self._pdftype = self._pdftypes[2]

    def FCLimits(self, x):
        """
        The actual function to calculate the confidence intervals.

        Parameters
        ----------
        x: scalar or array-like
            An observed value or an array of observed values.

        Returns
        -------
        The lower and upper confidence limits as tuple or array with shape (len(x),2)
        depending on the input shape.
        """
        x = np.atleast_1d(x)

        if len(x) > 1:
            return np.asfarray(np.vectorize(self.FCLimitsScalarX)(x)).T
        else:
            return self.FCLimitsScalarX(x[0])

    def FCLimitsScalarX(self, x):
        mucont = np.linspace(
            self._murange[0], self._murange[-1], (self._murange[-1] - self._murange[0]) / self._mustep)
        mulow, muup = self._murange

        found = 0
        for mu in mucont:
            xlow, xup = self.GetVariableLimits(mu)
            if found == 0 and (xlow <= x <= xup):
                mulow = mu
                found |= 1
                continue
            if found == 1 and not (xlow <= x <= xup):
                muup = mu
                break

        return mulow, muup

    def EstimateOptimumParameter(self, x):
        """
        Maximum-likelihood estimation of the optimum parameter for fixed observation x.

        Used in the limit calculation when the analytic form is unknown.
        Internal function.
        """

        def minfct(pars):
            return -self._pdf(x, pars[0])

        self._m.SetLowerBounds([self._mumin - self._bg])
        self._m.SetUpperBounds([self._mumax])
        self._m.SetMethod("BOBYQA")
        return self._m(lambda p, g: minfct(p), np.asfarray([x]))[0]

    def GetVariableLimits(self, mu):
        """
        Calculate the confidence intervals on the variable x assuming a fixed parameter mu.

        Internal function.
        """
        if self._discrete_pdf:
            xcont = np.linspace(int(self._xrange[0]), int(
                self._xrange[-1]), int(self._xrange[-1] - self._xrange[0]) + 1)
        else:
            xcont = np.linspace(
                self._xrange[0], self._xrange[-1], (self._xrange[-1] - self._xrange[0]) / self._mustep)

        dx = xcont[1] - xcont[0]

        if self._pdftype == "poisson":
            mubest = self.Boundarize(xcont - self._bg)
        elif self._pdftype == "normal":
            mubest = self.Boundarize(xcont)
        elif self._pdftype == "lognormal":
            mubest = self.Boundarize(np.log(xcont))
        else:
            mubest = self.Boundarize(np.vectorize(
                self.EstimateOptimumParameter)(xcont))

        ps = self.Finitize(self._pdf(xcont, mu))
        psbest = self.Finitize(self._pdf(xcont, mubest))

        LR = self.Finitize(ps / psbest)

        # sorting in order of decreasing probability
        LRorder = np.argsort(LR)[::-1]

        xsort, psort = xcont[LRorder], ps[LRorder]

        psum = np.cumsum(psort * dx)

        cli = np.where(psum >= self._cl)[0]
        cli = cli[0] + 1 if len(cli) != 0. else len(psum)

        cxsort = np.sort(xsort[:cli])

        return cxsort[0], cxsort[-1]

    def Finitize(self, arr):
        arr[np.isfinite(arr) == False] = 0.0
        return arr

    def Boundarize(self, arr):
        arr[arr < self._murange[0]] = self._murange[0]
        arr[arr > self._murange[-1]] = self._murange[-1]
        return arr


def qprint(x, s, latex=False):
    """Pretty print numbers with uncertainties.

    Examples
    --------

    >>> qprint(12.3333,2.3333)
    '12.3 +/- 2.3'
    >>> qprint(12.3333,0.2333)
    '12.33 +/- 0.23'
    >>> qprint(12.3333,0.02333)
    '12.333 +/- 0.023'
    >>> qprint(123.3333,23.333)
    '123 +/- 23'
    >>> qprint(1233.3333,23.333)
    '(1.233 +/- 0.023) x 10^3'
    """

    x, s = float(x), float(s)

    nx = int(np.floor(np.log10(np.abs(x))))
    ns = int(np.floor(np.log10(s)))

    sExp = None
    if np.abs(nx) >= 3:
        x /= 10 ** nx
        s /= 10 ** nx
        sExp = "(%i)" % nx if nx < 0 else "%i" % nx
        ns -= nx
    n = max(0, -ns + 1)
    if latex:
        pm = r"\pm"
    else:
        pm = "+/-"
    if sExp:
        return ("(%%.%if %%s %%.%if) x 10^%%s" % (n, n)) % (x, pm, s, sExp)
    else:
        return ("%%.%if %%s %%.%if" % (n, n)) % (x, pm, s)


def multivariate_gaussian(x, mu, cov):
    """
    Multivariate gaussian pdf with expectation vector mu and covariance matrix sigma.

    Parameters
    ----------
    x: array of n-floats
      Point where to estimate the pdf
    mu: array of n-floats
      Expectation vector of the pdf
    cov: 2d array of n x n -floats
      Covariance matrix

    Returns
    -------
    Float value corresponding to the probability density at x

    Example
    --------
    >>> mu,cov = np.asfarray([1.,1.]),np.asfarray([[0.5, 0.],[0., 0.3]])
    >>> x = np.asfarray([2.,2.])
    >>> from scipy.stats import norm
    >>> "%.6f" % (norm.pdf(x[0],mu[0],cov[0][0]**0.5)*norm.pdf(x[1],mu[1],cov[1][1]**0.5))
    '0.028553'
    >>> "%.6f" % multivariate_gaussian(x,mu,cov)
    '0.028553'
    """

    n = len(x)
    if len(mu) != n or cov.shape != (n, n):
        raise AssertionError("Error! Input dimensions are not matching!")

    det = np.linalg.det(cov)
    if det == 0:
        raise ValueError("Error! Covariance matrix is singular!")

    norm = ((2 * np.pi) ** n * np.absolute(det)) ** 0.5
    d = x - mu
    return np.exp(-0.5 * np.dot(np.dot(d.T, np.linalg.inv(cov)), d)) / norm


class multivariate_gaussian_evaluator(object):
    """
    Convenience class to utilize the multivariate_gaussian function.

    It will return the probabilities of select sample in relation to a greater distribution.

    If coverage is specified, it will return the mean, length of the axes of the hyperellipsoid, directional
    vector for orientation of the hyperellipsoid, and boolean array saying if points in (True) or
    outside of hyperellipsoid.

    Parameters
    ----------
    data: m-parameter by n-d array
      vector of parameters from which the multivariate gaussian will be made

    points: a-parameter by b-d array
      vector of parameters from which the probability of the multivariate gaussian will be calculated, default is data

    coverage: float, default None
      requested coverage of the hypervolume, associated with m-d multivariate gaussian

    quantiles: boolean
      if True, uses medians for mean vector; this should be more stable wrt outliers.
      if False, uses means for mean vector

    Returns
    -------
    default:
      array of probabilities for specified points

    if coverage specified:
      mean vector, length vector, directional vector, isin

    Notes
    -----
    Equation for coverage and explanation of MVN can be found at:
    http://jonathantemplin.com/files/multivariate/mv11icpsr/mv11icpsr_lecture04.pdf

    Examples
    --------
    Using coverage
    >>> m1 = [-0.9, -0.1, -0.0, 0.7, 1.3, 0.4, 0.6, -1.9, 0.2, -1.1]
    >>> m2 = [ 0.1, 0.7, -0.9, -0.1, -0.5, -0.7, -0.9, -0.2, -0.2, -0.5]
    >>> mvn = multivariate_gaussian_evaluator([m1, m2], coverage=[0.682])
    >>> mean, length, direct, isin = mvn()

    Draw the ellipse
    >>> from matplotlib.patches import Ellipse
    >>> ell2 = Ellipse(xy=(mean[0], mean[1]), width=length[0]*2, height=length[1]*2, angle=np.degrees(np.arctan2(*direct[:,0][::-1])))
    
    You need to manually add the Ellipse to axes 'ax': ax.add_artist(ell2)

    Without coverage
    >>> m1 = [-0.9, -0.1, -0.0, 0.7, 1.3, 0.4, 0.6, -1.9, 0.2, -1.1]
    >>> m2 = [ 0.1, 0.7, -0.9, -0.1, -0.5, -0.7, -0.9, -0.2, -0.2, -0.5]
    >>> xmin = np.min(m1)
    >>> xmax = np.max(m1)
    >>> ymin = np.min(m2)
    >>> ymax = np.max(m2)
    >>> X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    >>> positions = np.vstack([X.ravel(), Y.ravel()])

    Draw as color mesh
    >> mvn = multivariate_gaussian_evaluator([m1, m2], points = positions)
    >> val = mvn()
    >> Z = np.reshape(val.T, X.shape)
    >> plt.imshow(np.rot(Z,2), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    >> plt.plot(self.x, self.y, 'k.', markersize=2)
    >> plt.xlim([xmin, xmax])
    >> plt.ylim([ymin, ymax])
    """

    def __init__(self, data, points=None, wts=None, coverage=None, quantiles=True):
        self.data = np.array(data)

        if quantiles:
            if wts is None:
                self.mean = np.median(self.data, axis=1)
            else:
                assert self.data.ndim == 2, "Only 2-d calculation of weighted median is currently supported!"
                m1 = median(self.data[0], weights=wts)
                m2 = median(self.data[1], weights=wts)
                self.mean = np.array([m1, m2])

        else:
            self.mean = np.average(self.data, axis=1, weights=wts)
        self.cov = np.cov(self.data, aweights=wts)

        if points is None:
            self.points = np.array(data)
        else:
            self.points = np.array(points)

        self.coverage = coverage

    def __call__(self):
        if self.coverage is not None:
            length, direct = self.coverage_axes()
            isin = self.is_in_coverage()
            return self.mean, length, direct, isin
        else:
            return self.multivariate_gauss_prob()

    def multivariate_gauss_prob(self):
        points = self.points.copy()
        prob = [0] * len(points[0])
        pos = points.T
        for i in range(len(pos)):
            prob[i] = multivariate_gaussian(pos[i], self.mean, self.cov)
        return np.array(prob)

    def coverage_axes(self):
        from scipy.stats import chi2
        w, u = np.linalg.eig(self.cov.copy())
        mult = chi2(len(self.data)).ppf(self.coverage)
        return np.sqrt(mult * w), u

    # identical to IsInUncertaintyEllipse, defined here to reduce computational time
    def is_in_coverage(self):
        from scipy.stats import chi2
        points = self.points.copy()
        points = points.T
        w, u = np.linalg.eig(self.cov.copy())

        isin = [1] * len(points)
        for p in range(len(points)):
            x = np.dot(u.T, points[p] - self.mean.copy()) / np.sqrt(w)
            isin[p] = int(np.sum(x * x) <=
                          chi2(len(self.mean.copy())).ppf(self.coverage))
        return np.array(isin)


def LikelihoodRatioSignificance(LLnull, LLalt, ndof=1):
    """
    Test the significance given two fit results of different models to the SAME data assuming an approximate chi2 statistic.

    Parameters
    ----------
    LLnull: int
      value of log likelihood of null hypothesis data
    LLalt: int
      value of log likelihood of alt hypothesis data

    Returns
    -------
    int, p-value of probability of null hypothesis being true compared to alt hypothesis
      calculating 1 - cdf(d, 1), this is the p-value connected with correctness of the null hypothesis
      a small value of sf means that the probability of the null hypothesis being true compared to the alternative is small
      however, the p-value is not the probability itself!
      neither the one of null being false, nor the one of alt being true!

    Example
    --------
    >>> from scipy.stats import gaussian_kde
    >>> import numpy as np
    >>> from pyik.numpyext import multivariate_gaussian, LikelihoodRatioSignificance
    >>> m1 = [-0.9, -0.1, -0.0, 0.7, 1.3, 0.4, 0.6, -1.9, 0.2, -1.1]
    >>> m2 = [ 0.1, 0.7, -0.9, -0.1, -0.5, -0.7, -0.9, -0.2, -0.2, -0.5]
    >>> data = np.array([m1, m2])
    >>> kernel = gaussian_kde(data)
    >>> kde_values = kernel(data)
    >>> LLalt = np.sum(np.log(kde_values))
    >>>
    >>> cov = np.cov(data)
    >>> mu = np.mean(data, axis=1)
    >>> gauss = [0]*len(data[0])
    >>> data = data.T
    >>> for row in range(len(data)): gauss[row] = multivariate_gaussian(data[row], mu, cov)
    >>> LLnull = np.sum(np.log(gauss))
    >>> round(LikelihoodRatioSignificance(LLnull,LLalt), 4)
    0.15

    Authors
    -------
    Alexander Schulz
    """
    from scipy.stats import chi2

    d = 2 * LLalt - 2 * LLnull  # the log-likelihood ratio
    # this is approximately distributed according to Chi2 dist with degrees of freedom 1
    if d < 0:
        raise AssertionError(
            "The log-likelihood ratio is negative, you are probably doing it wrong :D!")

    sf = chi2.sf(d, ndof)
    return sf
