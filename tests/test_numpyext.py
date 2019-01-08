import numpy as np
from numpy.testing import assert_allclose, assert_equal


def test_profile():
    from pyik.numpyext import profile
    x = [0.0, 1.0, 2.0, 3.0]
    assert_equal(x, (0, 1.0, 2.0, 3.0))
    yavg, ystd, n, xe = profile(x, x, bins=2)
    assert_allclose(yavg, (0.5, 2.5), 1e-3)
    assert_allclose(ystd, (0.5, 0.5), 1e-3)
    assert_equal(n, (2, 2))
    assert_allclose(xe, (0., 1.5, 3.))


def test_centers():
    from pyik.numpyext import centers
    c, w = centers([0, 1, 3])
    assert_equal(c, (0.5, 2.0))
    assert_equal(w, (0.5, 1.0))


def test_derivative():
    from pyik.numpyext import derivative
    def f(x):
        return 2 + x + 2 * x ** 2 + x ** 3
    def fp(x):
        return 1 + 4 * x + 3 * x ** 2
    def fpp(x):
        return 4 + 6 * x

    assert_allclose(derivative(f, 1.0), fp(1.0))
    assert_allclose(derivative(f, 1.0, step=1e-3), fp(1.0))
    assert_allclose(derivative(f, 1.0, order=2), fpp(1.0))
    ones = np.ones(2)
    assert_allclose(derivative(f, ones), fp(ones))
    assert_allclose(derivative(f, ones, order=2), fpp(ones))


def test_derivativeND():
    from pyik.numpyext import derivativeND
    def f(xy):
        x, y = xy.T
        return x ** 2 + y ** 2
    
    xy = ((0., 0.), (1., 0.), (0., 1.))
    result = derivativeND(f, xy)
    assert_allclose(result, ((0., 0.), (2., 0.), (0., 2.)))

