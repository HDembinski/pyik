# -*- coding: utf-8 -*-
from six.moves import range
import numpy as np


def fit_ellipse(x, y, centered=False):
    """
    Non-iterative least-squares fit of an ellispe to data points.

    Parameters
    ----------
    x : array of x coordinates
    y : array of y coordinates
    centered : optional, set True if ellipse has known center at (0, 0)

    Returns
    -------
    Ellipse in matrix form xT M x = 1.
    If centered == False:
        2x2 matrix M, center point
    If centered == True:
        2x2 matrix M

    Notes
    -----
    Algorithm is fast and robust, but biased towards smaller ellipses.
    This is because it minimizes algebraic distances instead of
    geometric distances. For high-precision results, an iterative method
    can be used with this result as starting point.

    Original algorithm presented in:
    Halir, Flusser,
    Proc. 6th Int. Conf. in Central Europe on
    Computer Graphics and Visualization,
    WSCG, vol. 98 (1998)

    Added in this version:
    - treatment of known centre
    - fix for very low number of points
    - conversion to matrix representation
    """

    dot = np.dot
    inv = np.linalg.inv
    eig = np.linalg.eig
    cols = lambda *args: np.column_stack(args)
    rows = lambda *args: np.row_stack(args)

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    n = x.shape[0]

    d1 = cols(x ** 2, x * y, y ** 2)
    if centered:
        d2 = cols(np.ones(n),)
    else:
        d2 = cols(x, y, np.ones(n))

    s1 = dot(d1.T, d1)
    s2 = dot(d1.T, d2)
    s3 = dot(d2.T, d2)

    if centered:
        t = -s2.T / s3
    else:
        t = -dot(inv(s3), s2.T)

    m = s1 + dot(s2, t)
    m = rows(m[2, :] / 2.0, -m[1, :], m[0, :] / 2.0)

    w, v = eig(m)
    v = np.real(v)
    cond = 4.0 * v[0] * v[2] - v[1] ** 2 > 0
    for ind in range(len(cond)):
        if cond[ind] > 0:
            break
    a1 = np.real(np.squeeze(v[:, ind]))
    a2 = dot(t, a1)

    m = np.empty((2, 2))
    m[0, 0] = a1[0]
    m[0, 1] = m[1, 0] = 0.5 * a1[1]
    m[1, 1] = a1[2]

    if centered:
        r2 = -a2[-1]
        return m / r2
    else:
        c = -0.5 * dot(inv(m), a2[:2])
        r2 = dot(dot(c, m), c) - a2[-1]
        return m / r2, c
