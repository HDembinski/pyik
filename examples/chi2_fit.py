# -*- coding: utf-8 -*-

"""
This example demonstrates fitting a model to a (simulated) dataset using
numpyext.chi2_fit, which wraps Minuit.
"""

import numpy as np
from matplotlib import pyplot
from pyik.fit import ChiSquareFunction
from pyik.mplext import cornertext

np.random.seed(1)


def model(x, pars):
  """A slightly complex function. Needs to be vectorized."""
  a0, a1, x_break = pars  # unpack parameter vector
  x = np.atleast_1d(x)  # x needs to be a numpy array
  y = np.empty_like(x)
  mask = x <= x_break
  y[mask] = a0 * x[mask]
  y[~mask] = a0 * x[~mask] + a1 * (x[~mask] - x_break)**2
  return y

# Simulate a dataset of n measurements
n = 20
parsTrue = (2.0, 0.5, 13.0)

xs = np.linspace(0, 20, n)
ys = model(xs, parsTrue)

# Add some noise to the points
eys = 1.5 * np.ones(n)
ys += np.random.randn(n) * eys

# Perform a fit to the data points; reuse the errors used to generate the noise
# Note: fits to data without xerrors are much faster
starts = (1.0, 1.0, 10.0)  # starting values
# define bounds for parameter "x_break"
lower_bounds = (-np.inf, -np.inf, 0.0)
upper_bounds = (np.inf, np.inf, 20.0)
pars, cov, chi2, ndof = \
  ChiSquareFunction(model, xs, ys, eys) \
  .Minimize(starts,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds)

# Generate a plot of the fit
new_xs = np.linspace(0., 20, 1000)

figure = pyplot.figure()

pyplot.plot(new_xs, model(new_xs, pars), 'b')
pyplot.errorbar(xs, ys, eys, fmt='ok')

pyplot.xlim(-1, 21)
pyplot.xlabel("x")
pyplot.ylabel("y")

s = "Fit Example:\n"
for i, label in enumerate(("a_0", "a_1", "x_{brk}")):
  s += "$%s = %.2f \pm %.2f$ (True: $%.1f$)\n" % (label, pars[i], cov[i, i]**0.5, parsTrue[i])
s += "$\\chi^2 / n_{dof} = %.3f$" % (chi2 / ndof)

cornertext(s)

pyplot.show()
