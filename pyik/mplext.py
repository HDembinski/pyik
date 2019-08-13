# -*- coding: utf-8 -*-
"""
Contains extensions to matplotlib.

Special functions to extend matplotlib in areas where it lacks certain functionality.

"""
from __future__ import print_function
from six.moves import range
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import Formatter as MPLFormatter
import matplotlib.colors as mplcolors
import colorsys


def lighten_color(color, amount):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple. Can also darken.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    c = mplcolors.cnames.get(color, color)
    c = colorsys.rgb_to_hls(*mplcolors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], min(max(c[1] + amount, 0.0), 1.0), c[2])


def plot_bracket(x, y, yerr, xerr=None, capsize=3, axes=None, **kwargs):
    """
    Plot brackets to indicate errors.

    Parameters
    ----------
    x,y,yerr,xerr: central value and errors
    markersize: size of the bracket
    capsize: length of the tips of the bracket
    """

    if axes is None:
        axes = plt.gca()

    for k in ("mec", "markeredgecolor", "marker", "m"):
        if k in kwargs:
            raise ValueError("keyword %s not allowed" % k)

    col = "k"
    for k in ("c", "color"):
        if k in kwargs:
            col = kwargs[k]
            del kwargs[k]
    kwargs['ls'] = 'None'

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    if yerr is not None:
        yerr = np.atleast_1d(yerr)
        if len(yerr.shape) == 1:
            yd = y - yerr
            yu = y + yerr
        elif len(yerr.shape) == 2 and yerr.shape[0] == 2:
            yd = y - yerr[0]
            yu = y + yerr[1]
        else:
            raise ValueError("yerr has unexpected shape")

        dx = 0.01
        dy = 0.05
        t = 2 * dy * capsize
        w = 0.5
        m1 = ((-w - dx, t + dy), (-w - dx, -dy), (w + dx, -dy), (w + dx, t + dy),
              (w - dx, t + dy), (w - dx, dy), (-w + dx, dy), (-w + dx, t + dy))
        m2 = ((-w - dx, -t - dy), (-w - dx, dy), (w + dx, dy), (w + dx, -t - dy),
              (w - dx, -t - dy), (w - dx, -dy), (-w + dx, -dy), (-w + dx, -t - dy))
        axes.plot(x, yd, marker=m1, color=col, mec=col, **kwargs)
        axes.plot(x, yu, marker=m2, color=col, mec=col, **kwargs)

    if xerr is not None:
        xerr = np.atleast_1d(xerr)
        if len(xerr.shape) == 1:
            xd = x - xerr
            xu = x + xerr
        elif len(xerr.shape) == 2 and xerr.shape[0] == 2:
            xd = x - xerr[0]
            xu = x + xerr[1]
        else:
            raise ValueError("xerr has unexpected shape")

        dx = 0.05
        dy = 0.01
        t = 2 * dx * capsize
        h = 0.5
        m1 = ((t + dx, -h - dy), (-dx, -h - dy), (-dx, h + dy), (t + dx, h + dy),
              (t + dx, h - dy), (dx, h - dy), (dx, -h + dy), (t + dx, -h + dy))
        m2 = ((-t - dx, -h - dy), (dx, -h - dy), (dx, h + dy), (-t - dx, h + dy),
              (-t - dx, h - dy), (-dx, h - dy), (-dx, -h + dy), (-t - dx, -h + dy))
        axes.plot(xd, y, marker=m1, color=col, mec=col, **kwargs)
        axes.plot(xu, y, marker=m2, color=col, mec=col, **kwargs)


def plot_hist(xedges, ws, axes=None, **kwargs):
    """
    Plot histogram data in ROOT style.

    Parameters
    ----------
    xedge: lower bin boundaries + upper boundary of last bin
    w: content of the bins
    facecolor: a matplotlib color definition to fill the histogram
    axes: the axes to draw on (defaults to the current axes)
    """
    if axes is None:
        axes = plt.gca()

    m = len(ws)
    n = 2 * m + 2

    xy = np.zeros((2, n))

    xy[0][0] = xedges[0]
    xy[0][-1] = xedges[-1]

    for i in range(m):
        xy[0][1 + 2 * i] = xedges[i]
        xy[1][1 + 2 * i] = ws[i]
        xy[0][1 + 2 * i + 1] = xedges[i + 1]
        xy[1][1 + 2 * i + 1] = ws[i]

    if "fc" in kwargs:
        kwargs["facecolor"] = kwargs["fc"]
        del kwargs["fc"]
    if "c" in kwargs:
        kwargs["color"] = kwargs["c"]
        del kwargs["c"]

    if "facecolor" in kwargs:
        if "color" in kwargs:
            kwargs["edgecolor"] = kwargs["color"]
            del kwargs["color"]
        if "label" in kwargs:
            # label hack
            from matplotlib.patches import Rectangle
            r = Rectangle((0, 0), 0, 0, **kwargs)
            axes.add_patch(r)
            del kwargs["label"]
        return axes.fill_between(xy[0], 0, xy[1], **kwargs)
    else:
        return axes.plot(xy[0], xy[1], **kwargs)


def plot_boxerrors(xedges, ys, yes, axes=None, **kwargs):
    """
    Plot error boxes for a histogram.

    (Recommended way to show systematic uncertainties).

    Parameters
    ----------
    xedge: array of floats
      Lower bin boundaries + upper boundary of last bin as returned
      by numpy.histogram.
    ys: array of floats
      Center of the box.
    yes: array of floats
      Distance of the edge of the box from the center. Maybe one-dimensional
      for symmetric boxes or two-dimensional for asymmetric boxes.
    axes: Axes (optional, default: None)
      The axes to draw on (defaults to the current axes).

    Optional keyword arguments are forwarded to the matplotlib.patch.Rectangle
    objects. Useful keywords are: facecolor, edgecolor, alpha, zorder.
    """

    from matplotlib.patches import Rectangle

    if axes is None:
        axes = plt.gca()

    xedges = np.atleast_1d(xedges)
    ys = np.atleast_1d(ys)
    yes = np.atleast_1d(yes)

    n = len(ys)
    isAsymmetric = len(yes.shape) == 2
    rs = []
    for i in range(n):
        x0 = xedges[i]
        y0 = ys[i] - yes[i][0] if isAsymmetric else ys[i] - yes[i]
        xw = xedges[i + 1] - xedges[i]
        yw = yes[i][0] + yes[i][1] if isAsymmetric else 2 * yes[i]
        if yw > 0:
            r = Rectangle((x0, y0), xw, yw, **kwargs)
            rs.append(r)
            axes.add_artist(r)
    return rs


def cornertext(text, loc=2, color=None, frameon=False,
               axes=None, **kwargs):
    """
    Conveniently places text in a corner of a plot.

    Parameters
    ----------
    text: string or sequence of strings
      Text to be placed in the plot. May be a sequence of strings to get
      several lines of text.
    loc: integer or string
      Location of text, same as in legend(...).
    color: color or sequence of colors
      For making colored text. May be a sequence of colors to color
      each text line differently.
    frameon: boolean (optional)
      Whether to draw a border around the text. Default is False.
    axes: Axes (optional, default: None)
      Axes object which houses the text (defaults to the current axes).
    fontproperties: matplotlib.font_manager.FontProperties object
      Change the font style.

    Other keyword arguments are forwarded to the text instance.
    """

    from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, TextArea
    from matplotlib import rcParams
    from matplotlib.font_manager import FontProperties
    import warnings

    if axes is None:
        axes = plt.gca()

    locTranslate = {
        'upper right': 1,
        'upper left': 2,
        'lower left': 3,
        'lower right': 4,
        'right': 5,
        'center left': 6,
        'center right': 7,
        'lower center': 8,
        'upper center': 9,
        'center': 10
    }

    if isinstance(loc, str):
        if loc in locTranslate:
            loc = locTranslate[loc]
        else:
            message = ('Unrecognized location "%s". Falling back on "upper left"; valid '
                       'locations are\n\t%s') % (loc, '\n\t'.join(locTranslate.keys()))
            warnings.warn(message)
            loc = 2

    if "borderpad" in kwargs:
        borderpad = kwargs["borderpad"]
        del kwargs["borderpad"]
    else:
        borderpad = rcParams["legend.borderpad"]

    if "borderaxespad" in kwargs:
        borderaxespad = kwargs["borderaxespad"]
        del kwargs["borderaxespad"]
    else:
        borderaxespad = rcParams["legend.borderaxespad"]

    if "handletextpad" in kwargs:
        handletextpad = kwargs["handletextpad"]
        del kwargs["handletextpad"]
    else:
        handletextpad = rcParams["legend.handletextpad"]

    if "fontproperties" in kwargs:
        fontproperties = kwargs["fontproperties"]
        del kwargs["fontproperties"]
    else:
        if "size" in kwargs:
            size = kwargs["size"]
            del kwargs["size"]
        elif "fontsize" in kwargs:
            size = kwargs["fontsize"]
            del kwargs["fontsize"]
        else:
            size = rcParams["legend.fontsize"]
        fontproperties = FontProperties(size=size)

    texts = [text] if isinstance(text, str) else text

    colors = [color for t in texts] if (
        isinstance(color, str) or color is None) else color

    tas = []
    for t, c in zip(texts, colors):
        ta = TextArea(t,
                      textprops={"color": c, "fontproperties": fontproperties},
                      multilinebaseline=True,
                      minimumdescent=True,
                      **kwargs)
        tas.append(ta)

    vpack = VPacker(children=tas, pad=0, sep=handletextpad)

    aob = AnchoredOffsetbox(loc, child=vpack,
                            pad=borderpad,
                            borderpad=borderaxespad,
                            frameon=frameon)

    axes.add_artist(aob)
    return aob


def uncertainty_ellipse(par, cov, cfactor=1.51, axes=None, **kwargs):
    """
    Draw a 2D uncertainty ellipse.

    Parameters
    ----------
    par: array-like
      The parameter vector.
    cov: array-like
      The covariance matrix.
    cfactor: float (optional, default: 1.51 for 68 % coverage)
      Scaling factor to give the ellipse a desired coverage.
    axes: Axes (optional, default: None)
      The axes to draw on (defaults to the current axes).
    Other keyword-based arguments may be given, which are forwarded to
    the ellipse object.

    Returns
    -------
    An ellipse patch.

    Notes
    -----
    To compute the coverage factor with scipy, do
    >>> from scipy.stats import chi2
    >>> p_coverage = 0.68 # desired coverage
    >>> round(chi2(2).ppf(p_coverage) ** 0.5, 4)
    1.5096
    """

    from math import atan2, pi, sqrt
    from matplotlib.patches import Ellipse

    if axes is None:
        axes = plt.gca()

    u, s, v = np.linalg.svd(cov)

    angle = atan2(u[1, 0], u[0, 0]) * 180.0 / pi
    s0 = cfactor * sqrt(s[0])
    s1 = cfactor * sqrt(s[1])

    ellipse = Ellipse(xy=par, width=2.0 * s0, height=2.0 * s1,
                      angle=angle, **kwargs)
    axes.add_patch(ellipse)

    return ellipse


def ViolinPlot(x, y, bins=10, range=None, offsetX=0, offsetY=0,
               color="k", marker="o", draw="amv", xmean=False,
               extend=3, outliers=True, textpos=None, axes=None, **kwargs):
    """
    Draw a violin (kernel density estimate) plot with mean and median profiles.

    Adapted from http://pyinsci.blogspot.de/2009/09/violin-plot-with-matplotlib.html.
    Updated and simplified version to maximize data/ink ratio according to Tufte.

    Parameters
    ----------
    x: array of type float
      Data dimension to bin in
    y: array of type float
      Data to create profile from
    bins: int or array of type float or None
      Number of x bins or array of bin edges
      If None: Take x as bin centers and y as already binned values
    range (optional):
      The range in x used for binning

    color: Matplotlib-compatible color value
      Color that is used to draw markers and violins

    marker: Matplotlib-compatible marker
      Marker that is used for mean profile

    draw: string (default: "amv")
      What to draw: a (mean), m (median), v (violins), s (only 1 sigma violins), c (number of entries within bins)

    extend: float in units of standard deviation (default: 3)
      If a float x is given, the violins are drawn from -l to +u
      The values l, u are possibly asymmetric quantiles with respect to x sigma (normal distribution)
      If None is given, the violins are drawn in between the most extreme points

    outliers: bool (default: True)
      If true, will draw outlier points outside of extend

    axes: Axes (optional, default: None)
      The axes to draw on (defaults to the current axes)

    Other keyword-based arguments may be given, which are forwarded to
    the individual plot/errorbar/fill_between calls.

    Returns
    -------
    Calculated profile values
    """

    if axes is None:
        axes = plt.gca()
    else:
        plt.sca(axes)

    from scipy.stats import gaussian_kde, norm
    from pyik.numpyext import bin, centers, mad
    from scipy.stats.mstats import mquantiles

    s1 = norm.cdf(-1)
    sx = norm.cdf(-3 if extend is None else -extend)

    if bins is not None:
        ybins, xedgs = bin(x, np.column_stack((x, y)), bins=bins, range=range)
        xcens, xhws = centers(xedgs)
    else:
        xcens = x
        # xhws = (x[1:] - x[:-1]) / 2.
        # xhws = np.append(xhws, xhws[-1])
        xhws = np.ones(len(x)) * min((x[1:] - x[:-1]) / 2.)
        ybins = y

    l = len(ybins)
    means, stds, meds, mads, ns = np.zeros(l), np.zeros(l), np.zeros(l), \
        np.zeros(l), np.zeros(l)

    for i, ybin in enumerate(ybins):

        ybind = np.asfarray(ybin)

        if bins is not None:

            if len(ybind) < 1:
                continue

            if len(ybind) == 1:

                xbinh = np.atleast_1d(ybind[0][0])
                ybinh = np.atleast_1d(ybind[0][1])

            else:
                m = np.isfinite(ybind.T[1])
                xbinh = ybind.T[0][m]
                ybinh = ybind.T[1][m]

            if xmean:
                xcens[i] = np.mean(xbinh)
                xhws[i] = np.std(xbinh)

        else:

            ybinh = ybind[np.isfinite(ybind)]

        means[i] = np.mean(ybinh)
        stds[i] = np.std(ybinh, ddof=1)
        meds[i] = np.median(ybinh)
        mads[i] = mad(ybinh)
        ns[i] = len(ybinh)
        qs = mquantiles(ybinh, prob=[sx, s1, 1 - s1, 1 - sx])

        if len(ybinh) > 1:
            # calculates the kernel density
            try:
                k = gaussian_kde(ybinh)
            except:
                print("Warning! Error in estimating kernel density for data in bin %s! Skipping bin..." % i)
                continue

            # support of violins
            if extend is None:
                m = k.dataset.min()
                M = k.dataset.max()
                y = np.arange(m, M, (M - m) / 200.)
            else:
                y = np.linspace(qs[0], qs[-1], extend * 100)

            # scaling violins
            v = k.evaluate(y)
            vmax = v.max()
            v = v / vmax * xhws[i] * 0.8

            # violins
            if "v" in draw and "s" not in draw:
                plt.fill_betweenx(y, xcens[i] - v + offsetX, xcens[i] + offsetX, facecolor=color,
                                  edgecolor="none", lw=0.5, zorder=0, alpha=0.1)

                # # hack to remove (overdraw) the inner white line that looks ugly
                # plt.fill_betweenx(y, xcens[i], xcens[i] + v, facecolor="None",
                #                   edgecolor="white", lw=2, zorder=0)

            # median
            if "m" in draw:

                # mean uncertainty violin part
                mask = (y > meds[i] - (meds[i] - qs[1]) / np.sqrt(ns[i])
                        ) & (y < meds[i] + (qs[2] - meds[i]) / np.sqrt(ns[i]))
                plt.fill_betweenx(y[mask], xcens[i] - v[mask] + offsetX, xcens[i] + offsetX, facecolor=color, alpha=0.5,
                                  edgecolor="None", zorder=3)

                if "v" in draw:  # 1 sigma violin part
                    a = 0.25
                    if "s" in draw:
                        a = 0.15
                    mask = (y > qs[1]) & (y < qs[2])
                    plt.fill_betweenx(y[mask], xcens[i] - v[mask] + offsetX, xcens[i] + offsetX, facecolor=color,
                                      edgecolor="none", lw=0.5, zorder=1, alpha=a)

                # # and to remove inner line again
                # plt.fill_betweenx(y[mask], xcens[i], xcens[i] + v[mask], facecolor="none",
                #                   edgecolor="white", lw=4, zorder=1)

                wm = xhws[i] * 0.8 * k.evaluate(meds[i]) / vmax
                plt.plot((xcens[i] - wm + offsetX, xcens[i] + offsetX), (meds[i], meds[i]), ls="-", lw=1,
                         color=color, zorder=3)

            if outliers:

                youts = ybinh[(ybinh < qs[0]) | (ybinh > qs[-1])]
                xouts = np.ones(len(youts)) * xcens[i]

                plt.plot(xouts + offsetX, youts, marker=".",
                         ls="None", ms=2, color=color, zorder=1)

    # Mean profile
    if "a" in draw:
        zero_mask = (ns > 0)
        merrbar = plt.errorbar(xcens[zero_mask] + offsetX, means[zero_mask], stds[zero_mask] / np.sqrt(ns[zero_mask]), marker=marker, ls="None",
                               elinewidth=1, mew=2, mfc="white", mec=color, color=color,
                               capsize=0, zorder=4, **kwargs)
        # matplotlib is fucking up the zorder for me if not explicitly told what to do
        for el in merrbar[2]:
            el.set_zorder(1)

    if textpos is None:
        textpos = y.min()

    if "c" in draw:
        for n, x in zip(ns, xcens + offsetX):
            plt.annotate(str(n.astype(int)), xy=(x, textpos + offsetY), xycoords=('data', 'data'), rotation=90,
                         xytext=(0, 0), textcoords='offset points', va='top', ha='center', color=color, size=9)

    # to bring all the violins into visible x-range
    plt.xlim(min(xcens - 2 * xhws), max(xcens + 2 * xhws))

    return xcens, xhws, means, stds, meds, mads, ns


def plot_labeled_vspan(x0, x1, label, y=0.5, axes=None,
                       color="k", facecolor=None, fontsize=None, zorder=0):
    """
    Draw a vspan with a label in the center.

    Parameters
    ----------
    x0: float
        data coordinate where span starts.
    x1: float
        data coordinate where span ends.
    label: str
        Text label.
    y (optional): float
        Vertical axes coordinate around which to center label. Default: 0.5.
    axes (optional): Axes instance
        Axes instance to draw onto. Default: matplotlib.pyplot.gca().
    color (optional): str or sequence
        Color for the text and the span (if facecolor is not set).
        Default: black.
    facecolor (optional): str or sequence
        Color for the span. Default: color lightened by 0.75.
    fontsize (optional): str or float
        Fontsize for the text.
    zorder (optional): int
        z-placement of span. Default: zorder=0.
    """
    if axes is None:
        axes = plt.gca()
    facecolor = lighten_color(color, 0.75) if facecolor is None else facecolor
    span = axes.axvspan(x0, x1,
        facecolor=facecolor,
        zorder=zorder)
    text = axes.text(0.5 * (x0 + x1), y, label,
        transform=axes.get_xaxis_transform(),
        ha="center", va="center",
        fontsize=fontsize, rotation=90,
        zorder=zorder+1)
    return span, text
