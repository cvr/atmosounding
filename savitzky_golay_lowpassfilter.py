#!/usr/bin/python
# vim: set fileencoding=utf-8 fileformat=unix :
# -*- coding: utf-8 -*-
# vim: set ts=8 et sw=4 sts=4 sta :
########################################################################
#
# Copyright (C) 2017 by Carlos Veiga Rodrigues. All rights reserved.
# author: Carlos Veiga Rodrigues <cvrodrigues@gmail.com>
#
# This program can be redistribuited and modified
# under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation,
# either version 3 of the License or any later version.
# This program is distributed WITHOUT ANY WARRANTY,
# without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.
# For more details consult the GNU Lesser General Public License
# at http://www.gnu.org/licenses/lgpl.html.
#
########################################################################

__version__ = "1.0"
__date__ = "2017Jul"
__author__ = "Carlos Veiga Rodrigues <cvrodrigues@gmail.com>"
__email__ = "cvrodrigues@gmail.com"
__copyright__ = "Copyright (C) 2017 by Carlos Veiga Rodrigues. All rights reserved."
__license__ = "LGPL3"

import numpy as np

def savgol_irregular_brute (f, x, xwindow, order=4, deriv=0):
    """
    Applies a Savitzky-Golay low-pass filter to irregularly sampled data.
    
    This function applies the Savitzky-Golay filter in a brute force way.
    For each record in the input data, a polynomial is fitted, meaning
    a matrix is pseudo-inverted to solve the least-squares problem.
    
    The original technique assumes a signal with uniform sampling and only
    one matrix is pseudo-inverted, independently of the number of records
    in the input data.

    Parameters
    ----------
    f : array_like, shape (M,)
        Input data, consisting of a sampled signal function of a dimension
        such as space or time, i.e. `f(x)`.
    x : array_like, shape (M,)
        Input data, with the samping time or position of the `f` values.
    xwindow : float
        Defines the window size, with dimension of `x`, used to filter the
        data. For each position `x[i]`, a polynomial will be fitted for
        all records of `f` that fall inside this window, i.e.:
        .. math ::
            - x_{window} / 2  <  x - x_i  <  x_{window} / 2

    order : int, optional
        Degree of the fitting polynomial.
    deriv : int, optional
        The order of the derivative to compute. This must be a nonnegative
        integer. The default is 0, which means to filter the data without
        differentiating.

    Returns
    -------
    g : array_like, shape (deriv, M) or (M,)
        The filtered signal and its derivatives in order to `x`. If input
        parameter `deriv` is 0 a 1-D array is return with  shape (M,).

    See also
    --------
    scipy.signal.savgol_filter
    savgol_irregular_interp
    numpy.linalg.pinv

    References
    ----------
    A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by
    Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8),
    pp 1627-1639.
    """
    try:
        order = np.abs(np.int(order))
        deriv = np.abs(np.int(deriv))
    except ValueError as msg:
        raise ValueError("'order' and 'deriv' must be of type int")
    if order < 2:
        raise TypeError("polynomial 'order' should be >= 2")
    if deriv > order:
        print("Warning: 'deriv' > 'order'")
        print("derivatives higher than polynomial order will be 0")
    window = order + 1  # minimum number of points
    half = (window - window % 2) // 2
    xhalf = xwindow / 2.
    o = np.arange(0, order + 1)
    g = []  # g[o,...] where 'o' represents the n'th derivative
    for d in range(deriv + 1):
        g.append(np.zeros(np.shape(f), float) * np.nan)
    ## western boundary
    ii = np.flatnonzero(x <= x[0] + xwindow).max() + 1
    ii = max(ii, window + 1)
    iw = np.flatnonzero(x < x[0] + xhalf).max() + 1
    for i in range(0, iw):
        p = np.polyfit(x[:ii] - x[i], f[:ii], order)
        g[0][i] = p[-1]
        for d in range(1, min(deriv, order) + 1):
            g[d][i] = p[-1-d] * np.math.factorial(d)
    ## eastern boundary
    ii = np.flatnonzero(x >= x[-1] - xwindow).min()
    ii = min(ii, np.size(x) - window)
    ie = np.flatnonzero(x > x[-1] - xhalf).min()
    for i in range(ie, np.size(x)):
        p = np.polyfit(x[ii:] - x[i], f[ii:], order)
        g[0][i] = p[-1]
        for d in range(1, min(deriv, order) + 1):
            g[d][i] = p[-1-d] * np.math.factorial(d)
    ## in-between
    for i in range(iw, ie):
        ii = (x >= x[i] - xhalf) & (x <= x[i] + xhalf)
        if sum(ii) < window:
            ## number of points not enough
            ii = np.flatnonzero(ii)
            iadd = half - (ii.size - ii.size % 2) // 2
            ii = slice(max(ii.min() - iadd, 0), \
                min(ii.max() + iadd + 1, x.size))
        p = np.polyfit(x[ii] - x[i], f[ii], order)
        g[0][i] = p[-1]
        aux = 1
        for d in range(1, min(deriv, order) + 1):
            g[d][i] = p[-1-d] * np.math.factorial(d)
    ## for deriv > order, derivative is 0
    for d in range(min(deriv, order) + 1, deriv + 1):
        g[d][:] = 0
    if len(g) == 1:
        g = g[0]
    return g


def savgol_irregular_interp (f, x, xwindow, order=4, deriv=0, dx=None):
    """
    Applies a Savitzky-Golay low-pass filter to irregularly sampled data.
    
    This function applies the Savitzky-Golay filter, linearly interpolating
    the input data to regularize it. Although a polynomial is fitted to
    each record of the input data, the least-squares problem requires only
    one coefficients matrix, thus the pseudo-invertion of the matrix is
    performed only once. The exception is for recors at the boundaries,
    where a polynomial is fitted for each reacord.
    
    Parameters
    ----------
    f : array_like, shape (M,)
        Input data, consisting of a sampled signal function of a dimension
        such as space or time, i.e. `f(x)`.
    x : array_like, shape (M,)
        Input data, with the samping time or position of the `f` values.
    xwindow : float
        Defines the window size, with dimension of `x`, used to filter the
        data. For each position `x[i]`, a polynomial will be fitted for
        all records of `f` that fall inside this window, i.e.:
        .. math ::
            - x_{window} / 2  <  x - x_i  <  x_{window} / 2

    order : int, optional
        Degree of the fitting polynomial.
    deriv : int, optional
        The order of the derivative to compute. This must be a nonnegative
        integer. The default is 0, which means to filter the data without
        differentiating.
    dx : float, optional
        Specify the sampling space `dx`. The algorithm computes a value
        for `dx` based on the mode of `x[1:] - x[:-1]`. This parameter
        may be used if another value for `dx` is preferred.
        
    Returns
    -------
    g : array_like, shape (deriv, M) or (M,)
        The filtered signal and its derivatives in order to `x`. If input
        parameter `deriv` is 0 a 1-D array is return with  shape (M,).

    See also
    --------
    scipy.signal.savgol_filter
    savgol_irregular_brute
    numpy.linalg.pinv

    References
    ----------
    A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by
    Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8),
    pp 1627-1639.
    """
    try:
        order = np.abs(np.int(order))
        deriv = np.abs(np.int(deriv))
    except ValueError as msg:
        raise ValueError("'order' and 'deriv' must be of type int")
    if order < 2:
        raise TypeError("polynomial 'order' should be >= 2")
    if deriv > order:
        print("Warning: 'deriv' > 'order'")
        print("derivatives higher than polynomial order will be 0")
    ## compute delta and window
    if None == dx:
        digits = int(abs(np.ceil(np.log10(np.finfo(float).eps)))) - 1
        from scipy.stats import mode
        dx = np.asscalar(mode(np.round(np.diff(x), digits))[0])
    window = int(np.ceil(xwindow / dx))
    window = max(window, order + 1)  # window >= order + 1
    window += (window + 1) % 2  # make it an odd integer
    print("window: %d points" % window)
    half = (window - 1) // 2
    owindow = order + 1 + (order + 2) % 2
    ohalf = (owindow - 1) // 2
    g = []  # g[o,...] where 'o' represents the n'th derivative
    for d in range(deriv + 1):
        g.append(np.zeros(np.shape(f), float) * np.nan)
    ## from dx and window update spacing
    xwindow = (window - 1) * dx
    # xhalf = half * dx
    xhalf = xwindow / 2.
    j = np.arange(-half, half + 1)
    xj = j * dx
    print("updated xwindow: %g" % xwindow)
    ## west boundary
    ii = np.sum(x <= x[0] + xwindow)  # points inside xwindow
    ii += (ii + 1) % 2  # make it an odd integer
    iw = np.flatnonzero(x >= x[0] + xhalf).min()  # points up to xhalf
    #ii = max(ii, window)  # increase if smaller than window
    #iw = max(iw, half)  # increase if smaller than half
    ii = max(ii, owindow)  # increase if smaller than owindow
    iw = max(iw, ohalf)  # increase if smaller than ohalf
    for i in range(0, iw):
        p = np.polyfit(x[:ii]-x[i], f[:ii], order)
        g[0][i] = p[-1]
        for d in range(min(deriv, order) + 1):
            g[d][i] = p[-1-d] * np.math.factorial(d)
    ## east boundary
    ii = np.sum(x >= x[-1] - xwindow)  # points inside xwindow
    ii += (ii + 1) % 2  # make it an odd integer
    ie = np.flatnonzero(x <= x[-1] - xhalf).max() # index of xhalf
    #ii = max(ii, window)  # increase if smaller than window
    #ie = min(ie, np.size(x) - half - 1)  # decrease if higher than half
    ii = max(ii, owindow)  # increase if smaller than owindow
    ie = min(ie, np.size(x) - ohalf - 1)  # decrease if higher than ohalf
    for i in range(ie + 1, np.size(x)):
        p = np.polyfit(x[-ii:]-x[i], f[-ii:], order)
        g[0][i] = p[-1]
        for d in range(min(deriv, order) + 1):
            g[d][i] = p[-1-d] * np.math.factorial(d)
    ## inner points, compute coefficients
    o = np.arange(order + 1)
    j = np.arange(-half, half + 1)
    A = np.array([[jj**oo for oo in o] for jj in j])
    Apinv = np.linalg.pinv(A)
    xj = j * dx
    nim1 = np.size(x) - 1
    for i in range(iw, ie + 1):
        ki = max(0, np.flatnonzero(x > x[i] - xhalf).min() - 1)
        ke = min(nim1, np.flatnonzero(x < x[i] + xhalf).max() + 1) + 1
        xdiff = np.diff(x[ki:ke])
        if np.allclose(dx, np.diff(x[ki:ke])):
            ff = f[j + i]
        else:
            ff = np.interp(xj, x[ki:ke] - x[i], f[ki:ke])
        g[0][i] = np.dot(Apinv[0], ff)
        for d in range(1, min(deriv, order) + 1):
            g[d][i] = np.dot(Apinv[d], ff) * np.math.factorial(d) / dx**d
    ## for deriv > order, derivative is 0
    for d in range(min(deriv, order) + 1, deriv + 1):
        g[d][:] = 0
    if len(g) == 1:
        g = g[0]
    return g



if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    dx = 3.
    window = 11
    xwindow = (window - 1) * dx
    print("window  = %s" % window)
    print("xwindow = %s" % xwindow)
    k = np.arange(0, 6 * window + 5)
    u = np.random.randint(-9, 9, k.size)
    #u += k**2 / (4*window)
    x = np.arange(k.size) * dx
    #dudx = np.diff(u) / np.diff(x)
    dudx = (u[2:] - u[:-2]) / (x[2:] - x[:-2])
    U1, dU1, = savgol_irregular_brute (u, x, xwindow, order=4, deriv=1)
    U2, dU2, = savgol_irregular_interp (u, x, xwindow, order=4, deriv=1)
    from scipy.signal import savgol_filter
    U0 = savgol_filter(u, window, 4, deriv=0, delta=dx)
    dU0 = savgol_filter(u, window, 4, deriv=1, delta=dx)
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title(r"Constant $\Delta x$")
    ax[0].plot(x, u, label=r"$u$", c='k', ls='-')
    ax[0].plot(x, U0, label=r"$<U>_0$", c='magenta', ls='-')
    ax[0].plot(x, U1, label=r"$<U>_1$", c='forestgreen', ls='-')
    ax[0].plot(x, U2, label=r"$<U>_2$", c='goldenrod', ls='-')
    ax[0].set_xlim(xmin=0, xmax=np.max(x))
    ax[0].set_xlabel(r"$x$")
    ax[0].set_xlabel(r"$u(x)$")
    ax[0].legend(loc='upper left', borderaxespad=.2, frameon=True)
    ax[1].plot(x[1:-1], dudx, label=r"$u$", c='k', ls='-')
    ax[1].plot(x, dU0, label=r"$dU_0/dx$", c='magenta', ls='-')
    ax[1].plot(x, dU1, label=r"$dU_1/dx$", c='forestgreen', ls='-')
    ax[1].plot(x, dU2, label=r"$dU_2/dx$", c='goldenrod', ls='-')
    ax[1].set_xlim(xmin=0, xmax=np.max(x))
    ax[1].set_xlabel(r"$x$")
    ax[1].set_xlabel(r"$du/dx$")
    ax[1].legend(loc='upper left', borderaxespad=.2, frameon=True)
    fig.tight_layout(w_pad=.15, h_pad=.15, rect=(0,0,1,1))
    plt.show()
    for i in range(0, x.size, window // 2):
        x[i:] += .5*dx
    #dudx = np.diff(u) / np.diff(x)
    dudx = (u[2:] - u[:-2]) / (x[2:] - x[:-2])
    U1, dU1, = savgol_irregular_brute (u, x, xwindow, order=4, deriv=1)
    U2, dU2, = savgol_irregular_interp (u, x, xwindow, order=4, deriv=1)
    from scipy.signal import savgol_filter
    U0 = savgol_filter(u, window, 4, deriv=0, delta=dx)
    dU0 = savgol_filter(u, window, 4, deriv=1, delta=dx)
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title(r"Variable $\Delta x$")
    ax[0].plot(x, u, label=r"$u$", c='k', ls='-')
    ax[0].plot(x, U0, label=r"$<U>_0$", c='magenta', ls='-')
    ax[0].plot(x, U1, label=r"$<U>_1$", c='forestgreen', ls='-')
    ax[0].plot(x, U2, label=r"$<U>_2$", c='goldenrod', ls='-')
    ax[0].set_xlim(xmin=0, xmax=np.max(x))
    ax[0].set_xlabel(r"$x$")
    ax[0].set_xlabel(r"$u(x)$")
    ax[0].legend(loc='upper left', borderaxespad=.2, frameon=True)
    ax[1].plot(x[1:-1], dudx, label=r"$u$", c='k', ls='-')
    ax[1].plot(x, dU0, label=r"$dU_0/dx$", c='magenta', ls='-')
    ax[1].plot(x, dU1, label=r"$dU_1/dx$", c='forestgreen', ls='-')
    ax[1].plot(x, dU2, label=r"$dU_2/dx$", c='goldenrod', ls='-')
    ax[1].set_xlim(xmin=0, xmax=np.max(x))
    ax[1].set_xlabel(r"$x$")
    ax[1].set_xlabel(r"$du/dx$")
    ax[1].legend(loc='upper left', borderaxespad=.2, frameon=True)
    fig.tight_layout(w_pad=.15, h_pad=.15, rect=(0,0,1,1))
    plt.show()
    U3, dU3, = savgol_irregular_interp (u, x, xwindow, order=4, deriv=1,\
        dx=.2*dx)
    from scipy.signal import savgol_filter
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title(r"Variable $\Delta x$")
    ax[0].plot(x, u, label=r"$u$", c='k', ls='-')
    ax[0].plot(x, U2, label=r"$<U>$", c='goldenrod', ls='-')
    ax[0].plot(x, U3, label=r"$<U>$, with $0.2 * \Delta x$",\
            c='blue', ls='-')
    ax[0].set_xlim(xmin=0, xmax=np.max(x))
    ax[0].set_xlabel(r"$x$")
    ax[0].set_xlabel(r"$u(x)$")
    ax[0].legend(loc='upper left', borderaxespad=.2, frameon=True)
    ax[1].plot(x[1:-1], dudx, label=r"$u$", c='k', ls='-')
    ax[1].plot(x, dU2, label=r"$dU_2/dx$", c='goldenrod', ls='-')
    ax[1].plot(x, dU3, label=r"$dU_3/dx$, with $0.2 * \Delta x$",\
        c='blue', ls='-')
    ax[1].set_xlim(xmin=0, xmax=np.max(x))
    ax[1].set_xlabel(r"$x$")
    ax[1].set_xlabel(r"$du/dx$")
    ax[1].legend(loc='upper left', borderaxespad=.2, frameon=True)
    fig.tight_layout(w_pad=.15, h_pad=.15, rect=(0,0,1,1))
    plt.show()

