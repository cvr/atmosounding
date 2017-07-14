#!/usr/bin/env python
# vim: set fileencoding=utf-8 fileformat=unix :
# -*- coding: utf-8 -*-
# vim: set ts=8 et sw=4 sts=4 sta :
########################################################################
#
# Copyright (C) 2011 by Carlos Veiga Rodrigues. All rights reserved.
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

def d1 (u, x, stencil=5):
    """
    Computes first derivative of an irregularly sampled signal through
    finite differences, considering a stencil of 3 or 5 points.
    
    Parameters
    ----------
    u : array_like, shape (M,)
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

    stencil : int, optional
        Choose the size of the computational stencil used to compute
        the finite differences. A `stencil` of 3 or 5 (default) may be used.

    Returns
    -------
    du : array_like, shape (M,)
        The first derivative in respect to `x`.
    """
    du = np.zeros(u.shape, float) * np.nan
    x = np.array(x, float)
    if stencil == 3:
        ## central finite differences with stencil [i-1, i, i+1]
        ## indices
        w = slice(0, -2)  # west
        p = slice(1, -1)  # centre
        e = slice(2, None)  # east
        f = p  # where to evaluate the derivative
        # cw = (x[f]-x[p]) * (x[f]-x[e]) / ((x[w]-x[p]) * (x[w]-x[e]))
        # cp = (x[f]-x[w]) * (x[f]-x[e]) / ((x[p]-x[w]) * (x[p]-x[e]))
        # ce = (x[f]-x[w]) * (x[f]-x[p]) / ((x[e]-x[w]) * (x[e]-x[p]))
        dcw = ((x[f]-x[p]) + (x[f]-x[e])) / ((x[w]-x[p]) * (x[w]-x[e]))
        #dcp = ((x[f]-x[w]) + (x[f]-x[e])) / ((x[p]-x[w]) * (x[p]-x[e]))
        dce = ((x[f]-x[w]) + (x[f]-x[p])) / ((x[e]-x[w]) * (x[e]-x[p]))
        dcp = - dcw - dce
        du[f] = u[w]*dcw + u[p]*dcp + u[e]*dce
    elif stencil == 5:
        ## central finite differences with stencil [i-2, i-1, i, i+1, i+2]
        ## indices
        W = slice(0, -4)  # west west
        w = slice(1, -3)  # west
        p = slice(2, -2)  # central
        e = slice(3, -1)  # east
        E = slice(4, None)  # east east
        f = p  # where to evaluate the derivative
        # cW = (x[f]-x[w]) * (x[f]-x[p]) * (x[f]-x[e]) * (x[f]-x[E]) \
        #     / ((x[W]-x[w]) * (x[W]-x[p]) * (x[W]-x[e]) * (x[W]-x[E]))
        # cw = (x[f]-x[W]) * (x[f]-x[p]) * (x[f]-x[e]) * (x[f]-x[E]) \
        #     / ((x[w]-x[W]) * (x[w]-x[p]) * (x[w]-x[e]) * (x[w]-x[E]))
        # #cp = (x[f]-x[W]) * (x[f]-x[w]) * (x[f]-x[e]) * (x[f]-x[E]) \
        # #    / ((x[p]-x[W]) * (x[p]-x[w]) * (x[p]-x[e]) * (x[p]-x[E]))
        # ce = (x[f]-x[W]) * (x[f]-x[w]) * (x[f]-x[p]) * (x[f]-x[E]) \
        #     / ((x[e]-x[W]) * (x[e]-x[w]) * (x[e]-x[p]) * (x[e]-x[E]))
        # cE = (x[f]-x[W]) * (x[f]-x[w]) * (x[f]-x[p]) * (x[f]-x[e]) \
        #     / ((x[E]-x[W]) * (x[E]-x[w]) * (x[E]-x[p]) * (x[E]-x[e]))
        # cp = 1 - cW - cw - ce - cE
        dcW = ( \
            #+ (x[f]-x[p]) * (x[f]-x[e]) * (x[f]-x[E]) \
            + (x[f]-x[w]) * (x[f]-x[e]) * (x[f]-x[E]) \
            #+ (x[f]-x[w]) * (x[f]-x[p]) * (x[f]-x[E]) \
            #+ (x[f]-x[w]) * (x[f]-x[p]) * (x[f]-x[e]) \
            ) / ((x[W]-x[w]) * (x[W]-x[p]) * (x[W]-x[e]) * (x[W]-x[E]))
        dcw = ( \
            #+ (x[f]-x[p]) * (x[f]-x[e]) * (x[f]-x[E]) \
            + (x[f]-x[W]) * (x[f]-x[e]) * (x[f]-x[E]) \
            #+ (x[f]-x[W]) * (x[f]-x[p]) * (x[f]-x[E]) \
            #+ (x[f]-x[W]) * (x[f]-x[p]) * (x[f]-x[e]) \
            ) / ((x[w]-x[W]) * (x[w]-x[p]) * (x[w]-x[e]) * (x[w]-x[E]))
        #dcp = ( \
        #    + (x[f]-x[w]) * (x[f]-x[e]) * (x[f]-x[E]) \
        #    + (x[f]-x[W]) * (x[f]-x[e]) * (x[f]-x[E]) \
        #    + (x[f]-x[W]) * (x[f]-x[w]) * (x[f]-x[E]) \
        #    + (x[f]-x[W]) * (x[f]-x[w]) * (x[f]-x[e]) \
        #    ) / ((x[p]-x[W]) * (x[p]-x[w]) * (x[p]-x[e]) * (x[p]-x[E]))
        dce = ( \
            #+ (x[f]-x[w]) * (x[f]-x[p]) * (x[f]-x[E]) \
            #+ (x[f]-x[W]) * (x[f]-x[p]) * (x[f]-x[E]) \
            + (x[f]-x[W]) * (x[f]-x[w]) * (x[f]-x[E]) \
            #+ (x[f]-x[W]) * (x[f]-x[w]) * (x[f]-x[p]) \
            ) / ((x[e]-x[W]) * (x[e]-x[w]) * (x[e]-x[p]) * (x[e]-x[E]))
        dcE = ( \
            #+ (x[f]-x[w]) * (x[f]-x[p]) * (x[f]-x[e]) \
            #+ (x[f]-x[W]) * (x[f]-x[p]) * (x[f]-x[e]) \
            + (x[f]-x[W]) * (x[f]-x[w]) * (x[f]-x[e]) \
            #+ (x[f]-x[W]) * (x[f]-x[w]) * (x[f]-x[p]) \
            ) / ((x[E]-x[W]) * (x[E]-x[w]) * (x[E]-x[p]) * (x[E]-x[e]))
        dcp = -dcW - dcw - dce - dcE
        du[f] = u[W]*dcW + u[w]*dcw + u[p]*dcp + u[e]*dce + u[E]*dcE
        ## near boundary points - reduce stencil
        f, w, p, e = 1, 0, 1, 2
        dcw = ((x[f]-x[p]) + (x[f]-x[e])) / ((x[w]-x[p]) * (x[w]-x[e]))
        #dcp = ((x[f]-x[w]) + (x[f]-x[e])) / ((x[p]-x[w]) * (x[p]-x[e]))
        dce = ((x[f]-x[w]) + (x[f]-x[p])) / ((x[e]-x[w]) * (x[e]-x[p]))
        dcp = - dcw - dce
        du[f] = u[w]*dcw + u[p]*dcp + u[e]*dce
        f, w, p, e = -2, -3, -2, -1
        dcw = ((x[f]-x[p]) + (x[f]-x[e])) / ((x[w]-x[p]) * (x[w]-x[e]))
        # dcp = ((x[f]-x[w]) + (x[f]-x[e])) / ((x[p]-x[w]) * (x[p]-x[e]))
        dce = ((x[f]-x[w]) + (x[f]-x[p])) / ((x[e]-x[w]) * (x[e]-x[p]))
        dcp = - dcw - dce
        du[f] = u[w]*dcw + u[p]*dcp + u[e]*dce
    ## one-sided derivatives for boundary points
    du[0] = (u[1] - u[0]) / (x[1] - x[0])
    du[-1] = (u[-1] - u[-2]) / (x[-1] - x[-2])
    return du

def d2 (u, x, stencil=5):
    """
    Computes second derivative of an irregularly sampled signal through
    finite differences, considering a stencil of 3 or 5 points.
    
    Parameters
    ----------
    u : array_like, shape (M,)
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

    stencil : int, optional
        Choose the size of the computational stencil used to compute
        the finite differences. A `stencil` of 3 or 5 (default) may be used.

    Returns
    -------
    ddu : array_like, shape (M,)
        The second derivative in respect to `x`.
    """
    ddu = np.zeros(u.shape, float) * np.nan
    x = np.array(x, float)
    if stencil == 3:
        ## central finite differences with stencil [i-1, i, i+1]
        ## indices
        w = slice(0, -2)  # west
        p = slice(1, -1)  # centre
        e = slice(2, None)  # east
        f = p  # where to evaluate the derivative
        ddcw = 2. / ((x[w]-x[p]) * (x[w]-x[e]))
        #ddcp = 2. / ((x[p]-x[w]) * (x[p]-x[e]))
        ddce = 2. / ((x[e]-x[w]) * (x[e]-x[p]))
        ddcp = - ddcw - ddce
        ddu[f] = u[w]*ddcw + u[p]*ddcp + u[e]*ddce
    elif stencil == 5:
        ## central finite differences with stencil [i-2, i-1, i, i+1, i+2]
        ## indices
        W = slice(0, -4)  # west west
        w = slice(1, -3)  # west
        p = slice(2, -2)  # central
        e = slice(3, -1)  # east
        E = slice(4, None)  # east east
        f = p  # where to evaluate the derivative
        ddcW = 2 * ( \
            + (x[f]-x[e]) * (x[f]-x[E]) + (x[f]-x[p]) * (x[f]-x[E]) \
            + (x[f]-x[p]) * (x[f]-x[e]) + (x[f]-x[w]) * (x[f]-x[E]) \
            + (x[f]-x[w]) * (x[f]-x[e]) + (x[f]-x[w]) * (x[f]-x[p]) \
            ) / ((x[W]-x[w]) * (x[W]-x[p]) * (x[W]-x[e]) * (x[W]-x[E]))
        ddcw = 2 * ( \
            + (x[f]-x[e]) * (x[f]-x[E]) + (x[f]-x[p]) * (x[f]-x[E]) \
            + (x[f]-x[p]) * (x[f]-x[e]) + (x[f]-x[W]) * (x[f]-x[E]) \
            + (x[f]-x[W]) * (x[f]-x[e]) + (x[f]-x[W]) * (x[f]-x[p]) \
            ) / ((x[w]-x[W]) * (x[w]-x[p]) * (x[w]-x[e]) * (x[w]-x[E]))
        ddcp = 2 * ( \
            + (x[f]-x[e]) * (x[f]-x[E]) + (x[f]-x[w]) * (x[f]-x[E]) \
            + (x[f]-x[w]) * (x[f]-x[e]) + (x[f]-x[W]) * (x[f]-x[E]) \
            + (x[f]-x[W]) * (x[f]-x[e]) + (x[f]-x[w]) * (x[f]-x[E]) \
            ) / ((x[p]-x[W]) * (x[p]-x[w]) * (x[p]-x[e]) * (x[p]-x[E]))
        #ddcp = 2 * ( \
        #    + (x[f]-x[e]) * (x[f]-x[E]) + (x[f]-x[w]) * (x[f]-x[E]) \
        #    + (x[f]-x[w]) * (x[f]-x[e]) + (x[f]-x[W]) * (x[f]-x[E]) \
        #    + (x[f]-x[W]) * (x[f]-x[e]) + (x[f]-x[w]) * (x[f]-x[E]) \
        #    ) / ((x[p]-x[W]) * (x[p]-x[w]) * (x[p]-x[e]) * (x[p]-x[E]))
        ddce = 2 * ( \
            + (x[f]-x[p]) * (x[f]-x[E]) + (x[f]-x[w]) * (x[f]-x[E]) \
            + (x[f]-x[w]) * (x[f]-x[p]) + (x[f]-x[W]) * (x[f]-x[E]) \
            + (x[f]-x[W]) * (x[f]-x[p]) + (x[f]-x[W]) * (x[f]-x[w]) \
            ) / ((x[e]-x[W]) * (x[e]-x[w]) * (x[e]-x[p]) * (x[e]-x[E]))
        ddcE = 2 * ( \
            + (x[f]-x[p]) * (x[f]-x[e]) + (x[f]-x[w]) * (x[f]-x[e]) \
            + (x[f]-x[w]) * (x[f]-x[p]) + (x[f]-x[W]) * (x[f]-x[e]) \
            + (x[f]-x[W]) * (x[f]-x[p]) + (x[f]-x[W]) * (x[f]-x[w]) \
            ) / ((x[E]-x[W]) * (x[E]-x[w]) * (x[E]-x[p]) * (x[E]-x[e]))
        ddcp = - ddcW - ddcw - ddce - ddcE
        ddu[f] = u[W]*ddcW + u[w]*ddcw + u[p]*ddcp + u[e]*ddce + u[E]*ddcE
        ## near boundary points - reduce stencil
        f, w, p, e = 1, 0, 1, 2
        ddcw = 2. / ((x[w]-x[p]) * (x[w]-x[e]))
        #ddcp = 2. / ((x[p]-x[w]) * (x[p]-x[e]))
        ddce = 2. / ((x[e]-x[w]) * (x[e]-x[p]))
        ddcp = - ddcw - ddce
        ddu[f] = u[w]*ddcw + u[p]*ddcp + u[e]*ddce
        f, w, p, e = -2, -3, -2, -1
        ddcw = 2. / ((x[w]-x[p]) * (x[w]-x[e]))
        # ddcp = 2. / ((x[p]-x[w]) * (x[p]-x[e]))
        ddce = 2. / ((x[e]-x[w]) * (x[e]-x[p]))
        ddcp = - ddcw - ddce
        ddu[f] = u[w]*ddcw + u[p]*ddcp + u[e]*ddce
    ## one-sided derivatives for boundary points
    ddu[0] = ddu[1]
    ddu[-1] = ddu[-2]
    return ddu



if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    dx = 3.
    window = 11
    k = np.arange(0, 6 * window + 5)
    x = np.arange(k.size) * dx
    u = x**3 / 10. + np.exp(.05*x)
    dudx = 3 * x**2 / 10. + .05*np.exp(.05*x)
    d2udx2 = 3 * 2 * x / 10. + .05*.05*np.exp(.05*x)
    dudx_3 = d1 (u, x, stencil=3)
    dudx_5 = d1 (u, x, stencil=5)
    d2udx2_3 = d2 (u, x, stencil=3)
    d2udx2_5 = d2 (u, x, stencil=5)
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title(r"Constant $\Delta x$")
    ax[0].plot(x, dudx, label=r"exact", c='k', ls='-')
    ax[0].plot(x, dudx_3, label=r"$du/dx$, stencil=3", c='goldenrod', ls='-')
    ax[0].plot(x, dudx_5, label=r"$du/dx$, stencil=5", c='forestgreen', ls='-')
    ax[0].set_xlim(xmin=0, xmax=np.max(x))
    ax[0].set_xlabel(r"$x$")
    ax[0].set_xlabel(r"$du/dx$")
    ax[0].legend(loc='upper left', borderaxespad=.2, frameon=True)
    ax[1].plot(x, d2udx2, label=r"exact", c='k', ls='-')
    ax[1].plot(x, d2udx2_3, label=r"stencil=3", c='goldenrod', ls='-')
    ax[1].plot(x, d2udx2_5, label=r"stencil=5", c='forestgreen', ls='-')
    ax[1].set_xlim(xmin=0, xmax=np.max(x))
    ax[1].set_xlabel(r"$x$")
    ax[1].set_xlabel(r"$d^2u/dx^2$")
    ax[1].legend(loc='upper left', borderaxespad=.2, frameon=True)
    fig.tight_layout(w_pad=.15, h_pad=.15, rect=(0,0,1,1))
    plt.show()

