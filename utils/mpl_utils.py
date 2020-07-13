#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 5 May 2020

@author: Przemyslaw Zielinski
"""

from matplotlib import rcParams
from matplotlib.gridspec import GridSpec


def set_figsize(textwidth_fraction=1.0, height_to_width_ratio=0.5):
    """
    Set aesthetic figure dimensions to avoid scaling in latex.

    This function assumes that the initial width stored in
    the first coordinate of rcParams["figure.figsize"]
    is equal to \textwidth size of the latex dacument.

    Parameters
    ----------
    fraction: float
    Fraction of the document's textwidth, equal to rcParams["figure.figsize"],
    which you wish the current figure to occupy. To import the figure in latex
    use "\includegraphics[width=fraction\textwidth]{...}"

    ratio: float
    The desired aspect ratio of width / height.

    Returns
    -------
    fig_dim: tuple
    Dimensions of figure in inches
    """
    document_width = rcParams["figure.figsize"][0]
    fig_width = textwidth_fraction * document_width
    fig_height = fig_width * height_to_width_ratio

    return (fig_width, fig_height)

def coord_grid(fig, darray, xylim=[-1.1, 1.1], var='x'):
    gs = GridSpec(3, 3, figure=fig, hspace=0.05, wspace=0.05)

    for n, coord in enumerate(darray.T[:-1]):
        for m in range(n+1):
            ax = fig.add_subplot(gs[n, m])
    #         ax.set_title(f"{n}, {m-1}")
            ax.scatter(darray[:,m-1], coord, s=0.5)
            ax.set_xlim(xylim)
            ax.set_ylim(xylim)
            ax.set_aspect('equal')
            if n == 2:
                ax.set_xlabel(f'${var}_{(m-1) % 4}$')
    #             ax.set_xticks([-.25, 0.0, +.25])
            else:
                ax.set_xticklabels([])
            if m == 0:
                ax.set_ylabel(f'${var}_{n}$', rotation=0)
    #             ax.set_yticks([-.25, 0.0, +.25])
            else:
                ax.set_yticklabels([])
    return gs
