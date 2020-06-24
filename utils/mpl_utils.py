#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 5 May 2020

@author: Przemyslaw Zielinski
"""

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
    from matplotlib import rcParams

    document_width = rcParams["figure.figsize"][0]
    fig_width = textwidth_fraction * document_width
    fig_height = fig_width * height_to_width_ratio

    return (fig_width, fig_height)
