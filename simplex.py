"""
simplex
=======

Barycentric colour map for visualising output of 3-class probabilistic models.
Maps a 3-class prediction (p0, p1, p2) onto the probability simplex.
Each class is assigned a base colour and a prediction is rendered as
the probability-weighted blend of those colours.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from matplotlib.colors import to_rgb

# Equilateral-triangle corners
_CORNERS_XY = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.5, np.sqrt(3) / 2]])


class SimplexColormap:
    """
    Map 3-class probability vectors to RGBA colours.

    Parameters
    ----------
    colors : sequence of 3 matplotlib colours, optional
        Base colour for each class. Defaults to a red / green / blue triad.
    sharpen : float, default 1.0
        Exponent applied to the probabilities before blending (then
        renormalised). > 1 pushes colours toward the winning class, tightening
        the apparent boundaries; < 1 softens them. 1.0 is a faithful blend.
    bg : matplotlib colour, default "white"
        Colour that maximal uncertainty blends toward. With saturated base
        colours and bg="white", uncertain regions look pale.
    gamma : float, default 1.0
        Optional gamma applied to the final RGB for perceptual tuning.

    Notes
    -----
    The call signature mirrors a matplotlib ``Colormap`` (``cmap(values)`` -> RGBA),
    except ``values`` is a (N, 3) array of probabilities rather than a
    scalar field, so it slots into the same plotting idioms.
    """

    def __init__(self, colors=None, sharpen=1.0, bg="white", gamma=1.0):
        if colors is None:
            colors = ["#d62728", "#2ca02c", "#1f77b4"]  # red, green, blue
        self.colors = np.array([to_rgb(c) for c in colors])  # (3, 3)
        if self.colors.shape != (3, 3):
            raise ValueError("Provide exactly 3 base colours for 3 classes.")
        self.sharpen = float(sharpen)
        self.bg = np.array(to_rgb(bg))
        self.gamma = float(gamma)

    def __call__(self, probs, alpha=1.0):
        """probs: (N, 3) or (3,) probabilities -> (N, 4) or (4,) RGBA."""
        P = np.asarray(probs, dtype=float)
        single = P.ndim == 1
        P = np.atleast_2d(P)
        P = P / P.sum(axis=1, keepdims=True)  # guard against tiny drift

        if self.sharpen != 1.0:
            P = P ** self.sharpen
            P = P / P.sum(axis=1, keepdims=True)

        rgb = P @ self.colors                 # weighted blend of class colours
        # Pull toward the background by how uncertain the point is.
        # certainty = (max p - 1/3) / (1 - 1/3)  in [0, 1]
        certainty = (P.max(axis=1) - 1 / 3) / (1 - 1 / 3)
        rgb = self.bg + (rgb - self.bg) * certainty[:, None]

        if self.gamma != 1.0:
            rgb = np.clip(rgb, 0, 1) ** self.gamma

        rgba = np.empty((rgb.shape[0], 4))
        rgba[:, :3] = np.clip(rgb, 0, 1)
        rgba[:, 3] = alpha
        return rgba[0] if single else rgba


def _as_proba_fn(model):
    """Accept a callable returning probs, or an object with predict_proba."""
    if callable(model):
        return model
    if hasattr(model, "predict_proba"):
        return model.predict_proba
    raise TypeError("Pass a callable -> (N, 3) probs, or an object with "
                    "predict_proba.")


def plot_decision_regions(model, xlim, ylim, ax=None, cmap=None, n=300,
                          boundaries=True, boundary_kw=None):
    """Colour a 2D feature space by a 3-class model's predicted probabilities.

    Parameters
    ----------
    model : callable or estimator
        ``model(XY)`` or ``model.predict_proba(XY)`` must return (N, 3) probs
        for XY of shape (N, 2).
    xlim, ylim : (lo, hi) tuples giving the plotting extent.
    cmap : SimplexColormap, optional. A default red/green/blue map is used.
    n : grid resolution per axis.
    boundaries : if True, draw argmax decision boundaries on top.
    boundary_kw : dict of kwargs forwarded to ax.contour for the boundaries.

    Returns the Axes.
    """
    proba = _as_proba_fn(model)
    cmap = cmap or SimplexColormap()
    ax = ax or plt.gca()

    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    xx, yy = np.meshgrid(xs, ys)
    XY = np.column_stack([xx.ravel(), yy.ravel()])

    P = np.asarray(proba(XY), dtype=float)
    rgba = cmap(P).reshape(n, n, 4)
    ax.imshow(rgba, origin="lower", extent=[*xlim, *ylim],
              aspect="auto", interpolation="bilinear")

    if boundaries:
        labels = P.argmax(axis=1).reshape(n, n).astype(float)
        bkw = dict(levels=[0.5, 1.5], colors="k", linewidths=1.2)
        if boundary_kw:
            bkw.update(boundary_kw)
        ax.contour(xx, yy, labels, **bkw)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


def barycentric_to_xy(P):
    """(N, 3) probabilities -> (N, 2) equilateral-triangle coordinates."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    return P @ _CORNERS_XY


def plot_simplex(model=None, ax=None, cmap=None, n=300, scalar=None,
                 scalar_cmap="viridis", labels=("class 0", "class 1", "class 2")):
    """Render the probability triangle itself.

    By default each point is coloured by the SimplexColormap (its own
    probability vector). Pass ``scalar`` (a function (N,3)->(N,)) to colour by a
    scalar field with an ordinary 1D colormap instead.
    """
    ax = ax or plt.gca()
    cmap = cmap or SimplexColormap()

    g = np.linspace(0, 1, n)
    a, b = np.meshgrid(g, g)
    a, b = a.ravel(), b.ravel()
    c = 1 - a - b
    keep = c >= 0
    P = np.column_stack([a[keep], b[keep], c[keep]])
    xy = barycentric_to_xy(P)

    triang = tri.Triangulation(xy[:, 0], xy[:, 1])
    if scalar is not None:
        ax.tripcolor(triang, scalar(P), shading="gouraud", cmap=scalar_cmap)
    else:
        # tripcolor needs a scalar; emulate per-vertex RGBA via a fine scatter.
        ax.scatter(xy[:, 0], xy[:, 1], c=cmap(P), s=6, marker="s",
                   edgecolors="none")

    closed = np.vstack([_CORNERS_XY, _CORNERS_XY[0]])
    ax.plot(closed[:, 0], closed[:, 1], "k-", lw=1)
    ha = ("right", "left", "center")
    va = ("top", "top", "bottom")
    for (x, y), lab, h, v in zip(_CORNERS_XY, labels, ha, va):
        ax.text(x, y, f" {lab} ", ha=h, va=v)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def simplex_legend(ax=None, cmap=None, n=120,
                   labels=("class 0", "class 1", "class 2")):
    """Draw a small triangular colour key so viewers can decode the blend."""
    return plot_simplex(ax=ax, cmap=cmap, n=n, labels=labels)
