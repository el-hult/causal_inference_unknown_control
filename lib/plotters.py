import numbers
import warnings
from typing import Sequence, Callable

import matplotlib
import networkx as nx
import numpy as np

from matplotlib import pyplot as plt, colors as mcolors, cm as mcolormaps


def draw_graph(w, title, out_path=None):
    """Take a adjacency matrix w, and plot it as a graph.

    Args:
        out_path (Optional[pathlike]): a place to save the plot
    """
    G = nx.from_numpy_array(create_using=nx.DiGraph, A=w)
    d = w.shape[0]
    nx.set_node_attributes(
        G=G,
        name="label",
        values={0: "X", 1: "Y", **{i: f"Z{i - 1}" for i in range(2, d)}},
    )

    layout = "circ"
    connectionstyle = "arc3,rad=0.1"

    fig, ax1 = plt.subplots()

    weights = list(nx.get_edge_attributes(G, "weight").values())
    if len(weights) == 0:
        cmax = 1
    else:
        cmax = max([abs(w) for w in weights])
    cmin = -cmax
    ccenter = 0
    norm = mcolors.TwoSlopeNorm(vmin=cmin, vmax=cmax, vcenter=ccenter)
    cmap = mcolormaps.get_cmap("PiYG")
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.abs(np.linspace(-1, 1, cmap.N))
    my_cmap = mcolors.ListedColormap(my_cmap)
    colors_as_rgba_tuples = [my_cmap(norm(w)) for w in weights]
    if layout is None or layout == "random":
        pos = nx.random_layout(G)
    elif layout == "circ":
        pos = nx.circular_layout(G)
    else:
        raise ValueError

    with warnings.catch_warnings():
        """networkx uses some deprecated matplotlib functions....
         I'm ignoring that...
        """
        warnings.filterwarnings(
            action="ignore",
            category=matplotlib.MatplotlibDeprecationWarning,
            module="networkx",
        )
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(
            G, pos, edge_color=colors_as_rgba_tuples, connectionstyle=connectionstyle
        )
        label_dict = nx.get_node_attributes(G, "label")
        nx.draw_networkx_labels(G, pos, labels=label_dict)

    fig.colorbar(mcolormaps.ScalarMappable(norm=norm, cmap=cmap))
    ax1.set_title(title)

    if out_path is not None:
        fig.savefig(out_path)
        plt.close(fig)


def plot_contours_in_2d(
    constrs: Sequence[Callable], ax, box_scale, n_points, contour_opts=None
):
    if isinstance(box_scale, numbers.Number):
        l, r, t, b = (box_scale,) * 4
    elif len(box_scale) == 4:
        l, r, t, b = box_scale
    else:
        raise ValueError
    contour_opts = contour_opts or {}
    x = np.linspace(l, r, n_points)
    y = np.linspace(b, t, n_points)
    X, Y = np.meshgrid(x, y)
    fs = np.zeros((n_points, n_points, len(constrs)))
    for j in range(n_points):
        for k in range(n_points):
            for idx, f in enumerate(constrs):
                fs[j, k, idx] = f(np.array([X[j, k], Y[j, k]]))

    for idx, constr in enumerate(constrs):
        ax.contour(X, Y, fs[:, :, idx], **contour_opts)
