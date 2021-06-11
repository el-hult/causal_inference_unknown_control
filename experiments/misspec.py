"""
This experiment checks how various degrees of misspecification in the latent covariance
structure affects the confidence interval width and coverage.
It also computes the bias.
"""
# Standard library
import datetime
import json
import pathlib
import argparse
import os

# Third party party
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.stats as sps
import pandas as pd
import tqdm
import seaborn as sns

# this codebase
from lib.daggnn_util import simulate_random_dag
from lib.misc import printt, RandGraphSpec
from lib.relaxed_notears import (
    relaxed_notears,
    mest_covarance,
)
from lib.linear_algebra import make_Z_clear_first, make_L_no_diag
from lib.linear_sem import ace, ace_grad, selected_graphs

expname = "misspec"
fname_config = "config.json"
fname_raw = "raw.csv"
fname_txt = "results.txt"


def default_encoder(obj):
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if isinstance(obj, RandGraphSpec):
        return str(obj)
    else:
        raise TypeError


def daggify(G: nx.DiGraph):
    """In place removal of absolut0-smallest edges until we have a DAG"""
    while not nx.algorithms.dag.is_directed_acyclic_graph(G):
        u, v, _ = min(
            G.edges(data=True), key=lambda u_v_data: np.abs(u_v_data[2]["weight"])
        )
        G.remove_edge(u, v)
    return G


def shd(W1: np.ndarray, W2: np.ndarray, threshold=1e-3, prune=True):
    """Compute structural hamming distance between two graphs encoded by weight matrices
    
    How many edges must be added, removed or flipped to go from W1 to W2?
    """
    G1 = nx.from_numpy_array(W1, create_using=nx.DiGraph)
    G2 = nx.from_numpy_array(W2, create_using=nx.DiGraph)
    for G in G1, G2:
        to_remove = [
            (u, v)
            for u, v, data in G.edges(data=True)
            if np.abs(data["weight"]) < threshold
        ]
        for u, v in to_remove:
            G.remove_edge(u, v)

        if prune:
            daggify(G)

    assert nx.algorithms.dag.is_directed_acyclic_graph(G1)
    assert nx.algorithms.dag.is_directed_acyclic_graph(G2)

    s1 = set(G1.edges())
    s1back = set(G1.reverse().edges())
    s2 = set(G2.edges())

    inserts = len(s2 - s1)
    removals = len(s1 - s2)
    flips = len(s1back & s2)
    shd_ = inserts + removals - flips  # flips are double counted in inserts+removals
    return shd_


class SingleExperimentRun:
    def __init__(self, opts):
        self.opts = opts

    def __call__(self, seed):
        np.random.seed(seed)
        opts = self.opts
        if opts.rand_graph is not None:
            G = simulate_random_dag(
                d=opts.rand_graph.d,
                degree=opts.rand_graph.k * 2,
                graph_type="erdos-renyi",
            )
            W = nx.to_numpy_array(G)
        else:
            W = selected_graphs[opts.named_graph]
            G = nx.from_numpy_array(W, create_using=nx.DiGraph)
        sigma = np.diag(
            np.random.uniform(
                low=2 / (1 + opts.condmax),
                high=2 * opts.condmax / (1 + opts.condmax),
                size=G.number_of_nodes(),
            )
        )

        notears_options = dict()

        # True ACE
        d_nodes = W.shape[0]
        Z = make_Z_clear_first(d_nodes)
        id = np.eye(d_nodes)
        M = np.linalg.pinv(id - Z @ W.T)
        ace_true = M[1, 0]

        # full data limit ACE
        M2 = np.linalg.pinv(id - W.T)
        L_no_diag = make_L_no_diag(d_nodes)
        infinite_data_cov = M2 @ sigma @ M2.T
        w_initial = np.zeros_like(W)
        result_circ = relaxed_notears(
            infinite_data_cov,
            L=L_no_diag,
            W_initial=w_initial,
            dag_tolerance=opts.dag_tolerance_epsilon,
            optim_opts=notears_options,
        )
        assert result_circ["success"]
        theta_circ, W_circ, f_circ = (
            result_circ["theta"],
            result_circ["w"],
            result_circ["f_final"],
        )
        ace_circ = ace(theta_circ, L_no_diag)

        # finite data ACE-n and ACE-n standard error
        data = (
            np.random.multivariate_normal(
                mean=np.zeros(d_nodes), cov=sigma, size=opts.n_data
            )
            @ M2.T
        )
        finite_data_cov = np.cov(data, rowvar=False)
        result_n = relaxed_notears(
            finite_data_cov,
            L_no_diag,
            w_initial,
            opts.dag_tolerance_epsilon,
            notears_options,
        )
        assert result_n["success"]
        theta_n, w_n, f_n = result_n["theta"], result_n["w"], result_n["f_final"]
        ace_n = ace(theta_n, L_no_diag)

        noise_scale = f_n * 2 / d_nodes
        noise_cov = noise_scale * id
        covariance_matrix = mest_covarance(
            w_n, finite_data_cov, L_no_diag, True, noise_cov=noise_cov
        )
        gradient_of_ace_with_respect_to_theta = ace_grad(theta=theta_n, L=L_no_diag)
        ace_var = (
            gradient_of_ace_with_respect_to_theta
            @ covariance_matrix
            @ gradient_of_ace_with_respect_to_theta
            / opts.n_data
        )
        ace_n_se = np.sqrt(ace_var)

        # report results
        dic = dict(
            ace_circ=ace_circ,
            ace_true=ace_true,
            ace_n_se=ace_n_se,
            ace_n=ace_n,
            f_circ=f_circ,
            f_n=f_n,
        )
        dic["shd_circ"] = shd(W_circ, W)
        dic["condition"] = np.linalg.cond(sigma)
        return dic


def run_experiment(opts):
    ofolder = opts.ofolder

    from multiprocessing import Pool

    with Pool(6) as p:
        lazy_results = p.imap(SingleExperimentRun(opts), range(opts.n_repetitions))
        resdicts = list(
            tqdm.tqdm(lazy_results, total=opts.n_repetitions, smoothing=0.1)
        )

    df = pd.DataFrame(resdicts)
    df.to_csv(os.path.join(ofolder, fname_raw))


def post_process(opts):
    """Presentation code"""
    ofolder = opts.ofolder
    df = pd.read_csv(os.path.join(ofolder, fname_raw))
    q = sps.norm.ppf(1 - np.array(opts.confidence_level) / 2)
    df["q"] = q
    df["z_score"] = (df["ace_n"] - df["ace_circ"]) / df["ace_n_se"]
    df["is_covered"] = np.abs(df["z_score"]) < df["q"]

    fig, ax = plt.subplots()
    if df.ace_true.unique().size == 1:
        ax.axhline(df.ace_true.unique(), color="black")
    ax.scatter(
        df.condition, df.ace_circ,
    )
    ax.set_ylabel("z-score")
    ax.set_xlabel("True ACE")
    fig.savefig(os.path.join(ofolder, "scatter.png"))
    df[["condition", "ace_circ"]].to_csv(os.path.join(ofolder, "misspec_scatter.csv"))

    sns.pairplot(
        df[["ace_true", "ace_circ", "is_covered", "condition", "shd_circ"]],
        hue="is_covered",
        diag_kind="hist",
        diag_kws={"bins": "sqrt"},
    )
    fig = plt.gcf()
    fig.savefig(os.path.join(ofolder, "pairplot.png"))

    fig, ax = plt.subplots()
    n_bins = 10
    cuts = np.quantile(df.condition, np.linspace(0, 1, num=n_bins))
    cuts[0], cuts[-1] = 1, opts.condmax
    # cuts = np.linspace(start=1, stop=opts.condmax, num=10+1, endpoint=True)
    df["cond_bin"] = pd.cut(df.condition, cuts)
    tmp = df.groupby("cond_bin").agg(
        x=("condition", "mean"), y=("is_covered", "mean"), s=("is_covered", "size"),
    )
    tmp["yerr"] = 2 * np.sqrt(tmp.y * (1 - tmp.y) / tmp.s)
    vals = np.array([tmp.y[0], *tmp.y])
    errs = np.array([tmp.yerr[0], *tmp.yerr])
    ax.step(cuts, vals, linestyle="solid", linewidth=1, where="pre")
    ax.fill_between(
        cuts,
        vals - errs,
        vals + errs,
        linestyle="solid",
        linewidth=1,
        step="pre",
        color="red",
        alpha=0.4,
    )
    ax.axhline(1 - opts.confidence_level, color="black")
    ax.set_xlabel("Condition number")
    ax.set_ylabel("Empirical coverage rate")
    fig.savefig(os.path.join(ofolder, "coverage.png"))
    pd.DataFrame(
        {
            "x": cuts,
            "y": vals,
            "yUpper": vals + errs,
            "yLower": vals - errs,
            "target": 1 - opts.confidence_level,
        }
    ).to_csv(os.path.join(opts.ofolder, f"{expname}_coverage_{opts.n_data}.csv"))

    with open(os.path.join(ofolder, fname_txt), "w") as f:
        m = df.is_covered.mean()
        ress = {"Mean coverage": m}
        if len(df["ace_true"].unique()) == 1:
            ress["ace_true"] = df.loc[0, "ace_true"]
        json.dump(ress, f, indent=4)


def init():
    """Parse options, create output folder, document settings"""

    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--rand_graph",
        type=RandGraphSpec,
        help=(
            "Specification of a random graph. "
            "<Number of nodes>,<expected number of edges per node>"
            " The default case is 4,1"
        ),
    )
    grp.add_argument(
        "--named_graph",
        type=str,
        choices=selected_graphs.keys(),
        help="A predefined graph by name. Default is a random graph instead.",
    )
    p.add_argument(
        "--n_repetitions", default=300, type=int, help="The number of datasets to run"
    )
    p.add_argument(
        "--condmax",
        default=1.1,
        type=float,
        help="Worst condition number to be generated for the latent noise structure",
    )
    p.add_argument(
        "--confidence_level",
        default=0.05,
        type=float,
        help="95percent confidence ==> alpha=confidence_level=.05",
    )
    p.add_argument(
        "--dag_tolerance_epsilon",
        type=float,
        default=1e-7,
        help="The max value of h(W)",
    )
    p.add_argument(
        "--n_data",
        default=1_000,
        type=int,
        help="Number of data points/observations to compute ace_n",
    )
    opts = p.parse_args()
    if opts.named_graph is None and opts.rand_graph is None:
        opts.rand_graph = RandGraphSpec("4,1")

    ofolder = pathlib.Path("output").joinpath(
        f"{expname}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    printt(f"Outputting to {ofolder}")
    os.makedirs(ofolder)
    opts.ofolder = ofolder

    with open(ofolder.joinpath(fname_config), "w") as f:
        json.dump(vars(opts), fp=f, default=default_encoder, indent=2)

    return opts


def main():
    t0 = printt("== Starting ==")
    opts = init()
    run_experiment(opts)
    post_process(opts)
    printt(f"== END RUN ===", t0=t0)


if __name__ == "__main__":
    main()
