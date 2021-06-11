import argparse
import datetime
import pathlib
import os
import itertools
import pprint


import numpy as np
import pandas as pd
import lingam
import scipy.stats
import networkx as nx
from tqdm import trange
import matplotlib.pyplot as plt
import colorama
import seaborn as sns
import statsmodels.api as sm

from lib.daggnn_util import simulate_random_dag, simulate_sem
from lib.linear_algebra import make_L_no_diag, make_Z_clear_first
from lib.linear_sem import ace, ace_grad, selected_graphs
from lib.misc import CheckUniqueStore, RandGraphSpec, printt, cross_moment_4
from lib.plotters import draw_graph
from lib.relaxed_notears import relaxed_notears, mest_covarance, ace_circ

colorama.init()


raw_fname = "raw.csv"
summary_fname = "summary.txt"
config_fname = "config.txt"
graph_fname = "graph.png"
plot_fname = "summary.png"

variances = {
    "linear-gauss": 1,
    "linear-exp": 1,
    "linear-gumbel": np.pi ** 2 / 6,
}


def main():
    tstart = datetime.datetime.now()
    printt("Starting!")

    printt("Parsing options")
    opts = parse_args()
    output_folder = pathlib.Path(
        "output", f"baselines_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    os.makedirs(output_folder)
    pp = pprint.PrettyPrinter(indent=4)
    with open(output_folder.joinpath(config_fname), "w") as f:
        f.write(pp.pformat(vars(opts)) + "\n")
    printt("Config:\n" + pp.pformat(vars(opts)))

    printt("Running experiment")
    raw_path = output_folder.joinpath(raw_fname)
    df, W_true = run_experiment(opts)
    df.to_csv(raw_path)
    draw_graph(W_true, "Adjacency matrix for SEM", output_folder.joinpath(graph_fname))

    printt("Processing experiment output")
    post_process(output_folder)

    printt("Done!")
    tend = datetime.datetime.now()
    printt(f"Total runtime was {tend-tstart}")


def lingam_once(data):
    model = lingam.DirectLiNGAM()
    model.fit(data)
    W = model.adjacency_matrix_.T
    d = W.shape[0]
    Z = make_Z_clear_first(d)
    assert all(np.diag(W) == 0), "W matrix has nonzero diagonal - L not appropriate"
    id = np.eye(d)
    M = np.linalg.pinv(id - Z @ W.T)
    ace = M[1, 0]
    return W, ace


def lingam_stuff(data, opts):
    W, ace = lingam_once(data)
    aces = np.zeros(opts.n_bootstrap)
    n = data.shape[0]
    for b in trange(opts.n_bootstrap, desc="LiNGAM Bootstrap", leave=False):
        bootstrap_samples = np.random.randint(n, size=n)
        _, ace = lingam_once(data[bootstrap_samples, :])
        aces[b] = ace
    lo, hi = np.percentile(
        aces, [100 * opts.confidence_level / 2, 100 - 100 * opts.confidence_level / 2]
    )
    return W, ace, lo, hi


def notears_stuff(data: np.ndarray, opts: argparse.Namespace, normal):
    data_cov = np.cov(data, rowvar=False)
    n, d = data.shape
    L = make_L_no_diag(d)
    res = relaxed_notears(
        data_cov,
        L=L,
        W_initial=np.zeros((d, d)),
        dag_tolerance=opts.dag_tolerance,
        optim_opts=dict(lbfgs_ftol=opts.ftol, lbfgs_gtol=opts.gtol),
    )
    assert res["success"]
    w = res["w"]
    theta = res["theta"]
    if normal:
        covariance_matrix = mest_covarance(w, data_cov, L, normal)
    else:
        cm4 = cross_moment_4(data)
        covariance_matrix = mest_covarance(w, data_cov, L, normal, cm4)
    gradient_of_ace_with_respect_to_theta = ace_grad(theta=theta, L=L)
    ace_variance = (
        gradient_of_ace_with_respect_to_theta
        @ covariance_matrix
        @ gradient_of_ace_with_respect_to_theta
    )
    ace_standard_error = np.sqrt(ace_variance / n)
    ace_value = ace(theta, L=L)
    q = scipy.stats.norm.ppf(1 - opts.confidence_level / 2)
    lo, hi = ace_value + ace_standard_error * np.array([-q, q])
    return w, ace_value, ace_standard_error, lo, hi


def simulate_linear_sem(G, sem_type, n):
    """Simulates a linear SEM as in DAG-GNN-article, and centers the data.
    Forces the noise to have variance 1"""
    data = simulate_sem(
        G,
        n=n,
        x_dims=1,
        sem_type=sem_type,
        linear_type="linear",
        noise_scale=1 / np.sqrt(variances[sem_type]),
    ).squeeze()
    data = data - data.mean(axis=0)
    return data


def generate_random_graph(opts):
    for seed in itertools.count(0):
        np.random.seed(seed)
        G = simulate_random_dag(
            d=opts.rand_graph.d, degree=opts.rand_graph.k * 2, graph_type="erdos-renyi",
        )
        W = nx.to_numpy_array(G)
        d = W.shape[0]
        Z = make_Z_clear_first(d)
        id = np.eye(d)
        M = np.linalg.pinv(id - Z @ W.T)
        ace = M[1, 0]
        if not np.isclose(ace, 0):
            printt(f"Seed {seed} gave a graph of causal effect {ace}")
            return G, W, ace


def get_selected_graph(name, opts):
    if name not in selected_graphs.keys():
        raise ValueError
    W = selected_graphs[name]
    d = W.shape[0]
    Z = make_Z_clear_first(d)
    id = np.eye(d)
    M = np.linalg.pinv(id - Z @ W.T)
    ace = M[1, 0]
    G = nx.DiGraph(W)
    return G, W, ace


def run_experiment(opts):

    if opts.named_graph is not None:
        G, W_true, ace_true = get_selected_graph(opts.named_graph, opts)
    else:
        G, W_true, ace_true = generate_random_graph(opts)
    d_nodes = G.number_of_nodes()

    ress = []
    for noise in opts.noise:
        sem_type = "linear-" + noise
        normal = noise == "gauss"
        printt("Computing ace_circ")
        W_our_lim, ace_our_lim = ace_circ(W_true, np.eye(d_nodes), opts.dag_tolerance)
        printt("Computing LiNGAM large data limit")
        W_lingam_lim, ace_lingam_lim = lingam_once(
            simulate_linear_sem(G, sem_type, opts.n_data_lingam_lim)
        )
        printt(f"Starting data draws for {sem_type}")
        for n_data in opts.n_data:
            printt(f"Working wth sample size = {n_data}")
            for data_draw in trange(opts.n_repetitions, desc=f"Runs {sem_type}"):
                data = simulate_linear_sem(G, sem_type, n_data)
                (
                    W_lingam,
                    ace_lingam,
                    ace_ci_lingam_low,
                    ace_ci_lingam_high,
                ) = lingam_stuff(data, opts)
                ress.append(
                    dict(
                        n_data=n_data,
                        data_draw=data_draw,
                        sem_type=sem_type,
                        Method="LiNGAM",
                        ace_true=ace_true,
                        ace_lim=ace_lingam_lim,
                        ace=ace_lingam,
                        ace_ci_low=ace_ci_lingam_low,
                        ace_ci_high=ace_ci_lingam_high,
                        confidence_level=opts.confidence_level,
                    )
                )
                (
                    W_our,
                    ace_our,
                    ace_our_se,
                    ace_ci_our_low,
                    ace_ci_our_high,
                ) = notears_stuff(data, opts, normal)
                ress.append(
                    dict(
                        n_data=n_data,
                        data_draw=data_draw,
                        sem_type=sem_type,
                        Method="our",
                        ace_true=ace_true,
                        ace_lim=ace_our_lim,
                        ace=ace_our,
                        ace_ci_low=ace_ci_our_low,
                        ace_ci_high=ace_ci_our_high,
                        confidence_level=opts.confidence_level,
                    )
                )

    df = pd.DataFrame(ress)
    return df, W_true


def post_process(output_folder):
    dimensions = ["sem_type", "Method", "n_data"]
    df = pd.read_csv(output_folder.joinpath(raw_fname))
    df[f"ci_cover"] = (
        (df[f"ace_ci_low"] < df[f"ace_lim"]) & (df[f"ace_ci_high"] > df[f"ace_lim"])
    ).astype("float")
    df[f"ci_width"] = df[f"ace_ci_high"] - df[f"ace_ci_low"]

    nobs = df.groupby(dimensions)["ci_cover"].count().to_frame("nobs")
    count = df.groupby(dimensions)["ci_cover"].sum().to_frame("count")
    p = df.groupby(dimensions)["ci_cover"].mean().to_frame("p")
    dfs = pd.concat([nobs, count], axis=1)
    pminmax = dfs.apply(
        lambda s: pd.Series(
            sm.stats.proportion_confint(
                s["count"], s["nobs"], method="beta", alpha=0.05
            ),
            index=["pmin", "pmax"],
        ),
        axis=1,
    )
    dfs = pd.concat([p, pminmax], axis=1)
    cr_nice = dfs.apply(
        lambda x: f"{x['p']:7.2%} ({x['pmin']:7.2%},{x['pmax']:7.2%})", axis=1
    ).to_frame("CR")
    avg_w = df.groupby(dimensions)["ci_width"].mean().to_frame("Avg CI width")
    avg_ace = df.groupby(dimensions)["ace"].mean().to_frame("Avg ACE est")
    summary = pd.concat([cr_nice, avg_w, avg_ace], axis=1)

    summ_path = output_folder.joinpath(summary_fname)
    with open(summ_path, "w") as f:
        f.write("Latex output\n")
        s = summary.to_latex(float_format="%.2f")
        f.write(s)
        f.write("\n\n")
        f.write("Easy readable output\n")
        f.write(str(summary))
    print(summary)

    sns.displot(df, row="Method", col="sem_type", x="ace")
    plt.savefig(output_folder.joinpath(plot_fname))


def parse_args():

    p = argparse.ArgumentParser(
        description="Compare our proposed method with baselines - currenctly LiNGAM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dag_tolerance", default=1e-7, type=float, help="The epsilon-value we aim for"
    )
    p.add_argument(
        "--n_data",
        default=[1000],
        type=int,
        nargs="+",
        help=(
            "The number of data points. Specify several amounts to anayze the behavior "
            "of sample size on the coverage rate"
        ),
        action=CheckUniqueStore,
    )
    p.add_argument(
        "--n_data_lingam_lim",
        default=100_000,
        type=int,
        help="Data points for the large-sample lingam computation",
    )
    p.add_argument(
        "--n_repetitions",
        default=100,
        type=int,
        help="How many data sets to draw from the graph per noise type?",
    )
    p.add_argument(
        "--confidence_level",
        default=0.05,
        type=float,
        help="(95%% coverage ==> confidence_level 5%%)",
    )
    p.add_argument(
        "--n_bootstrap",
        default=100,
        type=int,
        help="Number of bootstrap repetitions for LiNGAM",
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--rand_graph",
        type=RandGraphSpec,
        help=(
            "Specification of a random graph. "
            "<Number of nodes>,<expected number of edges per node>"
        ),
    )
    grp.add_argument(
        "--named_graph",
        type=str,
        choices=selected_graphs.keys(),
        help="A predefined graph by name. Default is a random graph instead.",
    )
    p.add_argument(
        "--noise",
        default=["gauss", "exp", "gumbel"],
        type=str,
        nargs="+",
        choices=["gauss", "exp", "gumbel"],
        help="How many expected edges per node?",
        action=CheckUniqueStore,
    )
    p.add_argument(
        "--ftol",
        default=1e-10,
        type=float,
        help="The ftol parameter to pass to L-BFGS-B",
    )
    p.add_argument(
        "--gtol",
        default=1e-6,
        type=float,
        help="The ftol parameter to pass to L-BFGS-B",
    )
    opts = p.parse_args()
    return opts


if __name__ == "__main__":
    main()
