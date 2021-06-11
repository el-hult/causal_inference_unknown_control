"""Code to see the performance of the method under nonlinear data generating process
"""
import os
import itertools
import pathlib
import warnings
import argparse
import pprint
import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.stats
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from lib.daggnn_util import simulate_random_dag, simulate_sem
from lib.linear_algebra import make_L_no_diag
from lib.linear_sem import ace, ace_grad
from lib.misc import cross_moment_4, printt
from lib.ols import myOLS
from lib.relaxed_notears import relaxed_notears, mest_covarance

raw_fname = "raw.csv"
summary_fname = "summary.txt"
config_fname = "config.txt"


def post_process(folder_path):
    df = pd.read_csv(folder_path.joinpath(raw_fname))
    with open(folder_path.joinpath(summary_fname), "w") as f:
        f.write(
            str(
                df.groupby("linear_type")["is_covered"]
                .mean()
                .apply(lambda v: f"{v:.2%}")
            )
        )
    sns.displot(data=df, x="z_score", row="linear_type")
    plt.savefig(folder_path.joinpath("scores.png"))


def run_experiment(opts, output_folder):
    resdicts = []
    reps = opts.repetitions
    for linear_type, _ in tqdm(
        itertools.product(["linear", "nonlinear_1", "nonlinear_2"], range(reps)),
        total=3 * reps,
    ):

        dic = random_G_and_ace(opts, linear_type=linear_type)
        dic["linear_type"] = linear_type
        resdicts.append(dic)

    df = pd.DataFrame(resdicts)
    q = scipy.stats.norm.ppf(1 - np.array(opts.confidence_level) / 2)
    df["q"] = q
    df["z_score"] = (df["ace_n"] - df["ace_circ"]) / df["ace_n_se"]
    df["is_covered"] = np.abs(df["z_score"]) < df["q"]

    df.to_csv(output_folder.joinpath(raw_fname))


def ace_mc_naive(G, sim_args, absprec=0.01):
    """Make a naive MC computation until the SE is less than absprec"""
    G_int = G.copy()
    G_int.remove_edges_from([e for e in G.edges if e[1] == 0])
    res = ace_mc = ace_mc_se = None
    for k in 10 ** np.arange(4, 10):
        kwds = {**sim_args, "n": k}
        intervention_data = simulate_sem(G_int, **kwds).squeeze()
        res = myOLS(intervention_data[:, [0]], intervention_data[:, 1])
        ace_mc = res["params"][0]
        ace_mc_se = res["HC0_se"][0]
        if ace_mc_se < absprec:
            break
    if ace_mc_se >= absprec:
        warnings.warn(f"MC computation lacks precision, HC0_se={res['HC0_se']}")
    return ace_mc, ace_mc_se


def ace_notears(G, sim_args, m_obs, dag_tolerance, notears_options, linear_type):
    data = simulate_sem(G, **sim_args, n=m_obs).squeeze()
    d_nodes = G.number_of_nodes()
    L_no_diag = make_L_no_diag(d_nodes)
    w_initial = np.zeros((d_nodes, d_nodes))
    data_cov = np.cov(data, rowvar=False)
    result_circ = relaxed_notears(
        data_cov, L_no_diag, w_initial, dag_tolerance, notears_options,
    )
    assert result_circ["success"]
    theta_n, w_n = result_circ["theta"], result_circ["w"]
    ace_n = (ace(theta_n, L_no_diag)).item()
    if linear_type == "linear":
        covariance_matrix = mest_covarance(w_n, data_cov, L_no_diag, True)
    else:
        cm4 = cross_moment_4(data)
        covariance_matrix = mest_covarance(w_n, data_cov, L_no_diag, False, cm4)
    gradient_of_ace_with_respect_to_theta = ace_grad(theta=theta_n, L=L_no_diag)
    ace_var = (
        gradient_of_ace_with_respect_to_theta
        @ covariance_matrix
        @ gradient_of_ace_with_respect_to_theta
        / m_obs
    )
    ace_se = np.sqrt(ace_var)
    return w_n, ace_n, ace_se


def random_G_and_ace(opts, linear_type):
    """Generate a random graph and see if the CI covers the true value"""
    m = opts.k_edge_multiplier * opts.d_nodes  # expected no of edges.
    notears_options = dict()

    G = simulate_random_dag(
        d=opts.d_nodes, degree=m * 2 / opts.d_nodes, graph_type="erdos-renyi"
    )

    sim_args = dict(x_dims=1, sem_type="linear-gauss", linear_type=linear_type,)
    ace_mc, ace_mc_se = ace_mc_naive(G, sim_args)
    w_circ, ace_circ, _ = ace_notears(
        G,
        sim_args,
        opts.n_data_circ,
        opts.dag_tolerance_epsilon,
        notears_options,
        linear_type,
    )
    w_n, ace_n, ace_n_se = ace_notears(
        G,
        sim_args,
        opts.n_data,
        opts.dag_tolerance_epsilon,
        notears_options,
        linear_type,
    )

    return dict(ace_mc=ace_mc, ace_circ=ace_circ, ace_n=ace_n, ace_n_se=ace_n_se,)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute coverage rate in linear and nonlinear SEMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--repetitions",
        default=100,
        type=int,
        help="Number of random graphs/data sets tested",
    )
    p.add_argument(
        "--d_nodes", default=4, type=int, help="Number of nodes in the graphs"
    )
    p.add_argument(
        "--n_data",
        default=1_000,
        type=int,
        help="Number of data points/observations to compute ace_n",
    )
    p.add_argument(
        "--n_data_circ",
        default=1_000_000,
        type=int,
        help="Number of data points/observations to approximate ace_circ",
    )
    p.add_argument(
        "--dag_tolerance_epsilon",
        type=float,
        default=1e-7,
        help="The max value of h(W)",
    )
    p.add_argument(
        "--k_edge_multiplier",
        default=1,
        type=int,
        help=(
            "The number in a ER1, ER2, ER4 graph. It is the number of expected edges "
            "divided by the number of nodes. It is equal to the expected out-degree or "
            "the expected in-degree in the graph."
        ),
    )
    p.add_argument(
        "--confidence_level",
        default=0.05,
        type=float,
        help="95percent confidence ==> alpha=confidence_level=.05",
    )
    opts = p.parse_args()
    return opts


def main():
    tstart = datetime.datetime.now()
    printt("Starting!")

    printt("Parsing options")
    opts = parse_args()
    output_folder = pathlib.Path(
        "output", f"nonlinear_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    os.makedirs(output_folder)
    pp = pprint.PrettyPrinter(indent=4)
    with open(output_folder.joinpath(config_fname), "w") as f:
        f.write(pp.pformat(vars(opts)) + "\n")
    printt(pp.pformat(vars(opts)))

    printt("Running experiment")
    run_experiment(opts, output_folder)

    printt("Processing experiment output")
    post_process(output_folder)

    printt("Done!")
    tend = datetime.datetime.now()
    printt(f"Total runtime was {tend-tstart}")


if __name__ == "__main__":
    main()
