"""
This experiment computes the confidence interval quite directly by
the method described in the article: point estimation plus the delta method.
"""
import datetime
import pickle
import os
from pathlib import Path
import argparse
import pprint

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import networkx as nx

from lib.relaxed_notears import relaxed_notears, mest_covarance, h
from lib.linear_algebra import make_L_no_diag
from lib.linear_sem import ace, ace_grad, selected_graphs
from lib.ols import myOLS
from lib.plotters import draw_graph
from lib.daggnn_util import simulate_sem

fname_c = "config.txt"


def generate_data(W, n):
    return simulate_sem(
        G=nx.DiGraph(W),
        n=n,
        x_dims=1,
        sem_type="linear-gauss",
        linear_type="linear",
        noise_scale=1,
    ).squeeze()


def plot_trend(df, gamma_circ, fname=None):
    ace_color = "C0"
    ols_color = "C1"
    fig, ax = plt.subplots(num="trend")
    ax.axhline(
        gamma_circ, label=r"$\approx \gamma_\circ$", color="k", linestyle="dashed"
    )
    ax.errorbar(
        x=df["m_obs"],
        y=df["ace_value"],
        yerr=df["q_ace_standard_error"],
        label=r"$\Gamma_{\alpha,n}$",
        color=ace_color,
        linestyle="None",
        capsize=3,
    )
    ax.errorbar(
        x=df["m_obs"],
        y=df["ols_value"],
        yerr=df["q_ols_standard_error"],
        label=r"$B_{\alpha,n}$",
        color=ols_color,
        linestyle="None",
        capsize=3,
    )
    ax.legend()
    ax.set_xlabel(r"Number of observations $n$")
    ax.set_xscale("log")
    if fname:
        fig.savefig(fname)
        plt.close(fig)


def run_experiment(opts, output_folder: Path):
    w_true = selected_graphs[opts.named_graph]
    d_nodes = w_true.shape[0]
    L_no_diag = make_L_no_diag(d_nodes)
    w_initial = np.zeros((d_nodes, d_nodes))
    m_obss = np.logspace(
        start=np.log10(opts.n_data_min),
        stop=np.log10(opts.n_data_max),
        num=opts.n_n_data,
        dtype="int",
    )

    m_obs = m_obss[-1] * 10
    data = generate_data(n=m_obs, W=w_true)
    data_cov = np.cov(data, rowvar=False)
    notears_options = {}
    result_circ = relaxed_notears(
        data_cov, L_no_diag, w_initial, opts.dag_tolerance, notears_options,
    )
    theta_circ, w_circ, success = (
        result_circ["theta"],
        result_circ["w"],
        result_circ["success"],
    )
    assert success
    ace_circ = (ace(theta_circ, L_no_diag)).item()
    draw_graph(
        w=w_circ,
        title=f"$\\gamma_{{{m_obs}}}={ace_circ:.2f}$",
        out_path=output_folder.joinpath(f"w_notears_{m_obs}.png"),
    )
    with open(output_folder.joinpath(f"ace_circ.pkl"), mode="wb") as f:
        pickle.dump(file=f, obj=ace_circ)

    result_dicts = []
    for m_obs in m_obss:
        print(f"Starting handling of {opts.named_graph}, n={m_obs}.")
        data = generate_data(W=w_true, n=m_obs)
        data_cov = np.cov(data, rowvar=False)
        result = relaxed_notears(
            data_cov, L_no_diag, w_initial, opts.dag_tolerance, notears_options,
        )
        theta_notears, w_notears, success = (
            result["theta"],
            result["w"],
            result["success"],
        )
        draw_graph(
            w=w_notears,
            title=f"$\\gamma_{{{m_obs}}}={(ace(theta_notears, L_no_diag)).item():.2f}$",
            out_path=output_folder.joinpath(f"w_notears_{m_obs}.png"),
        )
        print(f"w_notears: {w_notears}")
        print(
            f"h(w_notears): {h(w_notears)}, compare with"
            f" DAG tolerance {opts.dag_tolerance}"
        )
        print(f"$\\gamma_{{{m_obs}}}$={(ace(theta_notears, L_no_diag))}")

        covariance_matrix = mest_covarance(w_notears, data_cov, L_no_diag)
        gradient_of_ace_with_respect_to_theta = ace_grad(
            theta=theta_notears, L=L_no_diag
        )
        ace_variance = (
            gradient_of_ace_with_respect_to_theta
            @ covariance_matrix
            @ gradient_of_ace_with_respect_to_theta
        )
        ace_standard_error = np.sqrt(ace_variance / m_obs)
        ace_value = ace(theta_notears, L=L_no_diag)

        print("Computing the OLS solution")
        regressors = np.delete(data, 1, axis=1)
        outcomes = data[:, 1]
        ols_result = myOLS(X=regressors, y=outcomes)
        ols_direct_causal_effect, ols_standard_error = (
            ols_result["params"][0],
            ols_result["HC0_se"][0],
        )

        q = scipy.stats.norm.ppf(1 - np.array(opts.confidence_level) / 2)

        d = dict(
            m_obs=m_obs,
            ace_value=ace_value,
            ace_standard_error=ace_standard_error,
            q_ace_standard_error=q * ace_standard_error,
            ols_value=ols_direct_causal_effect,
            ols_standard_error=ols_standard_error,
            q_ols_standard_error=q * ols_standard_error,
            q=q,
            ace_circ=ace_circ,
            confidence_level=opts.confidence_level,
        )
        result_dicts.append(d)

    print("Completed the optimization. On to plotting!")
    df = pd.DataFrame(result_dicts)
    df.to_csv(output_folder.joinpath("summary.csv"))

    print(f"start plotting trend graph")
    fname = output_folder.joinpath("asymptotics.png")
    plot_trend(df, ace_circ, fname)
    print(f"Finished plotting the trend graph")

    #
    #
    #   Post process after completion of all runs (produce plots and save)
    #
    #
    gamma_true = ace(w_true.T.flatten(), L=np.eye(d_nodes ** 2)).item()
    print(f"\\gamma={gamma_true}")
    draw_graph(
        w=w_true,
        title=f"$\\gamma={gamma_true:.2f}",
        out_path=output_folder.joinpath(f"w_true.png"),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--named_graph", default="4collider", type=str)
    p.add_argument("--confidence_level", default=0.05, type=float)
    p.add_argument("--n_data_min", default=10 ** 2, type=int)
    p.add_argument("--n_data_max", default=10 ** 5, type=int)
    p.add_argument("--n_n_data", default=10, type=int)
    p.add_argument(
        "--dag_tolerance", default=1e-7, type=float, help="The epsilon-value we aim for"
    )
    opts = p.parse_args()

    return opts


def main():
    t_start = datetime.datetime.now()
    opts = parse_args()
    output_folder = Path("output").joinpath(
        f"interval_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        f" {opts.named_graph}"
    )
    os.makedirs(output_folder)
    pp = pprint.PrettyPrinter(indent=4)
    with open(output_folder.joinpath(fname_c), "w") as f:
        f.write(pp.pformat(opts) + "\n")
    run_experiment(opts, output_folder)
    t_end = datetime.datetime.now()
    print(f"END RUN === Run time: {str(t_end - t_start).split('.')[0]}")


if __name__ == "__main__":
    main()
