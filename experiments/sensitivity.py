""" Analysis of how sensitive the problem is with respect to the DAG tolerance epsilon
"""
import datetime
import os
import pathlib
import pprint
import argparse

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.linalg
import networkx as nx
from tqdm import tqdm
import colorama

from lib.daggnn_util import simulate_random_dag
from lib.relaxed_notears import relaxed_notears, make_h_paramterized, make_notears_loss
from lib.linear_algebra import make_L_no_diag
from lib.linear_sem import ace
from lib.misc import RandGraphSpec, printt
from lib.plotters import draw_graph, plot_contours_in_2d

colorama.init()

opath = pathlib.Path("output")
fname_c = "config.txt"
fname_raw = "results.pkl"
fname_pgf = "dagtol_pgfplots.csv"


def run_experiment(opts, output_folder):
    notears_options = dict()
    notears_options["tolerated_constraint_violation"] = 1e-12
    notears_options["lbfgs_ftol"] = opts.ftol
    notears_options["lbfgs_gtol"] = opts.gtol
    dag_tolerance_epsilons = np.logspace(
        np.log10(opts.eps_min), np.log10(opts.eps_max), opts.n_eps
    )
    w_trues = []
    for s in range(0, opts.n_graphs):
        np.random.seed(s)
        w_trues.append(
            nx.to_numpy_array(
                simulate_random_dag(
                    d=opts.rand_graph.d,
                    degree=opts.rand_graph.k * 2,
                    graph_type="erdos-renyi",
                )
            )
        )

    result_dfs = []
    for k, w_true in tqdm(list(enumerate(w_trues)), desc="graph W"):
        #
        #
        # Set up
        #
        #
        d_nodes = w_true.shape[0]
        L_no_diag = make_L_no_diag(d_nodes)
        w_initial = np.zeros((d_nodes, d_nodes))
        h, grad_h = make_h_paramterized(L_no_diag)
        theta_true = L_no_diag.T @ w_true.T.flatten()
        ace_true = ace(theta_true, L_no_diag)

        id = np.eye(d_nodes)
        M = np.linalg.pinv(id - w_true.T)
        data_cov = M @ M.T
        noise_cov = np.eye(data_cov.shape[0])
        noise_prec = np.linalg.pinv(noise_cov)
        Q = np.kron(noise_prec, data_cov)
        sQrt = scipy.linalg.sqrtm(Q)
        theta_star = np.linalg.pinv(sQrt @ L_no_diag) @ sQrt @ id.T.flatten()
        w_star = (L_no_diag @ theta_star).reshape(d_nodes, d_nodes).T
        h_star = h(theta_star)

        result_dicts = []
        for dag_tolerance in tqdm(
            dag_tolerance_epsilons, desc="tolerance epsilon", leave=False
        ):
            if dag_tolerance >= h_star:
                tqdm.write(
                    f"Optimum in the interior. {dag_tolerance:.2g} >= {h_star:.2g}"
                )
            result = relaxed_notears(
                data_cov, L_no_diag, w_initial, dag_tolerance, notears_options
            )
            theta_notears = result["theta"]
            w_notears = result["w"]
            h_notears = h(theta_notears)
            assert result["success"], "Solving failed!"
            assert (
                h_notears
                < dag_tolerance + notears_options["tolerated_constraint_violation"]
            ), (
                f"h_notears >= dag_tolerance + rho, {h_notears} >= "
                f"{dag_tolerance} + {notears_options['tolerated_constraint_violation']}"
            )
            d = dict(
                theta_notears=theta_notears,
                w_notears=w_notears,
                h_notears=h_notears,
                dag_tolerance=dag_tolerance,
                k=k,
                w_true=w_true,
                theta_true=theta_true,
                h_star=h_star,
                w_star=w_star,
                theta_star=theta_star,
                d_nodes=d_nodes,
                data_cov=data_cov,
                ace_notears=ace(theta_notears, L_no_diag),
                ace_true=ace_true,
                rho=result["rho"],
            )
            result_dicts.append(d)

        #   Process after completion of all runs (produce plots and save)
        #
        #
        df_inner = pd.DataFrame(result_dicts)
        result_dfs.append(df_inner)
        w_best = df_inner["w_notears"][
            df_inner["dag_tolerance"].idxmin()
        ]  # the one with smallest dag tolerance...

        draw_graph(
            w=w_true,
            title="$W_{true}$",
            out_path=output_folder.joinpath(f"{k}_w_true.png"),
        )
        draw_graph(
            w=w_best,
            title=f"$\\hat W$ for $\\epsilon={df_inner['dag_tolerance'].min()}$",
            out_path=output_folder.joinpath(f"{k}_w_best.png"),
        )
        fig, axs = plt.subplots(1, 3)
        ax1, ax2, ax3 = axs.flatten()
        absmax = max(np.abs(w_best).max(), np.abs(w_true).max(), np.abs(w_star).max())
        plotopts = dict(
            cmap="PiYG",
            norm=matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-absmax, vmax=absmax),
        )
        ax1.matshow(w_best, **plotopts)
        ax1.set_title("w_notears")
        ax2.matshow(w_true, **plotopts)
        ax2.set_title("w_true")
        ax3.matshow(w_star, **plotopts)
        ax3.set_title("w_star")
        fig.savefig(output_folder.joinpath(f"{k}_mats.png"))
        plt.close(fig)

    raw_path = output_folder.joinpath(fname_raw)
    df = pd.concat(result_dfs)
    df["w_err"] = df["w_notears"] - df["w_true"]
    df["w_err_star"] = df["w_star"] - df["w_true"]
    df["max-metric"] = df["w_err"].apply(lambda w: np.abs(w).max())
    df["2-metric"] = df["w_err"].apply(lambda w: np.linalg.norm(w, ord="fro"))
    df["1-metric"] = df["w_err"].apply(lambda w: np.abs(w).sum())
    df["max-metric_star"] = df["w_err_star"].apply(lambda w: np.abs(w).max())
    df["2-metric_star"] = df["w_err_star"].apply(lambda w: np.linalg.norm(w, ord="fro"))
    df["1-metric_star"] = df["w_err_star"].apply(lambda w: np.abs(w).sum())
    df["ace_abs"] = np.abs(df["ace_notears"])
    df["ace_abs_err"] = np.abs(df["ace_true"] - df["ace_notears"])
    df["ace_err"] = df["ace_true"] - df["ace_notears"]
    df["ace_true_zero"] = np.abs(df["ace_true"]) < 1e-14
    df.to_pickle(path=raw_path)


def post_process(output_folder):
    raw_path = output_folder.joinpath(fname_raw)
    df = pd.read_pickle(raw_path)
    generate_pgfplots_output(df, output_folder)

    df.groupby("k")["h_star"].unique().astype(float).to_csv(
        output_folder.joinpath("hstars.csv")
    )

    fig = plt.figure("00")
    ax = fig.subplots()
    for k in df.k.unique():
        a = df[df.k == k]
        ax.loglog(
            a.dag_tolerance,
            a["1-metric"],
            linestyle="dotted",
            marker=".",
            color=f"C{k}",
            label=f"$w(\\varepsilon)$, k={k}",
        )
        ax.loglog(
            a.h_star.unique(),
            a["1-metric_star"].unique(),
            linestyle="none",
            label=f"$w(\\infty)$, k={k}",
            color=f"C{k}",
            marker="*",
        )
        ax.axvline(a.h_star.unique(), linestyle="dotted", color=f"C{k}")
    ax.legend()
    fig.savefig(output_folder.joinpath(f"1-metric_thresh.png"))
    plt.close()

    fig = plt.figure("ace_by_eps")
    ax = fig.subplots()
    sns.lineplot(data=df, y="ace_notears", x="dag_tolerance", hue="k", ax=ax)
    ax.set_xscale("log")
    fig.savefig(output_folder.joinpath(f"ace_by_eps.png"))
    plt.close()

    fig = plt.figure("ace_abs_err")
    ax = fig.subplots()
    for k in df.k.unique():
        df_k = df[df.k == k]
        ax.plot(
            df_k.dag_tolerance,
            np.abs(a["ace_abs_err"]),
            linestyle="solid",
            marker=".",
            color=f"C{k}",
            label=f"$|\\gamma_{{true}} - \\gamma(\\hat w(\\varepsilon))|$, k={k}",
        )
        ax.axvline(df_k.h_star.unique(), linestyle="dotted", color=f"C{k}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig(output_folder.joinpath(f"ace_abs_err.png"))
    plt.close()

    plot_log_stuff(df, output_folder)
    plot_cross_sections([8, 11], df, output_folder)

    print(f"Finished outputting")


def generate_pgfplots_output(df, output_folder):
    pgf_path = output_folder.joinpath(fname_pgf)
    pivot = df.pivot(
        index="dag_tolerance",
        columns="k",
        values=["ace_abs_err", "max-metric", "h_notears", "ace_abs"],
    )
    pivot.columns = ["-".join(map(str, col)).strip() for col in pivot.columns.values]
    pivot.to_csv(pgf_path)


def plot_log_stuff(df, output_folder):
    df2 = (
        df[
            [
                "k",
                "ace_abs",
                "1-metric",
                "dag_tolerance",
                "h_notears",
                "ace_abs_err",
                "ace_true_zero",
                "2-metric",
                "max-metric",
            ]
        ]
        .melt(id_vars=["k", "dag_tolerance", "ace_true_zero"])
        .astype({"k": str})
    )
    fgrid = sns.relplot(
        data=df2,
        x="dag_tolerance",
        y="value",
        kind="line",
        col="variable",
        style="ace_true_zero",
        facet_kws={"sharex": True, "sharey": False},
        hue="k",
        col_wrap=2,
    )
    fgrid.set(xscale="log", yscale="log")
    fgrid.savefig(output_folder.joinpath(f"loglogs.png"))
    plt.close()


def plot_cross_sections(idxs, df, output_folder):
    for k in df.k.unique():
        df_k = df[df.k == k]
        thetas = np.array(df_k.theta_notears.to_list())
        d_nodes = df.d_nodes.unique().item()
        theta_true = df_k.theta_true[0]
        data_cov = df_k.data_cov[0]
        maximum = max(thetas[:, idxs].max(), theta_true[idxs].max())
        minimum = min(thetas[:, idxs].min(), theta_true[idxs].min())
        corner0 = maximum * 1.2 + 0.1
        corner1 = minimum * 1.2 - 0.1
        bbox = (corner1, corner0, corner1, corner0)
        h, _ = make_h_paramterized(make_L_no_diag(d_nodes))
        loss, _ = make_notears_loss(data_cov, make_L_no_diag(d_nodes))

        name_snap_pairs = [
            (f"eps{p}percent", int((thetas.shape[0] - 1) * p / 100))
            for p in [0, 20, 40, 60, 80, 100]
        ]
        for name, snap_idx in name_snap_pairs:
            # n.b. 0 % means the first (smallest) value

            def h_(theta_sub):
                theta = thetas[snap_idx, :].copy()
                theta[idxs] = theta_sub
                return h(theta)

            def loss_(theta_sub):
                theta = thetas[snap_idx, :].copy()
                theta[idxs] = theta_sub
                return loss(theta)

            fig = plt.figure(f"00{k}")
            ax = fig.subplots()
            norm = matplotlib.colors.LogNorm(
                vmin=df_k.dag_tolerance.min(), vmax=df_k.dag_tolerance.max()
            )
            cmap = plt.get_cmap("viridis")
            resolution = 100
            plot_contours_in_2d(
                [h_],
                ax,
                bbox,
                resolution,
                contour_opts=dict(
                    norm=norm,
                    levels=sorted(df_k.h_notears.unique()),
                    alpha=0.7,
                    cmap=cmap,
                ),
            )
            plot_contours_in_2d(
                [loss_],
                ax,
                bbox,
                resolution,
                contour_opts=dict(levels=10, alpha=0.5, colors="black"),
            )
            ax.plot(
                thetas[:, idxs[0]],
                thetas[:, idxs[1]],
                linestyle="solid",
                color="black",
                linewidth=0.2,
            )
            ax.scatter(
                x=theta_true[idxs[0]],
                y=theta_true[idxs[1]],
                marker="*",
                c="black",
                s=300,
            )
            s = ax.scatter(
                x=thetas[:, idxs[0]],
                y=thetas[:, idxs[1]],
                c=df_k.h_notears,
                norm=norm,
                cmap=cmap,
                marker="x",
            )
            ax.axvline(0, linestyle="solid", color="black")
            ax.axhline(0, linestyle="solid", color="black")
            fig.colorbar(s, label=r"$\epsilon$")
            ax.set_aspect("equal")
            fig.savefig(output_folder.joinpath(f"{k}_coords_{name}.png"))
            plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rand_graph",
        type=RandGraphSpec,
        help=(
            "Specification of a random graphs. "
            "<Number of nodes>,<expected number of edges per node>"
        ),
        default=RandGraphSpec("4,1"),
    )
    p.add_argument(
        "--eps_min",
        default=1e-9,
        type=float,
        help="The smallest dag tolerance under consideration",
    )
    p.add_argument(
        "--eps_max",
        default=10,
        type=float,
        help="The largest DAG tolerance under consideration",
    )
    p.add_argument(
        "--n_eps",
        default=20,
        type=int,
        help="The number of DAG tolerances to compute for",
    )
    p.add_argument(
        "--n_graphs",
        default=10,
        type=int,
        help="How many random graphs to compute for?",
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


def main():
    tstart = datetime.datetime.now()
    printt("Starting!")

    printt("Parsing options")
    opts = parse_args()
    output_folder = opath.joinpath(
        f"sensitivity_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    os.makedirs(output_folder)
    pp = pprint.PrettyPrinter(indent=4)
    with open(output_folder.joinpath(fname_c), "w") as f:
        f.write(pp.pformat(vars(opts)) + "\n")
    printt("Config:\n" + pp.pformat(vars(opts)))

    printt("Running experiment")
    run_experiment(opts, output_folder)

    printt("Processing experiment output")
    post_process(output_folder)

    printt("Done!")
    tend = datetime.datetime.now()
    printt(f"Total runtime was {tend-tstart}")


if __name__ == "__main__":
    main()
