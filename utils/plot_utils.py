from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from strategies import StrategyRunResult

from causal_models.scm import SCM


_linestyles_map = {
    "solid": "solid",
    "dotted": "dotted",
    "dashed": "dashed",
    "dashdot": "dashdot",
    "loosely dotted": (0, (1, 10)),
    "densely dotted": (0, (1, 1)),
    "long dash with offset": (5, (10, 3)),
    "loosely dashed": (0, (5, 10)),
    "densely dashed": (0, (5, 1)),
    "loosely dashdotted": (0, (3, 10, 1, 10)),
    "dashdotted": (0, (3, 5, 1, 5)),
    "densely dashdotted": (0, (3, 1, 1, 1)),
    "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
    "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
}


def _common_plot_setup() -> None:
    SIZE = 14
    plt.rc("font", size=SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE + 4)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE + 4)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SIZE)  # legend fontsize
    plt.rc("figure", titlesize=SIZE + 20)  # fontsize of the figure title


def plot_regret_curve(
    results_dict: dict,
    name_to_linestyle_color: dict,
    name_to_label: dict = {},
    plot_path: Optional[str] = None,
):
    _common_plot_setup()

    oracle_mses = {}
    for horizon, run_results in results_dict["oracle_with_true_nu"].items():
        run_results: list[StrategyRunResult]

        errors = [res.squared_errors[-1] for res in run_results]
        oracle_mses[horizon] = np.mean(errors)

    def plot(axs, name: str, plot_props, result: dict, with_var: bool = True) -> None:
        x_vals = []
        y_mean = []
        y_std = []

        for horizon, run_results in result.items():
            run_results: list[StrategyRunResult]

            x_vals.append(horizon)
            errors = [res.squared_errors[-1] for res in run_results]
            y_scaled = (errors - oracle_mses[horizon]) / oracle_mses[horizon] * 100
            y_mean.append(np.mean(y_scaled))
            y_std.append(np.std(y_scaled))

        x_vals = np.array(x_vals)
        y_mean = np.array(y_mean)
        y_std = np.array(y_std)

        x_sort_idx = np.argsort(x_vals)
        x_vals = x_vals[x_sort_idx]
        y_mean = y_mean[x_sort_idx]
        y_std = y_std[x_sort_idx]

        label = name_to_label[name] if name in name_to_label else name
        plt.plot(
            x_vals,
            y_mean,
            label=label,
            color=plot_props[1],
            linestyle=_linestyles_map[plot_props[0]],
            marker=plot_props[2],
        )

        if with_var:
            ci = 1.96 * y_std / np.sqrt(len(run_results))
            axs.errorbar(x_vals, y_mean, yerr=ci, ls="none", color=plot_props[1])

    plt.title("Relative regret vs Horizon")
    plt.xlabel("Horizon $T$ (total samples collected)")
    plt.ylabel("Relative regret (%)")
    for name, plot_props in name_to_linestyle_color.items():
        plot(plt, name, plot_props, results_dict[name], with_var=True)

    if plot_path is not None:
        plt.savefig(plot_path, bbox_inches="tight", pad_inches=0.0)
    plt.show()


def plot_mse_curve(
    results_dict: dict,
    name_to_linestyle_color: dict,
    name_to_label: dict = {},
    plot_path: Optional[str] = None,
):
    _common_plot_setup()

    true_scm: SCM = results_dict["true_scm"]

    def plot(axs, name: str, plot_props, result: dict, with_var: bool = True) -> None:
        x_vals = []
        y_mean = []
        y_std = []

        for horizon, run_results in result.items():
            run_results: list[StrategyRunResult]

            x_vals.append(horizon)
            y_errs = [
                (res.ate_hats[-1] - true_scm.get_true_ate()) ** 2 * horizon
                for res in run_results
            ]
            y_mean.append(np.mean(y_errs))
            y_std.append(np.std(y_errs))

        x_vals = np.array(x_vals)
        y_mean = np.array(y_mean)
        y_std = np.array(y_std)

        x_sort_idx = np.argsort(x_vals)
        x_vals = x_vals[x_sort_idx]
        y_mean = y_mean[x_sort_idx]
        y_std = y_std[x_sort_idx]

        label = name_to_label[name] if name in name_to_label else name
        plt.plot(
            x_vals,
            y_mean,
            label=label,
            color=plot_props[1],
            linestyle=_linestyles_map[plot_props[0]],
            marker=plot_props[2],
        )
        # plt.legend()

        if with_var:
            ci = 1.96 * y_std / np.sqrt(len(run_results))
            axs.errorbar(x_vals, y_mean, yerr=ci, ls="none", color=plot_props[1])

    plt.title("Scaled MSE vs Budget")
    plt.xlabel("Budget $B$")
    plt.ylabel("Scaled MSE ($B \\times \\text{MSE}$)")
    for name, plot_props in name_to_linestyle_color.items():
        plot(plt, name, plot_props, results_dict[name], with_var=True)

    if plot_path is not None:
        plt.savefig(plot_path, bbox_inches="tight", pad_inches=0.0)
    plt.show()


def plot_kappa_curve(
    results_dict: dict, name_to_linestyle_color: dict, name_to_label: dict = {}
):
    _common_plot_setup()

    def plot(axs, name: str, plot_props, result: dict, with_var: bool = True) -> None:
        x_vals = []
        y_mean = []
        y_std = []

        for horizon, run_results in result.items():
            run_results: list[StrategyRunResult]

            x_vals.append(horizon)
            current_kappas = [res.current_ks[-1] for res in run_results]
            y_scaled = current_kappas
            y_mean.append(np.mean(y_scaled))
            y_std.append(np.std(y_scaled))

        x_vals = np.array(x_vals)
        y_mean = np.array(y_mean)
        y_std = np.array(y_std)

        x_sort_idx = np.argsort(x_vals)
        x_vals = x_vals[x_sort_idx]
        y_mean = y_mean[x_sort_idx]
        y_std = y_std[x_sort_idx]

        label = name_to_label[name] if name in name_to_label else name
        plt.plot(
            x_vals,
            y_mean,
            label=label,
            color=plot_props[1],
            linestyle=plot_props[0],
            marker=plot_props[2],
        )
        plt.legend()

        if with_var:
            ci = 1.96 * y_std / np.sqrt(len(run_results))
            axs.errorbar(x_vals, y_mean, yerr=ci, ls="none", color=plot_props[1])

    plt.title("Final kappa vs horizon")
    plt.xlabel("Total samples collected (Horizon)")
    plt.ylabel("Final kappa")
    for name, plot_props in name_to_linestyle_color.items():
        plot(plt, name, plot_props, results_dict[name], with_var=True)

    plt.show()


def plot_coverage_curve(
    results_dict: dict,
    name_to_linestyle_color: dict,
    name_to_label: dict = {},
    plot_path: Optional[str] = None,
):
    _common_plot_setup()

    def is_covered(
        truth: float, estimate: float, variance: float, num_samples: int
    ) -> bool:
        width = 1.96 * np.sqrt(variance / num_samples)
        if truth <= estimate + width and truth >= estimate - width:
            return True

        return False

    def plot(axs, name: str, plot_props, result: dict, with_var: bool = True) -> None:
        x_vals = []
        y_mean = []
        y_std = []

        for horizon, run_results in result.items():
            run_results: list[StrategyRunResult]

            x_vals.append(horizon)
            coverage = [
                float(
                    is_covered(
                        truth=results_dict["true_scm"].get_true_ate(),
                        estimate=res.ate_hats[-1],
                        variance=res.var_hats[-1],
                        num_samples=horizon,
                    )
                )
                * 100
                for res in run_results
            ]

            y_mean.append(np.mean(coverage))
            y_std.append(np.std(coverage))

        x_vals = np.array(x_vals)
        y_mean = np.array(y_mean)
        y_std = np.array(y_std)

        x_sort_idx = np.argsort(x_vals)
        x_vals = x_vals[x_sort_idx]
        y_mean = y_mean[x_sort_idx]
        y_std = y_std[x_sort_idx]

        label = name_to_label[name] if name in name_to_label else name
        plt.plot(
            x_vals,
            y_mean,
            label=label,
            color=plot_props[1],
            linestyle=_linestyles_map[plot_props[0]],
            marker=plot_props[2],
        )

        if with_var:
            ci = 1.96 * y_std / np.sqrt(len(run_results))
            axs.errorbar(x_vals, y_mean, yerr=ci, ls="none", color=plot_props[1])

    plt.title("Coverage vs Horizon")
    plt.xlabel("Horizon $T$ (total samples collected)")
    plt.ylabel("Coverage (%)")

    for name, plot_props in name_to_linestyle_color.items():
        plot(plt, name, plot_props, results_dict[name], with_var=True)

    if plot_path is not None:
        plt.savefig(plot_path, bbox_inches="tight", pad_inches=0.0)
    plt.show()


def plot_CI_size_curve(
    results_dict: dict,
    name_to_linestyle_color: dict,
    name_to_label: dict = {},
    plot_path: Optional[str] = None,
):

    _common_plot_setup()

    def plot(axs, name: str, plot_props, result: dict, with_var: bool = True) -> None:
        x_vals = []
        y_mean = []
        y_std = []

        for horizon, run_results in result.items():
            run_results: list[StrategyRunResult]

            x_vals.append(horizon)
            ci_sizes = [1.96 * np.sqrt(res.var_hats[-1]) for res in run_results]
            y_mean.append(np.mean(ci_sizes))
            y_std.append(np.std(ci_sizes))

        x_vals = np.array(x_vals)
        y_mean = np.array(y_mean)
        y_std = np.array(y_std)

        x_sort_idx = np.argsort(x_vals)
        x_vals = x_vals[x_sort_idx]
        y_mean = y_mean[x_sort_idx]
        y_std = y_std[x_sort_idx]

        label = name_to_label[name] if name in name_to_label else name
        plt.plot(
            x_vals,
            y_mean,
            label=label,
            color=plot_props[1],
            linestyle=_linestyles_map[plot_props[0]],
            marker=plot_props[2],
        )

        if with_var:
            ci = 1.96 * y_std / np.sqrt(len(run_results))
            axs.errorbar(x_vals, y_mean, yerr=ci, ls="none", color=plot_props[1])

    plt.title("CI size vs Horizon")
    plt.xlabel("Horizon $T$ (total samples collected)")
    plt.ylabel("CI size (Scaled by $\\sqrt{T}$)")

    for name, plot_props in name_to_linestyle_color.items():
        plot(plt, name, plot_props, results_dict[name], with_var=True)

    if plot_path is not None:
        plt.savefig(plot_path, bbox_inches="tight", pad_inches=0.0)
    plt.show()
