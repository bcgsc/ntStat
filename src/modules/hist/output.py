import functools
import itertools
import os
import warnings

import matplotlib.animation
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
import pandas as pd
import tabulate
import termplotlib as tpl
import tqdm
from histogram import NtCardHistogram
from model import Model, score


class TablePrinter:

    @staticmethod
    def get_valid_formats():
        return tabulate.tabulate_formats

    def __init__(self, fmt) -> None:
        self.__fmt = fmt

    def print(
        self,
        title: str,
        *rows: list[list[str]],
        header: list[str] = (),
        **kwargs,
    ) -> None:
        if title:
            print(f"{title}:")
        print(tabulate.tabulate(rows, header, self.__fmt, ".3f", ",", **kwargs))
        print()


def print_hist(hist: numpy.typing.NDArray[np.uint64]) -> None:
    try:
        w = 3 * os.get_terminal_size().columns // 4
    except OSError:
        w = 80
        warnings.warn("OSError when getting terminal size, output width set to 80")
    x = np.arange(1, hist.shape[0] + 1)
    y = np.add.reduceat(hist, range(0, hist.shape[0], hist.shape[0] // w + 1))
    y = np.log(y + 1)
    y = y - y.min()
    fig = tpl.figure()
    fig.hist(y, x, max_width=w)
    fig.show()
    print()


def save_plot(
    hist: NtCardHistogram,
    model: Model,
    kmer_size: int,
    style: str,
    title: str,
    table_rows: list[list[str]],
    plot_range: tuple[int, int],
    y_log: bool,
    out_path: str,
):
    x_range = np.arange(plot_range[0], plot_range[1])
    plt.style.use(style)
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(f"$k$-mer count ($k$ = {kmer_size})")
    ax.set_ylabel("Frequency")
    ax.bar(x_range, hist.values[x_range - 1], width=1, label="Histogram")
    ax.plot([], [])  # shift the color map
    if model.converged:
        y_model = model.score_components(x_range) * hist.num_distinct
        ax.plot(x_range, y_model[0, :], label=f"Errors")
        ax.plot(x_range, y_model[1, :], label="Heterozygous")
        ax.plot(x_range, y_model[2, :], label="Homozygous")
        ax.plot(x_range, y_model.sum(axis=0), label="Fitted model", ls="--", lw=2.5)
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, matplotlib.lines.Line2D([], [], linestyle=""))
    labels.insert(0, "")
    table = tabulate.tabulate(table_rows, tablefmt="plain", colalign=("left", "right"))
    legend = ax.legend(
        handles=handles,
        labels=labels,
        ncols=1,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.02),
        frameon=False,
        title=table,
        alignment="left",
    )
    plt.setp(legend.get_title(), family="Monospace")
    if y_log:
        ax.set_yscale("log")
        ax.relim()
        ax.autoscale_view()
    else:
        ax.set_ylim(top=hist.values[hist.first_minima :].max() * 1.5)
    fig.savefig(out_path)


def save_probs(hist: NtCardHistogram, model: Model, out_path: str):
    counts = np.arange(1, hist.max_count + 1)
    scores = model.score_components(counts)
    pd.DataFrame.from_dict(
        {
            "count": counts,
            "frequency": hist.values,
            "error": scores[0, :],
            "heterozgous": scores[1, :],
            "homozygous": scores[2, :],
        }
    ).to_csv(out_path, index=False)


def plot_fit_state(data, plots, x_range, error_history):
    loss, group_data = data[0], list(data[1])
    error_history.extend([loss] * len(group_data))
    plots[0].set_data(range(1, len(error_history) + 1), error_history)
    y_model = score(x_range, group_data[0][0])
    plots[1].set_data(x_range, y_model[0, :])
    plots[2].set_data(x_range, y_model[1:, :].sum(axis=0))
    plots[3].set_data(x_range, y_model.sum(axis=0))
    return plots


def save_fit_animation(
    history,
    hist: NtCardHistogram,
    style: str,
    plot_range: tuple[int, int],
    y_log: bool,
    out_path: str,
):
    x_range = np.arange(plot_range[0], plot_range[1])
    fig, axs = plt.subplots(ncols=2)
    plt.style.use(style)
    axs[0].set_xlabel("Count")
    axs[0].set_ylabel("Density")
    axs[0].bar(x_range, hist.as_distribution()[x_range - 1], width=1, label="Histogram")
    axs[0].plot([], [])
    (weak_plot,) = axs[0].plot([], [], label=f"Weak k-mers")
    (robust_plot,) = axs[0].plot([], [], label="Robust k-mers")
    (fitted_plot,) = axs[0].plot([], [], label="Fitted model", ls="--", lw=2.5)
    axs[0].set_xlim(plot_range[0], plot_range[1])
    if y_log:
        axs[0].set_ylim(0, 1)
        axs[0].set_yscale("log")
    else:
        axs[0].set_ylim(top=hist.as_distribution()[hist.first_minima :].max() * 1.5)
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Best model's error (log)")
    axs[1].set_xlim((1, len(history) + 1))
    errors = [err for _, err in history]
    axs[1].set_ylim(min(errors) * 0.9, max(errors) + min(errors) * 0.9)
    axs[1].set_yscale("log")
    axs[1].plot([], [])
    (err_plot,) = axs[1].plot([], [])
    error_history = []
    plots = [err_plot, weak_plot, robust_plot, fitted_plot]
    frame_groups = itertools.groupby(history, lambda x: x[1])
    func = functools.partial(
        plot_fit_state,
        plots=plots,
        x_range=x_range,
        error_history=error_history,
    )
    progress_bar = tqdm.tqdm(
        desc="Saving gif",
        total=len(set(errors)),
        leave=False,
    )
    matplotlib.animation.FuncAnimation(
        fig,
        func,
        frame_groups,
        save_count=len(set(errors)),
        repeat=False,
        interval=1,
    ).save(
        out_path,
        writer="pillow",
        fps=24,
        progress_callback=lambda *_: progress_bar.update(),
    )
