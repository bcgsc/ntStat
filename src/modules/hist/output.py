import os
import warnings

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
import tabulate
import termplotlib as tpl
from histogram import NtCardHistogram
from model import Model


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
    x_intersect: int,
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
    ax.set_yscale("log" if y_log else ax.get_yscale())
    ax.set_xlabel(f"$k$-mer count ($k$ = {kmer_size})")
    ax.set_ylabel("Frequency")
    y_model = model.score_components(x_range) * hist.num_total
    bars = ax.bar(x_range, hist.values[x_range - 1], width=1, label="Histogram")
    ax.plot([], [])  # shift the color map
    ax.plot(x_range, y_model[0, :], label=f"Weak k-mers")
    ax.plot(x_range, y_model[1:, :].sum(axis=0), label="Solid k-mers")
    ax.plot(x_range, y_model.sum(axis=0), label="Fitted model", ls="--", lw=2.5)
    ax.set_ylim(bottom=1)
    thresholds = {
        "Heterozygous peak": np.rint(model.peaks[0]).astype(int),
        "Homozygous peak": np.rint(model.peaks[1]).astype(int),
        "First minima": hist.first_minima + 1,
        "Weak/solid intersection": x_intersect,
    }
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, matplotlib.lines.Line2D([], [], linestyle=""))
    labels.insert(0, "")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, x) in enumerate(thresholds.items()):
        if x_range[0] <= x <= x_range[-1]:
            color = colors[(i + 4) % len(colors)]
            bars[x - x_range[0]].set_color(color)
            handles.append(matplotlib.patches.Patch(facecolor=color))
            labels.append(name)
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
    fig.savefig(out_path)
