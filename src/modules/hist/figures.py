import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
import termplotlib as tpl
from histogram import NtCardHistogram


def print_hist(hist: numpy.typing.NDArray[np.uint64]) -> None:
    w = 3 * os.get_terminal_size().columns // 4
    x = np.arange(1, hist.shape[0] + 1)
    y = np.add.reduceat(hist, range(0, hist.shape[0], hist.shape[0] // w + 1))
    y = np.log(y + 1)
    y = y - y.min()
    fig = tpl.figure()
    fig.hist(y, x, max_width=w)
    fig.show()
    print()


def plot_thresholds(hist: NtCardHistogram, ylog: bool, out_path: str) -> None:
    fig, ax = plt.subplots()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.set_yscale("log" if ylog else ax.get_yscale())
    ax.set_xlabel("K-mer count")
    ax.set_ylabel("Frequency")
    ax.plot(np.arange(1, hist.max_count + 1), hist.values)
    thresholds = {
        "First minima": hist.thresholds.min0 + 1,
        "First peak": hist.thresholds.max0 + 1,
        "Median index": hist.thresholds.median + 1,
        "Elbow": hist.thresholds.elbow + 1,
    }
    for i, x in enumerate(hist.thresholds.otsu):
        thresholds[f"Otsu threshold {i + 1}"] = x + 1
    for i, (name, x) in enumerate(thresholds.items()):
        ax.axvline(x, label=name, c=colors[i + 1], linestyle="--")
    ax.legend()
    fig.savefig(os.path.join(out_path, "thresholds.png"))
