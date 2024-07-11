import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import scipy.stats
from histogram import NtCardHistogram


def plot(
    hist: NtCardHistogram,
    err_rv: scipy.stats.rv_continuous,
    gmm_rv: list[scipy.stats.rv_continuous],
    gmm_w: list[float],
    gmm_norm: float,
    style: str,
    x_min: int,
    x_max: int,
    y_log: bool,
    out_path: str,
):
    x_range = np.arange(x_min, x_max + 1)
    plt.style.use(style)
    fig, ax = plt.subplots()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.set_yscale("log" if y_log else ax.get_yscale())
    ax.set_xlabel("K-mer count")
    ax.set_ylabel("Frequency")

    bars = ax.bar(x_range, hist.values[x_range - 1], alpha=0.5)
    y_err = err_rv.pdf(x_range) * hist.values.sum()
    ax.plot(x_range, y_err, label=f"Weak k-mers ({err_rv.dist.name})")
    y_gmm = np.zeros(x_range.shape[0])
    for i, (rv, w) in enumerate(zip(gmm_rv, gmm_w)):
        y = w * rv.pdf(x_range) * gmm_norm
        y_gmm += y
        ax.plot(x_range, y, label=f"Component {i + 1} ({rv.dist.name})")
    ax.plot(x_range, y_gmm, label="Mixture model", linestyle="dashed")

    thresholds = {
        "First minima": hist.first_minima + 1,
        "Elbow": hist.elbow + 1,
    }
    for i, x in enumerate(hist.otsu_thresholds):
        thresholds[f"Otsu threshold {i + 1}"] = x + 1
    handles, labels = ax.get_legend_handles_labels()
    for i, (name, x) in enumerate(thresholds.items()):
        if x_range[0] <= x <= x_range[-1]:
            bars[x - 1].set_color(colors[i + 1])
            handles.append(matplotlib.patches.Patch(facecolor=colors[i + 1]))
            labels.append(name)

    ax.legend(handles=handles, labels=labels)
    fig.savefig(out_path)
