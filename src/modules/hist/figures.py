import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from histogram import NtCardHistogram


class HistogramPlotter:

    @staticmethod
    def get_valid_styles():
        return plt.style.available

    def __init__(
        self,
        hist: NtCardHistogram,
        style: str,
        x_min: int,
        x_max: int,
        y_log: bool,
        out_path: str,
    ) -> None:
        plt.style.use(style)
        self.__hist = hist
        self.__y_log = y_log
        self.__out_path = out_path
        self.__x_range = np.arange(x_min, x_max)

    def plot_thresholds(self) -> None:
        fig, ax = plt.subplots()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ax.set_yscale("log" if self.__y_log else ax.get_yscale())
        ax.set_xlabel("K-mer count")
        ax.set_ylabel("Frequency")
        ax.plot(self.__x_range, self.__hist.values[self.__x_range - 1])
        thresholds = {
            "First minima": self.__hist.first_minima + 1,
            "Elbow": self.__hist.elbow + 1,
        }
        for i, x in enumerate(self.__hist.otsu_thresholds):
            thresholds[f"Otsu threshold {i + 1}"] = x + 1
        for i, (name, x) in enumerate(thresholds.items()):
            if self.__x_range[0] <= x <= self.__x_range[-1]:
                ax.axvline(x, label=name, c=colors[i + 1], linestyle="--")
        ax.legend()
        fig.savefig(os.path.join(self.__out_path, "thresholds.png"))

    def plot_error_distribution(
        self,
        err_rv: scipy.stats.rv_continuous,
        gmm_rv: list[scipy.stats.rv_continuous],
        gmm_w: list[float],
    ) -> None:
        fig, ax = plt.subplots()
        ax.set_yscale("log" if self.__y_log else ax.get_yscale())
        ax.set_xlabel("K-mer count")
        ax.set_ylabel("Probability density")
        y_hist = self.__hist.values[self.__x_range - 1].astype(np.float64)
        y_hist /= y_hist.sum()
        y_err = err_rv.pdf(self.__x_range)
        ax.bar(self.__x_range, y_hist, label="Histogram", alpha=0.25)
        ax.plot(self.__x_range, y_err, label=f"Weak k-mers ({err_rv.dist.name})")
        y_gmm = np.zeros(self.__x_range.shape[0])
        for i, (rv, w) in enumerate(zip(gmm_rv, gmm_w)):
            y = w * rv.pdf(self.__x_range)
            y_gmm += y
            ax.plot(self.__x_range, y, label=f"Component {i + 1} ({rv.dist.name})")
        ax.plot(self.__x_range, y_gmm, label="Mixture model", linestyle="dashed")
        ax.legend()
        fig.savefig(os.path.join(self.__out_path, "distributions.png"))
