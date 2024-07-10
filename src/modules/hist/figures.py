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
        y_log: bool,
        out_path: str,
    ) -> None:
        plt.style.use(style)
        self.__hist = hist
        self.__y_log = y_log
        self.__out_path = out_path

    def plot_thresholds(self) -> None:
        fig, ax = plt.subplots()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ax.set_yscale("log" if self.__y_log else ax.get_yscale())
        ax.set_xlabel("K-mer count")
        ax.set_ylabel("Frequency")
        ax.plot(np.arange(1, self.__hist.max_count + 1), self.__hist.values)
        thresholds = {
            "First minima": self.__hist.first_minima + 1,
            "Elbow": self.__hist.elbow + 1,
        }
        for i, x in enumerate(self.__hist.otsu_thresholds):
            thresholds[f"Otsu threshold {i + 1}"] = x + 1
        for i, (name, x) in enumerate(thresholds.items()):
            ax.axvline(x, label=name, c=colors[i + 1], linestyle="--")
        ax.legend()
        fig.savefig(os.path.join(self.__out_path, "thresholds.png"))

    def plot_error_distribution(self, err_rv: scipy.stats.rv_continuous) -> None:
        fig, ax = plt.subplots()
        ax.set_yscale("log" if self.__y_log else ax.get_yscale())
        ax.set_xlabel("K-mer count")
        ax.set_ylabel("Frequency")
        x = np.arange(1, self.__hist.max_count + 1)
        y_hist = self.__hist.values / self.__hist.values.sum()
        y_err = err_rv.pdf(x)
        ax.plot(x, y_hist, label="Actual")
        ax.plot(x, y_err, label=f"Weak k-mers ({err_rv.dist.name})")
        ax.legend()
        fig.savefig(os.path.join(self.__out_path, "distributions.png"))
