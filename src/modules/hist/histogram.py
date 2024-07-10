import numpy as np
import numpy.typing
import pandas as pd
import scipy.stats
from thresholds import Thresholds


class NtCardHistogram:

    def __init__(self, path: str) -> None:
        data = pd.read_csv(path, delimiter=r"\s+", header=None)[1].values
        self.__total = int(data[0])
        self.__distinct = int(data[1])
        self.__hist = data[2:]
        self.__hist.setflags(write=False)
        self.__thresholds = Thresholds(self.__hist)

    def __getitem__(self, count):
        return self.__hist[count - 1]

    @property
    def values(self) -> numpy.typing.NDArray[np.uint64]:
        return self.__hist

    @property
    def total(self) -> int:
        return self.__total

    @property
    def distinct(self) -> int:
        return self.__distinct

    @property
    def max_count(self) -> int:
        return self.__hist.shape[0]

    @property
    def thresholds(self) -> Thresholds:
        return self.__thresholds

    def __fit_pdf(
        self,
        rv: scipy.stats.rv_continuous,
        p0: list[float],
    ) -> tuple[scipy.stats.rv_continuous, int]:
        x = np.arange(1, self.max_count + 1)
        y = self.__hist / self.__hist.sum()
        p, _, info, *_ = scipy.optimize.curve_fit(rv.pdf, x, y, p0, full_output=True)
        return rv(*p), info["nfev"]

    def fit_burr(self) -> tuple[scipy.stats.rv_continuous, int]:
        return self.__fit_pdf(scipy.stats.burr, [1, 1, 1])

    def fit_expon(self) -> tuple[scipy.stats.rv_continuous, int]:
        return self.__fit_pdf(scipy.stats.expon, [0.5, 0.5])

    def err_kl_div(self, err_rv: scipy.stats.rv_continuous) -> np.float64:
        x_err = np.arange(1, self.thresholds.min0)
        y_err = err_rv.pdf(np.arange(1, self.thresholds.min0))
        return scipy.stats.entropy(y_err, self.values[x_err - 1])
