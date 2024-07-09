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

    @property
    def as_scipy_rv(self) -> scipy.stats.rv_histogram:
        return self.__hist / self.__hist.sum()

    def __fit_pdf(
        self, rv: scipy.stats.rv_continuous
    ) -> tuple[scipy.stats.rv_continuous, int]:
        x = np.arange(1, self.max_count + 1)
        y = self.__hist / self.__hist.sum()
        n_args = 2 if rv.shapes is None else len(rv.shapes.split(", ")) + 1
        p0 = [0.5] * n_args
        p, _, info, *_ = scipy.optimize.curve_fit(rv.pdf, x, y, p0, full_output=True)
        return rv(*p), info["nfev"]

    def fit_burr(self) -> tuple[scipy.stats.rv_continuous, int]:
        return self.__fit_pdf(scipy.stats.burr)

    def fit_expon(self) -> tuple[scipy.stats.rv_continuous, int]:
        return self.__fit_pdf(scipy.stats.expon)
