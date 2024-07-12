import numpy as np
import numpy.typing
import pandas as pd
import scipy.stats
import skimage.filters
import utils


class NtCardHistogram:

    def __init__(self, path: str) -> None:
        data = pd.read_csv(path, delimiter=r"\s+", header=None)[1].values
        self.__total = int(data[0])
        self.__distinct = int(data[1])
        self.__hist = data[2:]
        self.__hist.setflags(write=False)
        self.__elbow = utils.find_elbow(self.__hist)
        self.__min0 = utils.find_first_minima(self.__hist)
        self.__otsu = utils.find_otsu_thresholds(self.__hist)
        self.__otsu.setflags(write=False)

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
    def elbow(self) -> int:
        return self.__elbow

    @property
    def first_minima(self) -> int:
        return self.__min0

    @property
    def otsu_thresholds(self) -> numpy.typing.NDArray[np.uint]:
        return self.__otsu

    def __fit_err_pdf(
        self,
        rv: scipy.stats.rv_continuous,
        p0: list[float],
    ) -> tuple[scipy.stats.rv_continuous, float, int]:
        x = np.arange(1, self.first_minima + 1)
        norm = self.__hist[x - 1].sum()
        y = self.__hist[x - 1] / norm
        p, _, info, *_ = scipy.optimize.curve_fit(rv.pdf, x, y, p0, full_output=True)
        return rv(*p), norm, info["nfev"]

    def fit_burr(self) -> tuple[scipy.stats.rv_continuous, float, int]:
        return self.__fit_err_pdf(scipy.stats.burr, [1, 1, 1])

    def fit_expon(self) -> tuple[scipy.stats.rv_continuous, float, int]:
        return self.__fit_err_pdf(scipy.stats.expon, [0.5, 0.5])

    def err_kl_div(self, err_rv: scipy.stats.rv_continuous) -> np.float64:
        x_err = np.arange(1, self.first_minima)
        y_err = err_rv.pdf(np.arange(1, self.first_minima))
        return scipy.stats.entropy(y_err, self.values[x_err - 1])

    def fit_gmm(self) -> tuple[list[scipy.stats.rv_continuous], list[float], float, int]:
        x = np.arange(self.first_minima, self.max_count)
        norm = self.__hist[x - 1].sum()
        y = self.__hist[x - 1] / norm
        otsu = skimage.filters.threshold_multiotsu(hist=(y, x - 1))
        p0 = [0.5, (otsu[0] - self.first_minima) / 2, 1, 0.5, otsu.mean(), 1]
        p, _, info, *_ = scipy.optimize.curve_fit(
            utils.gmm, x, y, p0, full_output=True, maxfev=5000
        )
        w = [p[0], p[3]]
        rvs = [scipy.stats.norm(p[1], p[2]), scipy.stats.norm(p[4], p[5])]
        return rvs, w, norm, info["nfev"]
