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

    def as_distribution(self) -> numpy.typing.NDArray[np.float64]:
        return self.values / self.values.sum()

    def fit_burr(self) -> tuple[scipy.stats.rv_continuous, int]:
        return self.__fit_err_pdf(scipy.stats.burr, [1, 1, 1])

    def fit_expon(self) -> tuple[scipy.stats.rv_continuous, int]:
        return self.__fit_err_pdf(scipy.stats.expon, [0.5, 0.5])

    def kl_div(
        self,
        err_rv: scipy.stats.rv_continuous,
        gmm_rv: list[scipy.stats.rv_continuous],
        gmm_w: list[float],
    ) -> np.float64:
        x = np.arange(1, self.max_count + 1)
        p_err = err_rv.pdf(x)
        p_gmm = gmm_w[0] * gmm_rv[0].pdf(x) + gmm_w[1] * gmm_rv[1].pdf(x)
        p = p_err + p_gmm
        q = self.as_distribution()
        return scipy.stats.entropy(p, q)

    def fit_gmm(
        self,
        err_rv: scipy.stats.rv_continuous,
    ) -> tuple[list[scipy.stats.rv_continuous], list[float], int]:
        x = np.arange(1, self.max_count + 1)
        y = np.clip(self.as_distribution() - err_rv.pdf(x), 0, None)
        d = skimage.filters.threshold_multiotsu(hist=(self.values, x - 1))[-1]
        p0 = [0.5, d / 2, 1, 0.5, d, 1]
        p, _, info, *_ = scipy.optimize.curve_fit(
            utils.gmm, x, y, p0, full_output=True, maxfev=5000
        )
        w = [p[0], p[3]]
        rvs = [scipy.stats.norm(p[1], p[2]), scipy.stats.norm(p[4], p[5])]
        return rvs, w, info["nfev"]

    def __fit_err_pdf(
        self,
        rv: scipy.stats.rv_continuous,
        p0: list[float],
    ) -> tuple[scipy.stats.rv_continuous, float, int]:
        x = np.arange(1, self.max_count + 1)
        y = self.as_distribution()
        p, _, info, *_ = scipy.optimize.curve_fit(rv.pdf, x, y, p0, full_output=True)
        return rv(*p), info["nfev"]
