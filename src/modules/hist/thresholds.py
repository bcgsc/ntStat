import dataclasses

import kneed
import numpy as np
import numpy.typing
import scipy.signal
import skimage.filters


def find_first_minima(hist: numpy.typing.NDArray[np.uint64]) -> int:
    return scipy.signal.argrelextrema(hist, np.less)[0][0]


def find_first_maxima(hist: numpy.typing.NDArray[np.uint64]) -> int:
    return scipy.signal.argrelextrema(hist, np.greater)[0][0]


def find_median(hist: numpy.typing.NDArray[np.uint64]) -> int:
    return np.argsort(hist)[hist.shape[0] // 2]


def find_knee(hist: numpy.typing.NDArray[np.uint64]) -> int:
    x = np.arange(hist.shape[0])
    locator = kneed.KneeLocator(x, hist, curve="convex", direction="decreasing")
    return locator.knee


def find_otsu_thresholds(
    hist: numpy.typing.NDArray[np.uint64],
) -> numpy.typing.NDArray[np.uint]:
    return skimage.filters.threshold_multiotsu(hist=hist)


@dataclasses.dataclass(frozen=True, init=False)
class Thresholds:
    min0: int
    max0: int
    median: int
    knee: int
    otsu: numpy.typing.NDArray[np.uint]

    def __init__(self, hist) -> None:
        object.__setattr__(self, "min0", find_first_minima(hist))
        object.__setattr__(self, "max0", find_first_maxima(hist))
        object.__setattr__(self, "median", find_median(hist))
        object.__setattr__(self, "knee", find_knee(hist))
        object.__setattr__(self, "otsu", find_otsu_thresholds(hist))
