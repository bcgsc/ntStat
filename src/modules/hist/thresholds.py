import dataclasses

import kneed
import numpy as np
import numpy.typing
import scipy.signal
import skimage.filters


def find_first_minima(hist: numpy.typing.NDArray[np.uint64]) -> int:
    return scipy.signal.argrelextrema(hist, np.less)[0][0]


def find_elbow(hist: numpy.typing.NDArray[np.uint64]) -> int:
    x = np.arange(hist.shape[0])
    locator = kneed.KneeLocator(x, hist, curve="convex", direction="decreasing")
    return locator.elbow


def find_otsu_thresholds(
    hist: numpy.typing.NDArray[np.uint64],
) -> numpy.typing.NDArray[np.uint]:
    return skimage.filters.threshold_multiotsu(hist=hist)


@dataclasses.dataclass(frozen=True, init=False)
class Thresholds:
    min0: int
    elbow: int
    otsu: numpy.typing.NDArray[np.uint]

    def __init__(self, hist) -> None:
        object.__setattr__(self, "min0", find_first_minima(hist))
        object.__setattr__(self, "elbow", find_elbow(hist))
        object.__setattr__(self, "otsu", find_otsu_thresholds(hist))
