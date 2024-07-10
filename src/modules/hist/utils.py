import kneed
import numpy as np
import numpy.typing
import scipy.signal
import scipy.stats
import skimage.filters


def scipy_rv_to_string(rv: scipy.stats.rv_continuous | scipy.stats.rv_discrete):
    dist_args_str = ", ".join(f"{x:.3f}" for x in rv.args)
    return f"X ~ {rv.dist.name.title()}({dist_args_str})"


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
