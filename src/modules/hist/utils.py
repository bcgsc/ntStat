import argparse
import re

import kneed
import numpy as np
import numpy.typing
import scipy.signal
import scipy.stats
import skimage.filters


def validate_plot_range_str(r_str: str) -> tuple[int, int]:
    pattern = re.compile(r"^\d+:\d+$")
    if not pattern.match(r_str):
        raise argparse.ArgumentTypeError("invalid value")
    return tuple(map(int, r_str.split(":")))


def scipy_rv_to_string(rv: scipy.stats.rv_continuous | scipy.stats.rv_discrete):
    dist_args_str = ", ".join(f"{x:.1f}" for x in rv.args)
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


def get_coverage(gmm_rv: list[scipy.stats.rv_continuous]) -> float:
    return max(gmm_rv[0].args[0], gmm_rv[1].args[0])


def format_bp(x):
    units = ["", "K", "M", "G", "T", "P"]
    magnitude = int(np.log10(x))
    return f"{x / 10 ** magnitude:.3f}{units[magnitude // 3]}bp"


def gmm(
    x: numpy.typing.NDArray[np.float64],
    w1: float,
    mu1: float,
    sigma1: float,
    w2: float,
    mu2: float,
    sigma2: float,
) -> numpy.typing.NDArray[np.float64]:
    p1 = scipy.stats.norm.pdf(x, mu1, sigma1)
    p2 = scipy.stats.norm.pdf(x, mu2, sigma2)
    return w1 * p1 + w2 * p2


def find_intersection(
    y1: numpy.typing.NDArray[np.float64],
    y2: numpy.typing.NDArray[np.float64],
) -> int:
    return np.argmin(np.abs(y1 - y2))
