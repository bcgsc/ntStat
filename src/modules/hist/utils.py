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


def format_bp(x):
    units = ["", "K", "M", "G", "T", "P"]
    magnitude = int(np.log10(x))
    return f"{x / 10 ** magnitude:.3f}{units[magnitude // 3]}bp"


def kl_div(hist, model):
    p = model.pdf(np.arange(1, hist.max_count + 1))
    q = hist.as_distribution()
    return scipy.stats.entropy(p, q)
