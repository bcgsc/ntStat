import argparse
import re

import kneed
import numpy as np
import numpy.typing
import scipy.signal
import scipy.stats
import skimage.filters


def validate_plot_range_str(r_str: str) -> list[int, int]:
    if r_str == "auto":
        return [1, 0]
    pattern = re.compile(r"^\d+:\d+$")
    if not pattern.match(r_str):
        raise argparse.ArgumentTypeError("invalid value")
    return list(map(int, r_str.split(":")))


def scipy_rv_to_string(rv: scipy.stats.rv_continuous | scipy.stats.rv_discrete):
    dist_args_str = ", ".join(f"{x:.3f}" for x in rv.args)
    return f"{rv.dist.name.title()}({dist_args_str})"


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


def format_float(x):
    d = np.floor(np.log10(x))
    return f"{x:.{-int(d)}f}"


def format_bp(x):
    units = ["", "K", "M", "G", "T", "P"]
    magnitude = int(np.log10(x)) // 3
    return f"{x / 1000 ** magnitude:.1f} {units[magnitude]}bp"


def kl_div(hist, model):
    p = model.pdf(np.arange(1, hist.max_count + 1))
    q = hist.as_distribution()
    return scipy.stats.entropy(p, q)


def count_robust_kmers(hist, model):
    x = np.arange(1, hist.max_count + 1)
    scores = model.score_components(x)
    y_rob = scores[1:, :].sum(axis=0)
    p = np.divide(y_rob, scores.sum(axis=0))
    return int((x * p * hist.values).sum())


def get_error_rate(num_robust: int, num_total: int, kmer_size: int) -> float:
    return (1 - num_robust / num_total) / kmer_size


def get_heterozygosity(hist, model):
    x = np.arange(1, hist.max_count + 1)
    scores = model.score_components(x)
    p = np.divide(scores[1, :], scores.sum(axis=0))
    num_het = int((x * p * hist.values).sum())
    num_robust = count_robust_kmers(hist, model)
    return num_het / num_robust
