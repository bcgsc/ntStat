import argparse
import json
import re

import numpy as np
import scipy.signal
import scipy.stats

from .histogram import NtCardHistogram
from .model import Model


def validate_plot_range_str(r_str: str) -> list[int, int]:
    if r_str == "auto":
        return [1, 0]
    pattern = re.compile(r"^\d+:\d+$")
    if not pattern.match(r_str):
        raise argparse.ArgumentTypeError("invalid value")
    return list(map(int, r_str.split(":")))


def validate_config_json(path: str) -> dict:
    with open(path) as fp:
        return json.load(fp)


def scipy_rv_to_string(rv: scipy.stats.rv_continuous | scipy.stats.rv_discrete):
    dist_args_str = ", ".join(f"{x:.3f}" for x in rv.args)
    return f"{rv.dist.name.title()}({dist_args_str})"


def format_float(x):
    d = np.floor(np.log10(x))
    return f"{x:.{-int(d)}f}"


def format_bp(x):
    units = ["", "K", "M", "G", "T", "P"]
    magnitude = int(np.log10(x)) // 3
    return f"{x / 1000 ** magnitude:.1f} {units[magnitude]}bp"


def kl_div(hist: NtCardHistogram, model: Model):
    p = model.pdf(np.arange(1, hist.max_count + 1))
    q = hist.as_distribution()
    return scipy.stats.entropy(p, q)


def sum_absolute_error(hist: NtCardHistogram, model: Model):
    y_pred = model.pdf(np.arange(1, hist.max_count + 1))
    y_true = hist.as_distribution()
    return np.abs(y_true - y_pred).sum()


def count_robust_kmers(hist: NtCardHistogram, model: Model):
    c = min(model.homozygous_rv[1].interval(0.9995)[1], hist.max_count)
    x = np.arange(1, c)
    h = hist.values[x.astype(int) - 1]
    y = model.get_responsibilities(x)[1:, :].sum(axis=0) * h
    return int(sum(x * y) / sum(x * h) * hist.num_total)


def count_heterozygous_kmers(hist: NtCardHistogram, model: Model):
    c = min(model.homozygous_rv[1].interval(0.9995)[1], hist.max_count)
    x = np.arange(1, c)
    h = hist.values[x.astype(int) - 1]
    y = model.get_responsibilities(x)[1:-1, :].sum(axis=0) * h
    return int(sum(x * y) / sum(x * h) * count_robust_kmers(hist, model))
