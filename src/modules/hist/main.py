import argparse
import sys

import figures
import numpy as np
import scipy.stats
import utils
from histogram import NtCardHistogram
from table_printer import TablePrinter


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="k-mer spectrum file")
    parser.add_argument(
        "-f",
        "--table-format",
        help="stdout table format",
        choices=TablePrinter.get_valid_formats(),
        default="simple_outline",
    )
    parser.add_argument(
        "-e",
        "--err-dist",
        help="error distribution",
        choices=["burr", "expon"],
        default="burr",
    )
    parser.add_argument("-o", "--out-path", help="path for saving results", default=".")
    return parser.parse_args(argv)


def run(cmd_args: list[str]) -> int:
    args = parse_args(cmd_args)
    hist = NtCardHistogram(args.path)
    print("Histogram shape (y-axis in log scale):")
    figures.print_hist(hist.values)
    table_printer = TablePrinter(args.table_format)
    table_printer.print(
        "Basic stats",
        ["Maximum count", hist.max_count],
        ["Support", np.count_nonzero(hist.values)],
        ["Total number of k-mers", hist.total],
        ["Number of distinct k-mers", hist.distinct],
        ["Mean frequency", int(hist.values.mean())],
        ["Frequency standard deviation", int(hist.values.std())],
        ["Median frequency", int(np.median(hist.values))],
        ["Skewness", int(scipy.stats.skew(hist.values))],
        ["Kurtosis", int(scipy.stats.kurtosis(hist.values))],
    )
    table_printer.print(
        "Thresholds",
        ["First minima", hist.thresholds.min0 + 1],
        ["First peak", hist.thresholds.max0 + 1],
        ["Median index", hist.thresholds.median + 1],
        ["Knee", hist.thresholds.knee + 1],
        ["Otsu thresholds", ", ".join(map(str, hist.thresholds.otsu + 1))],
    )
    if args.err_dist == "burr":
        err_dist, num_iters = hist.fit_burr()
    elif args.err_dist == "expon":
        err_dist, num_iters = hist.fit_expon()
    table_printer.print(
        "Fitted error distribution",
        ["Distribution", utils.scipy_rv_to_string(err_dist)],
        ["Number of iterations", num_iters],
    )
    return 0


if __name__ == "__main__":
    exit(run(sys.argv[1:]))
