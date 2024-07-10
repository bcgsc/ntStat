import argparse
import sys

import figures
import matplotlib.pyplot as plt
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
    parser.add_argument(
        "-m",
        "--style",
        help="matplotlib style",
        default="ggplot",
    )
    parser.add_argument(
        "--y-log",
        help="plot y-axis in log scale",
        action=argparse.BooleanOptionalAction,
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
    )
    table_printer.print(
        "Thresholds",
        ["First minima", hist.thresholds.min0 + 1],
        ["Elbow", hist.thresholds.elbow + 1],
        ["Otsu thresholds", ", ".join(map(str, hist.thresholds.otsu + 1))],
    )
    if args.err_dist == "burr":
        err_rv, num_iters = hist.fit_burr()
    elif args.err_dist == "expon":
        err_rv, num_iters = hist.fit_expon()
    table_printer.print(
        "Fitted error distribution",
        ["Distribution", utils.scipy_rv_to_string(err_rv)],
        ["Number of iterations", num_iters],
    )
    plt.style.use(args.style)
    figures.plot_thresholds(hist, args.y_log, args.out_path)
    figures.plot_distributions(hist, err_rv, args.y_log, args.out_path)
    return 0


if __name__ == "__main__":
    exit(run(sys.argv[1:]))
