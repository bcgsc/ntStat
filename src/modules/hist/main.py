import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import stdout
import utils
import figures
from histogram import NtCardHistogram


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="k-mer spectrum file")
    parser.add_argument(
        "-f",
        "--table-format",
        help="stdout table format",
        choices=stdout.TablePrinter.get_valid_formats(),
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
        help="matplotlib style file, url, or one of "
        "available style names: ntstat.hist.default, ntstat.hist.paper, "
        f"{', '.join(plt.style.available)}",
        default="ntstat.hist.default",
    )
    parser.add_argument(
        "--y-log",
        help="plot y-axis in log scale",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-r",
        "--plot-range",
        help="x-axis limits in plots separated by a colon, "
        "i.e., use a:b to show results in the range a:b",
        type=utils.validate_plot_range_str,
    )
    parser.add_argument("-o", "--out-path", help="path to output plot", required=True)
    return parser.parse_args(argv)


def run(cmd_args: list[str]) -> int:
    args = parse_args(cmd_args)
    hist = NtCardHistogram(args.path)
    print("Histogram shape (y-axis in log scale):")
    stdout.print_hist(hist.values)
    table_printer = stdout.TablePrinter(args.table_format)
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
        ["First minima", hist.first_minima + 1],
        ["Elbow", hist.elbow + 1],
        ["Otsu thresholds", ", ".join(map(str, hist.otsu_thresholds + 1))],
    )
    if args.err_dist == "burr":
        err_rv, err_num_iters = hist.fit_burr()
    elif args.err_dist == "expon":
        err_rv, err_num_iters = hist.fit_expon()
    table_printer.print(
        "Fitted error distribution",
        ["Distribution", utils.scipy_rv_to_string(err_rv)],
        ["Number of iterations", err_num_iters],
        [f"KL Divergence (x <= {hist.first_minima + 1})", hist.err_kl_div(err_rv)],
    )
    gmm_rv, gmm_w, gmm_norm, gmm_num_iters = hist.fit_gmm()
    table_printer.print(
        "Fitted mixture model",
        ["Component 1", utils.scipy_rv_to_string(gmm_rv[0])],
        ["Component 2", utils.scipy_rv_to_string(gmm_rv[1])],
        ["Weights", ", ".join(f"{w:.3f}" for w in gmm_w)],
        ["Number of iterations", gmm_num_iters],
    )
    x_min, x_max = args.plot_range or (1, hist.max_count)
    figures.plot(
        hist,
        err_rv,
        gmm_rv,
        gmm_w,
        gmm_norm,
        args.style,
        x_min,
        x_max,
        args.y_log,
        args.out_path,
    )
    print(f"Saved plot to {args.out_path}")
    return 0


if __name__ == "__main__":
    exit(run(sys.argv[1:]))
