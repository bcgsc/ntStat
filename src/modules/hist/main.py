import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import output
import utils
from histogram import NtCardHistogram


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="k-mer spectrum file")
    parser.add_argument(
        "-f",
        "--table-format",
        help="stdout table format",
        choices=output.TablePrinter.get_valid_formats(),
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
    parser.add_argument(
        "-o",
        "--out-path",
        help="path to output directory",
        default=".",
    )
    return parser.parse_args(argv)


def run(cmd_args: list[str]) -> int:
    args = parse_args(cmd_args)
    hist = NtCardHistogram(args.path)
    if args.err_dist == "burr":
        err_rv, err_num_iters = hist.fit_burr()
    elif args.err_dist == "expon":
        err_rv, err_num_iters = hist.fit_expon()
    gmm_rv, gmm_w, gmm_num_iters = hist.fit_gmm(err_rv)
    print("Histogram shape (y-axis in log scale):")
    output.print_hist(hist.values)
    table_printer = output.TablePrinter(args.table_format)
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
        "Fitted model",
        ["Error distribution", utils.scipy_rv_to_string(err_rv)],
        [f"Heterozygous (w = {gmm_w[0]:.2f})", utils.scipy_rv_to_string(gmm_rv[0])],
        [f"Homozygous   (w = {gmm_w[1]:.2f})", utils.scipy_rv_to_string(gmm_rv[1])],
        ["Number of iterations", err_num_iters + gmm_num_iters],
        [f"KL Divergence", hist.kl_div(err_rv, gmm_rv, gmm_w)],
    )
    x = np.arange(1, hist.max_count + 1)
    y_err = err_rv.pdf(x)
    y_gmm = gmm_w[0] * gmm_rv[0].pdf(x) + gmm_w[1] * gmm_rv[1].pdf(x)
    x_intersect = utils.find_intersection(y_err, y_gmm) + 1
    table_printer.print(
        "Thresholds",
        ["First minima", hist.first_minima + 1],
        ["Elbow", hist.elbow + 1],
        ["Otsu thresholds", ", ".join(map(str, hist.otsu_thresholds + 1))],
        ["Weak/solid intersection", x_intersect + 1],
    )
    cov = utils.get_coverage(gmm_rv)
    genome_len = int(hist.total / cov)
    table_printer.print(
        "Dataset characteristics (estimated)",
        ["Coverage", f"{cov:.1f}x"],
        ["Genome length", utils.format_bp(genome_len)],
    )
    x_min, x_max = args.plot_range or (1, hist.max_count)
    plot_path = os.path.join(args.out_path, "plot.png")
    output.save_plot(
        hist,
        err_rv,
        gmm_rv,
        gmm_w,
        x_intersect,
        args.style,
        x_min,
        x_max,
        args.y_log,
        plot_path,
    )
    print(f"Saved plot to {plot_path}")
    return 0


if __name__ == "__main__":
    exit(run(sys.argv[1:]))
