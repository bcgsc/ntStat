import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import output
import utils
from histogram import NtCardHistogram
from model import Model


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
        "-m",
        "--style",
        help="matplotlib style file, url, or one of "
        "available style names: ntstat.hist.default, ntstat.hist.paper, "
        f"{', '.join(plt.style.available)}",
        default="ntstat.hist.default",
    )
    parser.add_argument("-t", "--title", help="title to put on plot")
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
    parser.add_argument("-o", "--out-path", help="path to output plot")
    return parser.parse_args(argv)


def run(cmd_args: list[str]) -> int:
    args = parse_args(cmd_args)
    hist = NtCardHistogram(args.path)
    model = Model()
    num_iters = model.fit(hist)
    print("Histogram shape (y-axis in log scale):")
    output.print_hist(hist.values)
    table_printer = output.TablePrinter(args.table_format)
    table_printer.print(
        "Fitted model",
        ["Error distribution", utils.scipy_rv_to_string(model.err_rv)],
        [
            f"Heterozygous (w = {model.heterozygous_rv[0]:.2f})",
            utils.scipy_rv_to_string(model.heterozygous_rv[1]),
        ],
        [
            f"Homozygous   (w = {model.homozygous_rv[0]:.2f})",
            utils.scipy_rv_to_string(model.homozygous_rv[1]),
        ],
        ["Number of iterations", num_iters],
        [f"KL Divergence", utils.kl_div(hist, model)],
    )
    num_solid = utils.count_solid_kmers(hist, model)
    table_printer.print(
        "k-mer statistics",
        ["Total number of k-mers", hist.total],
        ["Number of distinct k-mers", hist.distinct],
        ["Number of solid k-mers", num_solid],
    )
    x_intersect = model.get_solid_weak_intersection(np.arange(1, hist.max_count + 1))
    table_printer.print(
        "Thresholds",
        ["Elbow", hist.elbow + 1],
        ["First minima", hist.first_minima + 1],
        ["Weak/solid intersection", x_intersect],
        ["Otsu thresholds", ", ".join(map(str, hist.otsu_thresholds + 1))],
    )
    table_printer.print(
        "Dataset characteristics",
        ["Coverage", f"{model.coverage:.1f}x"],
        ["Error rate", f"{(1 - num_solid / hist.total) * 100:.3f}%"],
        ["Genome length", utils.format_bp(int(num_solid / model.coverage))],
    )
    x_min, x_max = args.plot_range or (1, hist.max_count)
    if args.out_path:
        output.save_plot(
            hist,
            model,
            x_intersect,
            args.style,
            args.title,
            x_min,
            x_max,
            args.y_log,
            args.out_path,
        )
    return 0


if __name__ == "__main__":
    exit(run(sys.argv[1:]))
