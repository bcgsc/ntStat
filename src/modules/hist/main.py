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
    parser.add_argument("-k", "--kmer-size", help="k-mer size", type=int, required=True)
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
    w_het, rv_het = model.heterozygous_rv
    w_hom, rv_hom = model.homozygous_rv
    num_solid = utils.count_solid_kmers(hist, model)
    err_rate = utils.get_error_rate(num_solid, hist.num_total, args.kmer_size)
    x_intersect = model.get_solid_weak_intersection(np.arange(1, hist.max_count + 1))

    print("Histogram shape (y-axis in log scale):")
    output.print_hist(hist.values)

    table_printer = output.TablePrinter(args.table_format)
    table_printer.print(
        "Fitted model",
        ["Error distribution", utils.scipy_rv_to_string(model.err_rv)],
        [f"Heterozygous (w = {w_het:.3f})", utils.scipy_rv_to_string(rv_het)],
        [f"Homozygous   (w = {w_hom:.3f})", utils.scipy_rv_to_string(rv_hom)],
        ["Number of iterations", num_iters],
        [f"KL Divergence", utils.kl_div(hist, model)],
    )
    table_printer.print(
        "k-mer statistics",
        ["Total number of k-mers", hist.num_total],
        ["Number of distinct k-mers", hist.num_distinct],
        ["Number of solid k-mers", num_solid],
    )
    table_printer.print(
        "Thresholds",
        ["Elbow", hist.elbow + 1],
        ["First minima", hist.first_minima + 1],
        ["Weak/solid intersection", x_intersect],
        ["Otsu thresholds", ", ".join(map(str, hist.otsu_thresholds + 1))],
    )
    dataset_table_rows = [
        ["Coverage", f"{model.coverage:.1f}x"],
        ["Error rate", f"{err_rate * 100:.2f}%"],
        ["Quality score", f"Q{int(-10 * np.log10(err_rate))}"],
        ["Genome size", utils.format_bp(int(hist.num_total / model.coverage))],
        ["Total size", f"{utils.format_bp(hist.num_total)}"],
    ]
    table_printer.print("Dataset characteristics", *dataset_table_rows)

    if args.out_path:
        output.save_plot(
            hist,
            model,
            x_intersect,
            args.style,
            args.title,
            dataset_table_rows,
            args.plot_range or (1, hist.max_count),
            args.y_log,
            args.out_path,
        )
    return 0


if __name__ == "__main__":
    exit(run(sys.argv[1:]))
