import argparse
import sys
import time

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
        help="plot x-axis limits (inclusive) separated by a colon, "
        "i.e., use a:b to show results in the range [a, b]. "
        "'auto' will automatically adjust the limits for better visibility.",
        type=utils.validate_plot_range_str,
    )
    parser.add_argument("-o", "--plot", help="path to output plot")
    parser.add_argument(
        "-p",
        "--probs",
        help="path to output probabilities in csv format",
    )
    parser.add_argument(
        "-g",
        "--fit-gif",
        help="path to output model fit history animation",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="path to differential evolution config file (json)",
        type=utils.validate_config_json,
    )
    parser.add_argument("--no-model", action="store_true")
    return parser.parse_args(argv)


def run(cmd_args: list[str]) -> int:
    args = parse_args(cmd_args)
    hist = NtCardHistogram(args.path)

    kmer_stats_rows = [
        ["Number of distinct k-mers", hist.num_distinct],
        ["Total number of k-mers", hist.num_total],
    ]
    thresh_rows = [
        ["Elbow", hist.elbow + 1],
        ["First minima", hist.first_minima + 1],
        ["Mode after first minima", hist.mode_after_first_minima + 1],
        ["Otsu thresholds", ", ".join(map(str, hist.otsu_thresholds + 1))],
    ]
    dataset_rows = [
        ["Dataset size (D)", f"{utils.format_bp(hist.num_total)}"],
    ]

    model = Model()
    if not args.no_model:
        t0 = time.time()
        num_iters, final_error, history = model.fit(hist)
        time_elapsed = time.time() - t0
    print("Histogram shape (y-axis in log scale):")
    output.print_hist(hist.values)

    x_crossover = 0
    table_printer = output.TablePrinter(args.table_format)
    if model.converged:
        w_err, rv_err = model.err_rv
        w_het, rv_het = model.heterozygous_rv
        w_hom, rv_hom = model.homozygous_rv
        model_components_rows = [
            [f"Errors       (w = {w_err:.3f})", utils.scipy_rv_to_string(rv_err)],
            [f"Heterozygous (w = {w_het:.3f})", utils.scipy_rv_to_string(rv_het)],
            [f"Homozygous   (w = {w_hom:.3f})", utils.scipy_rv_to_string(rv_hom)],
        ]
        for i, (w, rv) in enumerate(model.copy_rvs):
            row = [f"Copy {i + 3} (w = {w:.3f})", utils.scipy_rv_to_string(rv)]
            model_components_rows.append(row)
        table_printer.print(
            "Fitted model",
            *model_components_rows,
            ["Number of iterations", num_iters],
            ["Model error", final_error],
            ["Wall clock time", f"{time_elapsed:.3f}s"],
        )
        num_robust = utils.count_robust_kmers(hist, model)
        num_homozygous = utils.count_homozygous_kmers(hist, model)
        num_heterozygous = utils.count_heterozygous_kmers(hist, model)
        kmer_stats_rows.insert(1, ["Number of robust k-mers", num_robust])
        kmer_stats_rows.insert(2, ["Number of heterozygous k-mers", num_heterozygous])
        kmer_stats_rows.insert(3, ["Number of homozygous k-mers", num_homozygous])
        x_crossover = model.get_weak_robust_crossover(np.arange(1, hist.max_count + 1))
        thresh_rows.append(["Weak/robust crossover", x_crossover or "N/A"])
        thresh_rows.append(["Heterozygous peak", np.rint(model.peaks[0]).astype(int)])
        thresh_rows.append(["Homozygous peak", np.rint(model.peaks[1]).astype(int)])
        heterozygosity = num_heterozygous / num_robust
        genome_size = num_robust / model.coverage
        dataset_rows.append(["Est. coverage (C)", f"{model.coverage:.1f}x"])
        dataset_rows.append(["Robust k-mer rate (R)", f"{num_robust / hist.num_total * 100:.2f}%"])
        dataset_rows.append(["Heterozygosity", f"{heterozygosity * 100:.2f}%"])
        dataset_rows.append(["Genome size (RD/C)", utils.format_bp(genome_size)])

    table_printer.print("K-mer statistics", *kmer_stats_rows)
    table_printer.print("Thresholds", *thresh_rows)
    table_printer.print("Data/genome characteristics", *dataset_rows)

    plot_range = args.plot_range or [1, hist.max_count + 1]
    if plot_range[1] == 0 and model.converged:
        last_rv = model.copy_rvs[-1] if len(model.copy_rvs) > 0 else model.homozygous_rv
        plot_range[1] = int(last_rv[1].interval(0.99)[1])
    elif plot_range[1] == 0:
        plot_range[1] = int(hist.mode_after_first_minima * 3)
    plot_range[0] = max(plot_range[0], 0)
    plot_range[1] = min(plot_range[1], hist.max_count + 1)

    if args.plot:
        output.save_plot(
            hist,
            model,
            args.kmer_size,
            args.style,
            args.title,
            dataset_rows,
            plot_range,
            args.y_log,
            args.plot,
        )
        print(f"Saved plot to {args.plot}")
    if args.probs and model.converged:
        output.save_probs(hist, model, args.probs)
        print(f"Saved model probabilities to {args.probs}")
    if args.fit_gif and not args.no_model:
        output.save_fit_animation(
            history,
            hist,
            args.style,
            plot_range,
            args.y_log,
            args.fit_gif,
        )
        print(f"Saved fit history gif to {args.fit_gif}")
    return 0


if __name__ == "__main__":
    exit(run(sys.argv[1:]))
