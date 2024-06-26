#include <btllib/bloom_filter.hpp>
#include <btllib/counting_bloom_filter.hpp>
#include <fstream>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <signal.h>

#include "args.hpp"
#include "bf_utils.hpp"
#include "rarity.hpp"
#include "utils.hpp"

int
run(std::vector<std::string> argv)
{
  std::unique_ptr<ProgramArguments> args;
  try {
    args = std::make_unique<ProgramArguments>(argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << args.get()->get_help_message();
    return EXIT_FAILURE;
  }
  std::cout << args.get()->get_arg_summary();
  omp_set_num_threads(args.get()->num_threads);

  const auto num_elements = args.get()->histogram[1] * std::max(1UL, args.get()->seeds.size());

  size_t out_size;
  unsigned num_hashes;
  bool out_size_known = args.get()->out_size != 0;
  const auto cascade_fpr = 1 - std::cbrt(1 - args.get()->target_fpr);
  if (out_size_known) {
    out_size = args.get()->out_size;
    num_hashes = num_elements * log(2) / out_size;
  } else {
    num_hashes = 3;
    out_size = get_bf_size(num_elements, cascade_fpr, num_hashes);
  }

  std::cout << "number of hash functions: " << num_hashes << std::endl;
  std::cout << "output bloom filter size (each): " << human_readable(out_size) << std::endl;
  std::cout << "predicted memory usage: " << human_readable(2 * out_size) << std::endl;
  std::cout << std::setprecision(3);

  if (args.get()->seeds.empty()) {
    process_kmers(args.get()->reads_paths,
                  args.get()->long_mode,
                  args.get()->kmer_length,
                  args.get()->histogram[0],
                  out_size,
                  num_hashes,
                  args.get()->out_path);
  } else if (!args.get()->seeds.empty()) {
    process_seeds(args.get()->reads_paths,
                  args.get()->long_mode,
                  args.get()->seeds,
                  args.get()->histogram[0] * args.get()->seeds.size(),
                  out_size,
                  num_hashes,
                  args.get()->out_path);
  }
  return EXIT_SUCCESS;
}

PYBIND11_MODULE(rarity, m)
{
  m.def("run", &run);
}