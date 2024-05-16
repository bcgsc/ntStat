#pragma once

#include <argparse/argparse.hpp>
#include <btllib/counting_bloom_filter.hpp>
#include <btllib/nthash.hpp>
#include <btllib/seq_reader.hpp>
#include <cstdint>
#include <omp.h>

#include "utils.hpp"

namespace filter {

argparse::ArgumentParser*
get_argument_parser()
{
  const auto no_args = argparse::default_arguments::none;
  argparse::ArgumentParser* parser = new argparse::ArgumentParser("filter", "", no_args);
  parser->add_argument("-k").help("k-mer length").scan<'u', unsigned>();
  parser->add_argument("-s").help("path to spaced seeds file (one per line)");
  parser->add_argument("-t").help("number of threads").default_value(1U).scan<'u', unsigned>();
  parser->add_argument("-b").help("output BF/CBF size (bytes)").scan<'u', size_t>().required();
  parser->add_argument("-cmin")
    .help("minimum count threshold (>=1)")
    .default_value(1U)
    .scan<'u', unsigned>();
  parser->add_argument("-cmax")
    .help("maximum count threshold (<=255)")
    .default_value(255U)
    .scan<'u', unsigned>();
  parser->add_argument("-h", "--num-hashes")
    .help("Number of hashes to generate per k-mer/spaced seed")
    .default_value(3U)
    .scan<'u', unsigned>();
  parser->add_argument("--long")
    .help("optimize for long read data")
    .default_value(false)
    .implicit_value(true);
  parser->add_argument("-o").help("path to store output file").required();
  parser->add_argument("reads").help("path to sequencing data file(s)").required().remaining();
  return parser;
}

bool
validate(unsigned cmin, unsigned cmax)
{
  if (cmin == 1 && cmax == 255) {
    std::cerr << "need to specify -cmin (>1) or cmax (<255)" << std::endl;
    std::cerr << "use 'ntstat count' if not applying any filters" << std::endl;
    return false;
  }
  if (cmin == 0) {
    std::cerr << "-cmin=0 is invalid" << std::endl;
    return false;
  }
  if (cmax > 255) {
    std::cerr << "-cmax=" << cmax << "is invalid" << std::endl;
    return false;
  }
  return true;
}

int
main(const argparse::ArgumentParser& args)
{
  const auto num_threads = args.get<unsigned>("-t");
  const auto read_files = args.get<std::vector<std::string>>("reads");
  const auto out_cbf_size = args.get<size_t>("-b");
  const auto out_path = args.get("-o");
  const auto num_hashes = args.get<unsigned>("-h");
  const auto long_mode = args.get<bool>("--long");
  const auto cmin = args.get<unsigned>("-cmin");
  const auto cmax = args.get<unsigned>("-cmax");
  if (!validate(cmin, cmax)) {
    return EXIT_FAILURE;
  }
  omp_set_num_threads(num_threads);
  std::cout << "[-t] thread limit set to " << num_threads << std::endl;
  std::cout << "[--long] " << (long_mode ? "long" : "short") << " read data" << std::endl;
  if (args.is_used("-k")) {
    const auto kmer_length = args.get<unsigned>("-k");
  } else if (args.is_used("-s")) {
    const auto seeds_path = args.get<std::string>("-s");
    const auto seeds = read_file_lines(seeds_path);
  } else {
    std::cerr << "need to specify at least one of -k or -s" << std::endl;
    std::cerr << args;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}
