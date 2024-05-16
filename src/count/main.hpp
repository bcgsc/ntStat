#pragma once

#include <argparse/argparse.hpp>
#include <btllib/counting_bloom_filter.hpp>
#include <btllib/nthash.hpp>
#include <btllib/seq_reader.hpp>
#include <cstdint>
#include <omp.h>

#include "kmers.hpp"
#include "seeds.hpp"
#include "utils.hpp"

namespace count {

argparse::ArgumentParser*
get_argument_parser()
{
  const auto no_args = argparse::default_arguments::none;
  argparse::ArgumentParser* parser = new argparse::ArgumentParser("count", "", no_args);
  parser->add_argument("-k").help("k-mer length").scan<'u', unsigned>();
  parser->add_argument("-s").help("path to spaced seeds file (one per line)");
  parser->add_argument("-t").help("number of threads").default_value(1U).scan<'u', unsigned>();
  parser->add_argument("-b").help("output CBF size (bytes)").scan<'u', size_t>().required();
  parser->add_argument("-h", "--num-hashes")
    .help("Number of hashes to generate per k-mer/spaced seed")
    .default_value(3U)
    .scan<'u', unsigned>();
  parser->add_argument("--long")
    .help("optimize for long read data")
    .default_value(false)
    .implicit_value(true);
  parser->add_argument("-o").help("path to store output file").required();
  parser->add_argument("reads").help("path to sequence data file(s)").required().remaining();
  return parser;
}

void
main(const argparse::ArgumentParser& args)
{
  const auto num_threads = args.get<unsigned>("-t");
  const auto read_files = args.get<std::vector<std::string>>("reads");
  const auto out_cbf_size = args.get<size_t>("-b");
  const auto out_path = args.get("-o");
  const auto num_hashes = args.get<unsigned>("-h");
  const auto long_mode = args.get<bool>("--long");
  omp_set_num_threads(num_threads);
  std::cout << "[-t] thread limit set to " << num_threads << std::endl;
  std::cout << "[--long] " << (long_mode ? "long" : "short") << " read data" << std::endl;
  if (args.is_used("-k")) {
    const auto kmer_length = args.get<unsigned>("-k");
    all_kmer_counts(read_files, long_mode, kmer_length, num_hashes, out_cbf_size, out_path);
  } else if (args.is_used("-s")) {
    const auto seeds_path = args.get<std::string>("-s");
    const auto seeds = read_file_lines(seeds_path);
    all_seed_counts(read_files, long_mode, seeds, num_hashes, out_cbf_size, out_path);
  } else {
    std::cerr << "Need to specify at least one of -k or -s" << std::endl;
    std::cerr << args;
    std::exit(EXIT_FAILURE);
  }
}

}
