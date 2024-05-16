#pragma once

#include <argparse/argparse.hpp>
#include <btllib/counting_bloom_filter.hpp>
#include <btllib/nthash.hpp>
#include <btllib/seq_reader.hpp>
#include <cstdint>
#include <omp.h>

#include "timer.hpp"
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
  parser->add_argument("reads").help("path to sequencing data file(s)").required().remaining();
  return parser;
}

class SeedCountingBloomFilter
{
public:
  SeedCountingBloomFilter(size_t bytes, unsigned num_hashes, const std::vector<std::string>& seeds)
    : cbf(std::make_unique<btllib::CountingBloomFilter8>(bytes, num_hashes))
    , seeds(seeds)
  {
  }

  void insert(const std::string& seq)
  {
    for (const auto& seed : seeds) {
      btllib::SeedNtHash hash_fn(seq, { seed }, cbf.get()->get_hash_num(), seed.size());
      while (hash_fn.roll()) {
        cbf.get()->insert(hash_fn.hashes());
      }
    }
  }

  void save(const std::string& path) { cbf.get()->save(path); }

private:
  std::unique_ptr<btllib::CountingBloomFilter8> cbf;
  const std::vector<std::string>& seeds;
};

template<class BloomFilter>
inline void
insert_all(const std::vector<std::string>& read_files, BloomFilter& bf, bool long_mode)
{
  Timer timer;
  for (const auto& file : read_files) {
    std::cout << "processing " << file << "... " << std::flush;
    timer.start();
    const auto seq_reader_flag = get_seq_reader_flag(long_mode);
    btllib::SeqReader seq_reader(file, seq_reader_flag);
#pragma omp parallel shared(seq_reader)
    for (const auto& record : seq_reader) {
      bf.insert(record.seq);
    }
    timer.stop();
    std::cout << "done (" << timer.to_string() << ")" << std::endl;
  }
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
  omp_set_num_threads(num_threads);
  std::cout << "[-t] thread limit set to " << num_threads << std::endl;
  std::cout << "[--long] " << (long_mode ? "long" : "short") << " read data" << std::endl;
  if (args.is_used("-k")) {
    const auto kmer_length = args.get<unsigned>("-k");
    btllib::KmerCountingBloomFilter8 cbf(out_cbf_size, num_hashes, kmer_length);
    insert_all<btllib::KmerCountingBloomFilter8>(read_files, cbf, long_mode);
    cbf.save(out_path);
  } else if (args.is_used("-s")) {
    const auto seeds_path = args.get<std::string>("-s");
    const auto seeds = read_file_lines(seeds_path);
    SeedCountingBloomFilter cbf(out_cbf_size, num_hashes, seeds);
    insert_all<SeedCountingBloomFilter>(read_files, cbf, long_mode);
    cbf.save(out_path);
  } else {
    std::cerr << "Need to specify at least one of -k or -s" << std::endl;
    std::cerr << args;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}
