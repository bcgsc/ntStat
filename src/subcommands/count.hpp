#pragma once

#include <argparse/argparse.hpp>
#include <btllib/counting_bloom_filter.hpp>
#include <btllib/nthash.hpp>
#include <btllib/seq_reader.hpp>
#include <cstdint>
#include <omp.h>

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
  for (size_t i = 0; i < read_files.size(); i++) {
    std::string fraction_string = std::to_string(i + 1) + "/" + std::to_string(read_files.size());
    timer.start("[" + fraction_string + "] processing " + read_files[i]);
    const auto seq_reader_flag = get_seq_reader_flag(long_mode);
    btllib::SeqReader seq_reader(read_files[i], seq_reader_flag);
#pragma omp parallel shared(seq_reader)
    for (const auto& record : seq_reader) {
      bf.insert(record.seq);
    }
    timer.stop();
  }
}

inline void
print_seeds(const std::vector<std::string>& seeds)
{
  for (size_t i = 0; i < seeds.size(); i++) {
    std::cout << "[-s] seed " << i + 1 << ": " << seeds[i] << std::endl;
  }
}

int
main(const argparse::ArgumentParser& args)
{
  const auto num_threads = args.get<unsigned>("-t");
  const auto out_cbf_size = args.get<size_t>("-b");
  const auto num_hashes = args.get<unsigned>("-h");
  const auto long_mode = args.get<bool>("--long");
  const auto out_path = args.get("-o");
  const auto read_files = args.get<std::vector<std::string>>("reads");
  omp_set_num_threads(num_threads);
  std::cout << "[-t] thread limit set to " << num_threads << std::endl;
  std::cout << "[-h] using " << num_hashes << " hash functions" << std::endl;
  std::cout << "[--long] " << (long_mode ? "long" : "short") << " read data" << std::endl;
  Timer timer;
  if (args.is_used("-k")) {
    const auto kmer_length = args.get<unsigned>("-k");
    std::cout << "[-k] counting all " << kmer_length << "-mers" << std::endl;
    timer.start("[-b] initializing CBF (" + std::to_string(out_cbf_size) + " bytes)");
    btllib::KmerCountingBloomFilter8 cbf(out_cbf_size, num_hashes, kmer_length);
    timer.stop();
    insert_all<btllib::KmerCountingBloomFilter8>(read_files, cbf, long_mode);
    timer.start("[-o] saving to " + out_path);
    cbf.save(out_path);
    timer.stop();
  } else if (args.is_used("-s")) {
    const auto seeds_path = args.get<std::string>("-s");
    const auto seeds = read_file_lines(seeds_path);
    print_seeds(seeds);
    timer.start("[-b] initializing CBF (" + std::to_string(out_cbf_size) + " bytes)");
    SeedCountingBloomFilter cbf(out_cbf_size, num_hashes, seeds);
    timer.stop();
    insert_all<SeedCountingBloomFilter>(read_files, cbf, long_mode);
    timer.start("[-o] saving to " + out_path);
    cbf.save(out_path);
    timer.stop();
  } else {
    std::cerr << "need to specify at least one of -k or -s" << std::endl;
    std::cerr << args;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}
