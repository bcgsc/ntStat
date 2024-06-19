#pragma once

#include <argparse/argparse.hpp>
#include <atomic>
#include <btllib/bloom_filter.hpp>
#include <btllib/counting_bloom_filter.hpp>
#include <btllib/nthash.hpp>
#include <indicators/block_progress_bar.hpp>
#include <omp.h>

#include "utils.hpp"

namespace rarity {

template<class HashFunction, class OutputCountingBloomFilter>
inline void
process_read(size_t read_length,
             HashFunction& hash_fn,
             OutputCountingBloomFilter& counts,
             OutputCountingBloomFilter& depths)
{
  const auto num_elements = read_length - hash_fn.get_k();
  const auto bf_size = utils::get_bf_size(num_elements, 0.00001, hash_fn.get_hash_num());
  btllib::BloomFilter bf(bf_size, hash_fn.get_hash_num());
  while (hash_fn.roll()) {
    counts.insert(hash_fn.hashes());
    if (!bf.contains_insert(hash_fn.hashes())) {
      depths.insert(hash_fn.hashes());
    }
  }
}

inline void
process_kmers(const std::vector<std::string>& read_files,
              bool long_mode,
              unsigned kmer_length,
              size_t num_kmers,
              size_t out_size,
              unsigned num_hashes,
              const std::string& out_prefix)
{
  utils::Timer timer;
  timer.start("allocating output btllib::KmerCountingBloomFilter8s");
  btllib::KmerCountingBloomFilter8 counts(out_size, num_hashes, kmer_length);
  btllib::KmerCountingBloomFilter8 depths(out_size, num_hashes, kmer_length);
  timer.stop();
  const auto seq_reader_flag = utils::get_seq_reader_flag(long_mode);
  std::atomic<size_t> num_kmers_done{ 0 };
  std::atomic<unsigned> percent_done{ 0 };
  indicators::BlockProgressBar bar{
    indicators::option::BarWidth{ 50 },
    indicators::option::ShowElapsedTime{ true },
    indicators::option::ShowRemainingTime{ true },
  };
  for (const auto& file : read_files) {
    btllib::SeqReader seq_reader(file, seq_reader_flag);
#pragma omp parallel shared(seq_reader)
    for (const auto& record : seq_reader) {
      btllib::NtHash hash_fn(record.seq, num_hashes, kmer_length);
      process_read(record.seq.size(), hash_fn, counts, depths);
      num_kmers_done += record.seq.size() - kmer_length + 1;
#pragma omp critical
      if (num_kmers_done * 100 / num_kmers > percent_done) {
        percent_done = num_kmers_done * 100 / num_kmers;
        bar.set_progress(percent_done);
      }
    }
  }
  bar.set_progress(100);
  bar.mark_as_completed();
  timer.start("saving output files");
  counts.save(out_prefix + "counts.cbf");
  depths.save(out_prefix + "depths.cbf");
  timer.stop();
}

inline void
process_seeds(const std::vector<std::string>& read_files,
              bool long_mode,
              const std::vector<std::string>& seeds,
              size_t num_kmers,
              size_t out_size,
              unsigned num_hashes,
              const std::string& out_prefix)
{
  utils::Timer timer;
  timer.start("allocating output btllib::KmerCountingBloomFilter8s");
  btllib::CountingBloomFilter8 counts(out_size, num_hashes);
  btllib::CountingBloomFilter8 depths(out_size, num_hashes);
  timer.stop();
  const auto seq_reader_flag = utils::get_seq_reader_flag(long_mode);
  std::atomic<size_t> num_kmers_done{ 0 };
  std::atomic<unsigned> percent_done{ 0 };
  indicators::BlockProgressBar bar{
    indicators::option::BarWidth{ 50 },
    indicators::option::ShowElapsedTime{ true },
    indicators::option::ShowRemainingTime{ true },
  };
  for (const auto& file : read_files) {
    btllib::SeqReader seq_reader(file, seq_reader_flag);
#pragma omp parallel shared(seq_reader)
    for (const auto& record : seq_reader) {
      for (const auto& seed : seeds) {
        btllib::SeedNtHash hash_fn(record.seq, { seed }, num_hashes, seed.size());
        process_read(record.seq.size(), hash_fn, counts, depths);
        num_kmers_done += record.seq.size() - seed.size() + 1;
#pragma omp critical
        if (num_kmers_done * 100 / num_kmers > percent_done) {
          percent_done = num_kmers_done * 100 / num_kmers;
          bar.set_progress(percent_done);
        }
      }
    }
  }
  bar.set_progress(100);
  bar.mark_as_completed();
  timer.start("saving output files");
  counts.save(out_prefix + "counts.cbf");
  depths.save(out_prefix + "depths.cbf");
  timer.stop();
}

argparse::ArgumentParser*
get_argument_parser()
{
  argparse::ArgumentParser* parser = new argparse::ArgumentParser("rarity", "");
  parser->add_argument("-k").help("k-mer length").scan<'u', unsigned>();
  parser->add_argument("-s").help("path to spaced seeds file (one per line, if -k not specified)");
  parser->add_argument("-f").help("path to k-mer spectrum file (from ntCard)").required();
  parser->add_argument("-e")
    .help("target output false positive rate")
    .default_value(0.0001F)
    .scan<'g', float>();
  parser->add_argument("-b").help("output CBF size (bytes)").scan<'u', size_t>();
  parser->add_argument("--long")
    .help("optimize for long read data")
    .default_value(false)
    .implicit_value(true);
  parser->add_argument("-t").help("number of threads").default_value(1U).scan<'u', unsigned>();
  parser->add_argument("-o").help("prefix for storing output counting bloom filters").required();
  parser->add_argument("reads").help("path to sequencing data file(s)").required().remaining();
  return parser;
}

inline int
main(const argparse::ArgumentParser& args)
{
  unsigned kmer_length;
  std::vector<std::string> seeds;
  std::string element_name;
  std::cout << "parsed arguments:" << std::endl;
  if (args.is_used("-s")) {
    element_name = "spaced seed";
    const auto seeds_path = args.get("-s");
    seeds = utils::read_file_lines(seeds_path);
    std::cout << "[-s] counting spaced seeds" << std::endl;
    utils::print_seeds_list(seeds);
    kmer_length = seeds[0].size();
  } else if (args.is_used("-k")) {
    element_name = "k-mer";
    kmer_length = args.get<unsigned>("-k");
    std::cout << "[-k] counting " << kmer_length << "-mers" << std::endl;
  }

  const auto num_threads = args.get<unsigned>("-t");
  omp_set_num_threads(num_threads);
  std::cout << "[-t] thread limit set to " << num_threads << std::endl;

  const auto long_mode = args.get<bool>("--long");
  std::cout << "[--long] using " << (long_mode ? "long" : "short") << " read data" << std::endl;

  const auto target_fpr = args.get<float>("-e");
  std::cout << "[-e] target output false-positive rate: " << target_fpr << std::endl;

  const auto histogram_path = args.get("-f");
  const auto histogram = utils::read_ntcard_histogram(histogram_path);
  const auto num_elements = histogram[1] * std::max(1UL, seeds.size());

  unsigned num_hashes;
  size_t out_size;
  if (args.is_used("-b")) {
    num_hashes = utils::get_num_hashes(num_elements, out_size);
    out_size = args.get<size_t>("-b");
  } else {
    num_hashes = 3;
    out_size = utils::get_bf_size(num_elements, target_fpr, num_hashes);
  }
  std::cout << "number of hashes per " << element_name << ": " << num_hashes << std::endl;
  std::cout << "output bloom filter size (each): " << out_size << std::endl;

  const auto read_files = args.get<std::vector<std::string>>("reads");
  const auto out_path = args.get("-o");

  if (args.is_used("-k")) {
    process_kmers(args.get<std::vector<std::string>>("reads"),
                  long_mode,
                  kmer_length,
                  histogram[0],
                  out_size,
                  num_hashes,
                  args.get("-o"));
  } else if (args.is_used("-s")) {
    process_seeds(args.get<std::vector<std::string>>("reads"),
                  long_mode,
                  seeds,
                  histogram[0],
                  out_size,
                  num_hashes,
                  args.get("-o"));
  }
  return EXIT_SUCCESS;
}

}