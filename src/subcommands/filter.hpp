#pragma once

#include <argparse/argparse.hpp>
#include <btllib/counting_bloom_filter.hpp>
#include <btllib/nthash.hpp>
#include <btllib/seq_reader.hpp>
#include <cstdint>
#include <filesystem>
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
  parser->add_argument("--counts")
    .help("output counts (requires ~8x RAM for CBF)")
    .default_value(false)
    .implicit_value(true);
  parser->add_argument("-h", "--num-hashes")
    .help("number of hashes to generate per k-mer/spaced seed")
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
validate_thresholds(unsigned cmin, unsigned cmax)
{
  if (cmin == 1 && cmax == 255) {
    std::cerr << "need to specify cmin>1 or cmax<255" << std::endl;
    std::cerr << "use 'ntstat count' if not applying any filters" << std::endl;
    return false;
  }
  if (cmin == 0) {
    std::cerr << "cmin=0 is invalid" << std::endl;
    return false;
  }
  if (cmax > 255) {
    std::cerr << "cmax=" << cmax << "is invalid" << std::endl;
    return false;
  }
  if (cmin >= cmax) {
    std::cerr << "cmin cannot be greater than or equal to cmax" << std::endl;
    return false;
  }
  return true;
}

/* if cmin > 2 and cmax < 255 */
template<class HashFunction, class OutputBloomFilter>
void
process_read(HashFunction& hash_fn,
             OutputBloomFilter& out,
             unsigned min_count,
             unsigned max_count,
             btllib::BloomFilter& distinct,
             btllib::CountingBloomFilter8& intermediate)
{
  while (hash_fn.roll()) {
    if (!distinct.contains_insert(hash_fn.hashes())) {
      continue;
    }
    const unsigned count = intermediate.insert_contains(hash_fn.hashes()) + 1U;
    if (count == min_count) {
      out.insert(hash_fn.hashes(), min_count);
    } else if (count > min_count && count < max_count) {
      out.insert(hash_fn.hashes());
    } else if (count == max_count) {
      out.clear(hash_fn.hashes());
    }
  }
}

/* if cmin == 1 and cmax < 255 */
template<class HashFunction, class OutputBloomFilter>
void
process_read(HashFunction& hash_fn,
             OutputBloomFilter& out,
             unsigned max_count,
             btllib::CountingBloomFilter8& intermediate)
{
  while (hash_fn.roll()) {
    const unsigned count = intermediate.insert_contains(hash_fn.hashes());
    if (count < max_count) {
      out.insert(hash_fn.hashes());
    } else if (count == max_count) {
      out.clear(hash_fn.hashes());
    }
  }
}

/* if cmin == 2 and cmax == 255 */
template<class HashFunction, class OutputBloomFilter>
void
process_read(HashFunction& hash_fn, OutputBloomFilter& out, btllib::BloomFilter& distinct)
{
  while (hash_fn.roll()) {
    if (!distinct.contains_insert(hash_fn.hashes())) {
      out.insert(hash_fn.hashes());
    }
  }
}

/* if cmin > 2 and cmax == 255 */
template<class HashFunction, class OutputBloomFilter>
void
process_read(HashFunction& hash_fn,
             OutputBloomFilter& out,
             unsigned min_count,
             btllib::BloomFilter& distinct,
             btllib::CountingBloomFilter8& intermediate)
{
  while (hash_fn.roll()) {
    if (!distinct.contains_insert(hash_fn.hashes())) {
      continue;
    }
    const unsigned count = intermediate.insert_contains(hash_fn.hashes()) + 1;
    if (count == min_count) {
      out.insert(hash_fn.hashes(), min_count);
    } else if (count > min_count) {
      out.insert(hash_fn.hashes());
    }
  }
}

template<class OutputBloomFilter, typename... ReadProcessorArgs>
inline void
process_kmers(const std::vector<std::string>& read_files,
              bool long_mode,
              unsigned kmer_length,
              OutputBloomFilter& out,
              ReadProcessorArgs&... args)
{
  utils::Timer timer;
  const auto seq_reader_flag = utils::get_seq_reader_flag(long_mode);
  for (const auto& file : read_files) {
    timer.start("processing " + file);
    btllib::SeqReader seq_reader(file, seq_reader_flag);
#pragma omp parallel shared(seq_reader)
    for (const auto& record : seq_reader) {
      btllib::NtHash hash_fn(record.seq, out.get_hash_num(), kmer_length);
      process_read(hash_fn, out, args...);
    }
    timer.stop();
  }
}

template<class OutputBloomFilter, typename... ReadProcessorArgs>
inline void
process_seeds(const std::vector<std::string>& read_files,
              bool long_mode,
              const std::vector<std::string>& seeds,
              OutputBloomFilter& out,
              ReadProcessorArgs&... args)
{
  utils::Timer timer;
  const auto seq_reader_flag = utils::get_seq_reader_flag(long_mode);
  for (const auto& file : read_files) {
    timer.start("processing " + file);
    btllib::SeqReader seq_reader(file, seq_reader_flag);
#pragma omp parallel shared(seq_reader)
    for (const auto& record : seq_reader) {
      for (const auto& seed : seeds) {
        btllib::SeedNtHash hash_fn(record.seq, { seed }, out.get_hash_num(), seed.size());
        process_read(hash_fn, out, args...);
      }
    }
    timer.stop();
  }
}

template<class OutputBloomFilter>
void
process(const std::vector<std::string>& read_files,
        bool long_mode,
        unsigned kmer_length,
        const std::vector<std::string>& seeds,
        unsigned cmin,
        unsigned cmax,
        OutputBloomFilter& out)
{
  const bool using_kmers = seeds.size() == 0;
  if (cmin > 2 && cmax < 255) {
    btllib::BloomFilter bf(1024, out.get_hash_num());
    btllib::CountingBloomFilter8 cbf(1024, out.get_hash_num());
    if (using_kmers) {
      process_kmers(read_files, long_mode, kmer_length, out, cmin, cmax, bf, cbf);
    } else {
      process_seeds(read_files, long_mode, seeds, out, cmin, cmax, bf, cbf);
    }
  } else if (cmin == 1 && cmax < 255) {
    btllib::CountingBloomFilter8 cbf(1024, out.get_hash_num());
    if (using_kmers) {
      process_kmers(read_files, long_mode, kmer_length, out, cmax, cbf);
    } else {
      process_seeds(read_files, long_mode, seeds, out, cmax, cbf);
    }
  } else if (cmin == 2 && cmax == 255) {
    btllib::BloomFilter bf(1024, out.get_hash_num());
    if (using_kmers) {
      process_kmers(read_files, long_mode, kmer_length, out, bf);
    } else {
      process_seeds(read_files, long_mode, seeds, out, bf);
    }
  } else if (cmin > 2 and cmax == 255) {
    btllib::BloomFilter bf(1024, out.get_hash_num());
    btllib::CountingBloomFilter8 cbf(1024, out.get_hash_num());
    if (using_kmers) {
      process_kmers(read_files, long_mode, kmer_length, out, cmin, bf, cbf);
    } else {
      process_seeds(read_files, long_mode, seeds, out, cmin, bf, cbf);
    }
  }
}

template<class BloomFilter>
void
save(BloomFilter& bf, const std::string& path)
{
  utils::Timer timer;
  timer.start("[-o] saving to " + path);
  bf.save(path);
  timer.stop();
}

template<class BloomFilter>
class BloomFilterWrapper
{
public:
  BloomFilterWrapper(BloomFilter& out_include, BloomFilter& out_exclude)
    : out_include(out_include)
    , out_exclude(out_exclude)
  {
  }

  void insert(const uint64_t* hashes, uint8_t = 1) { out_include.insert(hashes); }

  void clear(const uint64_t* hashes) { out_exclude.insert(hashes); }

  void save(const std::string& path)
  {
    const std::string filename = std::filesystem::path(path).stem();
    const std::string extension = std::filesystem::path(path).extension();
    out_include.save(filename + "_include" + extension);
    out_exclude.save(filename + "_exclude" + extension);
  }

  unsigned get_hash_num() { return out_include.get_hash_num(); }

private:
  BloomFilter& out_include;
  BloomFilter& out_exclude;
};

int
main(const argparse::ArgumentParser& args)
{
  const auto num_threads = args.get<unsigned>("-t");
  const auto read_files = args.get<std::vector<std::string>>("reads");
  const auto out_size = args.get<size_t>("-b");
  const auto out_path = args.get("-o");
  const auto num_hashes = args.get<unsigned>("-h");
  const auto long_mode = args.get<bool>("--long");
  const auto cmin = args.get<unsigned>("-cmin");
  const auto cmax = args.get<unsigned>("-cmax");
  const auto counts = args.get<bool>("--counts");

  unsigned kmer_length;
  std::vector<std::string> seeds;
  if (args.is_used("-s")) {
    const auto seeds_path = args.get("-s");
    seeds = utils::read_file_lines(seeds_path);
    std::cout << "[-s] counting spaced seeds" << std::endl;
    utils::print_seeds_list(seeds);
    kmer_length = seeds[0].size();
  } else if (args.is_used("-k")) {
    kmer_length = args.get<unsigned>("-k");
    std::cout << "[-k] counting " << kmer_length << "-mers" << std::endl;
  }

  if (!validate_thresholds(cmin, cmax)) {
    std::cout << args;
    return EXIT_FAILURE;
  }

  omp_set_num_threads(num_threads);
  std::cout << "[-t] thread limit set to " << num_threads << std::endl;

  std::cout << "[--long] " << (long_mode ? "long" : "short") << " read data" << std::endl;

  utils::Timer timer;

  if (args.is_used("-k") && counts) {
    timer.start("initializing output btllib::KmerCountingBloomFilter8");
    btllib::KmerCountingBloomFilter8 out(out_size, num_hashes, kmer_length);
    timer.stop();
    process(read_files, long_mode, kmer_length, seeds, cmin, cmax, out);
    save(out, out_path);
  } else if (args.is_used("-k")) {
    timer.start("initializing output btllib::KmerBloomFilter");
    btllib::KmerBloomFilter out_include(out_size, num_hashes, kmer_length);
    btllib::KmerBloomFilter out_exclude(out_size, num_hashes, kmer_length);
    BloomFilterWrapper<btllib::KmerBloomFilter> out(out_include, out_exclude);
    timer.stop();
    process(read_files, long_mode, kmer_length, seeds, cmin, cmax, out);
    save(out, out_path);
  } else if (args.is_used("-s") && counts) {
    timer.start("initializing output btllib::CountingBloomFilter8");
    btllib::CountingBloomFilter8 out(out_size, num_hashes);
    timer.stop();
    process(read_files, long_mode, kmer_length, seeds, cmin, cmax, out);
    save(out, out_path);
  } else if (args.is_used("-s")) {
    timer.start("initializing output btllib::SeedBloomFilter");
    btllib::SeedBloomFilter out_include(out_size, kmer_length, seeds, num_hashes);
    btllib::SeedBloomFilter out_exclude(out_size, kmer_length, seeds, num_hashes);
    BloomFilterWrapper<btllib::SeedBloomFilter> out(out_include, out_exclude);
    timer.stop();
    process(read_files, long_mode, kmer_length, seeds, cmin, cmax, out);
    save(out, out_path);
  }

  std::cerr << "need to specify at least one of -k or -s" << std::endl;
  std::cerr << args;
  return EXIT_FAILURE;
}

}
