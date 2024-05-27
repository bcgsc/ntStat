#pragma once

#include <argparse/argparse.hpp>
#include <btllib/counting_bloom_filter.hpp>
#include <btllib/nthash.hpp>
#include <btllib/seq_reader.hpp>
#include <cstdint>
#include <filesystem>
#include <math.h>
#include <omp.h>
#include <tabulate/table.hpp>

#include "utils.hpp"

namespace filter {

argparse::ArgumentParser*
get_argument_parser()
{
  argparse::ArgumentParser* parser = new argparse::ArgumentParser("filter", "");
  parser->add_argument("-k").help("k-mer length").scan<'u', unsigned>();
  parser->add_argument("-s").help("path to spaced seeds file (one per line, if -k not specified)");
  parser->add_argument("-f").help("path to k-mer spectrum file (from ntCard)").required();
  parser->add_argument("-e")
    .help("target output false positive rate")
    .default_value(0.0001F)
    .scan<'g', float>();
  parser->add_argument("-b").help("output BF/CBF size (bytes)").scan<'u', size_t>();
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
  parser->add_argument("--long")
    .help("optimize for long read data")
    .default_value(false)
    .implicit_value(true);
  parser->add_argument("-t").help("number of threads").default_value(1U).scan<'u', unsigned>();
  parser->add_argument("-o").help("path to store output file").required();
  parser->add_argument("reads").help("path to sequencing data file(s)").required().remaining();
  return parser;
}

inline bool
validate_thresholds(unsigned cmin, unsigned cmax)
{
  if (cmin == 0) {
    std::cerr << "cmin should be greater than 0" << std::endl;
    return false;
  }
  if (cmax > 255) {
    std::cerr << "cmax should be less than 256" << std::endl;
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
inline void
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
inline void
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

/* if cmin == 1 and cmax == 255 */
template<class HashFunction, class OutputBloomFilter>
inline void
process_read(HashFunction& hash_fn, OutputBloomFilter& out)
{
  while (hash_fn.roll()) {
    out.insert(hash_fn.hashes());
  }
}

/* if cmin == 2 and cmax == 255 */
template<class HashFunction, class OutputBloomFilter>
inline void
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
inline void
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
  std::cout << "counting and filtering k-mers:" << std::endl;
  utils::Timer timer;
  const auto seq_reader_flag = utils::get_seq_reader_flag(long_mode);
  for (size_t i = 0; i < read_files.size(); i++) {
    std::string pfx = " [" + std::to_string(i + 1) + "/" + std::to_string(read_files.size()) + "] ";
    timer.start(pfx + "processing " + read_files[i]);
    btllib::SeqReader seq_reader(read_files[i], seq_reader_flag);
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
  std::cout << "counting and filtering spaced seeds:" << std::endl;
  utils::Timer timer;
  const auto seq_reader_flag = utils::get_seq_reader_flag(long_mode);
  for (size_t i = 0; i < read_files.size(); i++) {
    std::string pfx = " [" + std::to_string(i + 1) + "/" + std::to_string(read_files.size()) + "] ";
    timer.start(pfx + "processing " + read_files[i]);
    btllib::SeqReader seq_reader(read_files[i], seq_reader_flag);
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
inline void
process(const std::vector<std::string>& read_files,
        bool long_mode,
        unsigned kmer_length,
        const std::vector<std::string>& seeds,
        unsigned cmin,
        unsigned cmax,
        size_t bf_size,
        size_t cbf_size,
        OutputBloomFilter& out)
{
  utils::Timer timer;
  const bool using_kmers = seeds.size() == 0;
  const auto bf_size_str = utils::human_readable(bf_size);
  const auto cbf_size_str = utils::human_readable(cbf_size);
  if (cmin > 2 && cmax < 255) {
    timer.start(" allocating distincts bloom filter (" + bf_size_str + ")");
    btllib::BloomFilter bf(bf_size, out.get_hash_num());
    timer.stop();
    timer.start(" allocating intermediate counting bloom filter (" + cbf_size_str + ")");
    btllib::CountingBloomFilter8 cbf(cbf_size, out.get_hash_num());
    timer.stop();
    std::cout << std::endl;
    if (using_kmers) {
      process_kmers(read_files, long_mode, kmer_length, out, cmin, cmax, bf, cbf);
    } else {
      process_seeds(read_files, long_mode, seeds, out, cmin, cmax, bf, cbf);
    }
  } else if (cmin == 1 && cmax < 255) {
    timer.start(" allocating intermediate counting bloom filter (" + cbf_size_str + ")");
    btllib::CountingBloomFilter8 cbf(cbf_size, out.get_hash_num());
    timer.stop();
    std::cout << std::endl;
    if (using_kmers) {
      process_kmers(read_files, long_mode, kmer_length, out, cmax, cbf);
    } else {
      process_seeds(read_files, long_mode, seeds, out, cmax, cbf);
    }
  } else if (cmin == 1 && cmax == 255) {
    std::cout << std::endl;
    if (using_kmers) {
      process_kmers(read_files, long_mode, kmer_length, out);
    } else {
      process_seeds(read_files, long_mode, seeds, out);
    }
  } else if (cmin == 2 && cmax == 255) {
    timer.start(" allocating distincts bloom filter (" + bf_size_str + ")");
    btllib::BloomFilter bf(bf_size, out.get_hash_num());
    timer.stop();
    std::cout << std::endl;
    if (using_kmers) {
      process_kmers(read_files, long_mode, kmer_length, out, bf);
    } else {
      process_seeds(read_files, long_mode, seeds, out, bf);
    }
  } else if (cmin > 2 and cmax == 255) {
    timer.start(" allocating distincts bloom filter (" + bf_size_str + ")");
    btllib::BloomFilter bf(bf_size, out.get_hash_num());
    timer.stop();
    timer.start(" allocating intermediate counting bloom filter (" + cbf_size_str + ")");
    btllib::CountingBloomFilter8 cbf(cbf_size, out.get_hash_num());
    timer.stop();
    std::cout << std::endl;
    if (using_kmers) {
      process_kmers(read_files, long_mode, kmer_length, out, cmin, bf, cbf);
    } else {
      process_seeds(read_files, long_mode, seeds, out, cmin, bf, cbf);
    }
  }
  std::cout << std::endl;
}

template<class BloomFilter>
inline void
save(BloomFilter& bf, const std::string& path)
{
  std::cout << "actual output false positive rate: " << bf.get_fpr() << std::endl;
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

  double get_fpr() { return out_include.get_fpr(); }

private:
  BloomFilter& out_include;
  BloomFilter& out_exclude;
};

inline unsigned
get_num_hashes(double num_elements, double bf_size)
{
  return num_elements * log(2) / bf_size;
}

inline size_t
get_bf_size(double num_elements, double fpr, double num_hashes)
{
  double r = -num_hashes / log(1.0 - exp(log(fpr) / num_hashes));
  return ceil(num_elements * r);
}

inline uint64_t
get_num_elements(unsigned cmin, const std::vector<uint64_t> histogram, size_t num_seeds)
{
  uint64_t num_elements = histogram[1];
  for (unsigned i = 2; i < cmin + 1; i++) {
    num_elements -= histogram[i];
  }
  return num_elements * std::max(1UL, num_seeds);
}

inline void
get_intermediate_bf_sizes(unsigned cmin,
                          unsigned cmax,
                          const std::vector<u_int64_t>& histogram,
                          double out_fpr,
                          unsigned num_hashes,
                          size_t& bf_size,
                          size_t& cbf_size)
{
  double fpr = sqrt(out_fpr);
  bf_size = cmin > 1 ? get_bf_size(histogram[1], fpr, num_hashes) / 8 : 0;
  if (cmin == 1 && cmax < 255) {
    cbf_size = get_bf_size(histogram[1], fpr, num_hashes);
  } else if (cmin > 2 || cmax < 255) {
    cbf_size = get_bf_size(histogram[1] - histogram[2], fpr, num_hashes);
  } else {
    cbf_size = 0;
  }
}

inline double
get_cascade_fpr(double target_fpr)
{
  return 1 - std::cbrt(1 - target_fpr);
}

void
print_stats_table(uint64_t total,
                  uint64_t distinct,
                  uint64_t filtered,
                  const std::string& element_name)
{
  tabulate::Table table;
  std::cout << std::endl << "overall statistics:" << std::endl;
  table.add_row({ "total number of " + element_name + "s", utils::comma_sep(total) });
  table.add_row({ "number of distinct " + element_name + "s", utils::comma_sep(distinct) });
  table.add_row({ "number of unique filtered " + element_name + "s", utils::comma_sep(filtered) });
  table.format().border_top("").border_bottom("").border_left("").border_right("").corner("");
  table.column(1).format().font_align(tabulate::FontAlign::right);
  std::cout << table << std::endl << std::endl;
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

  const auto cmin = args.get<unsigned>("-cmin");
  std::cout << "[-cmin] minimum " << element_name << " count: " << cmin << std::endl;
  const auto cmax = args.get<unsigned>("-cmax");
  std::cout << "[-cmax] maximum " << element_name << " count: " << cmax << std::endl;
  const auto counts = args.get<bool>("--counts");

  const auto num_threads = args.get<unsigned>("-t");
  omp_set_num_threads(num_threads);
  std::cout << "[-t] thread limit set to " << num_threads << std::endl;

  const auto long_mode = args.get<bool>("--long");
  std::cout << "[--long] using " << (long_mode ? "long" : "short") << " read data" << std::endl;

  if (!validate_thresholds(cmin, cmax)) {
    std::cout << args;
    return EXIT_FAILURE;
  }

  const auto target_fpr = args.get<float>("-e");
  std::cout << "[-e] target output false-positive rate: " << target_fpr << std::endl;

  const auto histogram_path = args.get("-f");
  const auto histogram = utils::read_ntcard_histogram(histogram_path);
  const auto num_elements = get_num_elements(cmin, histogram, seeds.size());
  print_stats_table(histogram[0], histogram[1], num_elements, element_name);

  size_t out_size, excludes_size, bf_size, cbf_size;
  const auto cascade_fpr = get_cascade_fpr(target_fpr);
  unsigned num_hashes;
  if (args.is_used("-b")) {
    out_size = args.get<size_t>("-b");
    excludes_size = cmax < 255 && !counts ? out_size : 0;
    num_hashes = get_num_hashes(num_elements, out_size);
  } else {
    num_hashes = 3;
    out_size = get_bf_size(num_elements, cascade_fpr, num_hashes) / (counts ? 1UL : 8UL);
    const auto num_excludes = histogram[1] - num_elements;
    excludes_size = cmax < 255 && !counts ? get_bf_size(num_excludes, cascade_fpr, num_hashes) : 0;
  }
  get_intermediate_bf_sizes(cmin, cmax, histogram, cascade_fpr, num_hashes, bf_size, cbf_size);

  std::cout << "number of hashes per " << element_name << ": " << num_hashes << std::endl;
  std::cout << "cascade false positive rate: " << cascade_fpr << std::endl << std::endl;

  const auto read_files = args.get<std::vector<std::string>>("reads");
  const auto out_path = args.get("-o");

  utils::Timer timer;

  size_t ram_usage = bf_size + cbf_size + out_size + excludes_size;
  std::cout << "predicted memory usage: " << utils::human_readable(ram_usage) << std::endl;
  std::cout << std::setprecision(3);

  const auto out_size_str = utils::human_readable(out_size);
  const auto excludes_size_str = utils::human_readable(excludes_size);
  if (args.is_used("-k") && counts) {
    timer.start(" allocating output btllib::KmerCountingBloomFilter8 (" + out_size_str + ")");
    btllib::KmerCountingBloomFilter8 out(out_size, num_hashes, kmer_length);
    timer.stop();
    process(read_files, long_mode, kmer_length, seeds, cmin, cmax, bf_size, cbf_size, out);
    save(out, out_path);
  } else if (args.is_used("-k")) {
    timer.start(" allocating output btllib::KmerBloomFilter (" + out_size_str + ")");
    btllib::KmerBloomFilter out_include(out_size, num_hashes, kmer_length);
    timer.stop();
    timer.start(" allocating excludes btllib::KmerBloomFilter (" + excludes_size_str + ")");
    btllib::KmerBloomFilter out_exclude(out_size, num_hashes, kmer_length);
    timer.stop();
    BloomFilterWrapper<btllib::KmerBloomFilter> out(out_include, out_exclude);
    timer.stop();
    process(read_files, long_mode, kmer_length, seeds, cmin, cmax, bf_size, cbf_size, out);
    save(out, out_path);
  } else if (args.is_used("-s") && counts) {
    timer.start(" allocating output btllib::CountingBloomFilter8 (" + out_size_str + ")");
    btllib::CountingBloomFilter8 out(out_size, num_hashes);
    timer.stop();
    process(read_files, long_mode, kmer_length, seeds, cmin, cmax, bf_size, cbf_size, out);
    save(out, out_path);
  } else if (args.is_used("-s")) {
    timer.start(" allocating output btllib::SeedBloomFilter for includes (" + out_size_str + ")");
    btllib::SeedBloomFilter out_include(out_size, kmer_length, seeds, num_hashes);
    timer.stop();
    timer.start(" allocating excludes btllib::SeedBloomFilter (" + excludes_size_str + ")");
    btllib::SeedBloomFilter out_exclude(out_size, kmer_length, seeds, num_hashes);
    timer.stop();
    BloomFilterWrapper<btllib::SeedBloomFilter> out(out_include, out_exclude);
    process(read_files, long_mode, kmer_length, seeds, cmin, cmax, bf_size, cbf_size, out);
    save(out, out_path);
  } else {
    std::cerr << "need to specify at least one of -k or -s" << std::endl;
    std::cerr << args;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}
