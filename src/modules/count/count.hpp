#pragma once

#include <argparse/argparse.hpp>
#include <atomic>
#include <btllib/bloom_filter.hpp>
#include <btllib/counting_bloom_filter.hpp>
#include <btllib/nthash.hpp>
#include <indicators/block_progress_bar.hpp>
#include <omp.h>

#include "bf_utils.hpp"
#include "utils.hpp"

template<class HashFunction, class OutputCountingBloomFilter>
inline void
process_read(size_t read_length,
             HashFunction& hash_fn,
             OutputCountingBloomFilter& counts,
             OutputCountingBloomFilter& depths)
{
  const auto num_elements = read_length - hash_fn.get_k();
  const auto bf_size = get_bf_size(num_elements, 0.00001, hash_fn.get_hash_num());
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
              size_t total_elements,
              size_t out_size,
              unsigned num_hashes,
              const std::string& out_prefix)
{
  Timer timer;
  timer.start("allocating output btllib::KmerCountingBloomFilter8s");
  btllib::KmerCountingBloomFilter8 counts(out_size, num_hashes, kmer_length);
  btllib::KmerCountingBloomFilter8 depths(out_size, num_hashes, kmer_length);
  timer.stop();
  const auto seq_reader_flag = get_seq_reader_flag(long_mode);
  std::atomic<size_t> num_kmers_done{ 0 };
  std::atomic<size_t> num_reads{ 0 };
  indicators::BlockProgressBar bar{
    indicators::option::BarWidth{ 30 },
    indicators::option::ShowElapsedTime{ true },
    indicators::option::ShowRemainingTime{ true },
    indicators::option::PrefixText{ "processing " },
    indicators::option::MaxProgress{ total_elements },
  };
  const size_t progress_interval = total_elements / 100;
  size_t last_progress_threshold = 0;
  for (const auto& file : read_files) {
    btllib::SeqReader seq_reader(file, seq_reader_flag);
#pragma omp parallel shared(seq_reader)
    for (const auto& record : seq_reader) {
      ++num_reads;
      if (record.seq.size() < kmer_length) {
        continue;
      }
      btllib::NtHash hash_fn(record.seq, num_hashes, kmer_length);
      process_read(record.seq.size(), hash_fn, counts, depths);
      num_kmers_done += record.seq.size() - kmer_length + 1;
      if (num_kmers_done >= (last_progress_threshold + progress_interval)) {
        bar.set_progress(num_kmers_done);
        last_progress_threshold = num_kmers_done;
      }
    }
  }
  bar.set_progress(100);
  bar.mark_as_completed();
  std::cout << "total number of reads: " << num_reads << std::endl;
  timer.start("saving output files");
  std::locale::global(std::locale::classic());
  counts.save(out_prefix + "counts.cbf");
  depths.save(out_prefix + "depths.cbf");
  timer.stop();
}

inline void
process_seeds(const std::vector<std::string>& read_files,
              bool long_mode,
              const std::vector<std::string>& seeds,
              size_t total_elements,
              size_t out_size,
              unsigned num_hashes,
              const std::string& out_prefix)
{
  Timer timer;
  timer.start("allocating output btllib::KmerCountingBloomFilter8s");
  btllib::CountingBloomFilter8 counts(out_size, num_hashes);
  btllib::CountingBloomFilter8 depths(out_size, num_hashes);
  timer.stop();
  const auto seq_reader_flag = get_seq_reader_flag(long_mode);
  std::atomic<size_t> num_kmers_done{ 0 };
  std::atomic<size_t> num_reads{ 0 };
  indicators::BlockProgressBar bar{
    indicators::option::BarWidth{ 30 },
    indicators::option::ShowElapsedTime{ true },
    indicators::option::ShowRemainingTime{ true },
    indicators::option::PrefixText{ "processing " },
    indicators::option::MaxProgress{ total_elements },
  };
  const size_t progress_interval = total_elements / 100;
  size_t last_progress_threshold = 0;
  for (const auto& file : read_files) {
    btllib::SeqReader seq_reader(file, seq_reader_flag);
#pragma omp parallel shared(seq_reader)
    for (const auto& record : seq_reader) {
      ++num_reads;
      for (const auto& seed : seeds) {
        if (record.seq.size() < seed.size()) {
          continue;
        }
        btllib::SeedNtHash hash_fn(record.seq, { seed }, num_hashes, seed.size());
        process_read(record.seq.size(), hash_fn, counts, depths);
        num_kmers_done += record.seq.size() - seed.size() + 1;
        if (num_kmers_done >= (last_progress_threshold + progress_interval)) {
          bar.set_progress(num_kmers_done);
          last_progress_threshold = num_kmers_done;
        }
      }
    }
  }
  bar.set_progress(100);
  bar.mark_as_completed();
  std::cout << "total number of reads: " << num_reads << std::endl;
  timer.start("saving output files");
  std::locale::global(std::locale::classic());
  counts.save(out_prefix + "counts.cbf");
  depths.save(out_prefix + "depths.cbf");
  timer.stop();
}
