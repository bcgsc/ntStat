#pragma once

#include <btllib/bloom_filter.hpp>
#include <btllib/counting_bloom_filter.hpp>
#include <btllib/seq_reader.hpp>
#include <indicators/block_progress_bar.hpp>
#include <omp.h>

#include "utils.hpp"

/* if cmin >= 2 and cmax < 255 */
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
    if (distinct.contains_insert(hash_fn.hashes())) {
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
              size_t num_kmers,
              OutputBloomFilter& out,
              ReadProcessorArgs&... args)
{
  std::atomic<size_t> num_kmers_done{ 0 };
  indicators::BlockProgressBar bar{
    indicators::option::BarWidth{ 30 },
    indicators::option::ShowElapsedTime{ true },
    indicators::option::ShowRemainingTime{ true },
    indicators::option::PrefixText{ "processing " },
    indicators::option::MaxProgress{ num_kmers },
  };
  const size_t progress_interval = num_kmers / 100;
  size_t last_progress_threshold = 0;
  const auto seq_reader_flag = get_seq_reader_flag(long_mode);
  for (size_t i = 0; i < read_files.size(); i++) {
    btllib::SeqReader seq_reader(read_files[i], seq_reader_flag);
#pragma omp parallel shared(seq_reader)
    for (const auto& record : seq_reader) {
      if (record.seq.size() < kmer_length) {
        continue;
      }
      btllib::NtHash hash_fn(record.seq, out.get_hash_num(), kmer_length);
      process_read(hash_fn, out, args...);
      num_kmers_done += record.seq.size() - kmer_length + 1;
      if (num_kmers_done >= (last_progress_threshold + progress_interval)) {
        bar.set_progress(num_kmers_done);
        last_progress_threshold = num_kmers_done;
      }
    }
  }
  bar.set_progress(num_kmers);
  bar.mark_as_completed();
}

template<class OutputBloomFilter, typename... ReadProcessorArgs>
inline void
process_seeds(const std::vector<std::string>& read_files,
              bool long_mode,
              const std::vector<std::string>& seeds,
              size_t num_kmers,
              OutputBloomFilter& out,
              ReadProcessorArgs&... args)
{
  std::atomic<size_t> num_kmers_done{ 0 };
  indicators::BlockProgressBar bar{
    indicators::option::BarWidth{ 30 },
    indicators::option::ShowElapsedTime{ true },
    indicators::option::ShowRemainingTime{ true },
    indicators::option::PrefixText{ "processing " },
    indicators::option::MaxProgress{ num_kmers },
  };
  const size_t progress_interval = num_kmers / 100;
  size_t last_progress_threshold = 0;
  const auto seq_reader_flag = get_seq_reader_flag(long_mode);
  for (size_t i = 0; i < read_files.size(); i++) {
    btllib::SeqReader seq_reader(read_files[i], seq_reader_flag);
#pragma omp parallel shared(seq_reader)
    for (const auto& record : seq_reader) {
      for (const auto& seed : seeds) {
        if (record.seq.size() < seed.size()) {
          continue;
        }
        btllib::SeedNtHash hash_fn(record.seq, { seed }, out.get_hash_num(), seed.size());
        process_read(hash_fn, out, args...);
      }
      num_kmers_done += record.seq.size() - seeds[0].size() + 1;
      if (num_kmers_done >= (last_progress_threshold + progress_interval)) {
        bar.set_progress(num_kmers_done);
        last_progress_threshold = num_kmers_done;
      }
    }
  }
  bar.set_progress(100);
  bar.mark_as_completed();
}

template<class OutputBloomFilter>
inline void
process(const std::vector<std::string>& read_files,
        bool long_mode,
        unsigned kmer_length,
        const std::vector<std::string>& seeds,
        size_t total_num_kmers,
        unsigned cmin,
        unsigned cmax,
        size_t bf_size,
        size_t cbf_size,
        OutputBloomFilter& out)
{
  Timer timer;
  const bool using_kmers = seeds.size() == 0;
  const auto bf_size_str = human_readable(bf_size);
  const auto cbf_size_str = human_readable(cbf_size);
  if (cmin >= 2 && cmax < 255) {
    timer.start(" allocating distincts bloom filter (" + bf_size_str + ")");
    btllib::BloomFilter bf(bf_size, out.get_hash_num());
    timer.stop();
    timer.start(" allocating intermediate counting bloom filter (" + cbf_size_str + ")");
    btllib::CountingBloomFilter8 cbf(cbf_size, out.get_hash_num());
    timer.stop();
    std::cout << std::endl;
    if (using_kmers) {
      process_kmers(read_files, long_mode, kmer_length, total_num_kmers, out, cmin, cmax, bf, cbf);
    } else {
      process_seeds(read_files, long_mode, seeds, total_num_kmers, out, cmin, cmax, bf, cbf);
    }
  } else if (cmin == 1 && cmax < 255) {
    timer.start(" allocating intermediate counting bloom filter (" + cbf_size_str + ")");
    btllib::CountingBloomFilter8 cbf(cbf_size, out.get_hash_num());
    timer.stop();
    std::cout << std::endl;
    if (using_kmers) {
      process_kmers(read_files, long_mode, kmer_length, total_num_kmers, out, cmax, cbf);
    } else {
      process_seeds(read_files, long_mode, seeds, total_num_kmers, out, cmax, cbf);
    }
  } else if (cmin == 1 && cmax == 255) {
    if (using_kmers) {
      process_kmers(read_files, long_mode, kmer_length, total_num_kmers, out);
    } else {
      process_seeds(read_files, long_mode, seeds, total_num_kmers, out);
    }
  } else if (cmin == 2 && cmax == 255) {
    timer.start(" allocating distincts bloom filter (" + bf_size_str + ")");
    btllib::BloomFilter bf(bf_size, out.get_hash_num());
    timer.stop();
    std::cout << std::endl;
    if (using_kmers) {
      process_kmers(read_files, long_mode, kmer_length, total_num_kmers, out, bf);
    } else {
      process_seeds(read_files, long_mode, seeds, total_num_kmers, out, bf);
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
      process_kmers(read_files, long_mode, kmer_length, total_num_kmers, out, cmin, bf, cbf);
    } else {
      process_seeds(read_files, long_mode, seeds, total_num_kmers, out, cmin, bf, cbf);
    }
  }
}
