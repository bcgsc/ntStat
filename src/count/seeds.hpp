#pragma once

#include <btllib/counting_bloom_filter.hpp>
#include <btllib/nthash.hpp>
#include <btllib/seq_reader.hpp>
#include <omp.h>
#include <string>
#include <vector>

#include "timer.hpp"
#include "utils.hpp"

namespace count {

void
insert_all_seeds(const std::string& seq,
                 const std::vector<std::string>& seeds,
                 btllib::CountingBloomFilter8& counts)
{
  for (const auto& seed : seeds) {
    btllib::SeedNtHash hash_fn(seq, { seed }, counts.get_hash_num(), seed.size());
    while (hash_fn.roll()) {
      counts.insert(hash_fn.hashes());
    }
  }
}

void
all_seed_counts(const std::vector<std::string>& read_files,
                bool long_read_data,
                const std::vector<std::string>& seeds,
                unsigned num_hashes_per_seed,
                size_t cbf_size,
                const std::string& out_path)
{
  Timer timer;
  for (size_t i = 0; i < seeds.size(); i++) {
    std::cout << "[-s] seed " << i + 1 << ": " << seeds[i] << std::endl;
  }
  std::cout << "[-b] creating counting bloom filter (" << cbf_size << " bytes)... " << std::flush;
  timer.start();
  btllib::CountingBloomFilter8 counts(cbf_size, num_hashes_per_seed);
  timer.stop();
  std::cout << "done (" << timer.to_string() << ")" << std::endl;
  const auto seq_reader_flag = get_seq_reader_flag(long_read_data);
  for (const auto& file : read_files) {
    std::cout << "processing " << file << "..." << std::flush;
    timer.start();
    btllib::SeqReader seq_reader(file, seq_reader_flag);
    unsigned num_reads = 0;
#pragma omp parallel shared(seq_reader)
    for (const auto& record : seq_reader) {
      insert_all_seeds(record.seq, seeds, counts);
      ++num_reads;
    }
    timer.stop();
    std::cout << "done (" << num_reads << " reads in " << timer.to_string() << ")" << std::endl;
  }
  std::cout << "[-o] writing to " << out_path << "... " << std::flush;
  timer.start();
  counts.save(out_path);
  timer.stop();
  std::cout << "done (" << timer.to_string() << ")" << std::endl;
}

}