#pragma once

#include <btllib/counting_bloom_filter.hpp>
#include <btllib/seq_reader.hpp>
#include <omp.h>
#include <string>
#include <vector>

#include "timer.hpp"
#include "utils.hpp"

namespace count {

void
all_kmer_counts(const std::vector<std::string>& read_files,
                bool long_reads,
                unsigned kmer_length,
                unsigned num_hashes,
                size_t cbf_size,
                const std::string& out_path)
{
  Timer timer;
  std::cout << "[-k] counting all k-mers (k = " << kmer_length << ")" << std::endl;
  std::cout << "[-b] creating counting bloom filter (" << cbf_size << " bytes)... " << std::flush;
  timer.start();
  btllib::KmerCountingBloomFilter8 counts(cbf_size, num_hashes, kmer_length);
  timer.stop();
  std::cout << "done (" << timer.to_string() << ")" << std::endl;
  const auto seq_reader_flag = get_seq_reader_flag(long_reads);
  for (const auto& file : read_files) {
    std::cout << "processing " << file << "... " << std::flush;
    timer.start();
    btllib::SeqReader seq_reader(file, seq_reader_flag);
    unsigned num_reads = 0;
#pragma omp parallel shared(seq_reader)
    for (const auto& record : seq_reader) {
      counts.insert(record.seq);
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