#pragma once

#include <argparse/argparse.hpp>
#include <btllib/counting_bloom_filter.hpp>
#include <btllib/nthash.hpp>
#include <btllib/seq_reader.hpp>
#include <cstdint>
#include <fstream>
#include <omp.h>

#include "utils.hpp"

namespace query {

argparse::ArgumentParser*
get_argument_parser()
{
  const auto no_args = argparse::default_arguments::none;
  argparse::ArgumentParser* parser = new argparse::ArgumentParser("query", "", no_args);
  parser->add_argument("-s").help("path to spaced seeds file (one per line)");
  parser->add_argument("-t").help("number of threads").default_value(1U).scan<'u', unsigned>();
  parser->add_argument("-b").help("path to BF/CBF file").required();
  parser->add_argument("-o").help("path to output TSV file").required();
  parser->add_argument("--distinct").default_value(false).implicit_value(true);
  parser->add_argument("query").help("path to query data").required().remaining();
  return parser;
}

template<class BloomFilter, class HashFunction>
inline void
query(const BloomFilter& bf,
      HashFunction& hash_fn,
      const std::vector<std::string>& query_files,
      bool distinct,
      const std::string& out_path)
{
  Timer timer;
  timer.start("performing queries");
  btllib::BloomFilter all(distinct ? bf.get_bytes() : 1, bf.get_hash_num());
  std::ofstream out_file(out_path);
  out_file << std::boolalpha;
  for (const auto& file : query_files) {
    btllib::SeqReader seq_reader(file, btllib::SeqReader::Flag::SHORT_MODE);
#pragma omp parallel shared(seq_reader)
    for (const auto& record : seq_reader) {
      hash_fn.set_seq(record.seq);
      while (hash_fn.roll()) {
        if (!distinct || !all.contains_insert(hash_fn.hashes())) {
          const auto kmer = record.seq.substr(hash_fn.get_pos(), hash_fn.get_k());
#pragma omp critical
          out_file << kmer << "\t" << (unsigned)bf.contains(hash_fn.hashes()) << std::endl;
        }
      }
    }
  }
  timer.stop();
}

int
main(const argparse::ArgumentParser& args)
{
  const auto num_threads = args.get<unsigned>("-t");
  omp_set_num_threads(num_threads);
  std::cout << "[-t] thread limit set to " << num_threads << std::endl;
  const auto bf_path = args.get("-b");
  const auto distinct = args.get<bool>("--distinct");
  const auto files = args.get<std::vector<std::string>>("query");
  const auto out_path = args.get("-o");
  Timer timer;
  if (btllib::KmerBloomFilter::is_bloom_file(bf_path)) {
    timer.start("[-b] loading k-mer bloom filter from " + bf_path);
    btllib::KmerBloomFilter bf(bf_path);
    timer.stop();
    btllib::NtHash hash_fn(std::string(bf.get_k(), 'N'), bf.get_hash_num(), bf.get_k());
    query<btllib::KmerBloomFilter, btllib::NtHash>(bf, hash_fn, files, distinct, out_path);
  } else if (btllib::KmerCountingBloomFilter8::is_bloom_file(bf_path)) {
    timer.start("[-b] loading k-mer counting bloom filter from " + bf_path);
    btllib::KmerCountingBloomFilter8 bf(bf_path);
    timer.stop();
    btllib::NtHash hash_fn(std::string(bf.get_k(), 'N'), bf.get_hash_num(), bf.get_k());
    query<btllib::KmerCountingBloomFilter8, btllib::NtHash>(bf, hash_fn, files, distinct, out_path);
  } else if (btllib::SeedBloomFilter::is_bloom_file(bf_path)) {
    timer.start("[-b] loading seed bloom filter from " + bf_path);
    btllib::SeedBloomFilter bf(bf_path);
    timer.stop();
    btllib::SeedNtHash hash_fn(
      std::string(bf.get_k(), 'N'), bf.get_seeds(), bf.get_hash_num(), bf.get_k());
    query<btllib::SeedBloomFilter, btllib::SeedNtHash>(bf, hash_fn, files, distinct, out_path);
  } else {
    std::cerr << "invalid bloom filter type (-b)" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}
