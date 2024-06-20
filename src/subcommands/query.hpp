#pragma once

#include <argparse/argparse.hpp>
#include <btllib/counting_bloom_filter.hpp>
#include <btllib/nthash.hpp>
#include <btllib/seq_reader.hpp>
#include <cstdint>
#include <fstream>
#include <omp.h>

namespace query {

argparse::ArgumentParser*
get_argument_parser()
{
  argparse::ArgumentParser* parser = new argparse::ArgumentParser("query", "");
  parser->add_argument("-s").help("path to spaced seeds file (one per line)");
  parser->add_argument("-b").help("path to BF/CBF file").required();
  parser->add_argument("-o").help("path to output TSV file").required();
  parser->add_argument("query").help("path to query data").required().remaining();
  return parser;
}

template<class BloomFilter, class HashFunction>
inline void
query(const BloomFilter& bf,
      HashFunction& hash_fn,
      const std::vector<std::string>& query_files,
      const std::string& out_path)
{
  btllib::BloomFilter all(bf.get_bytes(), bf.get_hash_num());
  std::ofstream out_file(out_path);
  out_file << std::boolalpha;
  for (const auto& file : query_files) {
    btllib::SeqReader seq_reader(file, btllib::SeqReader::Flag::LONG_MODE);
    for (const auto& record : seq_reader) {
      hash_fn.set_seq(record.seq);
      while (hash_fn.roll()) {
        if (!all.contains_insert(hash_fn.hashes())) {
          const auto kmer = record.seq.substr(hash_fn.get_pos(), hash_fn.get_k());
          out_file << kmer << "\t" << (unsigned)bf.contains(hash_fn.hashes()) << std::endl;
        }
      }
    }
  }
}

int
main(const argparse::ArgumentParser& args)
{
  const auto bf_path = args.get("-b");
  const auto out_path = args.get("-o");
  const auto files = args.get<std::vector<std::string>>("query");
  if (btllib::KmerBloomFilter::is_bloom_file(bf_path)) {
    std::cerr << "[-b] detected btllib::KmerBloomFilter" << std::endl;
    std::cerr << "[-b] loading " << bf_path << "... " << std::flush;
    btllib::KmerBloomFilter bf(bf_path);
    std::cerr << "done" << std::endl;
    const std::string dummy_kmer(bf.get_k(), 'N');
    btllib::NtHash hash_fn(dummy_kmer, bf.get_hash_num(), bf.get_k());
    query(bf, hash_fn, files, out_path);
  } else if (btllib::KmerCountingBloomFilter8::is_bloom_file(bf_path)) {
    std::cerr << "[-b] detected btllib::KmerCountingBloomFilter8" << std::endl;
    std::cerr << "[-b] loading " << bf_path << "... " << std::flush;
    btllib::KmerCountingBloomFilter8 bf(bf_path);
    std::cerr << "done" << std::endl;
    const std::string dummy_kmer(bf.get_k(), 'N');
    btllib::NtHash hash_fn(dummy_kmer, bf.get_hash_num(), bf.get_k());
    query(bf, hash_fn, files, out_path);
  } else if (btllib::SeedBloomFilter::is_bloom_file(bf_path)) {
    std::cerr << "[-b] detected btllib::SeedBloomFilter" << std::endl;
    std::cerr << "[-b] loading " << bf_path << "... " << std::flush;
    btllib::SeedBloomFilter bf(bf_path);
    std::cerr << "done" << std::endl;
    const std::string dummy_kmer(bf.get_k(), 'N');
    btllib::SeedNtHash hash_fn(dummy_kmer, bf.get_seeds(), bf.get_hash_num(), bf.get_k());
    query(bf, hash_fn, files, out_path);
  } else {
    std::cerr << "invalid bloom filter type (-b)" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}
