#pragma once

#include <cstddef>
#include <filesystem>
#include <math.h>
#include <vector>

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

  bool contains(const uint64_t* hashes) { return out_include.contains(hashes); }

  void clear(const uint64_t* hashes) { out_exclude.insert(hashes); }

  void save(const std::string& path)
  {
    if (out_exclude.get_bytes() > 8) {
      const std::string filename = std::filesystem::path(path).stem();
      const std::string extension = std::filesystem::path(path).extension();
      out_include.save(filename + "_include" + extension);
      out_exclude.save(filename + "_exclude" + extension);
    } else {
      out_include.save(path);
    }
  }

  unsigned get_hash_num() { return out_include.get_hash_num(); }

  double get_fpr() { return out_include.get_fpr(); }

private:
  BloomFilter& out_include;
  BloomFilter& out_exclude;
};

inline size_t
get_bf_size(double num_elements, double fpr, double num_hashes)
{
  double r = -num_hashes / log(1.0 - exp(log(fpr) / num_hashes));
  return ceil(num_elements * r);
}
