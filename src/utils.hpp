#pragma once

#include <btllib/seq_reader.hpp>
#include <cmath>
#include <fstream>
#include <stdint.h>
#include <string>
#include <vector>

inline std::vector<std::string>
read_file_lines(const std::string& path)
{
  std::vector<std::string> lines;
  std::ifstream file(path);
  std::string line;
  while (file >> line) {
    lines.emplace_back(line);
  }
  return lines;
}

inline size_t
get_bf_size(double num_elements, double num_hashes, int num_seeds, double fpr)
{
  double r = -num_hashes / log(1.0 - exp(log(fpr) / num_hashes));
  return ceil(num_elements * std::max(num_seeds, 1) * r);
}

inline unsigned
get_seq_reader_flag(bool long_mode)
{
  if (long_mode) {
    return btllib::SeqReader::Flag::LONG_MODE;
  } else {
    return btllib::SeqReader::Flag::SHORT_MODE;
  }
}
