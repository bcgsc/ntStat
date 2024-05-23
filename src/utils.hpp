#pragma once

#include <btllib/seq_reader.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdint.h>
#include <string>
#include <vector>

namespace utils {

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

void
print_seeds_list(const std::vector<std::string>& seeds)
{
  for (size_t i = 0; i < seeds.size(); i++) {
    std::cout << "[-s] seed " << i + 1 << ": " << seeds[i] << std::endl;
  }
}

std::vector<uint64_t>
read_ntcard_histogram(const std::string& path)
{
  std::vector<uint64_t> hist;
  std::ifstream hist_file(path);
  std::string freq;
  uint64_t value;
  while (hist_file >> freq >> value) {
    hist.push_back(value);
  }
  return hist;
}

template<typename T>
inline std::string
comma_sep(T val)
{
  std::string val_str = std::to_string(val);
  size_t i = val_str.size() % 3;
  std::string result = i > 0 ? val_str.substr(0, i) + "," : "";
  for (; i + 3 <= val_str.size(); i += 3) {
    result += val_str.substr(i, 3) + ",";
  }
  return result.substr(0, result.size() - 1);
}

class Timer
{
private:
  std::chrono::time_point<std::chrono::system_clock> t_start;
  std::chrono::time_point<std::chrono::system_clock> t_end;

public:
  /**
   * Register the current time as the timer's starting point.
   */
  void start(const std::string& message)
  {
    std::cout << message << "... " << std::flush;
    this->t_start = std::chrono::system_clock::now();
  }

  /**
   * Register the current time as the timer's finish point.
   */
  void stop()
  {
    this->t_end = std::chrono::system_clock::now();
    std::cout << "done (" << this->elapsed_seconds() << "s)" << std::endl;
  }

  /**
   * Compute the difference between the start and stop points in seconds.
   */
  [[nodiscard]] long double elapsed_seconds() const
  {
    const std::chrono::duration<double> elapsed = (t_end - t_start);
    return elapsed.count();
  }
};

}