#pragma once

#include <btllib/seq_reader.hpp>
#include <cstddef>
#include <math.h>
#include <stdint.h>
#include <tabulate/table.hpp>
#include <vector>

inline uint64_t
get_num_elements(unsigned cmin, const std::vector<uint64_t>& histogram, size_t num_seeds)
{
  uint64_t num_elements = histogram[1];
  for (unsigned i = 2; i < cmin + 1; i++) {
    num_elements -= histogram[i];
  }
  return num_elements * std::max(1UL, num_seeds);
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

void
print_stats_table(uint64_t total, uint64_t distinct, uint64_t filtered)
{
  tabulate::Table table;
  std::cout << std::endl << "overall statistics:" << std::endl;
  table.add_row({ "total number of k-mers", comma_sep(total) });
  table.add_row({ "number of distinct k-mers", comma_sep(distinct) });
  table.add_row({ "number of unique filtered k-mers", comma_sep(filtered) });
  table.format().border_top("").border_bottom("").border_left("").border_right("").corner("");
  table.column(1).format().font_align(tabulate::FontAlign::right);
  std::cout << table << std::endl << std::endl;
}

inline std::string
human_readable(size_t bytes)
{
  unsigned o = 0;
  std::stringstream ss;
  double mantissa = bytes;
  while (mantissa >= 1024) {
    mantissa /= 1024.;
    ++o;
  }
  ss << std::ceil(mantissa * 10.) / 10. << "BKMGTPE"[o];
  ss << (o > 0 ? "B" : "");
  return ss.str();
}

inline unsigned
get_seq_reader_flag(bool long_mode)
{
  return long_mode ? btllib::SeqReader::Flag::LONG_MODE : btllib::SeqReader::Flag::SHORT_MODE;
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
