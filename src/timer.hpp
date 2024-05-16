#pragma once

#include <chrono>
#include <string>

class Timer
{
private:
  std::chrono::time_point<std::chrono::system_clock> t_start;
  std::chrono::time_point<std::chrono::system_clock> t_end;

public:
  /**
   * Register the current time as the timer's starting point.
   */
  void start() { this->t_start = std::chrono::system_clock::now(); }

  /**
   * Register the current time as the timer's finish point.
   */
  void stop() { this->t_end = std::chrono::system_clock::now(); }

  /**
   * Compute the difference between the start and stop points in seconds.
   */
  [[nodiscard]] long double elapsed_seconds() const
  {
    const std::chrono::duration<double> elapsed = (t_end - t_start);
    return elapsed.count();
  }

  /**
   * Get a human-readable representation of the elapsed time.
   */
  [[nodiscard]] std::string to_string() const
  {
    return std::to_string(this->elapsed_seconds()) + "s";
  }
};
