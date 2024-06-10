#pragma once

#include <argparse/argparse.hpp>
#include <tabulate/table.hpp>
#include <vector>

#include "utils.hpp"

namespace hist {

using histogram_t = std::vector<uint64_t>;

argparse::ArgumentParser*
get_argument_parser()
{
  argparse::ArgumentParser* parser = new argparse::ArgumentParser("hist", "");
  parser->add_argument("-k").help("k-mer length").scan<'u', unsigned>().required();
  parser->add_argument("histogram").help("path to k-mer spectrum file (from ntCard)").required();
  return parser;
}

unsigned
get_first_minima(const histogram_t histogram)
{
  int i = 2;
  while (i <= (int)histogram.size() - 2 && histogram[i] > histogram[i + 1]) {
    i++;
  }
  return i;
}

unsigned
get_solid_threshold(const histogram_t& histogram)
{
  return get_first_minima(histogram);
}

unsigned
get_repeat_threshold(const histogram_t& histogram)
{
  return (float)get_first_minima(histogram) * 1.75;
}

inline int
main(const argparse::ArgumentParser& args)
{
  utils::Timer timer;
  std::cout << "loading k-mer spectrum file... " << std::flush;
  const auto histogram = utils::read_ntcard_histogram(args.get("histogram"));
  std::cout << "done" << std::endl;
  std::cout << std::endl;
  tabulate::Table stats;
  stats.add_row({ "STATISTIC", "VALUE" });
  stats.add_row({ "first minima", std::to_string(get_first_minima(histogram)) });
  stats.add_row({ "solid k-mers threshold", std::to_string(get_solid_threshold(histogram)) });
  stats.add_row({ "repeat k-mers threshold", std::to_string(get_repeat_threshold(histogram)) });
  stats.column(1).format().font_align(tabulate::FontAlign::right);
  stats[0][1].format().font_align(tabulate::FontAlign::center);
  stats[0].format().border_top("-").border_bottom("-").border_left("|").border_right("|").corner(
    "+");
  std::cout << stats << std::endl;
  return EXIT_SUCCESS;
}

}