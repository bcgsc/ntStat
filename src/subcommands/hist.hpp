#pragma once

#include <argparse/argparse.hpp>
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
get_solid_threshold(const std::vector<uint64_t>& histogram)
{
  return get_first_minima(histogram);
}

inline int
main(const argparse::ArgumentParser& args)
{
  utils::Timer timer;
  std::cout << "loading k-mer spectrum file..." << std::flush;
  const auto histogram = utils::read_ntcard_histogram(args.get("histogram"));
  std::cout << "done" << std::endl;
  std::cout << "first minima = " << get_first_minima(histogram) << std::endl;
  std::cout << "solid k-mers threshold = " << get_solid_threshold(histogram) << std::endl;
  return EXIT_SUCCESS;
}

}