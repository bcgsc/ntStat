#pragma once

#include <argparse/argparse.hpp>
#include <fstream>
#include <sstream>

class ProgramArguments
{
public:
  std::vector<std::string> reads_paths;
  std::vector<std::string> seeds;
  std::vector<uint64_t> histogram;
  std::string out_path;
  size_t out_size;
  float target_err;
  unsigned kmer_length;
  unsigned num_threads;
  unsigned cmin, cmax;
  bool counts, long_mode;

  ProgramArguments(const std::vector<std::string>& argv)
  {
    argparse::ArgumentParser parser("filter");
    parser.add_argument("-k").help("k-mer length").scan<'u', unsigned>();
    parser.add_argument("-s").help("path to spaced seeds file (one per line, if -k not specified)");
    parser.add_argument("-f").help("path to k-mer spectrum file (from ntCard)").required();
    parser.add_argument("-e")
      .help("target output error rate")
      .default_value(0.001F)
      .scan<'g', float>();
    parser.add_argument("-b").help("output BF/CBF size (bytes)").scan<'u', size_t>();
    parser.add_argument("-cmin")
      .help("minimum count threshold (>=1, or 0 for first minimum)")
      .default_value(1U)
      .scan<'u', unsigned>();
    parser.add_argument("-cmax")
      .help("maximum count threshold (<=255)")
      .default_value(255U)
      .scan<'u', unsigned>();
    parser.add_argument("--counts")
      .help("output counts (requires ~8x RAM for CBF)")
      .default_value(false)
      .implicit_value(true);
    parser.add_argument("--long")
      .help("optimize for long read data")
      .default_value(false)
      .implicit_value(true);
    parser.add_argument("-t").help("number of threads").default_value(1U).scan<'u', unsigned>();
    parser.add_argument("-o").help("path to store output file").required();
    parser.add_argument("reads").help("path to sequencing data file(s)").required().remaining();

    help_message = parser.help().str();

    parser.parse_args(argv);

    reads_paths = parser.get<std::vector<std::string>>("reads");
    out_path = parser.get("-o");
    out_size = parser.is_used("-b") ? parser.get<size_t>("-b") : 0;
    target_err = parser.get<float>("-e");
    num_threads = parser.get<unsigned>("-t");
    cmin = parser.get<unsigned>("-cmin");
    cmax = parser.get<unsigned>("-cmax");
    counts = parser.get<bool>("--counts");
    long_mode = parser.get<bool>("--long");

    if (parser.is_used("-s")) {
      std::ifstream file(parser.get("-s"));
      std::string line;
      while (file >> line) {
        seeds.emplace_back(line);
      }
      kmer_length = seeds[0].size();
    } else if (parser.is_used("-k")) {
      kmer_length = parser.get<unsigned>("-k");
    } else {
      throw std::logic_error("at least one of -k or -s should be used");
    }

    std::ifstream hist_file(parser.get("-f"));
    std::string freq;
    uint64_t value;
    while (hist_file >> freq >> value) {
      histogram.emplace_back(value);
    }

    for (unsigned i = 2; cmin == 0 && i < histogram.size() - 1; i++) {
      if (histogram[i] <= histogram[i + 1]) {
        cmin = i - 1;
      }
    }

    if (cmax > 255) {
      throw std::runtime_error("cmax should be less than 256");
    }
    if (cmin >= cmax) {
      throw std::runtime_error("cmin cannot be greater than or equal to cmax");
    }
  }

  std::string get_help_message() { return help_message; }

  std::string get_arg_summary()
  {
    std::stringstream ss;
    if (seeds.empty()) {
      ss << "[-k] counting " << kmer_length << "-mers" << std::endl;
    } else {
      ss << "[-s] counting spaced seeds" << std::endl;
      for (size_t i = 0; i < seeds.size(); i++) {
        ss << "[-s] seed " << i + 1 << ": " << seeds[i] << std::endl;
      }
    }
    ss << "[-cmin] minimum count: " << cmin << std::endl;
    ss << "[-cmax] maximum count: " << cmax << std::endl;
    ss << "[-t] thread limit: " << num_threads << std::endl;
    ss << "[-e] target output error rate: " << target_err << std::endl;
    return ss.str();
  }

private:
  std::string help_message;
};