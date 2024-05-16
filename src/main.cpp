#include <argparse/argparse.hpp>
#include <functional>
#include <map>

#include "count/main.hpp"
#include "version.hpp"

constexpr char const* LOGO = "ntStat";

using subcommand_function = std::function<int(const argparse::ArgumentParser&)>;

std::pair<argparse::ArgumentParser*, subcommand_function>
get_subcommand(int argc, char* argv[])
{
  std::map<argparse::ArgumentParser*, subcommand_function> parser_functions = {
    { count::get_argument_parser(), count::main }
  };
  argparse::ArgumentParser parser("ntStat", VERSION);
  for (const auto& item : parser_functions) {
    parser.add_subparser(*item.first);
  }
  try {
    parser.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    for (const auto& item : parser_functions) {
      if (parser.is_subcommand_used(*item.first)) {
        std::cerr << *item.first;
      }
    }
    std::exit(EXIT_FAILURE);
  }
  for (const auto& item : parser_functions) {
    if (parser.is_subcommand_used(*item.first)) {
      return item;
    }
  }
  std::cerr << parser;
  std::exit(EXIT_SUCCESS);
}

int
main(int argc, char* argv[])
{
  const auto& subcommand = get_subcommand(argc, argv);
  std::cout << LOGO << std::endl;
  std::cout << "v" << VERSION << std::endl;
  return subcommand.second(*subcommand.first);
}