#!/usr/bin/env python3

import importlib
import signal
import sys

VERSION = "@PROJECT_VERSION@"
MODULES = ["--version", "count", "filter", "hist", "query"]


def print_help():
    print(f"usage: ntstat {{{', '.join(MODULES)}}} ...", file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        print("no module selected", file=sys.stderr)
        print_help()
        exit(1)
    module_name = sys.argv[1]
    if module_name == "--version":
        print(VERSION)
        exit(0)
    if module_name not in MODULES:
        print(f"invalid module: {module_name}", file=sys.stderr)
        print_help()
        exit(1)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    module = importlib.import_module(f"ntstat.{module_name}")
    exit(module.run(sys.argv[1:]))


if __name__ == "__main__":
    main()
