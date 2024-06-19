project(
    'ntStat',
    'cpp',
    version: '0.1',
    license: 'GPL-3',
    default_options: [
        'cpp_std=c++17',
        'werror=true',
        'warning_level=3',
        'optimization=3',
        'b_lto=true',
    ],
)

conf = configuration_data()
conf.set('PROJECT_VERSION', meson.project_version())
configure_file(
    input: 'src/version.hpp.in',
    output: 'version.hpp',
    configuration: conf,
)

compiler = meson.get_compiler('cpp')

dependencies = [
    dependency('argparse'),
    dependency('tabulate'),
    dependency('openmp'),
    compiler.find_library('btllib'),
]

include_directories = [
    'src',
    'vendor/indicators/include/',
]

sources = run_command(
    'scripts/list_sources.sh',
    check: true,
).stdout().strip().split()

executable(
    'ntstat',
    ['src/main.cpp'] + sources,
    dependencies: dependencies,
    include_directories: include_directories,
    install: true,
)