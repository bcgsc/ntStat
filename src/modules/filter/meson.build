deps = [
    dependency('argparse'),
    meson.get_compiler('cpp').find_library('btllib'),
    dependency('indicators'),
    dependency('openmp'),
    dependency('tabulate'),
]

py.extension_module(
    'filter',
    'main.cpp',
    dependencies: py_deps + deps,
    install: true,
    subdir: 'ntstat',
)