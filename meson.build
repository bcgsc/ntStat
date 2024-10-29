project(
    'ntStat',
    'cpp',
    version: '1.0.0',
    license: 'GPL-3',
    default_options: [
        'cpp_std=c++17',
        'werror=false',
        'optimization=3',
        'b_lto=true',
    ],
)

conf = configuration_data()
conf.set('PROJECT_VERSION', meson.project_version())
configure_file(
    input: 'src/ntstat.py',
    output: 'ntstat',
    configuration: conf,
    install: true,
    install_dir: 'bin',
    install_mode: 'rwxr-xr-x',
)

pybind11_conf = find_program('pybind11-config')
pybind11_conf_ret = run_command(pybind11_conf, ['--includes'], check: true)
pybind11_include = pybind11_conf_ret.stdout().split('-I')[-1].strip()
pybind11 = declare_dependency(include_directories: [pybind11_include])
py_deps = [pybind11, dependency('python3')]

py = import('python').find_installation('python3')

indicators = declare_dependency(include_directories: include_directories('vendor/indicators/include'))

subdir('src/modules/filter')
subdir('src/modules/hist')
subdir('src/modules/query')
subdir('src/modules/count')