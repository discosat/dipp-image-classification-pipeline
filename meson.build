project('e2e_pipeline', 'c', 'cpp', version: '0.0.1')

# Compiler and linker flags
cflags = ['-Wall']

# Dependencies
proto_c_dep = meson.get_compiler('c').find_library('protobuf-c', required: false)
rt_dep = meson.get_compiler('c').find_library('rt', required: false)
jxl_dep = meson.get_compiler('cpp').find_library('libjxl', required: true)
tflite_dep = dependency('tensorflow2-lite')
opencv_dep = dependency('opencv4')

# Source files
sources = [
    'src/utils/memory_util.c',
    'src/utils/config_util.c',
    'src/utils/batch_util.c',
    'src/utils/error_util.c',
    'src/utils/metadata_util.c',
    'src/utils/metadata.pb-c.c',
    'src/utils/logger.c',
]

dirs = include_directories(
    'src/include',
    'src/include/utils',
)

# Shared library (SO)
shared_library(
    'tflite_tiled_classification',
    ['src/tflite_module.cpp'] + sources,
    include_directories: dirs,
    c_args: cflags + ['-DSHARED_MEMORY=1'],
    dependencies: [rt_dep, proto_c_dep, tflite_dep],
)
# Shared library (SO)
shared_library(
    'jpegxl_encode',
    ['src/jpegxl_encode.cpp'] + sources,
    include_directories: dirs,
    c_args: cflags + ['-DSHARED_MEMORY=1'],
    dependencies: [rt_dep, proto_c_dep, jxl_dep],
)

# Shared library (SO)
shared_library(
    'demosaic',
    ['src/demosaic.cpp'] + sources,
    include_directories: dirs,
    c_args: cflags + ['-DSHARED_MEMORY=1'],
    dependencies: [rt_dep, proto_c_dep, opencv_dep],
)

# Shared library (SO)
shared_library(
    'single_module',
    ['src/single_module.cpp'] + sources,
    include_directories: dirs,
    c_args: cflags + ['-DSHARED_MEMORY=1'],
    dependencies: [rt_dep, proto_c_dep, opencv_dep, tflite_dep, jxl_dep],
)

# Check for cross-compiling
if not get_option('cross-compile')
    # If not cross-compiling, include the executable target
    sources += 'src/test.c'
    sources += 'src/utils/yaml_parser.c'

    libyaml_dep = dependency('yaml-0.1')
    m_dep = meson.get_compiler('c').find_library('m', required: false)

    # Executable
    executable(
        meson.project_name() + '-exec',
        sources,
        include_directories: dirs,
        c_args: cflags + ['-g', '-DSHARED_MEMORY=0'],
        dependencies: [libyaml_dep, m_dep, proto_c_dep],
    )
endif

sources_emulate = [
    'src/emulate_camera.c',
    'src/utils/memory_util.c',
    'src/utils/config_util.c',
    'src/utils/batch_util.c',
    'src/utils/error_util.c',
    'src/utils/metadata_util.c',
    'src/utils/metadata.pb-c.c',
    'src/utils/logger.c',
]

executable(
    'emulate_camera',
    sources_emulate,
    include_directories: dirs,
    c_args: cflags + ['-DSHARED_MEMORY=1'],
    dependencies: [proto_c_dep],
)