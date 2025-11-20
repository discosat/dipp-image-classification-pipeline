project('e2e_pipeline', 'c', 'cpp', version: '0.0.1')

# Compiler and linker flags
# Compiler and linker flags
cflags = ['-Wall', '-g', '-O3']
cppflags = ['-Wall', '-std=c++17', '-g', '-O3']

# Dependencies
proto_c_dep = meson.get_compiler('c').find_library('protobuf-c', required: false)
rt_dep = meson.get_compiler('c').find_library('rt', required: false)
# jxl_dep = meson.get_compiler('cpp').find_library('libjxl', required: true)
jxl_dep = dependency('libjxl', required: true)
jxl_threads_dep = dependency('libjxl_threads', required: true)
uuid_dep = meson.get_compiler('c').find_library('uuid', required: false)
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
    'tflite_classification',
    ['src/tflite_classification.cpp'] + sources,
    include_directories: dirs,
    c_args: cflags,
    cpp_args: cppflags,
    dependencies: [rt_dep, proto_c_dep, tflite_dep, uuid_dep],
)

# Shared library (SO)
shared_library(
    'tflite_binary_segmentation',
    ['src/tflite_binary_segmentation.cpp'] + sources,
    include_directories: dirs,
    c_args: cflags,
    cpp_args: cppflags,
    dependencies: [rt_dep, proto_c_dep, tflite_dep, uuid_dep],
)

# Shared library (SO)
shared_library(
    'jpegxl_encode',
    ['src/jpegxl_encode.cpp'] + sources,
    include_directories: dirs,
    c_args: cflags,
    cpp_args: cppflags,
    dependencies: [rt_dep, proto_c_dep, jxl_dep, jxl_threads_dep,uuid_dep],
)

# Shared library (SO)
shared_library(
    'jpegxl_module',
    ['src/jpegxl_module.c'] + sources,
    include_directories: dirs,
    c_args: cflags,
    dependencies: [rt_dep, proto_c_dep, jxl_dep, jxl_threads_dep, uuid_dep],
)

# Shared library (SO)
shared_library(
    'demosaic',
    ['src/demosaic_module.cpp'] + sources,
    include_directories: dirs,
    c_args: cflags,
    cpp_args: cppflags,
    dependencies: [rt_dep, proto_c_dep, opencv_dep, uuid_dep],
)

# Shared library (SO)
shared_library(
    'resize',
    ['src/resize_module.cpp'] + sources,
    include_directories: dirs,
    c_args: cflags,
    cpp_args: cppflags,
    dependencies: [rt_dep, proto_c_dep, opencv_dep, uuid_dep],
)

# Shared library (SO)
shared_library(
    'save_image',
    ['src/image_saver.cpp'] + sources,
    include_directories: dirs,
    c_args: cflags,
    cpp_args: cppflags,
    dependencies: [rt_dep, proto_c_dep, uuid_dep],
)

# Shared library (SO)
shared_library(
    'single_module',
    ['src/single_module.cpp'] + sources,
    include_directories: dirs,
    c_args: cflags,
    cpp_args: cppflags,
    dependencies: [rt_dep, proto_c_dep, opencv_dep, tflite_dep, jxl_dep, uuid_dep],
)

# Shared library (SO)
shared_library(
    'crash_and_burn',
    ['src/crash_and_burn.c'] + sources,
    include_directories: dirs,
    c_args: cflags,
    dependencies: [rt_dep, proto_c_dep, uuid_dep],
)

# Shared library (SO)
shared_library(
    'tile',
    ['src/tile_module.cpp'] + sources,
    include_directories: dirs,
    c_args: cflags,
    cpp_args: cppflags,
    dependencies: [rt_dep, proto_c_dep, uuid_dep],
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
        c_args: cflags + ['-g'],
        dependencies: [libyaml_dep, m_dep, proto_c_dep, uuid_dep],
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
    c_args: cflags,
    dependencies: [proto_c_dep, uuid_dep],
)