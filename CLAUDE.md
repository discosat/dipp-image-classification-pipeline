# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

**Configure and build shared libraries only (for deployment):**
```bash
./configure build                # Native build
./configure build --cross        # Cross-compile for AArch64
```

**Configure and build with test executable (for development/testing):**
```bash
./configure test                 # Native build with test executable
./configure test --cross         # Cross-compile test executable for AArch64
```

The configure script will:
- Generate meson_options.txt with appropriate cross-compile setting
- Clean and create builddir
- Set up meson with correct toolchain (native or cross-compile)
- Run ninja to compile

**Test the demosaic module from standalone buffer_dump.bin:**
```bash
./run.sh
```

## Architecture Overview

This is a DISCO2 Image Processing Pipeline module repository containing multiple image processing modules:

### Core Module Structure
Every module must implement the `run` function from `src/include/module.h`:
```c
ImageBatch run(ImageBatch *input_batch, ModuleParameterList *module_parameter_list, int *error_pipe);
```

### Key Components
- **ImageBatch**: Container for image collections with metadata, shared memory info
- **Metadata**: Per-image metadata (width, height, channels, timestamp, etc.)
- **Module Parameters**: YAML-based configuration accessed via `get_param_*` functions

### Current Modules
- `tflite_module.cpp`: TensorFlow Lite classification module (currently disabled in meson.build)
- `jpegxl_encode.cpp`: JPEG XL compression module  
- `demosaic.cpp`: Demosaicing using OpenCV (default test module)
- `single_module.cpp`: Combined module for benchmarking (currently disabled in meson.build)
- `crash_and_burn.c`: Test module for error handling
- `emulate_camera.c`: Camera emulation utility
- `webp_compression.c`: WebP compression module
- Module examples in `src/module_examples/`: dublicate_images.c, id.c, mirror_vertical.c, to_grayscale.c

### Module Development Rules
1. Input ImageBatch must remain unaltered - only modify result batch
2. Use utility functions from `src/include/utils/util.h` for batch manipulation
3. All external dependencies must be statically compiled
4. Error codes (1-99) should be defined in module's ERROR_CODE enum
5. Use `append_result_image()` to add processed images to result batch

### Testing
- Place `input.png` in workspace root
- Run test executable with image count: `./builddir/ITU-e2e_pipeline-exec [count]`
- Module parameters go in `config.yaml` (for tflite: `tflite_config.yaml`)
- By default, test executable is configured to test the demosaic module (see meson.build line 86)
- To test different modules, modify the test_sources line in meson.build

### Dependencies
Required packages:
```bash
sudo apt install build-essential libyaml-dev gcc python3-pip pkg-config gcc-aarch64-linux-gnu g++-aarch64-linux-gnu binutils-aarch64-linux-gnu
sudo pip3 install meson ninja
```

Module-specific dependencies are managed in `meson.build`.

### Current Branch Status
- Working on branch: `fix-half-image-display`
- Main branch: `main`
- For cross-compilation, requires sourcing Yocto environment: `/opt/poky/environment-setup-armv8a-poky-linux`

### Key Build Files
- `meson.build`: Defines all modules, dependencies, and build targets
- `configure`: Shell script handling build modes and cross-compilation
- `yocto_cross.ini`: Cross-compilation configuration for AArch64
- `meson_options.txt`: Generated dynamically by configure script
