# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

**Configure and build for target (AArch64):**
```bash
./configure build
```

**Configure and build for testing (host architecture):**
```bash
./configure test
```

The configure script will:
- Set up meson options
- Clean and create builddir
- Run ninja to compile

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
- `tflite_module.cpp`: TensorFlow Lite classification module
- `jpegxl_encode.cpp`: JPEG XL compression module  
- `demosaic.cpp`: Demosaicing using OpenCV
- `single_module.cpp`: Combined module for benchmarking
- `crash_and_burn.c`: Test module for error handling
- `emulate_camera.c`: Camera emulation utility

### Module Development Rules
1. Input ImageBatch must remain unaltered - only modify result batch
2. Use utility functions from `src/include/utils/util.h` for batch manipulation
3. All external dependencies must be statically compiled
4. Error codes (1-99) should be defined in module's ERROR_CODE enum
5. Use `append_result_image()` to add processed images to result batch

### Testing
- Place `input.png` in workspace root
- Run test executable with image count: `./builddir/e2e_pipeline-exec [count]`
- Module parameters go in `config.yaml` (for tflite: `tflite_config.yaml`)

### Dependencies
Required packages:
```bash
sudo apt install build-essential libyaml-dev gcc python3-pip pkg-config gcc-aarch64-linux-gnu g++-aarch64-linux-gnu binutils-aarch64-linux-gnu
sudo pip3 install meson ninja
```

Module-specific dependencies are managed in `meson.build`.