# Julia Set Fractal Generator (OpenCL)

This program generates Julia set fractal images by running the computation on the GPU using OpenCL. Each pixel is computed independently in parallel.

## Files

### `juliaSet.cpp`

The C++ host program implements the Julia Set fractal creation. It handles command-line argument parsing, GPU device discovery, OpenCL setup, kernel compilation, and save the image in PNG format. The program supports two operations:

- `list_gpu` — Enumerates all OpenCL-capable GPUs on the system and prints their vendor, and model in a formatted table.
- `generate` — Loads and compiles `julia_set.cl` at runtime, allocates GPU memory, launches the kernel, copies the pixels from device to host, and writes the result as a PNG image.

The user can select which GPU to run on via the `-device` flag (defaults to device 0) and optionally specify the output resolution (defaults to 1920×1080).

### `julia_set.cl`

The OpenCL kernel that runs on the GPU. Each work-item computes one pixel by:

1. Mapping its pixel coordinates to a point on the complex plane.
2. Iterating the formula `z = z^2 + c` until `|z| > 2` (escape) or the maximum iteration count is reached.
3. Applying smooth coloring to eliminate banding artifacts between iteration levels.
4. Writing an RGBA color value using a sine-wave palette.

The kernel receives the Julia constant `c`, iteration limit, zoom level, and pan offsets as arguments from the host.

### `helper_functions.h`

A header-only image I/O library with zero external dependencies. Provides two functions:

- `write_png()` — Writes RGBA pixel data to a valid PNG file. Constructs the PNG binary format manually (signature, IHDR, IDAT, IEND chunks) and uses DEFLATE stored blocks for the compression stream. 

## Build

### Linux

```bash
g++ -std=c++17 juliaSet.cpp -o juliaSet -lOpenCL
```

### macOS

```bash
clang++ -std=c++17 juliaSet.cpp -o juliaSet -framework OpenCL
```

### Windows (MSVC Developer Command Prompt)

```
cl juliaSet.cpp /std:c++17 /EHsc /I "C:\opencl-sdk\include" /link "C:\opencl-sdk\lib\OpenCL.lib" /out:juliaSet.exe
```

## Usage

```bash
# List available GPUs
./juliaSet -operation list_gpu

# Generate with default settings (device 0, 1920x1080)
./juliaSet -operation generate

# Generate on GPU #1
./juliaSet -operation generate -device 1

# Generate at 4K resolution on device 0
./juliaSet -operation generate -device 0 3840 2160
```

The output image is saved as `julia_set.png` in the current directory.

## Julia Set Parameters

The fractal shape is determined by the complex constant `c`. The default is `c = -0.7 + 0.27015i`, which produces classic spiraling tendrils. Other values can be set by modifying the `JuliaParams` struct in `juliaSet.cpp`:

| c value               | Pattern                  |
|------------------------|--------------------------|
| `-0.7 + 0.27015i`     | Spiraling tendrils       |
| `-0.8 + 0.156i`       | Dendritic branching      |
| `0.285 + 0.01i`       | Nested spirals           |
| `-0.4 + 0.6i`         | Connected "rabbit" set   |
| `0.355 + 0.355i`      | Symmetric cross          |

## Requirements

- An OpenCL-capable GPU with installed drivers and OpenCL runtime.
- A C++17 compiler.
- `julia_set.cl` must be in the same directory as the executable at runtime.
