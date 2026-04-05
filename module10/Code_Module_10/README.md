# add_gaussian_noise.cu

## Overview

`add_gaussian_noise.cu` is a CUDA program that adds Gaussian noise to BMP testing images at multiple configurable intensity levels. The images at different noise levels will be used to assess the edge detection capabilities of the Sobel and Canny algorithms. 

The noisy images can then be fed into an edge detector and compared against ground-truth edges using a metric like Pratt's Figure of Merit (FOM).

The program reads a set of original BMP images, applies Gaussian noise at five predefined standard-deviation levels (0, 5, 15, 30, and 50 in pixel-intensity units on the 0–255 scale), and writes the resulting noisy images to an output folder. All folder paths and input file names are read from a JSON configuration file (`configurations.json`), so no paths are hardcoded in the source.

Noise generation runs entirely on the GPU using NVIDIA's cuRAND library.


## Prerequisites

- NVIDIA CUDA Toolkit (nvcc compiler and CUDA runtime)
- cuRAND library (ships with the CUDA Toolkit)
- C++17-capable compiler (MSVC 2017+, GCC 8+, or Clang 7+)
- `helper_functions.h` header (provides BMP I/O and JSON config parsing)
- `configurations.json` file in the working directory


## Build

### Windows (MSVC + NVCC)

```
nvcc -O2 --std=c++17 -cudart=static add_gaussian_noise.cu -o add_gaussian_noise.exe -lcurand
```

### Linux

```
nvcc -O2 --std=c++17 add_gaussian_noise.cu -o add_gaussian_noise -lcurand
```


## Usage

The program takes no command-line arguments. All configuration is read from `configurations.json`:

```
./add_gaussian_noise
```

On Windows:

```
add_gaussian_noise.exe
```

For each original image (for example `img_testing_1.bmp`), five output files are produced — one per noise level:

```
img_testing_1_noise_0.bmp       (sigma = 0, identical copy)
img_testing_1_noise_5.bmp       (sigma = 5)
img_testing_1_noise_15.bmp      (sigma = 15)
img_testing_1_noise_30.bmp      (sigma = 30)
img_testing_1_noise_50.bmp      (sigma = 50)
```


## Configuration File

The program reads `configurations.json` from the current working directory. The two relevant sections are:

```json
{
  "testing_image_original": {
    "base_path": "pratt_fom/original",
    "original_images": [
      "img_testing_1.bmp",
      "img_testing_2.bmp"
    ]
  },
  "testing_image_noisy": {
    "base_path": "pratt_fom/noisy"
  }
}
```

- **`testing_image_original`** — specifies the input folder (`base_path`) and the list of BMP filenames to process (`original_images`).
- **`testing_image_noisy`** — specifies the output folder (`base_path`). If this folder does not exist, the program creates it automatically.


## Folder Structure

```
project_root/
├── add_gaussian_noise.cu        # this program
├── helper_functions.h           # BMP I/O, JSON config parser
├── configurations.json          # folder paths and file list
│
└── pratt_fom/
    ├── original/                   # INPUT: folder where the original BMP testing images are stored
    │   ├── img_testing_1.bmp
    │   └── img_testing_2.bmp
    │
    └── noisy/                      # OUTPUT: folder where the noisy testing images are stored
        ├── img_testing_1_noise_0.bmp
        ├── img_testing_1_noise_5.bmp
        ├── img_testing_1_noise_15.bmp
        ├── img_testing_1_noise_30.bmp
        ├── img_testing_1_noise_50.bmp
        ├── img_testing_2_noise_0.bmp
        ├── img_testing_2_noise_5.bmp
        ├── img_testing_2_noise_15.bmp
        ├── img_testing_2_noise_30.bmp
        └── img_testing_2_noise_50.bmp
```


## CUDA Kernel Memory Strategy

The `gaussianNoiseKernel` operates on a flat byte array where each element is a single color-channel value of a pixel (R, G, or B). Each thread processes exactly one byte, making the workload embarrassingly parallel with no data dependencies between threads.

### Global Memory

The input image (`d_in`) and the output image (`d_out`) are stored in **global memory** on the device because it is the only GPU memory space large enough to hold entire images which can be several megabytes each. The kernel reads one byte from `d_in` and writes one byte to `d_out` per thread, resulting in a simple streaming access pattern that hardware coalescing handles efficiently.

### Registers

Each thread's calculation values such as the loaded pixel value, the generated random noise sample, the noisy sum, and the clamped result are held in **registers**. Registers are the fastest storage on the GPU. Because every thread works on a single independent pixel or byte with no neighbor access, registers alone are sufficient for all per-thread computation.

### cuRAND State (Registers + Local Memory)

Each thread initializes a `curandState` structure via `curand_init()`, which the cuRAND library stores primarily in registers. If register pressure causes spills, the compiler moves the state to **local memory** (which is physically backed by global memory but is private to each thread and benefits from L1/L2 caching). The cuRAND state holds the PRNG sequence, offset, and internal counters needed to produce the normally distributed random number consumed by `curand_normal()`.
