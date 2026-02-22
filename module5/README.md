# CUDA Grayscale Benchmark (Module 5 Assignment)

**Date:** 2/20/2026  
**Student:** Herbert Schmidmeier  

This program generates a testing RGB image of width 7680 and height 4320, and converts it to grayscale using CUDA kernels.  
The goal is to compare the performance impact of reading the grayscale coefficients from different CUDA memory types
such as host memory, shared memory, constant memory, global memory, and registers.

The BMP image convertion will be part of my final project which is the implementation of the Sobel edge detection filter. 

---

## What the program does

1. **Creates a testing test image to avoid loading an actual BMP image:**
   - Image is stored as a **1D interleaved RGB array**: `RGBRGBRGB...`
   - Each pixel uses 3 bytes: **R, G, B**.

2. **Allocates pinned host memory** for the test image and grayscale output.

3. **Allocates device memory** for RGB input, grayscale output and copies RGB array to the GPU.

4. **Runs five CUDA kernels**, each using a different memory type for grayscale coefficients:
   - **Global memory**
   - **Constant memory**
   - **Shared memory**
   - **Registers**
   - **Mapped host memory**

5. **Timing approach**
   - Each kernel is launched **3 warmup runs**
   - Then launched **20 timed runs**
   - The runtime printed is the **average** over the 20 timed runs
   - Inside each kernel, the grayscale math is repeated `LOOP_ITERS = 500` times to amplify timing differences.

6. Copies the grayscale output back to host and prints a small sample.

---

## Usage

Two arguments can be passed to the program:

1. -threads: Number of threads per block (default: 256)
2. -blocks: Number of blocks. The number of threads and tile dimensions will be calculated based on this value.

  - If no argument is specified, the program calculates the number of blocks needed to cover all pixels based on the number of threads per block of 256.
  - These two arguments are mutually exclusive. If both are provided, the program will print an error and exit.
  - If `-threads` is provided, the program will check if it is valid value. It must be equal or smalled than 1,024. If not valid, the program will print an error and exit.

- Example usage:
```bash
./assignment.exe -threads 256
./assignment.exe -blocks 100000
```

---

## Compile

```bash
nvcc -cudart=static assignment.cu -o assignment.exe