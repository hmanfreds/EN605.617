# Module 6 Assignment — CUDA Streams + Events (Grayscale + Blur)

**Date:** 2/26/2026  
**Student:** Herbert Schmidmeier  

This program creates **two random “BMP-like” color images** in host memory (stored as 1D interleaved RGB bytes: `RGBRGB...`), copies them to the GPU, converts both images to **grayscale**, then applies a **3×3 box blur** average on the grayscale output.

The main goal is to demonstrate **concurrent GPU work using two CUDA streams**, and to use **CUDA events** to guarantee that the grayscale conversion completes before the blur starts **per stream**. The elapsed time for each stream is measured separately.

Regarding my final project, I have implemented in this assignment the CUDA kernel to perform the blur operation on the grayscale image. It averages the valid neighboring pixels (including itself) within a 3×3 window. The kernel also handles boundary conditions to avoid accessing out-of-bounds memory. 

---

## What the program does

For **Image 1** on **Stream 1** and **Image 2** on **Stream 2** (in parallel):

1. **Generate random RGB images on the host**
   - Each pixel is 3 bytes: `R, G, B`
   - Images are stored as 1D arrays (interleaved): `RGBRGBRGB...`

2. **Allocate pinned host memory**
   - Uses `cudaHostAlloc()` for faster async transfers

3. **Allocate device memory**
   - Separate device global memory for image 1 and image 2

4. **Copy RGB images to the GPU (async)**
   - `cudaMemcpyAsync(..., stream1)` and `cudaMemcpyAsync(..., stream2)`

5. **Convert RGB to grayscale (kernel)**
   - Kernel: `rgb_to_gray<<<grid, block, 0, stream>>>`
   - Uses a **grid-stride loop** to support large images

6. **Record “grayscale done” events**
   - `cudaEventRecord(event1, stream1)`
   - `cudaEventRecord(event2, stream2)`

7. **Wait for grayscale completion before blurring**
   - `cudaStreamWaitEvent(stream, event, 0)`
   - Ensures blur does not start early

8. **Blur grayscale using a 3×3 box average (kernel)**
   - Kernel: `blur_gray<<<grid, block, 0, stream>>>`
   - Averages valid neighbors inside image boundaries

9. **Measure elapsed time per stream**
   - Uses CUDA events (`start/stop`) recorded on each stream

10. **Copy grayscale + blurred outputs back to host (async)**
    - Then synchronizes both streams and prints sample values

11. **Print some pixel samples of the RGB, grayscale and blurred images**
    - The runtime for each stream is printed separately, showing the concurrent execution.

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

## Build

Compile:

```bash
nvcc -O2 -cudart=static assignment.cu -o assignment.exe