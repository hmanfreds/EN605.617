# Module 9 Assignment — NPP + cuRAND (Grayscale + Blur)

**Date:** 3/19/2026  
**Student:** Herbert Schmidmeier  

This program creates **two random "BMP-like" color images** directly on the GPU using **cuRAND** (stored as 1D interleaved RGB bytes: `RGBRGB...`), converts both images to **grayscale** using the **NPP library**, then applies a **3×3 box blur** average on the grayscale output also using the **NPP library**.

---

## What the program does

For **Image 1** on **Stream 1** and **Image 2** on **Stream 2** (in parallel):

1. **Generate random RGB images directly on the device using cuRAND**
   - Each pixel is 3 bytes: `R, G, B`
   - Images are stored as 1D arrays (interleaved): `RGBRGBRGB...`
   - Uses `curandGenerate()` with the XORWOW pseudo-random generator (seed: 1234)
   - Only the first 5 pixels (15 bytes) are copied to host for sample printing

2. **Allocate pinned host memory**
   - Uses `cudaHostAlloc()` for faster async transfers
   - Only output buffers (grayscale + blurred) are allocated on the host

3. **Allocate device memory**
   - Separate device global memory for RGB, grayscale, and blurred images for each stream

4. **Build NPP stream contexts manually**
   - `NppStreamContext` is populated from `cudaGetDeviceProperties()` and `cudaStreamGetFlags()`
   - Required because `nppGetStreamContext()` was removed in CUDA 13.1

5. **Convert RGB to grayscale (NPP)**
   - Function: `nppiRGBToGray_8u_C3C1R_Ctx()`
   - Uses ITU-R BT.601 luma coefficients: `0.299 R + 0.587 G + 0.114 B`
   - Runs concurrently on stream 1 and stream 2

6. **Record "grayscale done" events**
   - `cudaEventRecord(event1, stream1)`
   - `cudaEventRecord(event2, stream2)`

7. **Wait for grayscale completion before blurring**
   - `cudaStreamWaitEvent(stream, event, 0)`
   - Ensures blur does not start before grayscale is complete

8. **Blur grayscale using a 3×3 box average (NPP)**
   - Function: `nppiFilterBoxBorder_8u_C1R_Ctx()`
   - Border handling: `NPP_BORDER_REPLICATE` (clamps out-of-bounds to nearest edge pixel)
   - Runs concurrently on stream 1 and stream 2

9. **Copy grayscale + blurred outputs back to host (async)**
   - Then synchronizes both streams

10. **Print pixel samples of the RGB, grayscale, and blurred images**

---

## Program structure

| Function | Description |
|---|---|
| `create_image_device()` | Generates one random RGB image on device using cuRAND |
| `create_test_images_curand()` | Creates both images, destroys generator, copies 5-pixel samples to host |
| `rgb_to_gray_npp()` | Converts both RGB images to grayscale via NPP, records and waits on events |
| `blur_gray_npp()` | Blurs both grayscale images via NPP, copies results back to host |
| `print_sample_rgb()` | Prints the first 5 RGB pixel values |
| `print_sample()` | Prints the first 5 grayscale or blurred pixel values |

---

## Build

Compile:

```bash
nvcc assignment.cu -o assignment.exe -lnppc -lnppif -lnppicc -lcurand
```

---

## Run

```bash
./assignment.exe
```

The program uses a fixed image size of **7680 × 4320** (8K). No command-line arguments are supported in this version.

---

## Dependencies

| Library | Purpose |
|---|---|
| `cuRAND` | On-device random RGB image generation |
| `NPP` (`nppc`, `nppif`, `nppicc`) | RGB-to-grayscale conversion and box blur |
| CUDA Runtime | Streams, events, memory management |

Requires **CUDA 13.1** or later. The NPP stream context (`NppStreamContext`) must be populated manually since `nppGetStreamContext()` was removed in CUDA 13.1.
