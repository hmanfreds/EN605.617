// Johns Hopkins University
// GPU Programming Class
// Module 9 Assignment
// Herbert Schmidmeier
// Date: 3/19/2026

// ----------------------------------------------------------------------------------------

// Program to convert two randomly color BMP images to grayscale followed by blurring it.
// The test images are created as 1D vectors with the RGB pixles interleaved such as RGBRGB... 
// where each pixel is represented by 3 bytes (R, G, B). Each grayscale pixel uses 1 byte. 
// Another NPP function blurs the grayscale vector by averaging a 3x3 area around the pixel 
// being processed. This code uses 2 streams and 2 events to garantee that the grayscale 
// conversions are completed before the blurring starts. 
// - RGB test images are generated directly on the device using cuRAND.
// - The RGB image conversion to grayscale is done using the NPP library.
// - The grayscale blurring is also performed with the NPP library.

// ----------------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <curand.h>
#include <nppi.h>
#include <nppcore.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>


// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)

// Macro for checking NPP errors
#define NPP_CHECK(call) do {                                       \
    NppStatus status = (call);                                     \
    if (status != NPP_SUCCESS) {                                   \
        fprintf(stderr, "NPP error %s:%d: %d\n",                   \
                __FILE__, __LINE__, (int)status);                  \
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)

// Macro for checking cuRAND errors
#define CURAND_CHECK(call) do {                                    \
    curandStatus_t err = (call);                                   \
    if (err != CURAND_STATUS_SUCCESS) {                            \
        fprintf(stderr, "cuRAND error %s:%d: %d\n",                \
                __FILE__, __LINE__, (int)err);                     \
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)



// Generate a random RGB image directly on the device using cuRAND.
// -----------------------------------------------------------------------
void create_image_device(unsigned char* d_rgb, size_t rgb_bytes, curandGenerator_t gen)
{
    // Round up to the nearest multiple of 4 (curandGenerate requirement)
    size_t image_byte_count = (rgb_bytes + 3) / 4;

    unsigned int* d_tmp;
    CUDA_CHECK(cudaMalloc((void**)&d_tmp, image_byte_count * sizeof(unsigned int)));

    CURAND_CHECK(curandGenerate(gen, d_tmp, image_byte_count));

    // Reinterpret the random uint bytes directly as unsigned char RGB data
    CUDA_CHECK(cudaMemcpy(d_rgb, d_tmp, rgb_bytes, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(d_tmp));
}



// Wrapper to create random testing images using NPP
// -----------------------------------------------------------------------
void create_test_images_curand(unsigned char* d_rgb1, unsigned char* d_rgb2,
    size_t rgb_bytes,
    unsigned char* h_sample1, unsigned char* h_sample2)
{
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    // Calls function to create random image on the device memory
    create_image_device(d_rgb1, rgb_bytes, gen);
    create_image_device(d_rgb2, rgb_bytes, gen);

    CURAND_CHECK(curandDestroyGenerator(gen));

    // Copy just the first 5 pixels to host for the sample print
    CUDA_CHECK(cudaMemcpy(h_sample1, d_rgb1, 15, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sample2, d_rgb2, 15, cudaMemcpyDeviceToHost));
}



// Function converts RGB image to grayscale using the NPP library
// -----------------------------------------------------------------------
void rgb_to_gray_npp(unsigned char* d_rgb1, unsigned char* d_rgb2,
    unsigned char* d_gray1, unsigned char* d_gray2,
    int rgb_step, int gray_step, NppiSize roi,
    NppStreamContext nppCtx1, NppStreamContext nppCtx2,
    cudaEvent_t event1, cudaEvent_t event2)
{
    NPP_CHECK(nppiRGBToGray_8u_C3C1R_Ctx(
        d_rgb1, rgb_step, d_gray1, gray_step, roi, nppCtx1));

    NPP_CHECK(nppiRGBToGray_8u_C3C1R_Ctx(
        d_rgb2, rgb_step, d_gray2, gray_step, roi, nppCtx2));

    // Synchronize streams to ensure grayscale conversion is done before blurring
    CUDA_CHECK(cudaEventRecord(event1, nppCtx1.hStream));
    CUDA_CHECK(cudaEventRecord(event2, nppCtx2.hStream));
    CUDA_CHECK(cudaStreamWaitEvent(nppCtx1.hStream, event1, 0));
    CUDA_CHECK(cudaStreamWaitEvent(nppCtx2.hStream, event2, 0));
}



// Function blurs grayscale image using the NPP library
// -----------------------------------------------------------------------
void blur_gray_npp(unsigned char* d_gray1, unsigned char* d_gray2,
    unsigned char* d_blur1, unsigned char* d_blur2,
    unsigned char* h_gray1, unsigned char* h_gray2,
    unsigned char* h_blur1, unsigned char* h_blur2,
    int gray_step, NppiSize roi,
    NppStreamContext nppCtx1, NppStreamContext nppCtx2,
    size_t gray_bytes, size_t blur_bytes)
{
    NppiSize  mask = { 3, 3 };
    NppiPoint anchor = { 1, 1 };

    // Blur grayscale images using box filter by averaging 3x3 neighborhood
    NPP_CHECK(nppiFilterBoxBorder_8u_C1R_Ctx(
        d_gray1, gray_step, roi, { 0, 0 },
        d_blur1, gray_step, roi,
        mask, anchor, NPP_BORDER_REPLICATE, nppCtx1));

    NPP_CHECK(nppiFilterBoxBorder_8u_C1R_Ctx(
        d_gray2, gray_step, roi, { 0, 0 },
        d_blur2, gray_step, roi,
        mask, anchor, NPP_BORDER_REPLICATE, nppCtx2));

    // Copy results back to host
    CUDA_CHECK(cudaMemcpyAsync(h_gray1, d_gray1, gray_bytes, cudaMemcpyDeviceToHost, nppCtx1.hStream));
    CUDA_CHECK(cudaMemcpyAsync(h_gray2, d_gray2, gray_bytes, cudaMemcpyDeviceToHost, nppCtx2.hStream));
    CUDA_CHECK(cudaMemcpyAsync(h_blur1, d_blur1, blur_bytes, cudaMemcpyDeviceToHost, nppCtx1.hStream));
    CUDA_CHECK(cudaMemcpyAsync(h_blur2, d_blur2, blur_bytes, cudaMemcpyDeviceToHost, nppCtx2.hStream));
    CUDA_CHECK(cudaStreamSynchronize(nppCtx1.hStream));
    CUDA_CHECK(cudaStreamSynchronize(nppCtx2.hStream));
}

// Printing functions
// -----------------------------------------------------------------------

// Print the first 5 RGB pixels of the image
void print_sample_rgb(unsigned char* h_rgb, const std::string& text)
{
    printf("\nTest Image %s - Sample RGB values (RGB):", text.c_str());
    printf("\nRED pixels:   %u %u %u %u %u", h_rgb[0], h_rgb[3], h_rgb[6], h_rgb[9], h_rgb[12]);
    printf("\nGREEN pixels: %u %u %u %u %u", h_rgb[1], h_rgb[4], h_rgb[7], h_rgb[10], h_rgb[13]);
    printf("\nBLUE pixels:  %u %u %u %u %u\n", h_rgb[2], h_rgb[5], h_rgb[8], h_rgb[11], h_rgb[14]);
}

// Print the first 5 grayscale pixel values of the image
void print_sample(unsigned char* h_gray, const std::string& text)
{
    printf("%s pixels: %u %u %u %u %u\n", text.c_str(),
        h_gray[0], h_gray[1], h_gray[2], h_gray[3], h_gray[4]);
}



int main()
{
    // Default values
    int width = 7680;
    int height = 4320;

    // Print image size to be generated
    printf("\nImage size: %d x %d\n", width, height);

    // Memory sizes
    const size_t pixels = (size_t)width * (size_t)height;
    const size_t rgb_bytes = pixels * 3 * sizeof(unsigned char);
    const size_t gray_bytes = pixels * sizeof(unsigned char);
    const size_t blur_bytes = pixels * sizeof(unsigned char);

    // NPP row strides
    const int rgb_step = width * 3;
    const int gray_step = width * 1;

    // Host buffers for the first 5 RGB pixels of each image (sample printing only)
    unsigned char h_sample1[15], h_sample2[15];  // 5 pixels * 3 channels

    // Host pinned memory for output results (grayscale + blurred)
    unsigned char* h_gray1, * h_blur1;
    unsigned char* h_gray2, * h_blur2;
    CUDA_CHECK(cudaHostAlloc((void**)&h_gray1, gray_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_blur1, blur_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_gray2, gray_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_blur2, blur_bytes, cudaHostAllocDefault));

    // Device memory
    unsigned char* d_rgb1, * d_gray1, * d_blur1;
    unsigned char* d_rgb2, * d_gray2, * d_blur2;
    CUDA_CHECK(cudaMalloc((void**)&d_rgb1, rgb_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_gray1, gray_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_blur1, blur_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_rgb2, rgb_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_gray2, gray_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_blur2, blur_bytes));

    // Create random test images on device using cuRAND
    // -----------------------------------------------------------------------
    create_test_images_curand(d_rgb1, d_rgb2, rgb_bytes, h_sample1, h_sample2);


    // Create streams
    // -----------------------------------------------------------------------
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Create synchronization events (gray-done gates)
    cudaEvent_t event1, event2;
    CUDA_CHECK(cudaEventCreate(&event1));
    CUDA_CHECK(cudaEventCreate(&event2));

    // Build NPP stream contexts
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    NppStreamContext nppCtx1, nppCtx2;

    // Stream 1 context
    nppCtx1.hStream = stream1;
    nppCtx1.nCudaDeviceId = dev;
    nppCtx1.nMultiProcessorCount = prop.multiProcessorCount;
    nppCtx1.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    nppCtx1.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    nppCtx1.nSharedMemPerBlock = prop.sharedMemPerBlock;
    nppCtx1.nCudaDevAttrComputeCapabilityMajor = prop.major;
    nppCtx1.nCudaDevAttrComputeCapabilityMinor = prop.minor;
    CUDA_CHECK(cudaStreamGetFlags(stream1, &nppCtx1.nStreamFlags));

    // Stream 2 context
    nppCtx2.hStream = stream2;
    nppCtx2.nCudaDeviceId = dev;
    nppCtx2.nMultiProcessorCount = prop.multiProcessorCount;
    nppCtx2.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    nppCtx2.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    nppCtx2.nSharedMemPerBlock = prop.sharedMemPerBlock;
    nppCtx2.nCudaDevAttrComputeCapabilityMajor = prop.major;
    nppCtx2.nCudaDevAttrComputeCapabilityMinor = prop.minor;
    CUDA_CHECK(cudaStreamGetFlags(stream2, &nppCtx2.nStreamFlags));

    // NPP image ROI (Region of Interest) - full image dimensions
    NppiSize roi = { width, height };

    // Convert RGB to grayscale
    // -----------------------------------------------------------------------
    rgb_to_gray_npp(d_rgb1, d_rgb2, d_gray1, d_gray2,
        rgb_step, gray_step, roi,
        nppCtx1, nppCtx2, event1, event2);

    // Blur grayscale images and copy results back to host
    // -----------------------------------------------------------------------
    blur_gray_npp(d_gray1, d_gray2, d_blur1, d_blur2,
        h_gray1, h_gray2, h_blur1, h_blur2,
        gray_step, roi, nppCtx1, nppCtx2,
        gray_bytes, blur_bytes);

    // Print sample output
    print_sample_rgb(h_sample1, "1");
    print_sample(h_gray1, "Gray");
    print_sample(h_blur1, "Blurred");
    print_sample_rgb(h_sample2, "2");
    print_sample(h_gray2, "Gray");
    print_sample(h_blur2, "Blurred");

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaEventDestroy(event1));
    CUDA_CHECK(cudaEventDestroy(event2));

    CUDA_CHECK(cudaFree(d_rgb1));
    CUDA_CHECK(cudaFree(d_rgb2));
    CUDA_CHECK(cudaFree(d_gray1));
    CUDA_CHECK(cudaFree(d_gray2));
    CUDA_CHECK(cudaFree(d_blur1));
    CUDA_CHECK(cudaFree(d_blur2));

    CUDA_CHECK(cudaFreeHost(h_gray1));
    CUDA_CHECK(cudaFreeHost(h_gray2));
    CUDA_CHECK(cudaFreeHost(h_blur1));
    CUDA_CHECK(cudaFreeHost(h_blur2));

    return 0;
}
