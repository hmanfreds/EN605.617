/*
Date: 4/5/2026
Name: add_gaussian_noise_v3.cu

Description:
 Reads BMP images stored in the folder "pratt_fom/original", applies Gaussian
 noise at each level defined in the list NOISE_LEVELS[], and saves the noisy
 images to the folder "pratt_fom/noisy".

 5 files are created based on the pre-defined noise level 0, 5, 15, 30, 50.
 
 All folder paths and image file names come from configurations.json file.
 Folder structure:
 "pratt_fom/original": input folder containing the original BMP images
 "pratt_fom/noisy": output folder where the noisy files are stored
 
 If the "pratt_fom/noisy" folder does not exist it is created automatically.

 The output image file name will be appended according to the template:
 <original_image_file_name>_noise_<level>.bmp
 
 Noise generation runs as a CUDA kernel (one thread per pixel channel).
 The noise is generated using the CUDA library cuRAND.

 Requires C++17 for std::filesystem (MSVC 2017+, GCC 8+, Clang 7+).
 
 Build:
 nvcc -O2 -std=c++17 add_gaussian_noise.cu -o add_gaussian_noise -lcurand
*/


#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "helper_functions.h"   // functions to read/write BMP and load JSON config

namespace fs = std::filesystem;


//
// CONFIGURATIONS
//

// JSON file name
static const char CONFIG_FILE[] = "configurations.json";

// JSON section names applicable to this program
static const char SEC_ORIGINAL[] = "testing_image_original";  // where the original testing images are located (input)
static const char SEC_NOISY[] = "testing_image_noisy";  // where the noisy images will be saved (output)

// Noise levels in terms of Gaussian std-dev in pixel intensity units [0, 255]
static const float NOISE_LEVELS[] = { 0.0f, 5.0f, 15.0f, 30.0f, 50.0f };
static const int   NUM_LEVELS = (int)(sizeof(NOISE_LEVELS) / sizeof(NOISE_LEVELS[0]));


//
// CUDA KERNEL – add Gaussian noise to each image pixel independently
//

__global__ void gaussianNoiseKernel(const uint8_t* __restrict__ d_in,  // d_in input pixel pointer 
	uint8_t* __restrict__ d_out,  // d_out output pixel pointer
	int n,  // total number of bytes (width * height * channels)
	float sigma,  // Gaussian noise standard deviation in pixel intensity units
	unsigned long long seed)  // seed to random number generation
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // global index calculation
    if (idx >= n) return;  // bound check

    curandState state;  // allocates memory for cuRAND keep track of states
    curand_init(seed, (unsigned long long)idx, 0ULL, &state);  // initialize cuRAND

	const float noisy = (float)d_in[idx] + sigma * curand_normal(&state); // add Gaussian noise to the input pixel value
    d_out[idx] = (uint8_t)fminf(255.0f, fmaxf(0.0f, roundf(noisy))); // clamp pixel value between 0 and 255
}


//
// MAIN
//

int main() {

    // Load folder paths and file list from JSON configuration
    std::string inputDir, outputDir;
    std::vector<std::string> imageNames;

    // Load images
    if (!loadImagePaths(CONFIG_FILE, SEC_ORIGINAL, "original_images",
        SEC_NOISY, inputDir, imageNames, outputDir)) {
        return 1;
    }

    // Loop over each image loaded and add Gaussian noise

    for (const auto& imgName : imageNames) {

        // Build full input path using std::filesystem (cross-platform join).
        fs::path srcPath = fs::path(inputDir) / imgName;
        srcPath.make_preferred();  // normalize separators for the current OS

        // Load the image
        Image src;
        if (!loadBMP(srcPath, src)) {
            fprintf(stderr, "  Skipping '%s' (load failed)\n",
                srcPath.string().c_str());
            continue;
        }

        // Print image info
        const int n = src.width * src.height * src.channels;
        printf("\n  Adding Gaussian Noise to '%s'  (%dx%d, %dch, %d bytes)\n",
            srcPath.string().c_str(), src.width, src.height, src.channels, n);

        // Allocate memory on the device and copy image from host to device 
        uint8_t* d_in = nullptr, * d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, n));
        CUDA_CHECK(cudaMalloc(&d_out, n));
        CUDA_CHECK(cudaMemcpy(d_in, src.data.data(), n, cudaMemcpyHostToDevice));

        // Use the number of threads per block as 256 and calculate the grid size
        const int BLOCK = 256;
        const int grid = (n + BLOCK - 1) / BLOCK;

        // Extract string from the input image file name
        const std::string stem = srcPath.stem().string();

        // Create images with the Gaussian noise selected
        for (int li = 0; li < NUM_LEVELS; ++li) {
            const float sigma = NOISE_LEVELS[li];

            // Unique seed per noise level selected from the list
            const unsigned long long seed =
                0x1234 ^ (unsigned long long)((li + 1) * 1000003ULL);

			// Launch the kernel to add Gaussian noise to the image
            gaussianNoiseKernel << <grid, BLOCK >> > (d_in, d_out, n, sigma, seed);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

			// Copy the noisy image back to host and save it
            Image dst = src;
            CUDA_CHECK(cudaMemcpy(dst.data.data(), d_out, n,
                cudaMemcpyDeviceToHost));

            // Convert noise level from float to integer for the file name
            char levelStr[32];
            if (sigma == floorf(sigma))
                snprintf(levelStr, sizeof(levelStr), "%.0f", sigma);
            else
                snprintf(levelStr, sizeof(levelStr), "%g", sigma);

            // Save noisy image to file
            fs::path outPath = fs::path(outputDir) /
                (stem + "_noise_" + levelStr + ".bmp");
            outPath.make_preferred();

            if (saveBMP(outPath, dst))
                printf("    -> '%s'  (sigma = %.4g)\n",
                    outPath.string().c_str(), sigma);
        }

		// Free device memory
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }
    return 0;
}
