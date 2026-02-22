// Module 5 - Assignment
// Date: 2/20/2026
// Student: Herbert Schmidmeier


/* Program to convert a color BMP image to grayscale using CUDA.The program produces a random test image
in the format of a 1D interleaved RGB array such as RGBRGBRGB..., where each pixel is represented by
3 bytes (R, G, B). The program then launches sequentially the 5 CUDA kernels. To improve the time measuring
there is a 3-loop warmup followed by the launch of the same kernel 20 times. The runtimes are averaged
and printed after all kernels are executed. Considering that registers can't store a good size image
the reading of the grayscale coefficients was used to access the performance of each memory type.*/



#include <cuda_runtime.h>
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


// Parameters for timing
static const int LOOP_ITERS = 500;  // number of loops inside the kernel to read the gray coeffients multiple times
static const int WARMUP = 3;  // number of warmup loops before the kernel execution timing measuaring
static const int ITERS = 20;   // number of loops to average the kernel execution runtimes


// Clamp negative values to 0, large values to 255, then convert to unsigned char
__device__ unsigned char convert_float_char(float v) {
    if (v < 0.f){
        v = 0.f;
    }
    else if (v > 255.f){
        v = 255.f;
	}
    return static_cast<unsigned char>(v + 0.5f);  // convert the float number to 8-bit integer
}

///////////////////// HOST MEMORY /////////////////////

// Convert RGB to grayscale using the host memory (mapped memory)
__global__ void rgb_to_gray_host_mem(
    const unsigned char* rgb,
    unsigned char* gray,
    int width, int height,
    const float* coeffs)   // pointer to mapped host memory
{
    const float cr = coeffs[0];
    const float cg = coeffs[1];
    const float cb = coeffs[2];

    // 2D grid-stride loop
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += gridDim.y * blockDim.y)
    {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += gridDim.x * blockDim.x)
        {
            int idx3 = (y * width + x) * 3;
            float r = (float)rgb[idx3 + 0];
            float g = (float)rgb[idx3 + 1];
            float b = (float)rgb[idx3 + 2];

            float acc = 0.f;
            for (int k = 0; k < LOOP_ITERS; k++) {
                // acc * 0.00001f must be added otherwise the compiler will optimize the loop and execute only once imparing timing
                acc = acc * 0.00001f + cr * r + cg * g + cb * b;  // reading gray_coeff from mapped host memory
            }
            gray[y * width + x] = convert_float_char(acc);
        }
    }
}


///////////////////// GLOBAL MEMORY /////////////////////

// Storing the grayscale coefficients in the GLOBAL memory
__device__  float gmem_coeff_r = 0.299f;
__device__  float gmem_coeff_g = 0.587f;
__device__  float gmem_coeff_b = 0.114f;


// Convert RGB to grayscale using GLOBAL memory for the coefficients
__global__ void rgb_to_gray_global_mem(
	const unsigned char* rgb,  // input array: packed RGB (3 bytes per pixel)
	unsigned char*  gray,  // output array: grayscale (1 byte per pixel)
	int width, int height)  // image dimensions
{

    // 2D grid-stride loop
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += gridDim.y * blockDim.y)
    {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += gridDim.x * blockDim.x)
        {
            int idx3 = (y * width + x) * 3;
            float r = (float)rgb[idx3 + 0];
            float g = (float)rgb[idx3 + 1];
            float b = (float)rgb[idx3 + 2];

            // Repeat loop to amplify runtime (prevent full optimization with dependency)
            float acc = 0.f;
            for (int k = 0; k < LOOP_ITERS; k++) {
                // acc * 0.00001f must be added otherwise the compiler will optimize the loop and execute only once imparing timing
                acc = acc * 0.00001f + gmem_coeff_r * r + gmem_coeff_g * g + gmem_coeff_b * b;  // reading gray_coeff from global mem
            }
            gray[y * width + x] = convert_float_char(acc);
        }
    }
}


///////////////////// CONSTANT MEMORY /////////////////////

// Storing the grayscale coefficients in the CONSTANT memory
__constant__  float const_coeff_r = 0.299f;
__constant__  float const_coeff_g = 0.587f;
__constant__  float const_coeff_b = 0.114f;


// Convert RGB to grayscale using CONSTANT memory for the coefficients
__global__ void rgb_to_gray_const_mem(
    const unsigned char* rgb,
    unsigned char* gray,
    int width, int height)
{
    // 2D grid - stride loop
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += gridDim.y * blockDim.y)
    {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += gridDim.x * blockDim.x)
        {
            int idx3 = (y * width + x) * 3;
            float r = (float)rgb[idx3 + 0];
            float g = (float)rgb[idx3 + 1];
            float b = (float)rgb[idx3 + 2];

            float acc = 0.f;
            for (int k = 0; k < LOOP_ITERS; k++) {
                // acc * 0.00001f must be added otherwise the compiler will optimize the loop and execute only once imparing timing
				acc = acc * 0.00001f + const_coeff_r * r + const_coeff_g * g + const_coeff_b * b;  // reading gray_coeff from constant mem
            }
            gray[y * width + x] = convert_float_char(acc);
        }
    }
}


///////////////////// SHARED MEMORY /////////////////////

// Convert RGB to grayscale using SHARED memory for the coefficients
__global__ void rgb_to_gray_shared_mem(
    const unsigned char* rgb,
    unsigned char* gray,
    int width, int height)
{
    // Declare variable in shared memory
    __shared__ float sh_coeff_r;
    __shared__ float sh_coeff_g;
    __shared__ float sh_coeff_b;

	// Load coefficients into shared memory once per block
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sh_coeff_r = 0.299f;
        sh_coeff_g = 0.587f;
        sh_coeff_b = 0.114f;
    }
	__syncthreads();  // ensure all threads see the initialized shared memory values before start running

    // 2D grid-stride loop
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += gridDim.y * blockDim.y)
    {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += gridDim.x * blockDim.x)
        {
            int idx3 = (y * width + x) * 3;
            float r = (float)rgb[idx3 + 0];
            float g = (float)rgb[idx3 + 1];
            float b = (float)rgb[idx3 + 2];

            float acc = 0.f;
            for (int k = 0; k < LOOP_ITERS; k++) {
                // acc * 0.00001f must be added otherwise the compiler will optimize the loop and execute only once imparing timing
                acc = acc * 0.00001f + sh_coeff_r * r + sh_coeff_g * g + sh_coeff_b * b; // reading gray_coeff from shared memory
            }
            gray[y * width + x] = convert_float_char(acc);
        }
    }
}


///////////////////// REGISTER /////////////////////

// Convert RGB to grayscale using REGISTER for the coefficients
__global__ void rgb_to_gray_registers(
    const unsigned char* __restrict__ rgb,
    unsigned char* __restrict__ gray,
    int width, int height)
{
    // Cache coeff in registers once per thread
    float reg_coeff_r = 0.299f;
    float reg_coeff_g = 0.587f;
    float reg_coeff_b = 0.114f;

    // 2D grid-stride loop
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += gridDim.y * blockDim.y)
    {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += gridDim.x * blockDim.x)
        {
            int idx3 = (y * width + x) * 3;
            float r = (float)rgb[idx3 + 0];
            float g = (float)rgb[idx3 + 1];
            float b = (float)rgb[idx3 + 2];

            // Dependent loop to avoid being optimized away and to stress registers
            float acc = 0.f;
            for (int k = 0; k < LOOP_ITERS; k++) {
                // acc * 0.00001f must be added otherwise the compiler will optimize the loop and execute only once imparing timing
                acc = acc * 0.00001f + reg_coeff_r * r + reg_coeff_g * g + reg_coeff_b * b;  // reading gray_coeff from registers
            }
            gray[y * width + x] = convert_float_char(acc);
        }
    }
}

// Timing function
template <typename Kernel, typename... Args>
float time_kernel_ms(Kernel k, dim3 grid, dim3 block, Args... args)
{
    // Warmup kernel launch loops
    for (int i = 0; i < WARMUP; i++) {
		k << <grid, block >> > (args...);  // run the same kernel a few times as warmup to improve time accuracy
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

	// Timed kernel launch loops
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++) {
        k << <grid, block >> > (args...);  // run the same kernel multiple times to get average runtime
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

	return ms / ITERS;  // calculate average time per kernel for all runs
}


int main(int argc, char** argv)
{
    // Default values
    int width = 7680;               // default image width
	int height = 4320;              // default image height
    int threads_per_block = 256;   // default block size
    int tile_x = 16;                // default tile width
	int tile_y = 16;                // default tile height
	int num_blocks_x = 0;             // number of blocks in x direction
	int num_blocks_y = 0;             // number of blocks in y direction
	bool threads_arg_used = false;      // flag to check if threads_per_block argument was used
	bool num_blocks_arg_used = false;  // flag to check if num_blocks argument was used

	// calculate the number of blocks based on the default image size and tile size
    int num_blocks = ((width + tile_x - 1) / tile_x) * ((height + tile_y - 1) / tile_y);
	int grid_x = (width + tile_x - 1) / tile_x;  // calculate number of blocks in x direction
	int grid_y = (height + tile_y - 1) / tile_y;  // calculate number of blocks in y direction


    // Parse command line arguments - all arguments are optional
    // -threads and -num_blocks are multually exclusive
    for (int i = 1; i < argc; i++)
    {
        if (std::strcmp(argv[i], "-threads") == 0 && i + 1 < argc) {
            threads_arg_used = true;
            threads_per_block = std::atoi(argv[++i]);

            // Protect against invalid CUDA block sizes
            if (threads_per_block < 1 || threads_per_block > 1024) {
                std::cerr << "Error: -threads must be between 1 and 1024. You entered " << threads_per_block << ".\n";
                return EXIT_FAILURE;
            }
            // Calculate the tile dimensions
			tile_x = static_cast<int>(std::sqrt(threads_per_block));  // square threads_per_block and convert to integer
			tile_y = threads_per_block / tile_x;
			// Calculate the number of blocks based on the tile size and image dimensions
			grid_x = (width + tile_x - 1) / tile_x;  // calculate number of blocks in x direction
			grid_y = (height + tile_y - 1) / tile_y;  // calculate number of blocks in y direction
			// Check if block size meets the threads_per_block requirement
            if (tile_x * tile_y < threads_per_block) {
                tile_y += 1;  // adjust tile_y if the product is less than threads_per_block
			}
        }
        else if (std::strcmp(argv[i], "-num_blocks") == 0 && i + 1 < argc) {
            num_blocks_arg_used = true;
            num_blocks = std::atoi(argv[++i]);
            // Calculate the tile dimensions
            num_blocks_x = static_cast<int>(std::sqrt(num_blocks));  // square threads_per_block and convert to integer
            num_blocks_y = num_blocks / num_blocks_x;
            // Check if num_blocks calculated meets the requirement
            if (num_blocks_x * num_blocks_y < num_blocks) {
                num_blocks_y += 1;  // adjust num_blocks_y if the product is less than num_blocks
            }
			// Calculate the tile dimensions based on the number of blocks and image dimensions
			tile_x = (width + num_blocks_x - 1) / num_blocks_x;  // calculate tile width based on number of blocks in x direction
			tile_y = (height + num_blocks_y - 1) / num_blocks_y;  // calculate tile height based on number of blocks in y direction
            grid_x = (width + tile_x - 1) / tile_x;  // calculate number of blocks in x direction
            grid_y = (height + tile_y - 1) / tile_y;  // calculate number of blocks in y direction
			// Clamp number of threads per block to maximum of 1024 if the calculated tile size exceeds the limit
            if (tile_x * tile_y > 1024) {
				tile_x = 32;  // adjust tile_x to maximum allowed by CUDA
				tile_y = 32;  // adjust tile_y to maximum allowed by CUDA
            }

            }

        else {
            std::cout << "Warning: Unknown or incomplete argument: "
                << argv[i] << std::endl;
        }
    }

	// If user specified both -threads and -num_blocks, print an error and exit since they are mutually exclusive
    if (threads_arg_used && num_blocks_arg_used) {
        std::cerr << "Error: Cannot use -threads and -num_blocks at the same time.\n";
        return EXIT_FAILURE;
    }

	// Print values used for calculations
	printf("\nValues Used:\n");
	printf("Width  : %d\n", width);
	printf("Height : %d\n", height);
	printf("Image pixels: %d\n", width * height);
	printf("Number of blocks: %d\n", grid_x * grid_y);
	printf("Threads per block: %d\n", threads_per_block);
	printf("Tile (Block) dimensions: (%d x %d)\n", tile_x, tile_y);

	// Grid and block dimensions
    //dim3 block(tile_x, tile_y);
    //dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    dim3 block(tile_x, tile_y);
    dim3 grid(grid_x, grid_y);

	// Print image size and grid/block configuration
    printf("\nImage: %d x %d\n", width, height);
    printf("Grid: (%d, %d)  Block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);
    
    // Calculate memory sizes
    const size_t pixels = (size_t)width * (size_t)height;
    const size_t rgb_bytes = pixels * 3 * sizeof(unsigned char);
    const size_t gray_bytes = pixels * sizeof(unsigned char);

	// Host pointers for test image and grayscale output
    unsigned char* h_rgb;
    unsigned char* h_gray;
    CUDA_CHECK(cudaMallocHost((void**)&h_rgb, rgb_bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_gray, gray_bytes));

	// Create a random test BMP image as 1D interleaved RGB array
	// Fill RGB with random data to simulate a real image
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 255);

	// Sample RGB values are interleaved as RGBRGB... for memory coalescing
    for (size_t i = 0; i < pixels; i++) {
        h_rgb[i * 3 + 0] = static_cast<unsigned char>(dist(rng));
        h_rgb[i * 3 + 1] = static_cast<unsigned char>(dist(rng));
        h_rgb[i * 3 + 2] = static_cast<unsigned char>(dist(rng));
    }

	// Print sample RGB values
	printf("\nTest Image Sample RGB values (RGB):");
    printf("\nTest Image Sample RED pixels:   %u %u %u %u %u", h_rgb[0], h_rgb[3], h_rgb[6], h_rgb[9], h_rgb[12]);
    printf("\nTest Image Sample GREEN pixels: %u %u %u %u %u", h_rgb[1], h_rgb[4], h_rgb[7], h_rgb[10], h_rgb[13]);
    printf("\nTest Image Sample BLUE pixels:  %u %u %u %u %u\n", h_rgb[2], h_rgb[5], h_rgb[8], h_rgb[11], h_rgb[14]);

	// Device pointers for RGB test image and grayscale output
    unsigned char* d_rgb;
    unsigned char* d_gray;

	// Allocate and copy BMP test image to device
    CUDA_CHECK(cudaMalloc(&d_rgb, rgb_bytes));
    CUDA_CHECK(cudaMalloc(&d_gray, gray_bytes));
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, rgb_bytes, cudaMemcpyHostToDevice));

	// Create mapped host memory for grayscale coefficients
    // Allocate 3 floats in pinned host memory that is mapped into the device address space
    float* h_coeffs;
    CUDA_CHECK(cudaHostAlloc((void**)&h_coeffs, 3 * sizeof(float), cudaHostAllocMapped));

    // Fill coefficients host memory
    h_coeffs[0] = 0.299f;
    h_coeffs[1] = 0.587f;
    h_coeffs[2] = 0.114f;

    // Get the device pointer that maps to that same host allocation
    float* d_coeffs;
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_coeffs, (void*)h_coeffs, 0));

    // Run all kernels
	// Run kernel using GLOBAL MEMORY memory for grayscale coefficients
    float t_global = time_kernel_ms(rgb_to_gray_global_mem, grid, block,
        d_rgb, d_gray, width, height);

    // Run kernel using CONSTANT MEMORY for grayscale coefficients
    float t_const = time_kernel_ms(rgb_to_gray_const_mem, grid, block,
        d_rgb, d_gray, width, height);

    // Run kernel using SHARED MEMORY for grayscale coefficients
    float t_shared = time_kernel_ms(rgb_to_gray_shared_mem, grid, block,
        d_rgb, d_gray, width, height);

	// Run kernel using REGISTER for grayscale coefficients
    float t_regs = time_kernel_ms(rgb_to_gray_registers, grid, block,
        d_rgb, d_gray, width, height);

	// Run kernel using HOST MEMORY (mapped memory) for grayscale coefficients
    float t_host = time_kernel_ms(rgb_to_gray_host_mem, grid, block,
        d_rgb, d_gray, width, height, d_coeffs);

    // Copy d_gray back from device to host
    CUDA_CHECK(cudaMemcpy(h_gray, d_gray, gray_bytes, cudaMemcpyDeviceToHost));

    // Print results
	printf("\nAverage kernel time (%d warmup, %d kernel runs, %d loops inside kernel):\n", WARMUP, ITERS, LOOP_ITERS);
	printf("Global Mem    : %.3f ms\n", t_global);
	printf("Constant Mem  : %.3f ms\n", t_const);
	printf("Shared Mem    : %.3f ms\n", t_shared);
	printf("Registers     : %.3f ms\n", t_regs);
    printf("Host Mem      : %.3f ms\n", t_host);

    // Print grayscale output sample
	printf("\nTest Image Sample Grayscale values:\n");
	printf("Test Image Sample GRAY pixels: %u %u %u %u %u\n", h_gray[0], h_gray[1], h_gray[2], h_gray[3], h_gray[4]);

	// Free memory on host
    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));
    return 0;
}