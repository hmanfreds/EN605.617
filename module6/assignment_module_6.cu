// Module 6 - Assignment
// Date: 2/26/2026
// Student: Herbert Schmidmeier


/* Program to convert two randomly color BMP images to grayscale followed by blurring it. The test images 
are created as 1D vectors with the RGB pixles interleaved such as RGBRGB... where each pixel is represented
by 3 bytes (R, G, B). Each grayscale pixel uses 1 byte. Another kernel blurs the grayscale vector by averaging 
a 3x3 area around the pixel being processed. This code uses 2 streams and 2 events to garantee that the grayscale
conversions are completed before the blurring starts. The elapsed time for each stream is measured separately*/


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



// Convert RGB to grayscale
__global__ void rgb_to_gray(
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

            // Convert to grayscale and clamp to [0,255] range.
            float acc = 0.299f * r + 0.587f * g + 0.114f * b;  // reading gray_coeff from global mem
            gray[y * width + x] = convert_float_char(acc);
        }
    }
}


// Blur gayscale image using 3x3 box average
__global__ void blur_gray(
    const unsigned char* gray_in,  // input array: grayscale (1 byte per pixel)
    unsigned char* gray_out,  // output array: blurred grayscale (1 byte per pixel)
    int width, int height)  // image dimensions
{
    // 2D grid-stride loop
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += gridDim.y * blockDim.y)
    {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += gridDim.x * blockDim.x)
        {
            float sum = 0.f;
            int count = 0;
			// Loop over 3x3 neighborhood
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
					// Make sure pixels are inside the image boundaries
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        sum += (float)gray_in[ny * width + nx];
                        count++;  // count the number of pixels to average
                    }
                }
            }
            gray_out[y * width + x] = convert_float_char(sum / count);  // calculate average
        }
    }
}



void create_image(unsigned char* h_rgb, size_t pixels) {
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
}

void print_input_values(int width, int height, int threads_per_block, int tile_x, int tile_y, int grid_x, int grid_y)
{
    // Print values used for calculations
    printf("\nValues Used:\n");
    printf("Width  : %d\n", width);
    printf("Height : %d\n", height);
    printf("Image pixels: %d\n", width * height);
    printf("Number of blocks: %d\n", grid_x * grid_y);
    printf("Threads per block: %d\n", threads_per_block);
    printf("Tile (Block) dimensions: (%d x %d)\n", tile_x, tile_y);
}

void print_sample_rgb(unsigned char* h_rgb, const std::string& text) 
{
    // Print sample RGB values
    printf("\nTest Image %s - Sample RGB values (RGB):",text.c_str());
    printf("\nRED pixels:   %u %u %u %u %u", h_rgb[0], h_rgb[3], h_rgb[6], h_rgb[9], h_rgb[12]);
    printf("\nGREEN pixels: %u %u %u %u %u", h_rgb[1], h_rgb[4], h_rgb[7], h_rgb[10], h_rgb[13]);
    printf("\nBLUE pixels:  %u %u %u %u %u\n", h_rgb[2], h_rgb[5], h_rgb[8], h_rgb[11], h_rgb[14]);
}

void print_sample(unsigned char* h_gray, const std::string& text) 
{
	// Print sample of converted data - either grayscale or blurred values
    printf("%s pixels: %u %u %u %u %u\n", text.c_str(), h_gray[0], h_gray[1], h_gray[2], h_gray[3], h_gray[4]);
}

void print_grid_block_config(int width, int height, dim3 grid, dim3 block) 
{
    // Print image size and grid/block configuration
    printf("\nImage: %d x %d\n", width, height);
    printf("Grid: (%d, %d)  Block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);
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
    print_input_values(width, height, threads_per_block, tile_x, tile_y, grid_x, grid_y);

	// Grid and block dimensions
    //dim3 block(tile_x, tile_y);
    //dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    dim3 block(tile_x, tile_y);
    dim3 grid(grid_x, grid_y);

	// Print image size and grid/block configuration
    print_grid_block_config(width, height, grid, block);
    
    // Calculate memory sizes
    const size_t pixels = (size_t)width * (size_t)height;
    const size_t rgb_bytes = pixels * 3 * sizeof(unsigned char);
    const size_t gray_bytes = pixels * sizeof(unsigned char);
    const size_t blur_bytes = pixels * sizeof(unsigned char);

	// Create pointers for host memory
    unsigned char* h_rgb1;
    unsigned char* h_gray1;
    unsigned char* h_blur1;
    unsigned char* h_rgb2;
    unsigned char* h_gray2;
    unsigned char* h_blur2;

    // Allocate pinned memory on host
    CUDA_CHECK(cudaHostAlloc((void**)&h_rgb1, rgb_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_gray1, gray_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_blur1, blur_bytes, cudaHostAllocDefault));

    CUDA_CHECK(cudaHostAlloc((void**)&h_rgb2, rgb_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_gray2, gray_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_blur2, blur_bytes, cudaHostAllocDefault));

    // Create pointers for device memory
    unsigned char* d_rgb1;
    unsigned char* d_gray1;
    unsigned char* d_blur1;
    unsigned char* d_rgb2;
    unsigned char* d_gray2;
    unsigned char* d_blur2;

	// Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_rgb1, rgb_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_gray1, gray_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_blur1, blur_bytes));

    CUDA_CHECK(cudaMalloc((void**)&d_rgb2, rgb_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_gray2, gray_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_blur2, blur_bytes));

    // Create a random test image with the function create_image
    create_image(h_rgb1, pixels);
    create_image(h_rgb2, pixels);

	// Create Streams
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // Create Events
    cudaEvent_t event1, event2;
    CUDA_CHECK(cudaEventCreate(&event1));
    CUDA_CHECK(cudaEventCreate(&event2));

    // Copy color images to device
    CUDA_CHECK(cudaMemcpyAsync(d_rgb1, h_rgb1, rgb_bytes, cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_rgb2, h_rgb2, rgb_bytes, cudaMemcpyHostToDevice, stream2));

	// Create events for timing
    cudaEvent_t start1, stop1, start2, stop2;
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));
    CUDA_CHECK(cudaEventRecord(start1, stream1));
    CUDA_CHECK(cudaEventRecord(start2, stream2));

	// Lauch Kernels to convert RGB to grayscale in parallel on two streams
    rgb_to_gray <<<grid, block, 0, stream1 >>> (d_rgb1, d_gray1, width, height);
    rgb_to_gray <<<grid, block, 0, stream2 >>> (d_rgb2, d_gray2, width, height);
    CUDA_CHECK(cudaGetLastError());

    // Record “gray done” events
    CUDA_CHECK(cudaEventRecord(event1, stream1));
    CUDA_CHECK(cudaEventRecord(event2, stream2));

    // Make sure blur waits for gray completion
    CUDA_CHECK(cudaStreamWaitEvent(stream1, event1, 0));
    CUDA_CHECK(cudaStreamWaitEvent(stream2, event2, 0));

	// Lauch Kernels to blur grayscale images in parallel on two streams
    blur_gray <<<grid, block, 0, stream1 >>> (d_gray1, d_blur1, width, height);
	blur_gray <<<grid, block, 0, stream2 >>> (d_gray2, d_blur2, width, height);
    CUDA_CHECK(cudaGetLastError());

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop1, stream1));
    CUDA_CHECK(cudaEventRecord(stop2, stream2));
    CUDA_CHECK(cudaEventSynchronize(stop1));
    CUDA_CHECK(cudaEventSynchronize(stop2));
    CUDA_CHECK(cudaGetLastError());

	// Calculate elapsed time for each stream
    float ms1 = 0.f, ms2 = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms1, start1, stop1));
    CUDA_CHECK(cudaEventElapsedTime(&ms2, start2, stop2));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));

    // Copy grayscale image from device to host
    CUDA_CHECK(cudaMemcpyAsync(h_gray1, d_gray1, gray_bytes, cudaMemcpyDeviceToHost, stream1));
    CUDA_CHECK(cudaMemcpyAsync(h_gray2, d_gray2, gray_bytes, cudaMemcpyDeviceToHost, stream2));

	// Copy blurred image from device to host
    CUDA_CHECK(cudaMemcpyAsync(h_blur1, d_blur1, blur_bytes, cudaMemcpyDeviceToHost, stream1));
    CUDA_CHECK(cudaMemcpyAsync(h_blur2, d_blur2, blur_bytes, cudaMemcpyDeviceToHost, stream2));

    // Synchronize streams to ensure kernels have finished before copying results back to host
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // Print images samples
    print_sample_rgb(h_rgb1, "1");
    print_sample(h_gray1, "Gray");
    print_sample(h_blur1, "Blurred");
    print_sample_rgb(h_rgb2, "2");
    print_sample(h_gray2, "Gray");
    print_sample(h_blur2, "Blurred");

	// Print runtimes
    printf("\nStream1 kernel runtime: %.3f ms\n", ms1);
    printf("Stream2 kernel runtime: %.3f ms\n", ms2);
    printf("Total (approx max of both): %.3f ms\n", (ms1 > ms2 ? ms1 : ms2));

	// Destroy Streams and Events
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaEventDestroy(event1));
    CUDA_CHECK(cudaEventDestroy(event2));

	// Free DEVICE memory
    CUDA_CHECK(cudaFree(d_rgb1));
    CUDA_CHECK(cudaFree(d_gray1));
    CUDA_CHECK(cudaFree(d_blur1));
    CUDA_CHECK(cudaFree(d_rgb2));
    CUDA_CHECK(cudaFree(d_gray2));
    CUDA_CHECK(cudaFree(d_blur2));

    // Free HOST memory
    CUDA_CHECK(cudaFreeHost(h_rgb1));
    CUDA_CHECK(cudaFreeHost(h_gray1));
    CUDA_CHECK(cudaFreeHost(h_blur1));
    CUDA_CHECK(cudaFreeHost(h_rgb2));
    CUDA_CHECK(cudaFreeHost(h_gray2));
    CUDA_CHECK(cudaFreeHost(h_blur2));

    return 0;
}