// Module 3 - Assignment - 2/4/2026
// Student: Herbert Schmidmeier

// Compile: nvcc -cudart=static assignment.cu -o assignment.exe



#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>   
#include <cstddef>   


/* Transpose matrix A into AT on CPU */
void transpose_cpu(const int* A, int* AT, int m, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            AT[j * m + i] = A[i * n + j];
         }
    }
}


/* CUDA kernel to transpose matrix A into AT */
__global__ void transpose_gpu(const int* A, int* AT, int m, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..n-1
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0..m-1

    if (row < m && col < n) {
        AT[col * m + row] = A[row * n + col];
    }
}



/* Transpose matrix A into AT on CPU with branching */
void transpose_cpu_branching(const int* A, int* AT, int m, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {

            int v = 1;
            if (j % 2 == 1) {  // Match odd columns
                for (int k = 1; k <= 150; ++k) {  // Introduce looping and some calculations
                    int d = k + 1;
                    v = v / d;
                    v += k;
                }
                AT[j * m + i] = A[i * n + j];
            } else {
                AT[j * m + i] = A[i * n + j];
            }
        }
    }
}

/* CUDA kernel to transpose matrix A into AT with branching */
__global__ void transpose_gpu_branching(
    const int* A, int* AT, int m, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        AT[col * m + row] = A[row * n + col];

        int v = 1;        
        if (threadIdx.x % 2 == 1) {  // Match odd lanes
            for (int i = 1; i <= 150; ++i) {  // Introduce looping and some calculations
                int d = i + 1;      
                v = v / d;          
                v += i;             
            }
            AT[col * m + row] = A[row * n + col];
            }
        else {
            AT[col * m + row] = A[row * n + col];
            }
     }
}






int main(int argc, char** argv)
{
    // Matrix dimensions
    int m = 5000;
    int n = 5000;

    int totalThreads = (1 << 20); // default: 1M
    int blockSize    = 256;       // default

    if (argc >= 2) {  // Read total threads from command line
        totalThreads = std::atoi(argv[1]);
    }
    if (argc >= 3) {  // Read block size from command line
        blockSize = std::atoi(argv[2]);
    }
       if (argc >= 4) { // Read matrix dimension from command line
        m = std::atoi(argv[3]);
        n = std::atoi(argv[3]);
    }

    // Calculate the square matrix dimension if the dimension is not provided
    if (argc < 4) {
        int dimension = static_cast<int>(std::sqrt(static_cast<double>(totalThreads)));
        m = dimension;
        n = dimension;
    }

    // Validate totalThreads and Matrix size
        if (totalThreads < m * n) {
        std::cerr << "Warning: Total threads (" << totalThreads << ") is less than total matrix elements (" << (m * n) << "). Increase the totalThreads or decrease the matrix dimension.\n";
        std::cerr << "If the matrix dimension is omitted the matrix size will be calculated automatically to accomodate the totalThreads selected.\n";
        return 1; // Exit with error code
        }

    // Print the total number of threads and block size selected
    std::cout << "\nTotal number of threads selected: " << totalThreads << "\n";
    std::cout << "Block size selected: " << blockSize << "\n";
    std::cout << "Square Matrix Dim: " << m << " x " << n << "\n\n";
    
    // Convert 1D blockSize to 2D block dimensions
    int blockDim = static_cast<int>(std::floor(std::sqrt(blockSize)));

    // Ensure blockDim is a power of 2 and does not exceed 32
    if (blockDim > 32) {
        blockDim = 32;
    }
    else if (blockDim != 8 && blockDim != 16 && blockDim != 32) {
        blockDim++;
    }

    // Calculate grid dimensions based on block dimensions and matrix size
    dim3 block(blockDim, blockDim);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    // Print block and grid dimensions
    std::cout << "Using 2D block: " << blockDim << "x" << blockDim << " = " << (blockDim * blockDim) << " threads/block\n";
    std::cout << "Grid size : "<< grid.x << " x " << grid.y << " x " << grid.z << " blocks\n\n";

    // Host buffers
    std::vector<int> A(m * n);  // Original matrix
    std::vector<int> AT_cpu(m * n);  // Transposed matrix on CPU
    std::vector<int> AT_gpu(m * n);  // Transposed matrix on GPU

    // Populate matrix A with increasing integers
    for (int i = 0; i < m*n; i++) {
        A[i] = i + 1; }


    // ====== MATRIX TRANSPOSE ======

    std::cout << "Matrix Transpose:\n";

    // ====== Execute matrix transpose on CPU ======
    auto startCPU1 = std::chrono::high_resolution_clock::now();
    transpose_cpu(A.data(), AT_cpu.data(), m, n);  // Call CPU function
    auto stopCPU1 = std::chrono::high_resolution_clock::now();

    auto cpu1ElapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(stopCPU1 - startCPU1);
    std::cout << "Time elapsed CPU = " << cpu1ElapsedTime.count() << " ns\n";


    // ====== Execute matrix transpose on GPU ======

    int *d_A;  // Create device pointer for matrix A
    int *d_AT;  // Create device pointer for transposed matrix AT

    cudaMalloc((void**)&d_A, m * n * sizeof(int));  // Allocate memory on device for matrix A
    cudaMalloc((void**)&d_AT, m * n * sizeof(int)); // Allocate memory on device for transposed matrix AT

    cudaMemcpy(d_A, A.data(), m * n * sizeof(int), cudaMemcpyHostToDevice);   // Copy matrix A from Host to Device

    //dim3 block(16, 16);  // Create 16x16 threads per block
    //dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);  // Create grid

    auto startGPU1 = std::chrono::high_resolution_clock::now();
    transpose_gpu<<<grid, block>>>(d_A, d_AT, m, n);  // Launch kernel
    cudaDeviceSynchronize();
    auto stopGPU1 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(AT_gpu.data(), d_AT, m * n * sizeof(int), cudaMemcpyDeviceToHost);  // Copy transposed matrix from device to host

    cudaFree(d_A);  // Free device memory for matrix A
    cudaFree(d_AT); // Free device memory for transposed matrix AT

    auto gpu1ElapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(stopGPU1 - startGPU1);
    std::cout << "Time elapsed GPU = " << gpu1ElapsedTime.count() << " ns\n";

    double acceleration1 = static_cast<double>(cpu1ElapsedTime.count()) / static_cast<double>(gpu1ElapsedTime.count());
    std::cout << "GPU Acceleration = " << acceleration1 << "x\n\n";



    // ====== MATRIX TRANSPOSE with BRANCHING ======

    std::cout << "Matrix Transpose with Branching:\n";

    // ====== Execute matrix transpose with branching on CPU ======
    auto startCPU2 = std::chrono::high_resolution_clock::now();
    transpose_cpu_branching(A.data(), AT_cpu.data(), m, n);
    auto stopCPU2 = std::chrono::high_resolution_clock::now();

    auto cpu2ElapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(stopCPU2 - startCPU2);
    std::cout << "Time elapsed CPU = " << cpu2ElapsedTime.count() << " ns\n";


    // ====== Execute matrix transpose with branching on GPU ======

    //int *d_A;  // Create device pointer for matrix A
    //int *d_AT;  // Create device pointer for transposed matrix AT

    cudaMalloc((void**)&d_A, m * n * sizeof(int));  // Allocate memory on device for matrix A
    cudaMalloc((void**)&d_AT, m * n * sizeof(int)); // Allocate memory on device for transposed matrix AT

    cudaMemcpy(d_A, A.data(), m * n * sizeof(int), cudaMemcpyHostToDevice);   // Copy matrix A from Host to Device

    // dim3 block(16, 16);  // Create 16x16 threads per block
    // dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);  // Create grid

    auto startGPU2 = std::chrono::high_resolution_clock::now();
    transpose_gpu_branching<<<grid, block>>>(d_A, d_AT, m, n);  // Launch kernel
    cudaDeviceSynchronize();
    auto stopGPU2 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(AT_gpu.data(), d_AT, m * n * sizeof(int), cudaMemcpyDeviceToHost);  // Copy transposed matrix from device to host

    cudaFree(d_A);  // Free device memory for matrix A
    cudaFree(d_AT); // Free device memory for transposed matrix AT

    auto gpu2ElapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(stopGPU2 - startGPU2);
    std::cout << "Time elapsed GPU = " << gpu2ElapsedTime.count() << " ns\n";

    double acceleration2 = static_cast<double>(cpu2ElapsedTime.count()) / static_cast<double>(gpu2ElapsedTime.count());
    std::cout << "GPU Acceleration = " << acceleration2 << "x\n\n";

    return 0;
}
