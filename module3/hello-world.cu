// Module 3 - Practical Activities - 2/1/2026
// Student: Herbert Schmidmeier
// Modify the hello-world.cu CUDA code to execute 5 different numbers of threads, with various block sizes and numbers of blocks.

// To compile execute below:
// nvcc hello-world.cu -L /usr/local/cuda/lib -lcudart -o hello-world

#include <stdio.h>



// This kernel copies the block and thread indices into two arrays
__global__ 
void hello(unsigned int * block, unsigned int * thread)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[idx] = blockIdx.x;				// copy block index
	thread[idx] = threadIdx.x;				// copy thread index
}


void main_sub()
{
    // 5 variations: change array size and block size together
    const unsigned int n[5]         = {16,16,32,32,64};  // total number of elements in the array
    const unsigned int block_size[5] = {4,8,16,32,32};   // threads per block

    // Iterate through combinations
    for (int i = 0; i < 5; i++)
    {
        const unsigned int N = n[i];
        const unsigned int BLOCK_SIZE = block_size[i];
        const unsigned int NUM_BLOCKS = N/BLOCK_SIZE;

        // Host arrays
        unsigned int* cpu_block  = (unsigned int*)calloc(N, sizeof(unsigned int));
        unsigned int* cpu_thread = (unsigned int*)calloc(N, sizeof(unsigned int));

        // Create pointers for arrays
        unsigned int* gpu_block;
        unsigned int* gpu_thread;

        // Allocate memory for arrays in the GPU
        cudaMalloc((void**)&gpu_block,  sizeof(unsigned int) * N);
        cudaMalloc((void**)&gpu_thread, sizeof(unsigned int) * N);

        // Copy arrays from CPU memory to GPU memory
        cudaMemcpy(gpu_block,  cpu_block,  sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_thread, cpu_thread, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);

        // Launch kernel
        hello<<<NUM_BLOCKS, BLOCK_SIZE>>>(gpu_block, gpu_thread);

        // Copy arrays from GPU memory to CPU memory
        cudaMemcpy(cpu_block,  gpu_block,  sizeof(unsigned int) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_thread, gpu_thread, sizeof(unsigned int) * N, cudaMemcpyDeviceToHost);

        // Free GPU memory
        cudaFree(gpu_block);
        cudaFree(gpu_thread);

        // Print combination info
        printf("\nCombination %d\nArray size: %u - Blocks: %u - Threads per Block: %u - Total Threads Launched: %u\n\n",
               i + 1, N, NUM_BLOCKS, BLOCK_SIZE, NUM_BLOCKS * BLOCK_SIZE);

        // Print the blocks and threads for the combination
        for (unsigned int j = 0; j < N; j++) {
            printf("Block: %2u  Thread: %2u\n", cpu_block[j], cpu_thread[j]);
        }
    }
}


int main()
{
	main_sub();

	return EXIT_SUCCESS;
}
