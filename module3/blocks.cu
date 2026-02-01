// Module 3 - Practical Activities - 2/1/2026
// Student: Herbert Schmidmeier
//
// Modify the blocks.cu CUDA code, in the module3 folder in the course git project,
// to execute at least 5 different numbers of threads, with various block sizes, numbers of blocks,
// and grid dimensions.


#include <stdio.h>

#define ARRAY_SIZE 256
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

/* Declare  statically two arrays of ARRAY_SIZE each */
unsigned int cpu_block[ARRAY_SIZE];
unsigned int cpu_thread[ARRAY_SIZE];


// This kernel copies the block and thread indices into two arrays
__global__
void what_is_my_id(unsigned int * block, unsigned int * thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
}


void main_sub0()
{
    // 5 variations of block sizes
    const unsigned int block_size[5] = {16,32,64,128,256};   // threads per block - array size is fixed to 256

    // Iterate through combinations
    for (int i = 0; i < 5; i++) {

	    // Declare pointers for GPU based params
	    unsigned int *gpu_block;
	    unsigned int *gpu_thread;

        // Allocate memory for arrays in the GPU
	    cudaMalloc((void **)&gpu_block, ARRAY_SIZE_IN_BYTES);
	    cudaMalloc((void **)&gpu_thread, ARRAY_SIZE_IN_BYTES);

        // Copy arrays from CPU memory to GPU memory
	    cudaMemcpy( cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );
	    cudaMemcpy( cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice );

		// Print combination info
	    const unsigned int num_blocks = ARRAY_SIZE/block_size[i];
	    const unsigned int num_threads = ARRAY_SIZE/num_blocks;

	    // Execute our kernel
	    what_is_my_id<<<num_blocks, num_threads>>>(gpu_block, gpu_thread);

	    // Free the arrays on the GPU as now we're done with them
	    cudaMemcpy( cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );
	    cudaMemcpy( cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost );

        // Free GPU memory
	    cudaFree(gpu_block);
	    cudaFree(gpu_thread);

        // Print combination info
        printf("\nCombination %d\nArray size: %u - Blocks: %u - Threads per Block: %u - Total Threads Launched: %u\n\n",
               i + 1, ARRAY_SIZE, num_blocks, block_size[i], num_blocks * block_size[i] );

	    // Iterate through the arrays and print
	    for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	    {
		    printf("Thread: %2u - Block: %2u\n",cpu_thread[i],cpu_block[i]);
	    }
    }
}



int main()
{
	main_sub0();

	return EXIT_SUCCESS;
}
