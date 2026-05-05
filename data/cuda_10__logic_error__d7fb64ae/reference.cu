#include "gpu_recursive_reduce.h"

__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata, unsigned int isize)
{
    // Calculate thread ID for the block
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Dynamically allocated shared memory for partial sums
    extern __shared__ int sdata[];

    // Load data from global memory into shared memory
    if (idx < isize)
    {
        sdata[tid] = g_idata[idx];
    }
    else
    {
        sdata[tid] = 0;
    }
    __syncthreads();

    // Perform the tree-based reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Unroll the last warp to avoid extra synchronization
    if (tid < 32)
    {
        volatile int *smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }

    // Write the result from shared memory to global memory
    // Only the first thread writes the result
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }

    // If the grid size is larger than 1 block, we need to launch another
    // reduction to summarize all blocks' results. We do this only in the first block.
    if (blockIdx.x == 0 && gridDim.x > 1)
    {
        int nthreads = (isize + blockDim.x - 1) / blockDim.x;
        nthreads     = (nthreads + 1) / 2;   // next power of two for threads
        if (nthreads > 1)
        {
            gpuRecursiveReduce<<<1, nthreads, nthreads * sizeof(int)>>>(g_odata, g_odata,
                                                                        gridDim.x);
        }
        else
        {
            if (tid == 0) g_odata[0] = sdata[0];
        }
    }
}