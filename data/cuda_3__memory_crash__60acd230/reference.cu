// file: solution.cu
#include "kernel_launch.h"
#include <cuda.h>
#include <cuda_runtime.h>

void launch(int gridSizeX, int blockSizeX, int gridSizeY, int blockSizeY, int gridSizeZ,
            int blockSizeZ)
{
    int *output, *input;
    dim3 threadsPerBlock(blockSizeX, blockSizeY, blockSizeZ);
    dim3 numBlocks(gridSizeX, gridSizeY, gridSizeZ);

    // Kernel invocation with runtime cluster size
    {
        cudaLaunchConfig_t config = {0};
        config.gridDim            = numBlocks;
        config.blockDim           = threadsPerBlock;

        cudaLaunchKernelEx(&config, kernel, output, input);
    }
}
