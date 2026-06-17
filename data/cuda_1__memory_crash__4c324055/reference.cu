// file: solution.cu
#include "kernel_launch.h"
#include "kernel_helpers.h"

void launch(int gridSizeX, int blockSizeX, int gridSizeY, int blockSizeY, int gridSizeZ,
            int blockSizeZ)
{
    int *output, *input;
    // Allocate shared memory
    int sharedMemory = blockSizeX * blockSizeY * blockSizeZ * sizeof(int);

    // Launch kernel
    kernel<<<dim3(gridSizeX, gridSizeY, gridSizeZ), dim3(blockSizeX, blockSizeY, blockSizeZ), sharedMemory>>>(output, input);
}
