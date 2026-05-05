#include "heatmap.h"

__global__ void k_generateHeatmap(float * input_d, 
                                  unsigned char * output_d, 
                                  const float minValue, 
                                  const float maxValue,
                                  const int numElementsX,
                                  const int numElementsY) {
    const int threadX = threadIdx.x + blockIdx.x * blockDim.x;
    const int threadY = threadIdx.y + blockIdx.y * blockDim.y;
    const int strideSizeX = blockDim.x * gridDim.x;
    const int strideSizeY = blockDim.y * gridDim.y;
    const int numStrideLoopIterationsX = 1 + (numElementsX - 1) / strideSizeX;
    const int numStrideLoopIterationsY = 1 + (numElementsY - 1) / strideSizeY;
    const float reciprocalDifference = 1.0f / (maxValue - minValue);
    for(int strideY = 0; strideY < numStrideLoopIterationsY; strideY++) {
        for(int strideX = 0; strideX < numStrideLoopIterationsX; strideX++) {
            const int itemX = threadX + (strideX * strideSizeX);
            const int itemY = threadY + (strideY * strideSizeY);
            const int itemId = itemX + itemY * numElementsX;
            if(itemX < numElementsX && itemY < numElementsY) {
                const float data = input_d[itemId];
                // Clamping and normalizing the input.
                const float clampedData = fminf(fmaxf(data, minValue), maxValue);
                const float normalizedData = (clampedData - minValue) * reciprocalDifference;
                // In order to ensure that the result aligns with the designated output range, it is essential to round it to the nearest integer.
                const unsigned char output = round(normalizedData * SCALING_VALUE);
                output_d[itemId] = output;
            }
        }
    }
}