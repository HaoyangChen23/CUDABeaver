#include "include/heatmap.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void k_generateHeatmap(float * input_d,
                                  unsigned char * output_d,
                                  const float minValue,
                                  const float maxValue,
                                  const int numElementsX,
                                  const int numElementsY)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalElements = numElementsX * numElementsY;

    if (idx >= totalElements) return;

    float v = input_d[idx];

    if (v <= minValue) {
        output_d[idx] = 0;
        return;
    }

    if (v >= maxValue) {
        output_d[idx] = 255;
        return;
    }

    const float range = maxValue - minValue;
    if (range <= 0.0f) {
        output_d[idx] = 0;
        return;
    }

    const float normalized = (v - minValue) / range;
    const float mapped = normalized * 255.0f;
    int rounded = (int)floorf(mapped + 0.5f);

    if (rounded < 0) rounded = 0;
    if (rounded > 255) rounded = 255;

    output_d[idx] = (unsigned char)rounded;
}