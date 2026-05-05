#ifndef HEATMAP_H
#define HEATMAP_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call){                                      \
    cudaError_t error = call;                                  \
    if(error != cudaSuccess){                                  \
        fprintf(stderr, "CUDA error at %s: %d - %s \n",        \
                __FILE__, __LINE__, cudaGetErrorString(error));\
        exit(EXIT_FAILURE);                                    \
    }                                                          \
}

constexpr int NUM_ELEMENTS_X = 10;
constexpr int NUM_ELEMENTS_Y = 1;
constexpr int NUM_TOTAL_ELEMENTS = NUM_ELEMENTS_X * NUM_ELEMENTS_Y;
constexpr int SCALING_VALUE = 255;
constexpr int NUM_GRID_BLOCKS_X = 32;
constexpr int NUM_GRID_BLOCKS_Y = 8;
constexpr int NUM_BLOCK_THREADS_X = 16;
constexpr int NUM_BLOCK_THREADS_Y = 16;

__global__ void k_generateHeatmap(float * input_d, 
                                  unsigned char * output_d, 
                                  const float minValue, 
                                  const float maxValue,
                                  const int numElementsX,
                                  const int numElementsY);

#endif // HEATMAP_H