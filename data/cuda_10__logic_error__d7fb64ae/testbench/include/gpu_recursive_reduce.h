#ifndef GPU_RECURSIVE_REDUCE_H
#define GPU_RECURSIVE_REDUCE_H

__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata, unsigned int isize);

#endif // GPU_RECURSIVE_REDUCE_H