#include "gpu_recursive_reduce.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <nvtx3/nvToolsExt.h>

#define cudaCheckErrors(msg)                                                                   \
    do                                                                                         \
    {                                                                                          \
        cudaError_t __err = cudaGetLastError();                                                \
        if (__err != cudaSuccess)                                                              \
        {                                                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__);                                                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");                                        \
            exit(1);                                                                           \
        }                                                                                      \
    }                                                                                          \
    while (0)

void initializeArray(int *data, int size)
{
    // set random seed once before loop
    srand(42);
    for (int i = 0; i < size; i++)
    {
        data[i] = rand() % 100;
    }
}

int cpuReduce(int *data, int size)
{
    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += data[i];
    }
    return sum;
}

int launch(void)
{
    int isize    = 1 << 20;   // 2^20 elements
    int *h_idata = (int *)malloc(isize * sizeof(int));
    int *h_odata = (int *)malloc(isize * sizeof(int));
    int *d_idata, *d_odata;

    initializeArray(h_idata, isize);

    int cpu_sum = cpuReduce(h_idata, isize);

    cudaMalloc(&d_idata, isize * sizeof(int));
    cudaMalloc(&d_odata, isize * sizeof(int));

    cudaMemcpy(d_idata, h_idata, isize * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (isize + threads - 1) / threads;
    gpuRecursiveReduce<<<blocks, threads, threads * sizeof(int)>>>(d_idata, d_odata, isize);
    cudaCheckErrors("Kernel launch failure");

    cudaMemcpy(h_odata, d_odata, blocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy failure");

    int gpu_sum = cpuReduce(h_odata, blocks);

    assert(cpu_sum == gpu_sum);

    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    return 0;
}

int benchmark() {
    int isize    = 1 << 24;   // 2^24 elements for heavier workload
    int *h_idata = (int *)malloc(isize * sizeof(int));
    int *d_idata, *d_odata;

    initializeArray(h_idata, isize);

    cudaMalloc(&d_idata, isize * sizeof(int));
    cudaMalloc(&d_odata, isize * sizeof(int));

    int threads = 256;
    int blocks  = (isize + threads - 1) / threads;

    // Warmup: 3 iterations
    for (int i = 0; i < 3; i++)
    {
        cudaMemcpy(d_idata, h_idata, isize * sizeof(int), cudaMemcpyHostToDevice);
        gpuRecursiveReduce<<<blocks, threads, threads * sizeof(int)>>>(d_idata, d_odata, isize);
        cudaDeviceSynchronize();
    }

    // Timed region: 100 iterations
    nvtxRangePushA("bench_region");
    for (int i = 0; i < 100; i++)
    {
        cudaMemcpy(d_idata, h_idata, isize * sizeof(int), cudaMemcpyHostToDevice);
        gpuRecursiveReduce<<<blocks, threads, threads * sizeof(int)>>>(d_idata, d_odata, isize);
        cudaDeviceSynchronize();
    }
    nvtxRangePop();

    free(h_idata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0)
    {
        return benchmark();
    }
    launch();
    return 0;
}