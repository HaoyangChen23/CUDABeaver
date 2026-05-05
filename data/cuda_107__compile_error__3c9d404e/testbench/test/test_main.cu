#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include "heatmap.h"

void launch() {
    dim3 gridDim(NUM_GRID_BLOCKS_X, NUM_GRID_BLOCKS_Y, 1);
    dim3 blockDim(NUM_BLOCK_THREADS_X, NUM_BLOCK_THREADS_Y, 1);

    // Compute data dimensions that EXCEED grid coverage, requiring
    // a stride loop for correct results.
    // Grid covers: NUM_GRID_BLOCKS_X * NUM_BLOCK_THREADS_X in X (32*16 = 512)
    //              NUM_GRID_BLOCKS_Y * NUM_BLOCK_THREADS_Y in Y (8*16  = 128)
    const int numElementsX = NUM_GRID_BLOCKS_X * NUM_BLOCK_THREADS_X * 2;
    const int numElementsY = NUM_GRID_BLOCKS_Y * NUM_BLOCK_THREADS_Y * 2;
    const int numTotalElements = numElementsX * numElementsY;

    float * input_d;
    unsigned char * output_d;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMallocAsync(&input_d, sizeof(float) * numTotalElements, stream));
    CUDA_CHECK(cudaMallocAsync(&output_d, sizeof(unsigned char) * numTotalElements, stream));
    std::vector<float> input(numTotalElements);
    std::vector<unsigned char> output(numTotalElements);
    std::vector<unsigned char> expectedOutput(numTotalElements);
    float * input_h = input.data();
    unsigned char * output_h = output.data();
    // Test 1: Continuously increasing values with instances of partial overflow and underflow.
    {
        constexpr float minValue = 1900.0f;
        constexpr float maxValue = 2800.0f;
        void * args[6] = { &input_d, &output_d, (void*)&minValue, (void*)&maxValue, (void*)&numElementsX, (void*)&numElementsY };
        for(int i = 0; i < numTotalElements; i++) {
            input[i] = 1500 + i * 150.0f;
            expectedOutput[i] = round(255 * (min(max(input[i], minValue), maxValue) - minValue) / (maxValue - minValue));
        }
        CUDA_CHECK(cudaMemcpyAsync(input_d, input_h, sizeof(float) * numTotalElements, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_generateHeatmap, gridDim, blockDim, args, 0, stream));
        CUDA_CHECK(cudaMemcpyAsync(output_h, output_d, sizeof(unsigned char) * numTotalElements, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(int i = 0; i < numTotalElements; i++) {
            assert(expectedOutput[i] == output[i]);
        }
    }
    // Test 2: Duplicate Values.
    {
        constexpr float minValue = 1.0f;
        constexpr float maxValue = 2.0f;
        void * args[6] = { &input_d, &output_d, (void*)&minValue, (void*)&maxValue, (void*)&numElementsX, (void*)&numElementsY };
        for(int i = 0; i < numTotalElements; i++) {
            input[i] = 1.5f;
            expectedOutput[i] = round(255 * (min(max(input[i], minValue), maxValue) - minValue) / (maxValue - minValue));
        }
        CUDA_CHECK(cudaMemcpyAsync(input_d, input_h, sizeof(float) * numTotalElements, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_generateHeatmap, gridDim, blockDim, args, 0, stream));
        CUDA_CHECK(cudaMemcpyAsync(output_h, output_d, sizeof(unsigned char) * numTotalElements, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(int i = 0; i < numTotalElements; i++) {
            assert(expectedOutput[i] == output[i]);
        }
    }
    // Test 3: Values beyond the allowed range.
    {
        constexpr float minValue = 10.0f;
        constexpr float maxValue = 100.0f;
        void * args[6] = { &input_d, &output_d, (void*)&minValue, (void*)&maxValue, (void*)&numElementsX, (void*)&numElementsY };
        for(int i = 0; i < numTotalElements; i++) {
            input[i] = ((i & 1) ? 1.0f : 1000.0f);
            expectedOutput[i] = round(255 * (min(max(input[i], minValue), maxValue) - minValue) / (maxValue - minValue));
        }
        CUDA_CHECK(cudaMemcpyAsync(input_d, input_h, sizeof(float) * numTotalElements, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_generateHeatmap, gridDim, blockDim, args, 0, stream));
        CUDA_CHECK(cudaMemcpyAsync(output_h, output_d, sizeof(unsigned char) * numTotalElements, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(int i = 0; i < numTotalElements; i++) {
            assert(expectedOutput[i] == output[i]);
        }
    }
    // Test 4: Alternating values.
    {
        constexpr float minValue = 10.0f;
        constexpr float maxValue = 100.0f;
        void * args[6] = { &input_d, &output_d, (void*)&minValue, (void*)&maxValue, (void*)&numElementsX, (void*)&numElementsY };
        for(int i = 0; i < numTotalElements; i++) {
            input[i] = ((i & 1) ? 20.0f : 80.0f);
            expectedOutput[i] = round(255 * (min(max(input[i], minValue), maxValue) - minValue) / (maxValue - minValue));
        }
        CUDA_CHECK(cudaMemcpyAsync(input_d, input_h, sizeof(float) * numTotalElements, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_generateHeatmap, gridDim, blockDim, args, 0, stream));
        CUDA_CHECK(cudaMemcpyAsync(output_h, output_d, sizeof(unsigned char) * numTotalElements, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(int i = 0; i < numTotalElements; i++) {
            assert(expectedOutput[i] == output[i]);
        }
    }
    // Test 5: Increasing values within boundaries.
    {
        constexpr float minValue = 10.0f;
        constexpr float maxValue = 100.0f;
        void * args[6] = { &input_d, &output_d, (void*)&minValue, (void*)&maxValue, (void*)&numElementsX, (void*)&numElementsY };
        for(int i = 0; i < numTotalElements; i++) {
            input[i] = 10.0f + 10.0f * i;
            expectedOutput[i] = round(255 * (min(max(input[i], minValue), maxValue) - minValue) / (maxValue - minValue));
        }
        CUDA_CHECK(cudaMemcpyAsync(input_d, input_h, sizeof(float) * numTotalElements, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_generateHeatmap, gridDim, blockDim, args, 0, stream));
        CUDA_CHECK(cudaMemcpyAsync(output_h, output_d, sizeof(unsigned char) * numTotalElements, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(int i = 0; i < numTotalElements; i++) {
            assert(expectedOutput[i] == output[i]);
        }
    }
    // Test 6: Randomized inputs.
    {
        srand(1);
        constexpr float minValue = 0.0f;
        constexpr float maxValue = 1.0f;
        void * args[6] = { &input_d, &output_d, (void*)&minValue, (void*)&maxValue, (void*)&numElementsX, (void*)&numElementsY };
        for(int i = 0; i < numTotalElements; i++) {
            input[i] = 1.0f / rand();
            expectedOutput[i] = round(255 * (min(max(input[i], minValue), maxValue) - minValue) / (maxValue - minValue));
        }
        CUDA_CHECK(cudaMemcpyAsync(input_d, input_h, sizeof(float) * numTotalElements, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_generateHeatmap, gridDim, blockDim, args, 0, stream));
        CUDA_CHECK(cudaMemcpyAsync(output_h, output_d, sizeof(unsigned char) * numTotalElements, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(int i = 0; i < numTotalElements; i++) {
            assert(expectedOutput[i] == output[i]);
        }
    }
    // Test 7: A negative minimum value and a positive maximum value.
    {
        constexpr float minValue = -1.0f;
        constexpr float maxValue = 1.0f;
        void * args[6] = { &input_d, &output_d, (void*)&minValue, (void*)&maxValue, (void*)&numElementsX, (void*)&numElementsY };
        for(int i = 0; i < numTotalElements; i++) {
            input[i] = -1.0f + i * 0.2f;
            expectedOutput[i] = round(255 * (min(max(input[i], minValue), maxValue) - minValue) / (maxValue - minValue));
        }
        CUDA_CHECK(cudaMemcpyAsync(input_d, input_h, sizeof(float) * numTotalElements, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_generateHeatmap, gridDim, blockDim, args, 0, stream));
        CUDA_CHECK(cudaMemcpyAsync(output_h, output_d, sizeof(unsigned char) * numTotalElements, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(int i = 0; i < numTotalElements; i++) {
            assert(expectedOutput[i] == output[i]);
        }
    }
    CUDA_CHECK(cudaFreeAsync(input_d, stream));
    CUDA_CHECK(cudaFreeAsync(output_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    const int PERF_ELEMENTS_X = 8192;
    const int PERF_ELEMENTS_Y = 8192;
    const int PERF_TOTAL = PERF_ELEMENTS_X * PERF_ELEMENTS_Y;
    const int WARMUP_ITERS = 3;
    const int TIMED_ITERS = 100;

    dim3 gridDim(256, 32, 1);
    dim3 blockDim(16, 16, 1);

    float * input_d;
    unsigned char * output_d;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMallocAsync(&input_d, sizeof(float) * PERF_TOTAL, stream));
    CUDA_CHECK(cudaMallocAsync(&output_d, sizeof(unsigned char) * PERF_TOTAL, stream));

    std::vector<float> input_h(PERF_TOTAL);
    srand(42);
    for (int i = 0; i < PERF_TOTAL; i++) {
        input_h[i] = -100.0f + 200.0f * ((float)rand() / RAND_MAX);
    }
    CUDA_CHECK(cudaMemcpyAsync(input_d, input_h.data(), sizeof(float) * PERF_TOTAL, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const float minValue = -50.0f;
    const float maxValue = 50.0f;
    void * args[6] = { &input_d, &output_d, (void*)&minValue, (void*)&maxValue, (void*)&PERF_ELEMENTS_X, (void*)&PERF_ELEMENTS_Y };

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_generateHeatmap, gridDim, blockDim, args, 0, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_generateHeatmap, gridDim, blockDim, args, 0, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(input_d, stream));
    CUDA_CHECK(cudaFreeAsync(output_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}