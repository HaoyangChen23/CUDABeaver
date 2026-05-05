#undef NDEBUG
#include <assert.h>
#include <random>
#include <algorithm>
#include <string>
#include "polynomial_error.h"

void launch() {
    // Maximum number of different x values to calculate the average error of the polynomial function approximation f(x) = c1 + c2*x + c3*(x^2) + ... + cn*(x^(n-1)), where n = number of coefficients. This can be adjusted.
    constexpr int MAX_NUM_TRIALS = 20;
    // Maximum number of coefficients for polynomials; this can be adjusted.
    constexpr int MAX_NUM_COEFFICIENTS = 8;
    constexpr float ERROR_TOLERANCE = 1e-4f;
    constexpr int DETERMINISTIC_RANDOM_SEED = 42;
    // The minimum number of coefficients required in registers is 1, which is also the requirement for the calculations to form a polynomial (f(x) = c1). This must remain as 1.
    constexpr int MIN_NUM_COEFFICIENTS_IN_REGISTERS = 1;

    // Test settings.
    constexpr int NUM_TESTS = 7;

    // At least one coefficient is expected to exist as a polynomial (f(x) = c1).
    assert(NUM_COEFFICIENTS_IN_REGISTERS >= MIN_NUM_COEFFICIENTS_IN_REGISTERS);
    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceId));

    // Ensure the problem size exceeds maximum GPU occupancy,
    // requiring a grid-stride loop for correct results.
    int maxResidentThreads = deviceProperties.multiProcessorCount
                           * deviceProperties.maxThreadsPerMultiProcessor;
    int maxNumPolynomials = maxResidentThreads * 2;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Allocating memory for the algorithm on the host.
    float * xValues_h = new float[MAX_NUM_TRIALS];
    float * coefficients_h = new float[maxNumPolynomials * MAX_NUM_COEFFICIENTS];
    float * averageErrors_h = new float[maxNumPolynomials];
    float * expectedOutputs_h = new float[MAX_NUM_TRIALS];
    // Allocating memory for the tests.
    int * testNumTrials_h = new int[NUM_TESTS];
    int * testNumPolynomials_h = new int[NUM_TESTS];
    int * testNumCoefficients_h = new int[NUM_TESTS];
    float * testValuesForX_h = new float[NUM_TESTS * MAX_NUM_TRIALS];
    float * testCoefficients_h = new float[NUM_TESTS * maxNumPolynomials * MAX_NUM_COEFFICIENTS];
    float * testExpectedOutputs_h = new float[NUM_TESTS * MAX_NUM_TRIALS];
    
    // Allocating device memory.
    //  Input parameters for the f(x) function's polynomial approximations. Multiple x values are needed to calculate the average error for each polynomial.
    float * xValues_d;
    // The coefficients of each polynomial.
    float * coefficients_d;
    // The average error of the function approximation f(x) concerning the expected outputs.
    float * averageErrors_d;
    // Expected values of the function approximation f(x) for each given x value and per test.
    float * expectedPolynomialValues_d;
    CUDA_CHECK(cudaMallocAsync(&xValues_d, sizeof(float) * MAX_NUM_TRIALS, stream));
    CUDA_CHECK(cudaMallocAsync(&coefficients_d, sizeof(float) * maxNumPolynomials * MAX_NUM_COEFFICIENTS, stream));
    CUDA_CHECK(cudaMallocAsync(&averageErrors_d, sizeof(float) * maxNumPolynomials, stream));
    CUDA_CHECK(cudaMallocAsync(&expectedPolynomialValues_d, sizeof(float) * MAX_NUM_TRIALS, stream));
    
    // Random-number generator to simulate optimization heuristics such as simulated annealing, genetic algorithm, or particle swarm for the coefficients.
    std::mt19937 generator(DETERMINISTIC_RANDOM_SEED);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    // Preparing the test data.
    int testIndex = 0;
    // Test 1: sin(x)
    {
        int numTrials = MAX_NUM_TRIALS;
        int numPolynomials = maxNumPolynomials;
        int numCoefficients = MAX_NUM_COEFFICIENTS;
        testNumTrials_h[testIndex] = numTrials;
        testNumPolynomials_h[testIndex] = numPolynomials;
        testNumCoefficients_h[testIndex] = numCoefficients;
        // f(x) = sin(x) is chosen for the polynomial approximation error calculations.
        for(int i = 0; i < numTrials; i++) {
            float x = distribution(generator);
            testValuesForX_h[i + testIndex * MAX_NUM_TRIALS] = x;
            testExpectedOutputs_h[i + testIndex * MAX_NUM_TRIALS] = sin(x);
        }
        // Simulating the outcomes of heuristic optimization.
        for(int i = 0; i < numPolynomials * numCoefficients; i++) {
            testCoefficients_h[i + testIndex * maxNumPolynomials * MAX_NUM_COEFFICIENTS] = 0.5f;
        }
        testIndex++;
    }
    // Test 2: cos(x)
    {
        int numTrials = MAX_NUM_TRIALS;
        int numPolynomials = maxNumPolynomials;
        int numCoefficients = MAX_NUM_COEFFICIENTS;
        testNumTrials_h[testIndex] = numTrials;
        testNumPolynomials_h[testIndex] = numPolynomials;
        testNumCoefficients_h[testIndex] = numCoefficients;
        // f(x) = cos(x) is chosen for the polynomial approximation error calculations.
        for(int i = 0; i < numTrials; i++) {
            float x = distribution(generator);
            testValuesForX_h[i + testIndex * MAX_NUM_TRIALS] = x;
            testExpectedOutputs_h[i + testIndex * MAX_NUM_TRIALS] = cos(x);
        }
        // Simulating the outcomes of heuristic optimization.
        for(int i = 0; i < numPolynomials * numCoefficients; i++) {
            testCoefficients_h[i + testIndex * maxNumPolynomials * MAX_NUM_COEFFICIENTS] = (i % 2 == 0 ? 0.25f : 0.5f);
        }
        testIndex++;
    }
    // Test 3: sqrt(x)
    {
        int numTrials = MAX_NUM_TRIALS;
        int numPolynomials = maxNumPolynomials - 1;
        int numCoefficients = MAX_NUM_COEFFICIENTS;
        testNumTrials_h[testIndex] = numTrials;
        testNumPolynomials_h[testIndex] = numPolynomials;
        testNumCoefficients_h[testIndex] = numCoefficients;
        // f(x) = sqrt(x) is chosen for the polynomial approximation error calculations.
        for(int i = 0; i < numTrials; i++) {
            float x = distribution(generator);
            testValuesForX_h[i + testIndex * MAX_NUM_TRIALS] = x;
            testExpectedOutputs_h[i + testIndex * MAX_NUM_TRIALS] = sqrt(x);
        }
        // Simulating the outcomes of heuristic optimization.
        for(int i = 0; i < numPolynomials * numCoefficients; i++) {
            testCoefficients_h[i + testIndex * maxNumPolynomials * MAX_NUM_COEFFICIENTS] = distribution(generator);
        }
        testIndex++;
    }
    // Test 4: 1.0f / (1.0f + x)
    {
        constexpr float TEST4_CONSTANT = 1.0f;
        int numTrials = MAX_NUM_TRIALS;
        int numPolynomials = maxNumPolynomials - 10;
        int numCoefficients = MAX_NUM_COEFFICIENTS;
        testNumTrials_h[testIndex] = numTrials;
        testNumPolynomials_h[testIndex] = numPolynomials;
        testNumCoefficients_h[testIndex] = numCoefficients;
        // f(x) = 1.0f / (1.0f + x) is chosen for the polynomial approximation error calculations.
        for(int i = 0; i < numTrials; i++) {
            float x = distribution(generator);
            testValuesForX_h[i + testIndex * MAX_NUM_TRIALS] = x;
            testExpectedOutputs_h[i + testIndex * MAX_NUM_TRIALS] = TEST4_CONSTANT / (TEST4_CONSTANT + x);
        }
        // Simulating the outcomes of heuristic optimization.
        for(int i = 0; i < numPolynomials * numCoefficients; i++) {
            testCoefficients_h[i + testIndex * maxNumPolynomials * MAX_NUM_COEFFICIENTS] = distribution(generator);
        }
        testIndex++;
    }
    // Test 5: 1.0f - x * x
    {
        constexpr float TEST5_CONSTANT = 1.0f;
        int numTrials = MAX_NUM_TRIALS;
        int numPolynomials = maxNumPolynomials - 100;
        int numCoefficients = MAX_NUM_COEFFICIENTS;
        testNumTrials_h[testIndex] = numTrials;
        testNumPolynomials_h[testIndex] = numPolynomials;
        testNumCoefficients_h[testIndex] = numCoefficients;
        // f(x) = 1.0f - x * x is chosen for the polynomial approximation error calculations.
        for(int i = 0; i < numTrials; i++) {
            float x = distribution(generator);
            testValuesForX_h[i + testIndex * MAX_NUM_TRIALS] = x;
            testExpectedOutputs_h[i + testIndex * MAX_NUM_TRIALS] = TEST5_CONSTANT - x * x;
        }
        // Simulating the outcomes of heuristic optimization.
        for(int i = 0; i < numPolynomials * numCoefficients; i++) {
            testCoefficients_h[i + testIndex * maxNumPolynomials * MAX_NUM_COEFFICIENTS] = distribution(generator);
        }
        testIndex++;
    }
    // Test 6: x ^ 3
    {
        int numTrials = MAX_NUM_TRIALS;
        int numPolynomials = maxNumPolynomials - 1000;
        int numCoefficients = MAX_NUM_COEFFICIENTS;
        testNumTrials_h[testIndex] = numTrials;
        testNumPolynomials_h[testIndex] = numPolynomials;
        testNumCoefficients_h[testIndex] = numCoefficients;
        // f(x) = x ^ 3 is chosen for the polynomial approximation error calculations.
        for(int i = 0; i < numTrials; i++) {
            float x = distribution(generator);
            testValuesForX_h[i + testIndex * MAX_NUM_TRIALS] = x;
            testExpectedOutputs_h[i + testIndex * MAX_NUM_TRIALS] = x * x * x;
        }
        // Simulating the outcomes of heuristic optimization.
        for(int i = 0; i < numPolynomials * numCoefficients; i++) {
            testCoefficients_h[i + testIndex * maxNumPolynomials * MAX_NUM_COEFFICIENTS] = distribution(generator);
        }
        testIndex++;
    }
    // Test 7: 1.0f - sin(x)^2
    {
        constexpr float TEST7_CONSTANT = 1.0f;
        int numTrials = MAX_NUM_TRIALS;
        int numPolynomials = maxNumPolynomials;
        int numCoefficients = MAX_NUM_COEFFICIENTS;
        testNumTrials_h[testIndex] = numTrials;
        testNumPolynomials_h[testIndex] = numPolynomials;
        testNumCoefficients_h[testIndex] = numCoefficients;
        // f(x) = 1.0f - sin(x)^2 is chosen for the polynomial approximation error calculations.
        for(int i = 0; i < numTrials; i++) {
            float x = distribution(generator);
            testValuesForX_h[i + testIndex * MAX_NUM_TRIALS] = x;
            testExpectedOutputs_h[i + testIndex * MAX_NUM_TRIALS] = TEST7_CONSTANT - sin(x) * sin(x);
        }
        // Simulating the outcomes of heuristic optimization.
        for(int i = 0; i < numPolynomials * numCoefficients; i++) {
            testCoefficients_h[i + testIndex * maxNumPolynomials * MAX_NUM_COEFFICIENTS] = distribution(generator);
        }
        testIndex++;
    }
    // Iterating the tests.
    for(int test = 0; test < NUM_TESTS; test++)
    {
        int numTrials = testNumTrials_h[test];
        int numPolynomials = testNumPolynomials_h[test];
        int numCoefficients = testNumCoefficients_h[test];
        for(int i = 0; i < numTrials; i++) {
            xValues_h[i] = testValuesForX_h[i + test * MAX_NUM_TRIALS];
        }
        for(int i = 0; i < numPolynomials * numCoefficients; i++) {
            coefficients_h[i] = testCoefficients_h[i + test * maxNumPolynomials * MAX_NUM_COEFFICIENTS];
        }
        for(int i = 0; i < numTrials; i++) {
            expectedOutputs_h[i] = testExpectedOutputs_h[i + test * MAX_NUM_TRIALS];
        }
        CUDA_CHECK(cudaMemcpyAsync(xValues_d, xValues_h, sizeof(float) * numTrials, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(coefficients_d, coefficients_h, sizeof(float) * numPolynomials * numCoefficients, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(expectedPolynomialValues_d, expectedOutputs_h, sizeof(float) * numTrials, cudaMemcpyHostToDevice, stream));
        // Letting CUDA decide the best block size for the optimal shared memory allocation size since the allocation size depends on block size and the hardware.
        int minGridSize;
        int blockSize;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, 
                                                                  &blockSize, 
                                                                  (void*)k_computeAverageErrorsOfPolynomials, 
                                                                  [=](int blockSize) { 
                                                                      int size = blockSize * sizeof(float) * (numCoefficients - NUM_COEFFICIENTS_IN_REGISTERS);
                                                                      if(size < 0) {
                                                                          size = 0;
                                                                      }
                                                                      size += numTrials * 2 * sizeof(float);
                                                                      return size; 
                                                                  }));
        void * args[7] = { &numPolynomials, &numCoefficients, &numTrials, &xValues_d, &expectedPolynomialValues_d, &coefficients_d, &averageErrors_d };
        // (numCoefficients - NUM_COEFFICIENTS_IN_REGISTERS) represents the number of manual spillover to shared memory per thread.
        int sharedMem = blockSize * sizeof(float) * (numCoefficients - NUM_COEFFICIENTS_IN_REGISTERS);
        if(sharedMem < 0) {
            sharedMem = 0;
        }
        // The size of shared memory used for holding data pairs in average error calculations is numTrials * 2 * sizeof(float).
        sharedMem += numTrials * 2 * sizeof(float);
        assert((size_t)sharedMem <= deviceProperties.sharedMemPerBlock);
        // Grid: (minGridSize, 1, 1)
        // Block: (blockSize, 1, 1)
        CUDA_CHECK(cudaLaunchKernel((void*)k_computeAverageErrorsOfPolynomials, dim3(minGridSize, 1, 1), dim3(blockSize, 1, 1), args, sharedMem, stream));
        CUDA_CHECK(cudaMemcpyAsync(averageErrors_h, averageErrors_d, sizeof(float) * numPolynomials, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Comparing the results from device to the results computed on host.
        for(int pol = 0; pol < numPolynomials; pol++) {
            float averageError = 0.0f;
            for(int trial = 0; trial < numTrials; trial++) {
                float x = xValues_h[trial];
                float polynomial = coefficients_h[pol];
                for(int coeff = 1; coeff < numCoefficients; coeff++) {
                    float coefficient = coefficients_h[pol + coeff * numPolynomials];
                    polynomial = fmaf(polynomial, x, coefficient);
                }
                float error = fabsf(expectedOutputs_h[trial] - polynomial);
                averageError += error;
            }
            averageError /= numTrials;
            assert(fabsf(averageError - averageErrors_h[pol]) < ERROR_TOLERANCE);
        }
        
    }
    // Freeing resources.
    CUDA_CHECK(cudaFreeAsync(xValues_d, stream));
    CUDA_CHECK(cudaFreeAsync(coefficients_d, stream));
    CUDA_CHECK(cudaFreeAsync(averageErrors_d, stream));
    CUDA_CHECK(cudaFreeAsync(expectedPolynomialValues_d, stream));
    delete [] xValues_h;
    delete [] coefficients_h;
    delete [] averageErrors_h;
    delete [] expectedOutputs_h;
    delete [] testNumTrials_h;
    delete [] testNumPolynomials_h;
    delete [] testNumCoefficients_h;
    delete [] testValuesForX_h;
    delete [] testCoefficients_h;
    delete [] testExpectedOutputs_h;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    constexpr int BENCH_NUM_TRIALS = 20;
    constexpr int BENCH_NUM_POLYNOMIALS = 10000000;
    constexpr int BENCH_NUM_COEFFICIENTS = 8;
    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 1000;
    constexpr int DETERMINISTIC_RANDOM_SEED = 123;

    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceId));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float *xValues_h = new float[BENCH_NUM_TRIALS];
    float *coefficients_h = new float[BENCH_NUM_POLYNOMIALS * BENCH_NUM_COEFFICIENTS];
    float *expectedOutputs_h = new float[BENCH_NUM_TRIALS];

    std::mt19937 generator(DETERMINISTIC_RANDOM_SEED);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (int i = 0; i < BENCH_NUM_TRIALS; i++) {
        float x = distribution(generator);
        xValues_h[i] = x;
        expectedOutputs_h[i] = sin(x);
    }
    for (int i = 0; i < BENCH_NUM_POLYNOMIALS * BENCH_NUM_COEFFICIENTS; i++) {
        coefficients_h[i] = distribution(generator);
    }

    float *xValues_d, *coefficients_d, *averageErrors_d, *expectedPolynomialValues_d;
    CUDA_CHECK(cudaMallocAsync(&xValues_d, sizeof(float) * BENCH_NUM_TRIALS, stream));
    CUDA_CHECK(cudaMallocAsync(&coefficients_d, sizeof(float) * BENCH_NUM_POLYNOMIALS * BENCH_NUM_COEFFICIENTS, stream));
    CUDA_CHECK(cudaMallocAsync(&averageErrors_d, sizeof(float) * BENCH_NUM_POLYNOMIALS, stream));
    CUDA_CHECK(cudaMallocAsync(&expectedPolynomialValues_d, sizeof(float) * BENCH_NUM_TRIALS, stream));

    CUDA_CHECK(cudaMemcpyAsync(xValues_d, xValues_h, sizeof(float) * BENCH_NUM_TRIALS, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(coefficients_d, coefficients_h, sizeof(float) * BENCH_NUM_POLYNOMIALS * BENCH_NUM_COEFFICIENTS, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(expectedPolynomialValues_d, expectedOutputs_h, sizeof(float) * BENCH_NUM_TRIALS, cudaMemcpyHostToDevice, stream));

    int numPolynomials = BENCH_NUM_POLYNOMIALS;
    int numCoefficients = BENCH_NUM_COEFFICIENTS;
    int numTrials = BENCH_NUM_TRIALS;

    int minGridSize, blockSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize,
                                                              &blockSize,
                                                              (void*)k_computeAverageErrorsOfPolynomials,
                                                              [=](int blockSize) {
                                                                  int size = blockSize * sizeof(float) * (numCoefficients - NUM_COEFFICIENTS_IN_REGISTERS);
                                                                  if (size < 0) size = 0;
                                                                  size += numTrials * 2 * sizeof(float);
                                                                  return size;
                                                              }));

    void *args[7] = { &numPolynomials, &numCoefficients, &numTrials, &xValues_d, &expectedPolynomialValues_d, &coefficients_d, &averageErrors_d };
    int sharedMem = blockSize * sizeof(float) * (numCoefficients - NUM_COEFFICIENTS_IN_REGISTERS);
    if (sharedMem < 0) sharedMem = 0;
    sharedMem += numTrials * 2 * sizeof(float);

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_computeAverageErrorsOfPolynomials, dim3(minGridSize, 1, 1), dim3(blockSize, 1, 1), args, sharedMem, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_computeAverageErrorsOfPolynomials, dim3(minGridSize, 1, 1), dim3(blockSize, 1, 1), args, sharedMem, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(xValues_d, stream));
    CUDA_CHECK(cudaFreeAsync(coefficients_d, stream));
    CUDA_CHECK(cudaFreeAsync(averageErrors_d, stream));
    CUDA_CHECK(cudaFreeAsync(expectedPolynomialValues_d, stream));
    delete[] xValues_h;
    delete[] coefficients_h;
    delete[] expectedOutputs_h;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--perf") {
        benchmark();
    } else {
        launch();
    }
}