#ifndef POLYNOMIAL_ERROR_H
#define POLYNOMIAL_ERROR_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call) {                                     \
    cudaError_t error = call;                                  \
    if(error != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error at %s: %d - %s \n",        \
                __FILE__, __LINE__, cudaGetErrorString(error));\
        exit(EXIT_FAILURE);                                    \
    }                                                          \
}

// Offloading a small number of coefficients to registers decreases the 
// allocation size for shared memory and improves occupancy.
// Having MAX_NUM_COEFFICIENTS less than NUM_COEFFICIENTS_IN_REGISTERS is allowed.
constexpr int NUM_COEFFICIENTS_IN_REGISTERS = 2;

__global__ void k_computeAverageErrorsOfPolynomials(
    int numPolynomials, 
    int numCoefficients, 
    int numTrials, 
    float* xValues_d, 
    float* expectedPolynomialValues_d, 
    float* coefficients_d, 
    float* averageErrors_d);

#endif // POLYNOMIAL_ERROR_H