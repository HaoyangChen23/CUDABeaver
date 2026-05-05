#include "polynomial_error.h"

__global__ void k_computeAverageErrorsOfPolynomials(int numPolynomials, int numCoefficients, int numTrials, 
                                                    float * xValues_d, float * expectedPolynomialValues_d, float * coefficients_d, 
                                                    float * averageErrors_d) {
    // Each thread computes its own polynomial for multiple trial (x) values using the formula polynomial(x) = c1 + x * c2 + x^2 * c3 + ... + x^(n - 1) * cn.
    int localThreadIndex = threadIdx.x;
    int numLocalThreads = blockDim.x;
    int globalThreadIndex = localThreadIndex + blockIdx.x * blockDim.x;
    int numGlobalThreads = blockDim.x * gridDim.x;
    extern __shared__ float s_coefficients[];
    int coefficientSmemAllocation = numLocalThreads * (numCoefficients - NUM_COEFFICIENTS_IN_REGISTERS);
    if(coefficientSmemAllocation < 0) {
        coefficientSmemAllocation = 0;
    }
    float * s_ioDataPairs = reinterpret_cast<float *>(s_coefficients + coefficientSmemAllocation);
    int ioDataPairLoadSteps = (numTrials + numLocalThreads - 1) / numLocalThreads;
    for(int i = 0; i < ioDataPairLoadSteps; i++) {
        int pairIndex = i * numLocalThreads + localThreadIndex;
        if(pairIndex < numTrials) {
            s_ioDataPairs[pairIndex] = xValues_d[pairIndex];
            s_ioDataPairs[pairIndex + numTrials] = expectedPolynomialValues_d[pairIndex];
        }
    }
    __syncthreads();
    float r_coefficients[NUM_COEFFICIENTS_IN_REGISTERS];
    for(int polynomialIndex = globalThreadIndex; polynomialIndex < numPolynomials; polynomialIndex += numGlobalThreads) {
        // Storing some coefficients in the registers using static indexing and a static number of loop iterations.
        #pragma unroll 1
        for(int i = 0; i < NUM_COEFFICIENTS_IN_REGISTERS; i++) {
            r_coefficients[i] = (i < numCoefficients ? coefficients_d[polynomialIndex + i * numPolynomials] : 0.0f);
        }
        // The remaining coefficients are manually spilled over to shared memory.
        #pragma unroll 1
        for(int i = NUM_COEFFICIENTS_IN_REGISTERS; i < numCoefficients; i++) {
            s_coefficients[localThreadIndex + (i - NUM_COEFFICIENTS_IN_REGISTERS) * numLocalThreads] = coefficients_d[polynomialIndex + i * numPolynomials];
        }
        
        float averageError = 0.0f;
        #pragma unroll 1
        for(int i = 0; i < numTrials; i++) {
            float x = s_ioDataPairs[i];
            float polynomial = r_coefficients[0];
            // Coefficients from registers.
            #pragma unroll 1
            for(int j = 1; j < NUM_COEFFICIENTS_IN_REGISTERS; j++) {
                float coefficient = r_coefficients[j];
                polynomial = (j < numCoefficients ? fmaf(polynomial, x, coefficient) : polynomial);
            }
            // Coefficients from shared memory.
            #pragma unroll 1
            for(int j = NUM_COEFFICIENTS_IN_REGISTERS; j < numCoefficients; j++) {
                float coefficient = s_coefficients[localThreadIndex + (j - NUM_COEFFICIENTS_IN_REGISTERS) * numLocalThreads];
                polynomial = fmaf(polynomial, x, coefficient);
            }
            float error = fabsf(s_ioDataPairs[i + numTrials] - polynomial);
            averageError += error;
        }
        averageErrors_d[polynomialIndex] = averageError / numTrials;
    }
}