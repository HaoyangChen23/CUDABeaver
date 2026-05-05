#include "polynomial_error.h"

__global__ void k_computeAverageErrorsOfPolynomials(
    int numPolynomials,
    int numCoefficients,
    int numTrials,
    float* xValues_d,
    float* expectedPolynomialValues_d,
    float* coefficients_d,
    float* averageErrors_d)
{
    extern __shared__ float sharedCoeffs[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPolynomials) return;
    
    // Load coefficients into registers (up to NUM_COEFFICIENTS_IN_REGISTERS)
    float regCoeffs[NUM_COEFFICIENTS_IN_REGISTERS];
    
    int coeffsInRegs = (numCoefficients < NUM_COEFFICIENTS_IN_REGISTERS) ? 
                       numCoefficients : NUM_COEFFICIENTS_IN_REGISTERS;
    
    // Load register coefficients (strided layout: p + i * numPolynomials)
    for (int i = 0; i < coeffsInRegs; i++) {
        regCoeffs[i] = coefficients_d[tid + i * numPolynomials];
    }
    
    // Spill remaining coefficients to shared memory
    int coeffsInShared = numCoefficients - coeffsInRegs;
    
    if (coeffsInShared > 0) {
        // Each thread loads its spilled coefficients to shared memory
        // Use strided access pattern for coalesced memory access
        int sharedOffset = threadIdx.x * coeffsInShared;
        
        for (int i = 0; i < coeffsInShared; i++) {
            int coeffIdx = coeffsInRegs + i;
            sharedCoeffs[sharedOffset + i] = coefficients_d[tid + coeffIdx * numPolynomials];
        }
    }
    
    __syncthreads();
    
    float totalError = 0.0f;
    
    // Evaluate polynomial for each trial
    for (int t = 0; t < numTrials; t++) {
        float x = xValues_d[t];
        
        // Start with first coefficient
        float result = regCoeffs[0];
        
        // Process remaining register coefficients
        for (int i = 1; i < coeffsInRegs; i++) {
            result = result * x + regCoeffs[i];
        }
        
        // Process spilled coefficients from shared memory
        if (coeffsInShared > 0) {
            int sharedOffset = threadIdx.x * coeffsInShared;
            for (int i = 0; i < coeffsInShared; i++) {
                result = result * x + sharedCoeffs[sharedOffset + i];
            }
        }
        
        // Accumulate absolute error
        float expected = expectedPolynomialValues_d[t];
        totalError += fabsf(expected - result);
    }
    
    // Compute and store average error
    averageErrors_d[tid] = totalError / numTrials;
}