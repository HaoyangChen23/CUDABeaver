#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Gather-Scatter GEMM kernel
// D[scatter_D[i], j] = alpha * sum_k A[gather_A[i], k] * B[k, j] + beta * C[gather_C[i], j]
// Row-major layouts

template <int TILE_M = 8, int TILE_N = 8>
__global__ void gather_scatter_gemm_kernel(
    int M, int N, int K, float alpha,
    __half const *A, int lda, int const *gather_A,
    __half const *B, int ldb,
    float beta,
    float const *C, int ldc, int const *gather_C,
    float *D, int ldd, int const *scatter_D) {
    
    int i = blockIdx.x * TILE_M + threadIdx.y;  // row index in output (0 to M-1)
    int j = blockIdx.y * TILE_N + threadIdx.x;  // column index in output (0 to N-1)
    
    // Each thread computes one element (or handles boundary check)
    if (i < M && j < N) {
        // Gather indices for this row
        int a_row = gather_A[i];
        int c_row = gather_C[i];
        int d_row = scatter_D[i];
        
        // Compute dot product: sum_k A[a_row, k] * B[k, j]
        float acc = 0.0f;
        
        // Unroll by 4 for better instruction throughput
        int k = 0;
        for (; k + 3 < K; k += 4) {
            float a0 = __half2float(A[a_row * lda + k + 0]);
            float a1 = __half2float(A[a_row * lda + k + 1]);
            float a2 = __half2float(A[a_row * lda + k + 2]);
            float a3 = __half2float(A[a_row * lda + k + 3]);
            
            float b0 = __half2float(B[(k + 0) * ldb + j]);
            float b1 = __half2float(B[(k + 1) * ldb + j]);
            float b2 = __half2float(B[(k + 2) * ldb + j]);
            float b3 = __half2float(B[(k + 3) * ldb + j]);
            
            acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        }
        
        // Handle remaining elements
        for (; k < K; ++k) {
            float a_val = __half2float(A[a_row * lda + k]);
            float b_val = __half2float(B[k * ldb + j]);
            acc += a_val * b_val;
        }
        
        // Load C value with gather
        float c_val = C[c_row * ldc + j];
        
        // Compute final result with alpha/beta scaling
        float result = alpha * acc + beta * c_val;
        
        // Scatter write to D
        D[d_row * ldd + j] = result;
    }
}

// Host wrapper function
inline cudaError_t GatherScatterGemm(
    int M, int N, int K, float alpha,
    __half const *A, int lda, int const *gather_A,
    __half const *B, int ldb,
    float beta,
    float const *C, int ldc, int const *gather_C,
    float *D, int ldd, int const *scatter_D) {
    
    // Use 8x8 tiles for good occupancy and memory coalescing
    const int TILE_M = 8;
    const int TILE_N = 8;
    
    dim3 block(TILE_N, TILE_M);  // 8x8 = 64 threads per block
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    
    gather_scatter_gemm_kernel<TILE_M, TILE_N><<<grid, block>>>(
        M, N, K, alpha,
        A, lda, gather_A,
        B, ldb,
        beta,
        C, ldc, gather_C,
        D, ldd, scatter_D);
    
    return cudaGetLastError();
}
