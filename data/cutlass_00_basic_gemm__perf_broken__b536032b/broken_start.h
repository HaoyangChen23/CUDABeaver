#pragma once
#include <cuda_runtime.h>

// Tile dimensions - tuned for modern GPUs
#define TILE_M 128
#define TILE_N 128
#define TILE_K 8

// Thread block dimensions
#define BLOCK_M 16
#define BLOCK_N 16

// Number of threads per block
#define THREADS_PER_BLOCK (BLOCK_M * BLOCK_N)

__global__ void sgemm_nn_kernel(
    int M, int N, int K,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float *C, int ldc)
{
    // Each thread block computes a TILE_M x TILE_N tile of C
    // Each thread computes a (TILE_M/BLOCK_M) x (TILE_N/BLOCK_N) sub-tile
    
    const int TM = TILE_M / BLOCK_M;  // 8
    const int TN = TILE_N / BLOCK_N;  // 8
    
    // Block position in grid
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;
    
    // Thread position within block
    int thread_row = threadIdx.x / BLOCK_N;
    int thread_col = threadIdx.x % BLOCK_N;
    
    // Starting positions for this block
    int c_row = block_row * TILE_M;
    int c_col = block_col * TILE_N;
    
    // Allocate shared memory for A and B tiles
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    
    // Accumulator registers
    float Creg[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            Creg[i][j] = 0.0f;
        }
    }
    
    // Loop over K dimension in steps of TILE_K
    for (int k = 0; k < K; k += TILE_K) {
        // Load A tile (TILE_M x TILE_K) into shared memory
        // Each thread loads multiple elements
        #pragma unroll
        for (int i = 0; i < TILE_M; i += BLOCK_M) {
            int row = i + thread_row;
            int a_row = c_row + row;
            int a_col = k;
            
            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk += 1) {
                if (row < TILE_M && a_row < M && (k + kk) < K) {
                    As[row][kk] = A[a_row + (a_col + kk) * lda];
                } else {
                    As[row][kk] = 0.0f;
                }
            }
        }
        
        // Load B tile (TILE_K x TILE_N) into shared memory
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 1) {
            #pragma unroll
            for (int j = 0; j < TILE_N; j += BLOCK_N) {
                int col = j + thread_col;
                int b_row = k + kk;
                int b_col = c_col + col;
                
                if (col < TILE_N && b_row < K && b_col < N) {
                    Bs[kk][col] = B[b_row + b_col * ldb];
                } else {
                    Bs[kk][col] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial dot products
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            // Load A values for this thread's rows
            float Areg[TM];
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                Areg[i] = As[thread_row * TM + i][kk];
            }
            
            // Load B values for this thread's columns
            float Breg[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                Breg[j] = Bs[kk][thread_col * TN + j];
            }
            
            // Multiply-accumulate
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    Creg[i][j] += Areg[i] * Breg[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to C
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int c_row_idx = c_row + thread_row * TM + i;
        if (c_row_idx >= M) continue;
        
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int c_col_idx = c_col + thread_col * TN + j;
            if (c_col_idx >= N) continue;
            
            float c_val = alpha * Creg[i][j];
            if (beta != 0.0f) {
                c_val += beta * C[c_row_idx + c_col_idx * ldc];
            }
            C[c_row_idx + c_col_idx * ldc] = c_val;
        }
    }
}

// Alternative simpler kernel for smaller sizes
__global__ void sgemm_nn_simple_kernel(
    int M, int N, int K,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float *C, int ldc)
{
    // Each thread computes one element of C
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row + k * lda] * B[k + col * ldb];
    }
    
    float c_val = alpha * sum;
    if (beta != 0.0f) {
        c_val += beta * C[row + col * ldc];
    }
    C[row + col * ldc] = c_val;
}

cudaError_t SgemmNN(
    int M, int N, int K,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float *C, int ldc)
{
    cudaError_t err = cudaSuccess;
    
    // Choose kernel based on problem size
    // For large matrices, use tiled kernel; for small, use simple kernel
    
    const int threshold = 256;  // Use simple kernel for small matrices
    
    if (M <= threshold || N <= threshold || K <= threshold) {
        // Simple kernel
        dim3 threads(16, 16);
        dim3 blocks((M + threads.x - 1) / threads.x,
                    (N + threads.y - 1) / threads.y);
        
        sgemm_nn_simple_kernel<<<blocks, threads>>>(
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        // Tiled kernel
        dim3 threads(BLOCK_M * BLOCK_N);  // 256 threads
        dim3 blocks((M + TILE_M - 1) / TILE_M,
                    (N + TILE_N - 1) / TILE_N);
        
        sgemm_nn_kernel<<<blocks, threads>>>(
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    err = cudaGetLastError();
    return err;
}
