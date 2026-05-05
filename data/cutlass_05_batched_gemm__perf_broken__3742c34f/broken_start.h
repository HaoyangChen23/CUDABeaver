#pragma once
#include <cuda_runtime.h>

// Batched Strided SGEMM kernel with column-major layout
// Each thread block computes a tile of C
// Using shared memory for tiling A and B

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void strided_batched_sgemm_kernel(
    int m, int n, int k,
    float alpha,
    float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta)
{
    // Thread indices within block
    int thread_m = threadIdx.x % BLOCK_M;  // row in tile
    int thread_n = threadIdx.x / BLOCK_M;  // col in tile
    
    // Block indices
    int block_m = blockIdx.x;  // which M tile
    int block_n = blockIdx.y;  // which N tile
    int batch_id = blockIdx.z; // which batch
    
    // Early exit if this block is outside valid region
    if (batch_id >= gridDim.z) return;
    
    // Starting positions for this batch
    const float *A_batch = A + batch_id * batch_stride_A;
    const float *B_batch = B + batch_id * batch_stride_B;
    float *C_batch = C + batch_id * batch_stride_C;
    
    // Tile starting positions
    int start_m = block_m * BLOCK_M;
    int start_n = block_n * BLOCK_N;
    
    // Shared memory for tiles of A and B
    __shared__ float tile_A[BLOCK_K][BLOCK_M];  // A is M x K, so tile is BLOCK_M x BLOCK_K
    __shared__ float tile_B[BLOCK_K][BLOCK_N];  // B is K x N, so tile is BLOCK_K x BLOCK_N
    
    // Accumulator
    float accum = 0.0f;
    
    // Number of K tiles
    int num_k_tiles = (k + BLOCK_K - 1) / BLOCK_K;
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int start_k = k_tile * BLOCK_K;
        
        // Load tile of A (M x K, column-major)
        // A[i, k] = A[i + k * lda]
        // We need A[start_m + thread_m, start_k + k_idx]
        // Each thread loads multiple elements if needed
        for (int k_idx = thread_n; k_idx < BLOCK_K; k_idx += (blockDim.x / BLOCK_M)) {
            int global_m = start_m + thread_m;
            int global_k = start_k + k_idx;
            
            float val = 0.0f;
            if (global_m < m && global_k < k) {
                val = A_batch[global_m + global_k * lda];
            }
            tile_A[k_idx][thread_m] = val;
        }
        
        // Load tile of B (K x N, column-major)
        // B[k, j] = B[k + j * ldb]
        // We need B[start_k + k_idx, start_n + thread_n]
        for (int k_idx = thread_m; k_idx < BLOCK_K; k_idx += BLOCK_M) {
            int global_k = start_k + k_idx;
            int global_n = start_n + thread_n;
            
            float val = 0.0f;
            if (global_k < k && global_n < n) {
                val = B_batch[global_k + global_n * ldb];
            }
            tile_B[k_idx][thread_n] = val;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this K tile
        for (int k_idx = 0; k_idx < BLOCK_K && (start_k + k_idx) < k; k_idx++) {
            accum += tile_A[k_idx][thread_m] * tile_B[k_idx][thread_n];
        }
        
        __syncthreads();
    }
    
    // Write result to C
    int global_m = start_m + thread_m;
    int global_n = start_n + thread_n;
    
    if (global_m < m && global_n < n) {
        int idx = global_m + global_n * ldc;
        float c_val = C_batch[idx];
        C_batch[idx] = alpha * accum + beta * c_val;
    }
}

// Alternative kernel with better memory coalescing - using 2D thread blocks
template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void strided_batched_sgemm_kernel_v2(
    int m, int n, int k,
    float alpha,
    float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta)
{
    // 2D thread block: 16x16 = 256 threads
    int thread_m = threadIdx.x;  // 0 to BLOCK_M-1
    int thread_n = threadIdx.y;  // 0 to BLOCK_N-1
    
    int block_m = blockIdx.x;
    int block_n = blockIdx.y;
    int batch_id = blockIdx.z;
    
    // Starting positions for this batch
    const float *A_batch = A + batch_id * batch_stride_A;
    const float *B_batch = B + batch_id * batch_stride_B;
    float *C_batch = C + batch_id * batch_stride_C;
    
    int start_m = block_m * BLOCK_M;
    int start_n = block_n * BLOCK_N;
    
    __shared__ float tile_A[BLOCK_K][BLOCK_M];
    __shared__ float tile_B[BLOCK_N][BLOCK_K];  // Transposed for coalesced access
    
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Register blocking
    
    int num_k_tiles = (k + BLOCK_K - 1) / BLOCK_K;
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int start_k = k_tile * BLOCK_K;
        
        // Load A: coalesced along M dimension
        // Each thread loads (BLOCK_K / BLOCK_N) elements or uses multiple threads
        for (int k_offset = 0; k_offset < BLOCK_K; k_offset += 1) {
            int load_k = k_offset + (thread_n * BLOCK_M + thread_m) / BLOCK_M;
            int load_m = (thread_n * BLOCK_M + thread_m) % BLOCK_M;
            
            if (load_k < BLOCK_K) {
                int global_m = start_m + load_m;
                int global_k = start_k + load_k;
                float val = 0.0f;
                if (global_m < m && global_k < k) {
                    val = A_batch[global_m + global_k * lda];
                }
                tile_A[load_k][load_m] = val;
            }
        }
        
        // Load B: coalesced - B is column major so consecutive k are consecutive in memory
        for (int k_offset = 0; k_offset < BLOCK_K; k_offset += 1) {
            int load_k = k_offset + (thread_n * BLOCK_M + thread_m) / BLOCK_N;
            int load_n = (thread_n * BLOCK_M + thread_m) % BLOCK_N;
            
            if (load_k < BLOCK_K) {
                int global_k = start_k + load_k;
                int global_n = start_n + load_n;
                float val = 0.0f;
                if (global_k < k && global_n < n) {
                    val = B_batch[global_k + global_n * ldb];
                }
                tile_B[load_n][load_k] = val;  // Transposed
            }
        }
        
        __syncthreads();
        
        // Compute
        if (start_m + thread_m < m && start_n + thread_n < n) {
            for (int k_idx = 0; k_idx < BLOCK_K && (start_k + k_idx) < k; k_idx++) {
                accum[0] += tile_A[k_idx][thread_m] * tile_B[thread_n][k_idx];
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    int global_m = start_m + thread_m;
    int global_n = start_n + thread_n;
    
    if (global_m < m && global_n < n) {
        int idx = global_m + global_n * ldc;
        float c_val = C_batch[idx];
        C_batch[idx] = alpha * accum[0] + beta * c_val;
    }
}

// Optimized kernel with 1D thread block and proper tiling
template <int BM, int BN, int BK, int TM, int TN>
__global__ void strided_batched_sgemm_optimized(
    int m, int n, int k,
    float alpha,
    float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta)
{
    // Each block computes BM x BN tile of C
    // Each thread computes TM x TN elements
    
    const int num_threads = (BM * BN) / (TM * TN);
    
    int thread_idx = threadIdx.x;
    
    // Position within C tile
    int thread_row = (thread_idx / (BN / TN)) * TM;
    int thread_col = (thread_idx % (BN / TN)) * TN;
    
    int block_row = blockIdx.x * BM;
    int block_col = blockIdx.y * BN;
    int batch_id = blockIdx.z;
    
    // Batch offsets
    const float *A_batch = A + batch_id * batch_stride_A;
    const float *B_batch = B + batch_id * batch_stride_B;
    float *C_batch = C + batch_id * batch_stride_C;
    
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    
    float Creg[TM][TN] = {0};
    
    int num_k_tiles = (k + BK - 1) / BK;
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int k_start = k_tile * BK;
        
        // Load A into shared memory
        // A is M x K column major: A[i + k*lda]
        // Each thread loads multiple elements
        for (int load = thread_idx; load < BM * BK; load += num_threads) {
            int load_row = load % BM;
            int load_k = load / BM;
            
            int global_row = block_row + load_row;
            int global_k = k_start + load_k;
            
            float val = 0.0f;
            if (global_row < m && global_k < k) {
                val = A_batch[global_row + global_k * lda];
            }
            As[load_k][load_row] = val;
        }
        
        // Load B into shared memory  
        // B is K x N column major: B[k + j*ldb]
        for (int load = thread_idx; load < BK * BN; load += num_threads) {
            int load_k = load % BK;
            int load_col = load / BK;
            
            int global_k = k_start + load_k;
            int global_col = block_col + load_col;
            
            float val = 0.0f;
            if (global_k < k && global_col < n) {
                val = B_batch[global_k + global_col * ldb];
            }
            Bs[load_k][load_col] = val;
        }
        
        __syncthreads();
        
        // Compute
        for (int kk = 0; kk < BK && (k_start + kk) < k; kk++) {
            // Load A fragment
            float Afrag[TM];
            for (int i = 0; i < TM; i++) {
                Afrag[i] = As[kk][thread_row + i];
            }
            
            // Load B fragment
            float Bfrag[TN];
            for (int j = 0; j < TN; j++) {
                Bfrag[j] = Bs[kk][thread_col + j];
            }
            
            // Outer product
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    Creg[i][j] += Afrag[i] * Bfrag[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write to global memory
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int global_row = block_row + thread_row + i;
            int global_col = block_col + thread_col + j;
            
            if (global_row < m && global_col < n) {
                int idx = global_row + global_col * ldc;
                float c_val = C_batch[idx];
                C_batch[idx] = alpha * Creg[i][j] + beta * c_val;
            }
        }
    }
}

cudaError_t StridedBatchedSgemm(
    int m, int n, int k,
    float alpha,
    float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta,
    int batch_count)
{
    // Choose tile sizes based on problem size
    // Using 128x128 tiles with 8x8 per thread
    
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    
    dim3 block((BM * BN) / (TM * TN));  // 256 threads
    dim3 grid((m + BM - 1) / BM, (n + BN - 1) / BN, batch_count);
    
    strided_batched_sgemm_optimized<BM, BN, BK, TM, TN><<<grid, block>>>(
        m, n, k,
        alpha,
        A, lda, batch_stride_A,
        B, ldb, batch_stride_B,
        C, ldc, batch_stride_C,
        beta
    );
    
    return cudaGetLastError();
}
