#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// TF32 GEMM using Ampere Tensor Cores via WMMA
// C = alpha * A * B + beta * C
// A: row-major (M x K), B: column-major (K x N), C: row-major (M x N)

// Use WMMA for TF32 tensor core access
#include <mma.h>

using namespace nvcuda;

// Tile dimensions - using 64x64 output tiles with 16x16 WMMA operations
// Each warp computes a 64x64 tile using 4x4 WMMA tiles (16x16 each)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;  // TF32 uses K=8

// Thread block configuration
constexpr int BLOCK_M = 64;   // Rows of C per block
constexpr int BLOCK_N = 64;   // Cols of C per block  
constexpr int BLOCK_K = 32;   // K dimension per block

// Warps per block: 8 warps = 256 threads
constexpr int WARP_M = 32;    // Rows per warp in output tile
constexpr int WARP_N = 64;    // Cols per warp in output tile

// Each warp computes 32x64 using 2x4 WMMA tiles
constexpr int WARPS_M = 2;    // BLOCK_M / WARP_M = 64/32 = 2
constexpr int WARPS_N = 1;    // BLOCK_N / WARP_N = 64/64 = 1
constexpr int NUM_WARPS = WARPS_M * WARPS_N; // 2 warps

// Actually let's use 4 warps for better occupancy
// Reconfigure: BLOCK_M=64, BLOCK_N=64, with 4 warps (2x2 arrangement)
constexpr int WARP_M_4 = 32;  // 64/2
constexpr int WARP_N_4 = 32;  // 64/2

__global__ void tf32_gemm_kernel(
    int M, int N, int K,
    float alpha,
    float const * __restrict__ A, int lda,
    float const * __restrict__ B, int ldb,
    float beta,
    float * __restrict__ C, int ldc)
{
    // Block indices
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    
    // Thread index
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Warp position within block (2x2 arrangement)
    int warp_m = warp_id / 2;  // 0 or 1
    int warp_n = warp_id % 2;  // 0 or 1
    
    // Starting positions for this block
    int m_base = block_m * BLOCK_M;
    int n_base = block_n * BLOCK_N;
    
    // Starting positions for this warp
    int warp_m_base = m_base + warp_m * WARP_M_4;
    int warp_n_base = n_base + warp_n * WARP_N_4;
    
    // WMMA fragments
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag;
    
    // Initialize accumulators to zero
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }
    
    // Shared memory for A and B tiles
    // A tile: BLOCK_M x BLOCK_K, row-major
    // B tile: BLOCK_K x BLOCK_N, column-major (but stored as row-major in smem for coalescing)
    __shared__ float smem_A[BLOCK_M][BLOCK_K + 4];  // +4 for padding
    __shared__ float smem_B[BLOCK_N][BLOCK_K + 4];  // Transposed view: actually K x N, stored as N x K
    
    // Main loop over K
    for (int k_base = 0; k_base < K; k_base += BLOCK_K) {
        // Load A tile: BLOCK_M x BLOCK_K from global to shared
        // A is row-major: A[i,k] at A[i*lda + k]
        // Each thread loads multiple elements
        #pragma unroll
        for (int load = 0; load < (BLOCK_M * BLOCK_K + blockDim.x - 1) / blockDim.x; load++) {
            int idx = tid + load * blockDim.x;
            int smem_m = idx / BLOCK_K;
            int smem_k = idx % BLOCK_K;
            
            if (smem_m < BLOCK_M && k_base + smem_k < K) {
                int global_m = m_base + smem_m;
                int global_k = k_base + smem_k;
                if (global_m < M) {
                    smem_A[smem_m][smem_k] = A[global_m * lda + global_k];
                } else {
                    smem_A[smem_m][smem_k] = 0.0f;
                }
            }
        }
        
        // Load B tile: BLOCK_K x BLOCK_N from global to shared
        // B is column-major: B[k,j] at B[k + j*ldb]
        // Store in smem_B as row-major (j,k) for better access pattern
        #pragma unroll
        for (int load = 0; load < (BLOCK_N * BLOCK_K + blockDim.x - 1) / blockDim.x; load++) {
            int idx = tid + load * blockDim.x;
            int smem_n = idx / BLOCK_K;  // j index
            int smem_k = idx % BLOCK_K;  // k index
            
            if (smem_n < BLOCK_N && k_base + smem_k < K) {
                int global_n = n_base + smem_n;
                int global_k = k_base + smem_k;
                if (global_n < N) {
                    // B is column-major: B[k,j] = B[k + j*ldb]
                    smem_B[smem_n][smem_k] = B[global_k + global_n * ldb];
                } else {
                    smem_B[smem_n][smem_k] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Compute using WMMA
        // Each warp computes 32x32 output using 2x2 WMMA tiles (16x16 each)
        // WMMA_K = 8, so we need BLOCK_K/8 = 4 iterations
        
        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += WMMA_K) {
            // Load A fragments (2 per warp in M dimension)
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int a_row = warp_m * WARP_M_4 + i * WMMA_M;
                int a_col = k_step;
                
                // Load from shared memory to fragment
                wmma::load_matrix_sync(a_frag, &smem_A[a_row][a_col], BLOCK_K + 4);
                
                // Load B fragments (2 per warp in N dimension)
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    int b_row = warp_n * WARP_N_4 + j * WMMA_N;  // This is N dimension in output
                    // B in shared memory: smem_B[n][k], we need k as first dim for col_major WMMA
                    // Actually WMMA expects col_major B, so we load smem_B directly
                    // smem_B[b_row][k_step] gives us B[n,k] which matches col_major layout
                    wmma::load_matrix_sync(b_frag, &smem_B[b_row][k_step], BLOCK_K + 4);
                    
                    // Perform MMA
                    wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results to global memory with alpha and beta scaling
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int c_row = warp_m * WARP_M_4 + i * WMMA_M;
            int c_col = warp_n * WARP_N_4 + j * WMMA_N;
            
            // Use shared memory as intermediate storage for store
            __shared__ float smem_C[BLOCK_M][BLOCK_N + 4];
            
            // Store fragment to shared memory first
            wmma::store_matrix_sync(&smem_C[c_row][c_col], acc[i][j], BLOCK_N + 4, wmma::mem_row_major);
            
            __syncthreads();
            
            // Now write to global memory with alpha/beta
            // Each thread handles multiple elements
            #pragma unroll
            for (int store = 0; store < (BLOCK_M * BLOCK_N + blockDim.x - 1) / blockDim.x; store++) {
                int idx = tid + store * blockDim.x;
                int local_m = idx / BLOCK_N;
                int local_n = idx % BLOCK_N;
                
                if (local_m < BLOCK_M && local_n < BLOCK_N) {
                    int global_m = m_base + local_m;
                    int global_n = n_base + local_n;
                    
                    if (global_m < M && global_n < N) {
                        float result = alpha * smem_C[local_m][local_n];
                        if (beta != 0.0f) {
                            result += beta * C[global_m * ldc + global_n];
                        }
                        C[global_m * ldc + global_n] = result;
                    }
                }
            }
            
            __syncthreads();
        }
    }
}

// Alternative kernel with better structure
__global__ void tf32_gemm_kernel_v2(
    int M, int N, int K,
    float alpha,
    float const * __restrict__ A, int lda,
    float const * __restrict__ B, int ldb,
    float beta,
    float * __restrict__ C, int ldc)
{
    // Use 128 threads (4 warps) per block
    // Block tile: 64x64
    // Each warp: 32x32 output
    
    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int laneId = tid & 31;
    
    // 2x2 warp arrangement
    const int warpM = warpId >> 1;
    const int warpN = warpId & 1;
    
    const int blockM = blockIdx.y * 64;
    const int blockN = blockIdx.x * 64;
    
    const int warpMBase = blockM + warpM * 32;
    const int warpNBase = blockN + warpN * 32;
    
    // WMMA fragments - 2x2 grid per warp
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag[2][2];
    
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);
    
    // Shared memory
    __shared__ __align__(128) float smemA[64][36];  // 64x32 padded
    __shared__ __align__(128) float smemB[64][36];  // 64x32 padded (N x K layout)
    
    for (int k0 = 0; k0 < K; k0 += 32) {
        // Load A: 64x32 from global (row-major) to smem
        // Each thread loads 4 elements (128 threads * 4 = 512, but we need 64*32=2048)
        // Actually 128 threads, need 16 loads per thread for 2048 elements
        #pragma unroll
        for (int t = 0; t < 16; t++) {
            int idx = tid + t * 128;
            int row = idx >> 5;  // / 32
            int col = idx & 31;  // % 32
            
            if (row < 64) {
                int globalM = blockM + row;
                int globalK = k0 + col;
                float val = 0.0f;
                if (globalM < M && globalK < K) {
                    val = A[globalM * lda + globalK];
                }
                smemA[row][col] = val;
            }
        }
        
        // Load B: 32x64 from global (col-major) to smem
        // smemB layout: 64 rows (N), 32 cols (K)
        // B[k,j] at B[k + j*ldb]
        #pragma unroll
        for (int t = 0; t < 16; t++) {
            int idx = tid + t * 128;
            int row = idx >> 5;  // N index (0-63)
            int col = idx & 31;  // K index (0-31)
            
            if (row < 64) {
                int globalN = blockN + row;
                int globalK = k0 + col;
                float val = 0.0f;
                if (globalN < N && globalK < K) {
                    val = B[globalK + globalN * ldb];
                }
                smemB[row][col] = val;
            }
        }
        
        __syncthreads();
        
        // Compute with WMMA
        #pragma unroll
        for (int k1 = 0; k1 < 32; k1 += 8) {
            // Load A fragments
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int aRow = warpM * 32 + i * 16;
                wmma::load_matrix_sync(a_frag[i], &smemA[aRow][k1], 36);
            }
            
            // Load B fragments  
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int bRow = warpN * 32 + j * 16;
                wmma::load_matrix_sync(b_frag[j], &smemB[bRow][k1], 36);
            }
            
            // MMA
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store to global
    // Use shared memory for coalesced write
    __shared__ __align__(128) float smemC[64][68];
    
    // Store fragments to shared
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int cRow = warpM * 32 + i * 16;
            int cCol = warpN * 32 + j * 16;
            wmma::store_matrix_sync(&smemC[cRow][cCol], c_frag[i][j], 68, wmma::mem_row_major);
        }
    }
    
    __syncthreads();
    
    // Write to global with alpha/beta
    #pragma unroll
    for (int t = 0; t < 32; t++) {
        int idx = tid + t * 128;
        int row = idx >> 6;  // / 64
        int col = idx & 63;  // % 64
        
        if (row < 64 && col < 64) {
            int globalM = blockM + row;
            int globalN = blockN + col;
            if (globalM < M && globalN < N) {
                float val = alpha * smemC[row][col];
                if (beta != 0.0f) {
                    val += beta * C[globalM * ldc + globalN];
                }
                C[globalM * ldc + globalN] = val;
            }
        }
    }
}

// Simpler, more reliable kernel
__global__ void tf32_gemm_simple(
    int M, int N, int K,
    float alpha,
    float const * __restrict__ A, int lda,
    float const * __restrict__ B, int ldb,
    float beta,
    float * __restrict__ C, int ldc)
{
    // Block: 64x64 threads organized as 2 warps x 32 threads
    // Actually use 128 threads = 4 warps
    
    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    
    // Position in output tile
    // 4 warps in 2x2 arrangement
    const int warpRow = (warpId / 2) * 32;  // 0 or 32
    const int warpCol = (warpId % 2) * 32;  // 0 or 32
    
    const int blockRow = blockIdx.y * 64;
    const int blockCol = blockIdx.x * 64;
    
    // WMMA: 16x16x8 tiles
    // Each warp handles 32x32 = 2x2 WMMA tiles
    
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2][2];
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b_frag;
    
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);
    
    __shared__ float As[64][40];
    __shared__ float Bs[64][40];
    
    for (int kk = 0; kk < K; kk += 32) {
        // Load A tile (64x32, row-major)
        for (int t = tid; t < 64 * 32; t += 128) {
            int r = t / 32;
            int c = t % 32;
            int gr = blockRow + r;
            int gc = kk + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * lda + gc] : 0.0f;
        }
        
        // Load B tile (32x64, col-major) 
        // Store as 64x32 in shared (transposed for access)
        for (int t = tid; t < 64 * 32; t += 128) {
            int n = t / 32;  // 0-63 (N dim)
            int k = t % 32;  // 0-31 (K dim)
            int gn = blockCol + n;
            int gk = kk + k;
            // B is col-major: B[k,n] at B[k + n*ldb]
            Bs[n][k] = (gn < N && gk < K) ? B[gk + gn * ldb] : 0.0f;
        }
        
        __syncthreads();
        
        // Compute
        for (int k1 = 0; k1 < 32; k1 += 8) {
            for (int i = 0; i < 2; i++) {
                int aRow = warpRow + i * 16;
                wmma::load_matrix_sync(a_frag, &As[aRow][k1], 40);
                
                for (int j = 0; j < 2; j++) {
                    int bRow = warpCol + j * 16;
                    wmma::load_matrix_sync(b_frag, &Bs[bRow][k1], 40);
                    wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store
    __shared__ float Cs[64][68];
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int cRow = warpRow + i * 16;
            int cCol = warpCol + j * 16;
            wmma::store_matrix_sync(&Cs[cRow][cCol], acc[i][j], 68, wmma::mem_row_major);
        }
    }
    
    __syncthreads();
    
    for (int t = tid; t < 64 * 64; t += 128) {
        int r = t / 64;
        int c = t % 64;
        int gr = blockRow + r;
        int gc = blockCol + c;
        if (gr < M && gc < N) {
            float val = alpha * Cs[r][c];
            if (beta != 0.0f)
                val += beta * C[gr * ldc + gc];
            C[gr * ldc + gc] = val;
        }
    }
}

cudaError_t Tf32Gemm(
    int M, int N, int K,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float *C, int ldc)
{
    if (M <= 0 || N <= 0 || K <= 0) {
        return cudaSuccess;
    }
    
    dim3 block(128);  // 4 warps
    dim3 grid((N + 63) / 64, (M + 63) / 64);
    
    tf32_gemm_simple<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    
    return cudaGetLastError();
}
