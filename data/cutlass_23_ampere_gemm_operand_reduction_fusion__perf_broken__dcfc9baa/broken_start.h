#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>

// BF16 GEMM with operand reduction fusion
// D = alpha * A * B + beta * C
// row_reduction[k] = sum_{m=0..M-1} float(A[m, k]) for k = 0..K-1

// Tile dimensions
#define TILE_M 128
#define TILE_N 128
#define TILE_K 16

// Thread block dimensions
#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 16

// Warp dimensions
#define WARP_M 32
#define WARP_N 8

// Number of warps per block
#define WARPS_M 4
#define WARPS_N 2

// Warp tile dimensions
#define WARP_TILE_M (TILE_M / WARPS_M)  // 32
#define WARP_TILE_N (TILE_N / WARPS_N)  // 64

// Thread tile dimensions (each thread computes an 8x8 tile)
#define THREAD_M 8
#define THREAD_N 8

// Shared memory layout
#define SMEM_A_ROWS TILE_M
#define SMEM_A_COLS TILE_K
#define SMEM_B_ROWS TILE_K
#define SMEM_B_COLS TILE_N

// Use vectorized loads for better memory throughput
#define VECTOR_SIZE 4

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float val) {
    return __float2bfloat16(val);
}

// Kernel to compute row_reduction: sum over M dimension of A for each K
// A is M x K column-major: A[i + k * lda]
// row_reduction[k] = sum_{i=0}^{M-1} A[i, k]
__global__ void ComputeRowReduction(
    int M, int K,
    const __nv_bfloat16* __restrict__ A, int lda,
    float* __restrict__ row_reduction) {
    
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k < K) {
        float sum = 0.0f;
        // Each thread handles one column k, sum over all rows
        for (int i = 0; i < M; i++) {
            sum += bf16_to_float(A[i + k * lda]);
        }
        row_reduction[k] = sum;
    }
}

// Optimized reduction kernel using shared memory
__global__ void ComputeRowReductionOptimized(
    int M, int K,
    const __nv_bfloat16* __restrict__ A, int lda,
    float* __restrict__ row_reduction) {
    
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int k = blockIdx.x;
    
    float sum = 0.0f;
    
    // Each block handles one column k, threads cooperate to sum
    for (int i = tid; i < M; i += blockDim.x) {
        sum += bf16_to_float(A[i + k * lda]);
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        row_reduction[k] = sdata[0];
    }
}

// Main GEMM kernel
// D = alpha * A * B + beta * C
// A: M x K column-major, B: K x N row-major, C: M x N column-major, D: M x N column-major
__global__ void GemmKernel(
    int M, int N, int K, float alpha,
    const __nv_bfloat16* __restrict__ A, int lda,
    const __nv_bfloat16* __restrict__ B, int ldb,
    float beta,
    const __nv_bfloat16* __restrict__ C, int ldc,
    __nv_bfloat16* __restrict__ D, int ldd) {
    
    // Block and thread indices
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;
    
    // Starting positions for this thread block
    int m_start = block_m * TILE_M;
    int n_start = block_n * TILE_N;
    
    // Starting positions for this warp
    int warp_m_start = m_start + warp_m * WARP_TILE_M;
    int warp_n_start = n_start + warp_n * WARP_TILE_N;
    
    // Thread position within warp (8x8 thread tile)
    int thread_m_in_warp = (lane_id / WARP_N) * THREAD_M;
    int thread_n_in_warp = (lane_id % WARP_N) * THREAD_N;
    
    int m_thread = warp_m_start + thread_m_in_warp;
    int n_thread = warp_n_start + thread_n_in_warp;
    
    // Accumulator registers (8x8 = 64 elements per thread)
    float accum[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            accum[i][j] = 0.0f;
        }
    }
    
    // Shared memory for A and B tiles
    // A tile: TILE_M x TILE_K, B tile: TILE_K x TILE_N
    extern __shared__ char smem[];
    __nv_bfloat16* smem_A = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_B = (__nv_bfloat16*)&smem_A[TILE_M * TILE_K];
    
    // Main loop over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        int k_end = min(k_tile + TILE_K, K);
        int k_size = k_end - k_tile;
        
        // Load A tile: TILE_M x TILE_K from column-major A
        // Each thread loads multiple elements
        #pragma unroll
        for (int idx = threadIdx.x; idx < TILE_M * TILE_K; idx += blockDim.x) {
            int smem_m = idx / TILE_K;
            int smem_k = idx % TILE_K;
            
            int global_m = m_start + smem_m;
            int global_k = k_tile + smem_k;
            
            if (global_m < M && global_k < K) {
                smem_A[smem_m * TILE_K + smem_k] = A[global_m + global_k * lda];
            } else {
                smem_A[smem_m * TILE_K + smem_k] = __float2bfloat16_rn(0.0f);
            }
        }
        
        // Load B tile: TILE_K x TILE_N from row-major B
        #pragma unroll
        for (int idx = threadIdx.x; idx < TILE_K * TILE_N; idx += blockDim.x) {
            int smem_k = idx / TILE_N;
            int smem_n = idx % TILE_N;
            
            int global_k = k_tile + smem_k;
            int global_n = n_start + smem_n;
            
            if (global_k < K && global_n < N) {
                smem_B[smem_k * TILE_N + smem_n] = B[global_k * ldb + global_n];
            } else {
                smem_B[smem_k * TILE_N + smem_n] = __float2bfloat16_rn(0.0f);
            }
        }
        
        __syncthreads();
        
        // Compute on this tile
        #pragma unroll
        for (int k = 0; k < k_size; k++) {
            // Load A fragment for this thread
            float a_frag[THREAD_M];
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                int smem_m = thread_m_in_warp + i;
                a_frag[i] = bf16_to_float(smem_A[smem_m * TILE_K + k]);
            }
            
            // Load B fragment for this thread
            float b_frag[THREAD_N];
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                int smem_n = thread_n_in_warp + j;
                b_frag[j] = bf16_to_float(smem_B[k * TILE_N + smem_n]);
            }
            
            // Multiply-accumulate
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    accum[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results with alpha and beta scaling
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        int global_m = m_thread + i;
        if (global_m >= M) continue;
        
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            int global_n = n_thread + j;
            if (global_n >= N) continue;
            
            float result = alpha * accum[i][j];
            
            // Add beta * C if needed
            if (beta != 0.0f) {
                float c_val = bf16_to_float(C[global_m + global_n * ldc]);
                result += beta * c_val;
            }
            
            D[global_m + global_n * ldd] = float_to_bf16(result);
        }
    }
}

// Simpler kernel for better occupancy on smaller problems
__global__ void GemmKernelSimple(
    int M, int N, int K, float alpha,
    const __nv_bfloat16* __restrict__ A, int lda,
    const __nv_bfloat16* __restrict__ B, int ldb,
    float beta,
    const __nv_bfloat16* __restrict__ C, int ldc,
    __nv_bfloat16* __restrict__ D, int ldd) {
    
    // Each thread computes one element of D
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (m >= M || n >= N) return;
    
    float sum = 0.0f;
    
    // Compute dot product of row m of A (column-major) and row n of B (row-major)
    // A[m, k] = A[m + k * lda]
    // B[k, n] = B[k * ldb + n]
    for (int k = 0; k < K; k++) {
        float a = bf16_to_float(A[m + k * lda]);
        float b = bf16_to_float(B[k * ldb + n]);
        sum += a * b;
    }
    
    float result = alpha * sum;
    if (beta != 0.0f) {
        result += beta * bf16_to_float(C[m + n * ldc]);
    }
    
    D[m + n * ldd] = float_to_bf16(result);
}

// Shared memory GEMM kernel with better performance
__global__ void GemmKernelShared(
    int M, int N, int K, float alpha,
    const __nv_bfloat16* __restrict__ A, int lda,
    const __nv_bfloat16* __restrict__ B, int ldb,
    float beta,
    const __nv_bfloat16* __restrict__ C, int ldc,
    __nv_bfloat16* __restrict__ D, int ldd) {
    
    // Tile dimensions
    const int BM = 64;   // Block tile M
    const int BN = 64;   // Block tile N  
    const int BK = 16;   // Block tile K
    
    // Thread block dimensions
    const int TM = 8;    // Threads in M dimension
    const int TN = 8;    // Threads in N dimension
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Starting positions
    int m_start = by * BM;
    int n_start = bx * BN;
    
    // Thread's position within tile
    int m_thread = ty * 8 + (tx / 8);
    int n_thread = (tx % 8) * 8;
    
    // Actual global positions this thread computes
    int m_global[8];
    int n_global[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        m_global[i] = m_start + m_thread + i;
        n_global[i] = n_start + n_thread + i;
    }
    
    // Accumulators
    float accum[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            accum[i][j] = 0.0f;
        }
    }
    
    // Shared memory
    extern __shared__ char smem[];
    __nv_bfloat16* sA = (__nv_bfloat16*)smem;           // BM x BK
    __nv_bfloat16* sB = (__nv_bfloat16*)&sA[BM * BK];   // BK x BN
    
    // Loop over K tiles
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // Load A tile to shared memory (column-major)
        #pragma unroll
        for (int idx = ty * TM + tx; idx < BM * BK; idx += TM * TN) {
            int sm = idx / BK;
            int sk = idx % BK;
            int gm = m_start + sm;
            int gk = k_tile + sk;
            if (gm < M && gk < K) {
                sA[sm * BK + sk] = A[gm + gk * lda];
            } else {
                sA[sm * BK + sk] = __float2bfloat16_rn(0.0f);
            }
        }
        
        // Load B tile to shared memory (row-major)
        #pragma unroll
        for (int idx = ty * TM + tx; idx < BK * BN; idx += TM * TN) {
            int sk = idx / BN;
            int sn = idx % BN;
            int gk = k_tile + sk;
            int gn = n_start + sn;
            if (gk < K && gn < N) {
                sB[sk * BN + sn] = B[gk * ldb + gn];
            } else {
                sB[sk * BN + sn] = __float2bfloat16_rn(0.0f);
            }
        }
        
        __syncthreads();
        
        // Compute on this tile
        int k_end = min(BK, K - k_tile);
        #pragma unroll
        for (int k = 0; k < k_end; k++) {
            // Load A values for this thread
            float a_vals[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                a_vals[i] = bf16_to_float(sA[(m_thread + i) * BK + k]);
            }
            
            // Load B values for this thread
            float b_vals[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                b_vals[j] = bf16_to_float(sB[k * BN + n_thread + j]);
            }
            
            // Multiply-accumulate
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    accum[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (m_global[i] >= M) continue;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            if (n_global[j] >= N) continue;
            
            float result = alpha * accum[i][j];
            if (beta != 0.0f) {
                result += beta * bf16_to_float(C[m_global[i] + n_global[j] * ldc]);
            }
            D[m_global[i] + n_global[j] * ldd] = float_to_bf16(result);
        }
    }
}

cudaError_t GemmWithReduction(
    int M, int N, int K, float alpha,
    __nv_bfloat16 const *A, int lda,
    __nv_bfloat16 const *B, int ldb,
    float beta,
    __nv_bfloat16 const *C, int ldc,
    __nv_bfloat16 *D, int ldd,
    float *row_reduction) {
    
    cudaError_t err;
    
    // Compute row_reduction: sum over M dimension of A for each K
    // A is M x K column-major, so each column k is contiguous in memory
    // Use optimized reduction with one block per column
    
    // Choose block size based on M
    int reduction_block_size = 256;
    if (M < 256) reduction_block_size = 128;
    if (M < 128) reduction_block_size = 64;
    
    // Shared memory size for reduction
    size_t reduction_smem = reduction_block_size * sizeof(float);
    
    ComputeRowReductionOptimized<<<K, reduction_block_size, reduction_smem>>>(
        M, K, A, lda, row_reduction);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    // Choose GEMM kernel based on problem size
    // For large problems, use tiled shared memory kernel
    // For small problems, use simple kernel
    
    if (M <= 128 || N <= 128 || K <= 64) {
        // Small problem: use simple kernel
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        GemmKernelSimple<<<grid, block>>>(
            M, N, K, alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc,
            D, ldd);
    } else {
        // Large problem: use shared memory kernel
        // Tile size: 64x64 with 8x8 thread tiles
        const int BM = 64;
        const int BN = 64;
        const int BK = 16;
        
        dim3 block(8, 8);  // 64 threads per block
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        
        size_t smem_size = (BM * BK + BK * BN) * sizeof(__nv_bfloat16);
        
        GemmKernelShared<<<grid, block, smem_size>>>(
            M, N, K, alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc,
            D, ldd);
    }
    
    err = cudaGetLastError();
    return err;
}
