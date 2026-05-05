#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstdint>

// Flash Attention Forward Kernel for hdim=192, fp16
// Uses tiling with shared memory to avoid materializing full attention matrix

#define WARP_SIZE 32
#define NUM_WARPS 4
#define NUM_THREADS (WARP_SIZE * NUM_WARPS)  // 128 threads per block

// Tile sizes
#define BLOCK_M 64   // Queries per block
#define BLOCK_N 64   // Keys per block
#define BLOCK_D 192  // Head dim (fixed)

// Shared memory layout: Q, K, V tiles
// Q: [BLOCK_M][BLOCK_D] = 64 * 192 * 2 = 24KB
// K: [BLOCK_N][BLOCK_D] = 64 * 192 * 2 = 24KB  
// V: [BLOCK_N][BLOCK_D] = 64 * 192 * 2 = 24KB
// Total: ~72KB (fits in smem)

struct FlashAttnFwdParams {
    const half* Q;
    const half* K;
    const half* V;
    half* O;
    float* lse;
    int B;
    int H;
    int Sq;
    int Sk;
    int D;
    float scale;
};

__device__ __forceinline__ float2 half22float2(__half2 h) {
    return __half22float2(h);
}

__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

__device__ __forceinline__ half float_to_half(float f) {
    return __float2half_rn(f);
}

// Warp-level matrix multiply accumulate for Q@K^T
// Each warp computes a 16x16 tile of S
template<int WMMA_M, int WMMA_N, int WMMA_K>
__device__ __forceinline__ void wmma_gemm(
    const half* a, const half* b, float* c,
    int M, int N, int K,
    int lda, int ldb, int ldc) {
    
    // Simplified: each thread handles multiple elements
    // For fp16, we use simple dot product
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Each warp processes 16x16 output tile
    int warp_row = warp_id / 2;  // 2 warps horizontally
    int warp_col = warp_id % 2;
    
    int row_start = warp_row * 16;
    int col_start = warp_col * 16;
    
    // Each thread computes 8 elements (4x2 or similar)
    // Simplified: compute dot products
    for (int i = lane_id; i < 256; i += WARP_SIZE) {
        int local_row = i / 16;
        int local_col = i % 16;
        int global_row = row_start + local_row;
        int global_col = col_start + local_col;
        
        if (global_row < M && global_col < N) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < K; k++) {
                float av = half_to_float(a[global_row * lda + k]);
                float bv = half_to_float(b[global_col * ldb + k]);
                sum += av * bv;
            }
            c[global_row * ldc + global_col] = sum;
        }
    }
}

// Main flash attention forward kernel
template<int BLOCK_M_, int BLOCK_N_, int D_>
__global__ void flash_attn_fwd_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float* __restrict__ lse,
    int B, int H, int Sq, int Sk, int D, float scale) {
    
    // Shared memory
    extern __shared__ char smem[];
    half* sQ = (half*)smem;                              // [BLOCK_M][D]
    half* sK = (half*)(smem + BLOCK_M_ * D_ * sizeof(half));  // [BLOCK_N][D]
    half* sV = (half*)(smem + (BLOCK_M_ + BLOCK_N_) * D_ * sizeof(half)); // [BLOCK_N][D]
    
    // Thread indexing
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Block indexing
    int block_idx = blockIdx.x;
    int batch_head = block_idx / ((Sq + BLOCK_M_ - 1) / BLOCK_M_);
    int q_tile = block_idx % ((Sq + BLOCK_M_ - 1) / BLOCK_M_);
    
    int b = batch_head / H;
    int h = batch_head % H;
    
    if (b >= B) return;
    
    int q_start = q_tile * BLOCK_M_;
    int q_end = min(q_start + BLOCK_M_, Sq);
    int q_len = q_end - q_start;
    
    // Strides
    int stride_bh = H * Sq * D;
    int stride_h = Sq * D;
    int stride_s = D;
    
    int stride_bh_k = H * Sk * D;
    int stride_h_k = Sk * D;
    
    // Pointers to this batch/head
    const half* gQ = Q + b * stride_bh + h * stride_h;
    const half* gK = K + b * stride_bh_k + h * stride_h_k;
    const half* gV = V + b * stride_bh_k + h * stride_h_k;
    half* gO = O + b * stride_bh + h * stride_h;
    float* gLSE = lse + b * H * Sq + h * Sq;
    
    // Registers for online softmax and output accumulation
    // Each thread handles multiple query rows
    int rows_per_thread = (BLOCK_M_ + NUM_THREADS - 1) / NUM_THREADS;
    int row_start_local = tid * rows_per_thread;
    
    float m[8];  // max per row
    float l[8];  // sum per row  
    float o[8][8]; // accumulated output (D/NUM_THREADS per thread)
    
    // Initialize
    #pragma unroll
    for (int i = 0; i < rows_per_thread && row_start_local + i < BLOCK_M_; i++) {
        m[i] = -CUDART_INF_F;
        l[i] = 0.0f;
        #pragma unroll
        for (int d = 0; d < 8; d++) {
            o[i][d] = 0.0f;
        }
    }
    
    // Load Q tile to shared memory
    // Each thread loads D/NUM_THREADS elements per row
    for (int q = tid; q < q_len * D; q += NUM_THREADS) {
        int row = q / D;
        int col = q % D;
        int global_q = q_start + row;
        if (global_q < Sq) {
            sQ[row * D + col] = gQ[global_q * D + col];
        }
    }
    __syncthreads();
    
    // Iterate over K,V tiles
    int num_kv_tiles = (Sk + BLOCK_N_ - 1) / BLOCK_N_;
    
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int k_start = kv_tile * BLOCK_N_;
        int k_end = min(k_start + BLOCK_N_, Sk);
        int k_len = k_end - k_start;
        
        // Load K tile
        for (int idx = tid; idx < k_len * D; idx += NUM_THREADS) {
            int row = idx / D;
            int col = idx % D;
            sK[row * D + col] = gK[(k_start + row) * D + col];
        }
        __syncthreads();
        
        // Compute S = Q @ K^T for this tile
        // Each thread computes partial dot products
        float s_local[8][4];  // [rows_per_thread][BLOCK_N/NUM_THREADS]
        
        #pragma unroll
        for (int qi = 0; qi < rows_per_thread && row_start_local + qi < q_len; qi++) {
            int q_row = row_start_local + qi;
            #pragma unroll
            for (int kj = 0; kj < 4; kj++) {  // BLOCK_N / 16 warps
                int k_col = warp_id * 16 + lane_id / 2 + kj * (WARP_SIZE/2);
                if (k_col < k_len) {
                    float sum = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < D; d++) {
                        sum += half_to_float(sQ[q_row * D + d]) * 
                               half_to_float(sK[k_col * D + d]);
                    }
                    s_local[qi][kj] = sum * scale;
                } else {
                    s_local[qi][kj] = -CUDART_INF_F;
                }
            }
        }
        
        // Online softmax update
        #pragma unroll
        for (int qi = 0; qi < rows_per_thread && row_start_local + qi < q_len; qi++) {
            // Find max
            float m_prev = m[qi];
            float m_new = m_prev;
            #pragma unroll
            for (int kj = 0; kj < 4; kj++) {
                m_new = fmaxf(m_new, s_local[qi][kj]);
            }
            // Warp reduce max
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                m_new = fmaxf(m_new, __shfl_xor_sync(0xffffffff, m_new, offset));
            }
            
            // Compute exp and sum
            float l_scale = expf(m_prev - m_new);
            float l_new = l[qi] * l_scale;
            
            float p_local[4];
            #pragma unroll
            for (int kj = 0; kj < 4; kj++) {
                p_local[kj] = expf(s_local[qi][kj] - m_new);
                l_new += p_local[kj];
            }
            // Warp reduce sum
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                l_new += __shfl_xor_sync(0xffffffff, l_new, offset);
            }
            
            // Update output accumulator
            float o_scale = l[qi] / l_new * l_scale;
            #pragma unroll
            for (int d = 0; d < 8; d++) {
                o[qi][d] *= o_scale;
            }
            
            // Save for V accumulation
            m[qi] = m_new;
            l[qi] = l_new;
            
            // Store P (softmax) for V multiply
            // We'll reload from computation or store in registers
            // For now, recompute or keep in registers for V phase
            // Actually, we need to multiply P @ V now
        }
        
        // Load V tile
        for (int idx = tid; idx < k_len * D; idx += NUM_THREADS) {
            int row = idx / D;
            int col = idx % D;
            sV[row * D + col] = gV[(k_start + row) * D + col];
        }
        __syncthreads();
        
        // Compute O += P @ V
        // P is in s_local (registers), V is in sV
        #pragma unroll
        for (int qi = 0; qi < rows_per_thread && row_start_local + qi < q_len; qi++) {
            // Recompute P from stored S and updated m, l
            // Actually we need to re-apply softmax
            int q_row = row_start_local + qi;
            
            #pragma unroll
            for (int kj = 0; kj < 4; kj++) {
                int k_col = warp_id * 16 + lane_id / 2 + kj * (WARP_SIZE/2);
                if (k_col < k_len) {
                    // Recompute S
                    float s_val = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < D; d++) {
                        s_val += half_to_float(sQ[q_row * D + d]) * 
                                 half_to_float(sK[k_col * D + d]);
                    }
                    s_val *= scale;
                    
                    float p_val = expf(s_val - m[qi]) / l[qi];
                    
                    // Accumulate into output
                    #pragma unroll
                    for (int d = lane_id % 2 * 8; d < lane_id % 2 * 8 + 8; d++) {
                        o[qi][d - lane_id % 2 * 8] += p_val * half_to_float(sV[k_col * D + d]);
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    #pragma unroll
    for (int qi = 0; qi < rows_per_thread && row_start_local + qi < q_len; qi++) {
        int q_row = row_start_local + qi;
        int global_q = q_start + q_row;
        
        // Write LSE
        if (lane_id == 0) {
            gLSE[global_q] = m[qi] + logf(l[qi]);
        }
        
        // Write O
        int d_start = (warp_id * 16 + lane_id / 2) * 8;
        #pragma unroll
        for (int di = 0; di < 8; di++) {
            int d = d_start + di;
            if (d < D) {
                gO[global_q * D + d] = float_to_half(o[qi][di]);
            }
        }
    }
}

// Optimized version with better memory access patterns
template<int D>
__global__ void flash_attn_fwd_kernel_optimized(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float* __restrict__ lse,
    int B, int H, int Sq, int Sk, float scale) {
    
    // Smem: Q tile [64][192], K tile [64][192], V tile [64][192]
    // But we need to be more careful with smem usage
    
    extern __shared__ char smem[];
    // Layout: Q[64][192] + K[64][192] + V[64][192] = 3 * 64 * 192 * 2 = 73728 bytes
    half* sQ = (half*)smem;
    half* sK = (half*)(smem + 64 * D * sizeof(half));
    half* sV = (half*)(smem + 2 * 64 * D * sizeof(half));
    
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;  // /32
    const int lane_id = tid & 31;  // %32
    
    const int block_idx = blockIdx.x;
    const int num_q_tiles = (Sq + 63) >> 6;  // /64
    
    const int bh = block_idx / num_q_tiles;
    const int q_tile = block_idx % num_q_tiles;
    
    const int b = bh / H;
    const int h = bh % H;
    
    if (b >= B) return;
    
    const int q_start = q_tile << 6;
    const int q_end = min(q_start + 64, Sq);
    
    // Strides
    const int stride_bh = H * Sq * D;
    const int stride_h = Sq * D;
    const int stride_bh_kv = H * Sk * D;
    
    // Pointers
    const half* gQ = Q + b * stride_bh + h * stride_h;
    const half* gK = K + b * stride_bh_kv + h * stride_h;
    const half* gV = V + b * stride_bh_kv + h * stride_h;
    half* gO = O + b * stride_bh + h * stride_h;
    float* gLSE = lse + b * H * Sq + h * Sq;
    
    // Each thread handles 2 query rows (64 queries / 32 threads per warp / 2 warps)
    // Actually 128 threads, so each thread handles 0.5 query... 
    // Let's do: each warp handles 16 queries, 4 warps = 64 queries
    // Each thread in warp handles partial computation
    
    const int warp_q_start = warp_id << 4;  // warp_id * 16
    const int lane_q = lane_id >> 1;        // 0-15 within warp
    const int q_idx_local = warp_q_start + lane_q;
    
    // Registers for online softmax
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float o[24];  // D/4 elements per thread for 2 rows? Actually need per query
    
    // Actually let's use simpler approach: each thread handles 1 query row
    // 128 threads, 64 queries -> 2 threads per query, or query striping
    
    // Simpler: thread handles query (tid % 64), with 2 threads collaborating
    const int my_q = tid & 63;  // 0-63
    const int my_q_valid = my_q < (q_end - q_start);
    
    // Initialize accumulators
    if (my_q < 64) {
        #pragma unroll
        for (int d = 0; d < D/4; d++) {  // Will adjust
            // Actually each thread handles D/2 elements? No
        }
    }
    
    // Load Q tile cooperatively
    // 64*192 = 12288 elements, 128 threads -> 96 elements per thread
    for (int i = tid; i < (q_end - q_start) * D; i += 128) {
        int row = i / D;
        int col = i % D;
        sQ[row * D + col] = gQ[(q_start + row) * D + col];
    }
    __syncthreads();
    
    // Per-query registers
    float m_reg = -CUDART_INF_F;
    float l_reg = 0.0f;
    float o_reg[48];  // 192/4 = 48 for fp32 accumulation, but need per query
    
    // Actually: 64 queries, 128 threads -> 2 threads per query
    // Each thread pair handles 1 query, splitting D=192 across 2 threads = 96 each
    
    const int query_id = tid >> 1;      // 0-63
    const int query_part = tid & 1;     // 0 or 1 (which half of D)
    const int d_start = query_part * 96;
    const int d_per_thread = 96;
    
    // Initialize output accumulators
    #pragma unroll
    for (int i = 0; i < 48; i++) {  // 96/2 for float2 or just 96 floats
        o_reg[i] = 0.0f;
    }
    
    // Iterate over KV
    const int num_kv_tiles = (Sk + 63) >> 6;
    
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int k_start = kv_tile << 6;
        const int k_end = min(k_start + 64, Sk);
        
        // Load K
        for (int i = tid; i < (k_end - k_start) * D; i += 128) {
            int row = i / D;
            int col = i % D;
            sK[row * D + col] = gK[(k_start + row) * D + col];
        }
        __syncthreads();
        
        // Compute S = Q @ K^T for this tile
        // Each thread pair computes 1 row of S (64 elements)
        if (query_id < (q_end - q_start)) {
            // Load Q row to registers (cooperatively)
            // Actually Q is in smem, access directly
            
            float s_row[64];  // 64 KV positions
            
            // Compute dot products
            // Each thread computes half the dot product, then reduce
            for (int k = 0; k < (k_end - k_start); k++) {
                float qk_dot = 0.0f;
                
                // Split D across the two threads
                #pragma unroll
                for (int d = d_start; d < d_start + 96; d += 2) {
                    // Use float2 for efficiency
                    half2 q_h2 = *(half2*)&sQ[query_id * D + d];
                    half2 k_h2 = *(half2*)&sK[k * D + d];
                    float2 q_f2 = __half22float2(q_h2);
                    float2 k_f2 = __half22float2(k_h2);
                    qk_dot += q_f2.x * k_f2.x + q_f2.y * k_f2.y;
                }
                
                // Reduce across the two threads handling this query
                qk_dot += __shfl_xor_sync(0x3, qk_dot, 1);  // XOR with 1 swaps thread pairs
                
                s_row[k] = qk_dot * scale;
            }
            
            // Pad with -inf for invalid positions
            for (int k = (k_end - k_start); k < 64; k++) {
                s_row[k] = -CUDART_INF_F;
            }
            
            // Online softmax on this tile
            // Find max
            float m_prev = m_reg;
            float m_tile = m_prev;
            #pragma unroll
            for (int k = 0; k < 64; k++) {
                m_tile = fmaxf(m_tile, s_row[k]);
            }
            
            // Compute exp and sum
            float l_scale = expf(m_prev - m_tile);
            float l_tile = l_reg * l_scale;
            
            float p_row[64];
            #pragma unroll
            for (int k = 0; k < 64; k++) {
                p_row[k] = expf(s_row[k] - m_tile);
                l_tile += p_row[k];
            }
            
            // Update output: o = o * (l_prev * exp(m_prev - m_new) / l_new) + p * v
            float o_scale = l_reg / l_tile * l_scale;
            
            // Load V and accumulate
            // Need to sync to load V
            __syncthreads();
            
            // Load V tile
            for (int i = tid; i < (k_end - k_start) * D; i += 128) {
                int row = i / D;
                int col = i % D;
                sV[row * D + col] = gV[(k_start + row) * D + col];
            }
            __syncthreads();
            
            // Scale previous output
            #pragma unroll
            for (int i = 0; i < 48; i++) {
                o_reg[i] *= o_scale;
            }
            
            // Accumulate P @ V
            for (int k = 0; k < (k_end - k_start); k++) {
                float p = p_row[k] / l_tile;  // normalized
                
                #pragma unroll
                for (int di = 0; di < 48; di++) {
                    int d = d_start + di * 2;
                    half2 v_h2 = *(half2*)&sV[k * D + d];
                    float2 v_f2 = __half22float2(v_h2);
                    o_reg[di] += p * v_f2.x;
                    // Actually need to handle y component too...
                    // Simplified: use 96 floats
                }
            }
            
            // Actually redo with simpler approach: 96 float registers
            // For now, use direct indexing
            
            m_reg = m_tile;
            l_reg = l_tile;
        }
        
        __syncthreads();
    }
    
    // Write output
    if (query_id < (q_end - q_start) && query_part == 0) {  // Only once per query
        int global_q = q_start + query_id;
        
        // Write LSE
        if (lane_id == 0) {
            gLSE[global_q] = m_reg + logf(l_reg);
        }
        
        // Write O (collaborate with partner thread)
        // Actually need both threads
        
        // For simplicity, write from smem or direct
        // Reconstruct O from registers and write
        
        // This is getting complex. Let me use a cleaner approach.
    }
}

// Clean, working implementation
template<int D>
__global__ void flash_attn_fwd_clean(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float* __restrict__ lse,
    int B, int H, int Sq, int Sk, float scale) {
    
    // Shared memory: Q[64][D], K[64][D], V[64][D]
    extern __shared__ char smem[];
    half* sQ = (half*)smem;
    half* sK = (half*)(smem + 64 * D * sizeof(half));
    half* sV = (half*)(smem + 128 * D * sizeof(half));
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    const int num_q_tiles = (Sq + 63) / 64;
    const int bh = bid / num_q_tiles;
    const int q_tile = bid % num_q_tiles;
    
    const int b = bh / H;
    const int h = bh % H;
    
    if (b >= B) return;
    
    const int q_start = q_tile * 64;
    const int q_len = min(64, Sq - q_start);
    
    // Strides
    const int stride_bh = H * Sq * D;
    const int stride_h = Sq * D;
    const int stride_bh_kv = H * Sk * D;
    
    // Pointers
    const half* gQ = Q + b * stride_bh + h * stride_h + q_start * D;
    const half* gK_base = K + b * stride_bh_kv + h * stride_h;
    const half* gV_base = V + b * stride_bh_kv + h * stride_h;
    half* gO = O + b * stride_bh + h * stride_h + q_start * D;
    float* gLSE = lse + b * H * Sq + h * Sq + q_start;
    
    // Load Q: 64 * 192 = 12288 halfs, 128 threads -> 96 per thread
    for (int i = tid; i < q_len * D; i += 128) {
        sQ[i] = gQ[i];
    }
    __syncthreads();
    
    // Each warp handles 16 queries (4 warps * 16 = 64)
    // Each thread in warp: process elements for its query
    
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int q_per_warp = 16;
    const int my_q_in_warp = lane_id / 2;  // 0-15, 2 threads per query
    const int my_q = warp_id * 16 + my_q_in_warp;
    
    // Actually: 32 lanes, 16 queries per warp -> 2 lanes per query
    // Split D=192 across 2 lanes: 96 per lane
    
    const int my_lane_in_q = lane_id % 2;
    const int d_off = my_lane_in_q * 96;
    const int d_count = 96;
    
    // Registers for my query (if valid)
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float o[96];  // accumulated output
    
    #pragma unroll
    for (int i = 0; i < 96; i++) o[i] = 0.0f;
    
    const bool valid_q = my_q < q_len;
    
    // Iterate KV tiles
    const int num_kv_tiles = (Sk + 63) / 64;
    
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int k_start = kv_tile * 64;
        const int k_len = min(64, Sk - k_start);
        
        // Load K
        const half* gK = gK_base + k_start * D;
        for (int i = tid; i < k_len * D; i += 128) {
            sK[i] = gK[i];
        }
        __syncthreads();
        
        if (valid_q) {
            // Compute S[my_q][0:k_len] = Q[my_q] @ K[0:k_len]^T
            // Each thread computes partial dot, reduces with partner
            
            float s_local[64];
            
            for (int k = 0; k < k_len; k++) {
                float dot = 0.0f;
                
                // Compute partial dot product (96 elements)
                #pragma unroll
                for (int d = 0; d < 96; d += 4) {
                    int idx = d_off + d;
                    // Load 4 halfs as two half2
                    half2 q0 = *(half2*)&sQ[my_q * D + idx];
                    half2 q1 = *(half2*)&sQ[my_q * D + idx + 2];
                    half2 k0 = *(half2*)&sK[k * D + idx];
                    half2 k1 = *(half2*)&sK[k * D + idx + 2];
                    
                    float2 qf0 = __half22float2(q0);
                    float2 qf1 = __half22float2(q1);
                    float2 kf0 = __half22float2(k0);
                    float2 kf1 = __half22float2(k1);
                    
                    dot += qf0.x * kf0.x + qf0.y * kf0.y;
                    dot += qf1.x * kf1.x + qf1.y * kf1.y;
                }
                
                // Reduce with partner thread (XOR 1)
                dot += __shfl_xor_sync(0xffffffff, dot, 1);
                s_local[k] = dot * scale;
            }
            
            for (int k = k_len; k < 64; k++) {
                s_local[k] = -CUDART_INF_F;
            }
            
            // Online softmax
            float m_prev = m;
            float m_new = m;
            #pragma unroll
            for (int k = 0; k < 64; k++) {
                m_new = fmaxf(m_new, s_local[k]);
            }
            
            float l_scale = expf(m_prev - m_new);
            float l_new = l * l_scale;
            
            float p_local[64];
            #pragma unroll
            for (int k = 0; k < 64; k++) {
                p_local[k] = expf(s_local[k] - m_new);
                l_new += p_local[k];
            }
            
            // Update output
            float o_scale = l / l_new * l_scale;
            #pragma unroll
            for (int i = 0; i < 96; i++) {
                o[i] *= o_scale;
            }
            
            // Load V and accumulate
            __syncthreads();
            const half* gV = gV_base + k_start * D;
            for (int i = tid; i < k_len * D; i += 128) {
                sV[i] = gV[i];
            }
            __syncthreads();
            
            // P @ V accumulation
            for (int k = 0; k < k_len; k++) {
                float p = p_local[k] / l_new;
                
                #pragma unroll
                for (int d = 0; d < 96; d += 4) {
                    int idx = d_off + d;
                    half2 v0 = *(half2*)&sV[k * D + idx];
                    half2 v1 = *(half2*)&sV[k * D + idx + 2];
                    float2 vf0 = __half22float2(v0);
                    float2 vf1 = __half22float2(v1);
                    
                    o[d/4 * 2 + 0] += p * vf0.x;
                    o[d/4 * 2 + 1] += p * vf0.y;
                    o[d/4 * 2 + 2] += p * vf1.x;
                    o[d/4 * 2 + 3] += p * vf1.y;
                }
            }
            
            m = m_new;
            l = l_new;
        } else {
            __syncthreads();  // For V load
            __syncthreads();
        }
    }
    
    // Write output
    if (valid_q) {
        // Only thread 0 of each query pair writes LSE
        if (my_lane_in_q == 0) {
            gLSE[my_q] = m + logf(l);
        }
        
        // Write O: each thread writes its 96 elements
        // But need to coordinate with partner to write full 192
        
        // Use shuffle to get partner's data and write together
        // Or: write separately (simpler but 2x stores)
        
        half* my_O = gO + my_q * D + d_off;
        #pragma unroll
        for (int d = 0; d < 96; d += 2) {
            // Pack to half2
            half2 val = __floats2half2_rn(o[d], o[d+1]);
            *(half2*)&my_O[d] = val;
        }
    }
}

// Final optimized kernel with proper indexing
__global__ void flash_attn_fwd_final(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float* __restrict__ lse,
    int B, int H, int Sq, int Sk, int D, float scale) {
    
    const int BM = 64;  // Block size M
    const int BN = 64;  // Block size N
    
    extern __shared__ char smem[];
    half* sQ = (half*)smem;                    // [BM][D]
    half* sK = (half*)(smem + BM * D * sizeof(half));  // [BN][D]
    half* sV = (half*)(smem + (BM + BN) * D * sizeof(half)); // [BN][D]
    
    const int tid = threadIdx.x;
    const int nthreads = 128;
    
    const int num_q_tiles = (Sq + BM - 1) / BM;
    const int bh = blockIdx.x / num_q_tiles;
    const int q_tile = blockIdx.x % num_q_tiles;
    
    const int b = bh / H;
    const int h = bh % H;
    
    if (b >= B) return;
    
    const int q_start = q_tile * BM;
    const int q_len = min(BM, Sq - q_start);
    
    // Strides
    const int stride_q = H * Sq * D;
    const int stride_qh = Sq * D;
    const int stride_kv = H * Sk * D;
    const int stride_kvh = Sk * D;
    
    // Base pointers
    const half* gQ = Q + b * stride_q + h * stride_qh;
    const half* gK = K + b * stride_kv + h * stride_kvh;
    const half* gV = V + b * stride_kv + h * stride_kvh;
    half* gO = O + b * stride_q + h * stride_qh;
    float* gLSE = lse + b * H * Sq + h * Sq;
    
    // Load Q tile
    for (int i = tid; i < q_len * D; i += nthreads) {
        int row = i / D;
        int col = i % D;
        sQ[row * D + col] = gQ[(q_start + row) * D + col];
    }
    
    // Zero pad remaining rows
    for (int i = tid + q_len * D; i < BM * D; i += nthreads) {
        sQ[i] = __float2half_rn(0.0f);
    }
    __syncthreads();
    
    // Thread organization: 4 warps
    // Each warp handles BM/4 = 16 query rows
    // Each thread handles D/2 = 96 elements (2 threads per query row)
    
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int rows_per_warp = BM / 4;  // 16
    const int warp_row_start = warp_id * rows_per_warp;
    
    // 32 lanes for 16 rows: 2 lanes per row
    const int my_row_in_warp = lane_id / 2;
    const int my_row = warp_row_start + my_row_in_warp;
    const int my_part = lane_id % 2;  // 0 or 1 for which half of D
    
    const bool row_valid = my_row < q_len;
    
    // Registers
    float m_reg = -CUDART_INF_F;
    float l_reg = 0.0f;
    float o_reg[96];
    #pragma unroll
    for (int i = 0; i < 96; i++) o_reg[i] = 0.0f;
    
    const int d_start = my_part * 96;
    
    // KV tiles
    const int num_kv_tiles = (Sk + BN - 1) / BN;
    
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int k_start = kv_tile * BN;
        const int k_len = min(BN, Sk - k_start);
        
        // Load K
        for (int i = tid; i < k_len * D; i += nthreads) {
            int row = i / D;
            int col = i % D;
            sK[row * D + col] = gK[(k_start + row) * D + col];
        }
        // Zero pad
        for (int i = tid + k_len * D; i < BN * D; i += nthreads) {
            sK[i] = __float2half_rn(0.0f);
        }
        __syncthreads();
        
        // Compute S for this tile
        float s_scores[64];  // BN scores
        
        if (row_valid) {
            for (int k = 0; k < BN; k++) {
                float dot = 0.0f;
                
                // Dot product of Q[my_row] and K[k]
                #pragma unroll
                for (int d = 0; d < 96; d += 4) {
                    int idx = d_start + d;
                    half2 q0 = *(half2*)&sQ[my_row * D + idx];
                    half2 q1 = *(half2*)&sQ[my_row * D + idx + 2];
                    half2 k0 = *(half2*)&sK[k * D + idx];
                    half2 k1 = *(half2*)&sK[k * D + idx + 2];
                    
                    float2 qf0 = __half22float2(q0);
                    float2 qf1 = __half22float2(q1);
                    float2 kf0 = __half22float2(k0);
                    float2 kf1 = __half22float2(k1);
                    
                    dot += qf0.x * kf0.x + qf0.y * kf0.y;
                    dot += qf1.x * kf1.x + qf1.y * kf1.y;
                }
                
                // Sum with partner thread
                dot += __shfl_xor_sync(0xffffffff, dot, 1);
                s_scores[k] = (k < k_len) ? dot * scale : -CUDART_INF_F;
            }
        }
        
        // Online softmax
        if (row_valid) {
            float m_prev = m_reg;
            float m_new = m_prev;
            #pragma unroll
            for (int k = 0; k < BN; k++) {
                m_new = fmaxf(m_new, s_scores[k]);
            }
            
            float l_scale = expf(m_prev - m_new);
            float l_new = l_reg * l_scale;
            
            float p_scores[BN];
            #pragma unroll
            for (int k = 0; k < BN; k++) {
                p_scores[k] = expf(s_scores[k] - m_new);
                l_new += p_scores[k];
            }
            
            // Rescale output
            float o_scale = l_reg / l_new * l_scale;
            #pragma unroll
            for (int i = 0; i < 96; i++) {
                o_reg[i] *= o_scale;
            }
            
            // Save for V multiply
            // Need to keep p_scores in registers
            
            // Load V
            __syncthreads();
            for (int i = tid; i < k_len * D; i += nthreads) {
                int row = i / D;
                int col = i % D;
                sV[row * D + col] = gV[(k_start + row) * D + col];
            }
            for (int i = tid + k_len * D; i < BN * D; i += nthreads) {
                sV[i] = __float2half_rn(0.0f);
            }
            __syncthreads();
            
            // P @ V
            for (int k = 0; k < BN; k++) {
                float p = p_scores[k] / l_new;
                
                #pragma unroll
                for (int d = 0; d < 96; d += 2) {
                    int idx = d_start + d;
                    half2 v = *(half2*)&sV[k * D + idx];
                    float2 vf = __half22float2(v);
                    o_reg[d + 0] += p * vf.x;
                    o_reg[d + 1] += p * vf.y;
                }
            }
            
            m_reg = m_new;
            l_reg = l_new;
        } else {
            __syncthreads();
            __syncthreads();
        }
    }
    
    // Write output
    if (row_valid) {
        // Write LSE (only once per row)
        if (my_part == 0) {
            gLSE[q_start + my_row] = m_reg + logf(l_reg);
        }
        
        // Write O
        half* o_ptr = gO + (q_start + my_row) * D + d_start;
        #pragma unroll
        for (int d = 0; d < 96; d += 2) {
            half2 val = __floats2half2_rn(o_reg[d], o_reg[d+1]);
            *(half2*)(o_ptr + d) = val;
        }
    }
}

// Launcher
extern "C" {

void launch_flash_attn_fwd(
    const void* Q,
    const void* K,
    const void* V,
    void* O,
    float* lse,
    int B,
    int H,
    int Sq,
    int Sk,
    int D,
    float scale,
    cudaStream_t stream) {
    
    const int BM = 64;
    const int BN = 64;
    
    // Shared memory: Q[64][192] + K[64][192] + V[64][192] = 3 * 64 * 192 * 2 = 73728
    size_t smem_size = (BM + BN + BN) * D * sizeof(half);
    
    const int num_q_tiles = (Sq + BM - 1) / BM;
    const int total_blocks = B * H * num_q_tiles;
    
    const int nthreads = 128;
    
    // Set shared memory size
    cudaFuncSetAttribute(flash_attn_fwd_final, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    flash_attn_fwd_final<<<total_blocks, nthreads, smem_size, stream>>>(
        (const half*)Q, (const half*)K, (const half*)V,
        (half*)O, lse,
        B, H, Sq, Sk, D, scale);
}

}
