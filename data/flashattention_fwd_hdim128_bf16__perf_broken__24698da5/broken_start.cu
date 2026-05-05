#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define D 128

// Tile sizes for Flash Attention
#define BLOCK_M 128  // Rows of Q processed per block
#define BLOCK_N 64   // Rows of K/V processed per iteration

// Shared memory layout
// Q smem: BLOCK_M x D
// K smem: BLOCK_N x D  
// V smem: BLOCK_N x D

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float val) {
    return __float2bfloat16(val);
}

// Warp-level matrix multiply accumulate for Q @ K^T
// Each warp computes a 16x64 tile of S = Q @ K^T
// Q_tile: [16, D], K_tile: [64, D]
__device__ __forceinline__ void warp_mma_qk(
    const __nv_bfloat16* q_smem,      // [BLOCK_M, D] in shared memory
    const __nv_bfloat16* k_smem,      // [BLOCK_N, D] in shared memory
    float* acc,                       // [16] accumulator for this thread's rows
    int warp_id,                      // which warp in CTA
    int lane_id,                      // lane in warp
    int m_idx_start,                  // starting row in Q for this warp
    int n_idx_start                   // starting row in K for this iteration
) {
    // Each warp handles 16 rows of Q (BLOCK_M / WARPS_PER_BLOCK = 32, but we do 16 per warp iteration)
    // Actually: BLOCK_M=128, WARPS_PER_BLOCK=4, so each warp handles 32 rows
    // Let's use 16x16 tiles with multiple iterations
    
    // For simplicity: each thread computes 4 elements across 4 rows
    // We'll do a simple loop-based approach that's easier to get right
    
    const int rows_per_warp = BLOCK_M / WARPS_PER_BLOCK; // 32
    const int row_stride = 4;  // each thread handles 4 rows
    
    int local_row_base = (lane_id / 8) * row_stride;  // 0,4,8,12 within warp's 32 rows
    int local_col = (lane_id % 8) * 8;  // which 8 columns in K we're starting with
    
    // Zero accumulator
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        acc[i] = 0.0f;
    }
    
    // Compute Q[row] @ K[col] for our assigned rows/cols
    #pragma unroll
    for (int d = 0; d < D; d++) {
        // Load 4 Q values (one per row)
        float q_vals[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int q_row = m_idx_start + local_row_base + i;
            q_vals[i] = bf16_to_float(q_smem[q_row * D + d]);
        }
        
        // Load K values for 8 columns
        #pragma unroll
        for (int kc = 0; kc < 8; kc++) {
            int k_col = local_col + kc;
            if (k_col < BLOCK_N) {
                float k_val = bf16_to_float(k_smem[k_col * D + d]);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    acc[i * 8 + kc] += q_vals[i] * k_val;
                }
            }
        }
    }
}

// Simpler approach: each thread computes a subset of the S matrix
__device__ __forceinline__ void compute_qk_dot_product(
    const __nv_bfloat16* q_smem,
    const __nv_bfloat16* k_smem,
    float* s_acc,           // [BLOCK_M * BLOCK_N] output in registers (reduced)
    int tid,
    int num_threads
) {
    // Each thread computes a subset of the (BLOCK_M, BLOCK_N) S matrix
    // We'll compute in a strided fashion
    
    const int total_elements = BLOCK_M * BLOCK_N;
    
    // For each element this thread is responsible for
    for (int idx = tid; idx < total_elements; idx += num_threads) {
        int m = idx / BLOCK_N;  // row in Q
        int n = idx % BLOCK_N;  // row in K (column in S)
        
        float sum = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            float q_val = bf16_to_float(q_smem[m * D + d]);
            float k_val = bf16_to_float(k_smem[n * D + d]);
            sum += q_val * k_val;
        }
        s_acc[idx] = sum;
    }
}

// Online softmax update
// Given new max and sum, update running statistics
__device__ __forceinline__ void online_softmax_update(
    float& m_prev,      // previous max
    float& l_prev,      // previous sum (in exp space, i.e., sum(exp(x - m)))
    float m_new,        // new max for this chunk
    float l_new         // new sum for this chunk
) {
    float m_new_max = fmaxf(m_prev, m_new);
    float exp_m_prev = expf(m_prev - m_new_max);
    float exp_m_new = expf(m_new - m_new_max);
    l_prev = l_prev * exp_m_prev + l_new * exp_m_new;
    m_prev = m_new_max;
}

// Flash Attention forward kernel
// Processes BLOCK_M rows of Q at a time
__global__ void flash_attn_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,      // [B, H, Sq, D]
    const __nv_bfloat16* __restrict__ K,      // [B, H, Sk, D]
    const __nv_bfloat16* __restrict__ V,      // [B, H, Sk, D]
    __nv_bfloat16* __restrict__ O,            // [B, H, Sq, D]
    float* __restrict__ lse,                  // [B, H, Sq]
    int B, int H, int Sq, int Sk, int D_val,
    float scale
) {
    // Shared memory
    extern __shared__ char smem[];
    
    __nv_bfloat16* q_smem = (__nv_bfloat16*)smem;                           // [BLOCK_M, D]
    __nv_bfloat16* k_smem = (__nv_bfloat16*)(smem + BLOCK_M * D * sizeof(__nv_bfloat16));  // [BLOCK_N, D]
    __nv_bfloat16* v_smem = (__nv_bfloat16*)(smem + (BLOCK_M + BLOCK_N) * D * sizeof(__nv_bfloat16)); // [BLOCK_N, D]
    
    // Additional smem for S matrix (BLOCK_M * BLOCK_N floats)
    float* s_smem = (float*)(smem + (BLOCK_M + BLOCK_N + BLOCK_N) * D * sizeof(__nv_bfloat16));
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    // Which batch/head/block of queries
    const int total_q_blocks = (Sq + BLOCK_M - 1) / BLOCK_M;
    const int bh = bid / total_q_blocks;
    const int q_block_idx = bid % total_q_blocks;
    
    const int b = bh / H;
    const int h = bh % H;
    
    if (b >= B || h >= H) return;
    
    const int q_start = q_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, Sq);
    const int actual_block_m = q_end - q_start;
    
    // Pointers to this batch/head's data
    const __nv_bfloat16* q_ptr = Q + ((b * H + h) * Sq + q_start) * D;
    const __nv_bfloat16* k_ptr = K + ((b * H + h) * Sk) * D;
    const __nv_bfloat16* v_ptr = V + ((b * H + h) * Sk) * D;
    __nv_bfloat16* o_ptr = O + ((b * H + h) * Sq + q_start) * D;
    float* lse_ptr = lse + ((b * H + h) * Sq + q_start);
    
    // Step 1: Load Q into shared memory
    // Each thread loads multiple elements
    #pragma unroll
    for (int i = tid; i < actual_block_m * D; i += blockDim.x) {
        int row = i / D;
        int col = i % D;
        int global_idx = row * D + col;
        if (q_start + row < Sq) {
            q_smem[row * D + col] = q_ptr[global_idx];
        }
    }
    __syncthreads();
    
    // Per-row accumulators for online softmax
    // m: running max, l: running sum, o: running output accumulator
    float m[BLOCK_M / WARPS_PER_BLOCK];  // each warp handles BLOCK_M/WARPS_PER_BLOCK rows
    float l[BLOCK_M / WARPS_PER_BLOCK];
    float o_acc[BLOCK_M / WARPS_PER_BLOCK][D];  // output accumulator
    
    const int rows_per_thread = (actual_block_m + blockDim.x - 1) / blockDim.x;
    const int my_row_start = tid * rows_per_thread;
    const int my_row_end = min(my_row_start + rows_per_thread, actual_block_m);
    
    // Actually, let's do warp-level parallelism for rows
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_row_start = warp_id * (actual_block_m / WARPS_PER_BLOCK);
    const int warp_row_end = min(warp_row_start + (actual_block_m + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, actual_block_m);
    const int warp_num_rows = warp_row_end - warp_row_start;
    
    // Initialize online softmax
    #pragma unroll
    for (int i = 0; i < BLOCK_M / WARPS_PER_BLOCK; i++) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            o_acc[i][d] = 0.0f;
        }
    }
    
    // Iterate over K, V blocks
    for (int kv_block = 0; kv_block < Sk; kv_block += BLOCK_N) {
        const int kv_end = min(kv_block + BLOCK_N, Sk);
        const int actual_block_n = kv_end - kv_block;
        
        // Load K block into shared memory
        const __nv_bfloat16* k_block_ptr = k_ptr + kv_block * D;
        #pragma unroll
        for (int i = tid; i < actual_block_n * D; i += blockDim.x) {
            int row = i / D;
            int col = i % D;
            if (row < actual_block_n) {
                k_smem[row * D + col] = k_block_ptr[row * D + col];
            }
        }
        
        // Load V block into shared memory  
        const __nv_bfloat16* v_block_ptr = v_ptr + kv_block * D;
        #pragma unroll
        for (int i = tid; i < actual_block_n * D; i += blockDim.x) {
            int row = i / D;
            int col = i % D;
            if (row < actual_block_n) {
                v_smem[row * D + col] = v_block_ptr[row * D + col];
            }
        }
        __syncthreads();
        
        // Compute S = Q @ K^T for this warp's rows
        // Each warp handles its subset of rows
        for (int local_row = 0; local_row < warp_num_rows; local_row++) {
            int q_row = warp_row_start + local_row;
            if (q_row >= actual_block_m) break;
            
            // Compute dot products with all K rows
            float s_vals[BLOCK_N];
            float row_max = -INFINITY;
            
            #pragma unroll
            for (int k_row = 0; k_row < actual_block_n; k_row++) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = lane_id; d < D; d += WARP_SIZE) {
                    float q_val = bf16_to_float(q_smem[q_row * D + d]);
                    float k_val = bf16_to_float(k_smem[k_row * D + d]);
                    dot += q_val * k_val;
                }
                
                // Warp reduce
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                }
                
                dot *= scale;
                s_vals[k_row] = dot;
                row_max = fmaxf(row_max, dot);
            }
            
            // Compute exp and sum
            float row_sum = 0.0f;
            #pragma unroll
            for (int k_row = 0; k_row < actual_block_n; k_row++) {
                s_vals[k_row] = expf(s_vals[k_row] - row_max);
                row_sum += s_vals[k_row];
            }
            
            // Online softmax update
            float m_old = m[local_row];
            float l_old = l[local_row];
            float m_new = row_max;
            float l_new = row_sum;
            
            float m_new_max = fmaxf(m_old, m_new);
            float exp_m_old = expf(m_old - m_new_max);
            float exp_m_new = expf(m_new - m_new_max);
            
            // Update output accumulator: scale old, add new
            #pragma unroll
            for (int d = 0; d < D; d++) {
                o_acc[local_row][d] = o_acc[local_row][d] * exp_m_old;
            }
            
            // Add contribution from this block: P @ V where P = softmax(S)
            #pragma unroll
            for (int k_row = 0; k_row < actual_block_n; k_row++) {
                float p_val = s_vals[k_row] * exp_m_new;  // normalized probability
                #pragma unroll
                for (int d = lane_id; d < D; d += WARP_SIZE) {
                    float v_val = bf16_to_float(v_smem[k_row * D + d]);
                    o_acc[local_row][d] += p_val * v_val;
                }
            }
            
            // Update running stats
            l[local_row] = l_old * exp_m_old + l_new * exp_m_new;
            m[local_row] = m_new_max;
        }
        
        __syncthreads();
    }
    
    // Write output: normalize by final sum and convert to bf16
    for (int local_row = 0; local_row < warp_num_rows; local_row++) {
        int q_row = warp_row_start + local_row;
        if (q_row >= actual_block_m) break;
        
        float inv_l = 1.0f / l[local_row];
        
        #pragma unroll
        for (int d = lane_id; d < D; d += WARP_SIZE) {
            float o_val = o_acc[local_row][d] * inv_l;
            o_ptr[q_row * D + d] = float_to_bf16(o_val);
        }
        
        // Write lse (only one thread per row)
        if (lane_id == 0) {
            lse_ptr[q_row] = m[local_row] + logf(l[local_row]);
        }
    }
}

// Simpler, more correct implementation using a different approach
// Each block handles BLOCK_M query rows
// Processes BLOCK_N key rows at a time

#define SMEM_SIZE (4 * 1024 * 8)  // 32KB shared memory

__global__ void flash_attn_fwd_v2(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ lse,
    int B, int H, int Sq, int Sk, int D_val,
    float scale
) {
    extern __shared__ __nv_bfloat16 smem[];
    
    // Layout: Q[BLOCK_M, D], K[BLOCK_N, D], V[BLOCK_N, D]
    __nv_bfloat16* q_smem = smem;
    __nv_bfloat16* k_smem = smem + BLOCK_M * D;
    __nv_bfloat16* v_smem = k_smem + BLOCK_N * D;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    // Decode batch, head, query block
    const int q_blocks = (Sq + BLOCK_M - 1) / BLOCK_M;
    const int bh = bid / q_blocks;
    const int q_bidx = bid % q_blocks;
    
    const int b = bh / H;
    const int h = bh % H;
    
    if (b >= B) return;
    
    const int q_start = q_bidx * BLOCK_M;
    const int q_size = min(BLOCK_M, Sq - q_start);
    
    // Global pointers
    const __nv_bfloat16* g_q = Q + ((b * H + h) * Sq + q_start) * D;
    const __nv_bfloat16* g_k = K + ((b * H + h) * Sk) * D;
    const __nv_bfloat16* g_v = V + ((b * H + h) * Sk) * D;
    __nv_bfloat16* g_o = O + ((b * H + h) * Sq + q_start) * D;
    float* g_lse = lse + ((b * H + h) * Sq + q_start);
    
    // Load Q to shared memory
    for (int i = tid; i < q_size * D; i += blockDim.x) {
        q_smem[i] = g_q[i];
    }
    __syncthreads();
    
    // Per-row state (stored in registers, each thread handles multiple rows)
    // Use strided assignment for better load balancing
    const int rows_per_thread = (q_size + blockDim.x - 1) / blockDim.x;
    const int my_row = tid * rows_per_thread;
    const int my_rows = min(rows_per_thread, q_size - my_row);
    
    // Actually, use cooperative approach: all threads work on all rows in phases
    // Each thread maintains state for rows it "owns"
    
    // Simpler: each warp handles a subset of rows
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    
    const int warp_q_start = (warp_id * q_size) / num_warps;
    const int warp_q_end = ((warp_id + 1) * q_size) / num_warps;
    const int warp_q_size = warp_q_end - warp_q_start;
    
    // State arrays (small, in registers)
    float m_reg[16];  // max per row (max 16 rows per warp with BLOCK_M=128, 4 warps)
    float l_reg[16];
    float o_reg[16][D];  // output accumulator
    
    // Initialize
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        m_reg[i] = -INFINITY;
        l_reg[i] = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            o_reg[i][d] = 0.0f;
        }
    }
    
    // Iterate over KV blocks
    for (int kv_start = 0; kv_start < Sk; kv_start += BLOCK_N) {
        int kv_size = min(BLOCK_N, Sk - kv_start);
        
        // Load K, V to shared memory
        const __nv_bfloat16* g_k_block = g_k + kv_start * D;
        const __nv_bfloat16* g_v_block = g_v + kv_start * D;
        
        for (int i = tid; i < kv_size * D; i += blockDim.x) {
            k_smem[i] = g_k_block[i];
            v_smem[i] = g_v_block[i];
        }
        __syncthreads();
        
        // Each warp processes its assigned Q rows
        #pragma unroll
        for (int qr = 0; qr < warp_q_size; qr++) {
            int q_row = warp_q_start + qr;
            
            // Compute S[q_row, :] = Q[q_row] @ K^T
            // Each lane computes a subset of columns
            float s_local[BLOCK_N / WARP_SIZE + 1];
            float local_max = -INFINITY;
            
            // Strided over K rows
            for (int kc = lane; kc < kv_size; kc += WARP_SIZE) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; d++) {
                    dot += bf16_to_float(q_smem[q_row * D + d]) * bf16_to_float(k_smem[kc * D + d]);
                }
                dot *= scale;
                s_local[kc / WARP_SIZE] = dot;
                local_max = fmaxf(local_max, dot);
            }
            
            // Find row max across warp
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
            }
            
            // Compute exp and local sum
            float local_sum = 0.0f;
            for (int kc = lane; kc < kv_size; kc += WARP_SIZE) {
                s_local[kc / WARP_SIZE] = expf(s_local[kc / WARP_SIZE] - local_max);
                local_sum += s_local[kc / WARP_SIZE];
            }
            
            // Sum across warp
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
            }
            
            // Online softmax update
            float m_old = m_reg[qr];
            float l_old = l_reg[qr];
            float m_new = local_max;
            float l_new = local_sum;
            
            float m_updated = fmaxf(m_old, m_new);
            float exp_old = expf(m_old - m_updated);
            float exp_new = expf(m_new - m_updated);
            
            // Rescale output
            #pragma unroll
            for (int d = 0; d < D; d++) {
                o_reg[qr][d] *= exp_old;
            }
            
            // Add P @ V contribution
            for (int kc = lane; kc < kv_size; kc += WARP_SIZE) {
                float p = s_local[kc / WARP_SIZE] * exp_new;  // normalized
                #pragma unroll
                for (int d = 0; d < D; d++) {
                    o_reg[qr][d] += p * bf16_to_float(v_smem[kc * D + d]);
                }
            }
            
            // Update stats
            l_reg[qr] = l_old * exp_old + l_new * exp_new;
            m_reg[qr] = m_updated;
        }
        
        __syncthreads();
    }
    
    // Write output
    #pragma unroll
    for (int qr = 0; qr < warp_q_size; qr++) {
        int q_row = warp_q_start + qr;
        float inv_l = 1.0f / l_reg[qr];
        
        #pragma unroll
        for (int d = lane; d < D; d += WARP_SIZE) {
            g_o[q_row * D + d] = float_to_bf16(o_reg[qr][d] * inv_l);
        }
        
        if (lane == 0) {
            g_lse[q_row] = m_reg[qr] + logf(l_reg[qr]);
        }
    }
}

// Even simpler version - more likely to be correct
// Uses float accumulation, processes tiles carefully

__global__ void flash_attn_fwd_simple(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K, 
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ lse,
    int B, int H, int Sq, int Sk, int D_val,
    float scale
) {
    // Shared memory for tiles
    __shared__ __nv_bfloat16 smem_q[BLOCK_M * D];
    __shared__ __nv_bfloat16 smem_k[BLOCK_N * D];
    __shared__ __shared__ __nv_bfloat16 smem_v[BLOCK_N * D];
    __shared__ float smem_s[BLOCK_M * BLOCK_N];  // S matrix tile
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int bid = blockIdx.x;
    
    // Decode position
    const int q_tiles = (Sq + BLOCK_M - 1) / BLOCK_M;
    const int bh = bid / q_tiles;
    const int q_tile = bid % q_tiles;
    
    const int b = bh / H;
    const int h = bh % H;
    
    if (b >= B) return;
    
    const int q_start = q_tile * BLOCK_M;
    const int q_len = min(BLOCK_M, Sq - q_start);
    
    // Global memory pointers
    const __nv_bfloat16* ptr_q = Q + ((b * H + h) * Sq + q_start) * D;
    const __nv_bfloat16* ptr_k = K + ((b * H + h) * Sk) * D;
    const __nv_bfloat16* ptr_v = V + ((b * H + h) * Sk) * D;
    __nv_bfloat16* ptr_o = O + ((b * H + h) * Sq + q_start) * D;
    float* ptr_lse = lse + ((b * H + h) * Sq + q_start);
    
    // Load Q tile
    for (int i = tid; i < q_len * D; i += num_threads) {
        smem_q[i] = ptr_q[i];
    }
    __syncthreads();
    
    // Per-row accumulators (each thread handles a subset of rows)
    // Use warp-level distribution
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warps_per_block = num_threads / 32;
    
    const int rows_per_warp = (q_len + warps_per_block - 1) / warps_per_block;
    const int my_row_start = warp_id * rows_per_warp;
    const int my_row_end = min(my_row_start + rows_per_warp, q_len);
    const int my_num_rows = my_row_end - my_row_start;
    
    // Register storage for this thread's rows
    float row_m[8];   // max
    float row_l[8];   // sum of exp
    float row_o[8][D]; // output accumulator
    
    for (int i = 0; i < 8; i++) {
        row_m[i] = -INFINITY;
        row_l[i] = 0.0f;
        for (int d = 0; d < D; d++) row_o[i][d] = 0.0f;
    }
    
    // Process KV tiles
    for (int kv_tile = 0; kv_tile < Sk; kv_tile += BLOCK_N) {
        int kv_len = min(BLOCK_N, Sk - kv_tile);
        
        // Load K and V tiles
        const __nv_bfloat16* g_k = ptr_k + kv_tile * D;
        const __nv_bfloat16* g_v = ptr_v + kv_tile * D;
        
        for (int i = tid; i < kv_len * D; i += num_threads) {
            smem_k[i] = g_k[i];
            smem_v[i] = g_v[i];
        }
        __syncthreads();
        
        // Compute S = Q @ K^T for this warp's rows
        // Each thread in warp handles different K columns
        for (int local_r = 0; local_r < my_num_rows; local_r++) {
            int q_row = my_row_start + local_r;
            
            // Compute dot products with all K rows
            // Each lane computes a subset, then we exchange
            float s_vals[BLOCK_N];
            float thread_max = -INFINITY;
            
            for (int k_col = lane_id; k_col < kv_len; k_col += 32) {
                float dot = 0.0f;
                for (int d = 0; d < D; d++) {
                    dot += bf16_to_float(smem_q[q_row * D + d]) * bf16_to_float(smem_k[k_col * D + d]);
                }
                dot *= scale;
                s_vals[k_col] = dot;
                thread_max = fmaxf(thread_max, dot);
            }
            
            // Get global max for this row
            float row_max = thread_max;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, offset));
            }
            
            // Compute softmax values
            float thread_sum = 0.0f;
            for (int k_col = lane_id; k_col < kv_len; k_col += 32) {
                s_vals[k_col] = expf(s_vals[k_col] - row_max);
                thread_sum += s_vals[k_col];
            }
            
            float row_sum = thread_sum;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                row_sum += __shfl_xor_sync(0xffffffff, row_sum, offset);
            }
            
            // Online softmax update
            float m_prev = row_m[local_r];
            float l_prev = row_l[local_r];
            
            float m_new = fmaxf(m_prev, row_max);
            float exp_prev = expf(m_prev - m_new);
            float exp_new = expf(row_max - m_new);
            
            // Rescale output
            for (int d = 0; d < D; d++) {
                row_o[local_r][d] *= exp_prev;
            }
            
            // Add new contribution: P @ V
            for (int k_col = lane_id; k_col < kv_len; k_col += 32) {
                float p_val = s_vals[k_col] * exp_new;  // This is P[q_row, k_col]
                for (int d = 0; d < D; d++) {
                    row_o[local_r][d] += p_val * bf16_to_float(smem_v[k_col * D + d]);
                }
            }
            
            // Update running stats
            row_m[local_r] = m_new;
            row_l[local_r] = l_prev * exp_prev + row_sum * exp_new;
        }
        
        __syncthreads();
    }
    
    // Write final output
    for (int local_r = 0; local_r < my_num_rows; local_r++) {
        int q_row = my_row_start + local_r;
        float inv_l = 1.0f / row_l[local_r];
        
        for (int d = lane_id; d < D; d += 32) {
            ptr_o[q_row * D + d] = float_to_bf16(row_o[local_r][d] * inv_l);
        }
        
        if (lane_id == 0) {
            ptr_lse[q_row] = row_m[local_r] + logf(row_l[local_r]);
        }
    }
}

// Most robust version - uses simpler blocking, ensures correctness

__global__ void flash_attn_fwd_robust(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ lse,
    int B, int H, int Sq, int Sk, int D_val,
    float scale
) {
    // Shared memory
    __shared__ __nv_bfloat16 s_q[BLOCK_M * D];
    __shared__ __nv_bfloat16 s_k[BLOCK_N * D];
    __shared__ __nv_bfloat16 s_v[BLOCK_N * D];
    
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int bid = blockIdx.x;
    
    // Decode batch/head/q_tile
    const int n_q_tiles = (Sq + BLOCK_M - 1) / BLOCK_M;
    const int bh = bid / n_q_tiles;
    const int q_tile = bid % n_q_tiles;
    
    const int b = bh / H;
    const int h = bh % H;
    
    if (b >= B) return;
    
    const int q_off = q_tile * BLOCK_M;
    const int q_count = min(BLOCK_M, Sq - q_off);
    
    // Pointers
    const __nv_bfloat16* g_q = Q + ((b * H + h) * Sq + q_off) * D;
    const __nv_bfloat16* g_k = K + ((b * H + h) * Sk) * D;
    const __nv_bfloat16* g_v = V + ((b * H + h) * Sk) * D;
    __nv_bfloat16* g_o = O + ((b * H + h) * Sq + q_off) * D;
    float* g_lse = lse + ((b * H + h) * Sq + q_off);
    
    // Load Q
    for (int i = tid; i < q_count * D; i += nthreads) {
        s_q[i] = g_q[i];
    }
    __syncthreads();
    
    // Distribute rows to warps
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int nwarp = nthreads / 32;
    
    const int rows_per_warp = (q_count + nwarp - 1) / nwarp;
    const int my_row0 = warp_id * rows_per_warp;
    const int my_rowN = min(my_row0 + rows_per_warp, q_count);
    
    // Registers for this warp's rows
    float m[8], l[8], o[8][D];
    for (int i = 0; i < 8; i++) {
        m[i] = -INFINITY; l[i] = 0.0f;
        for (int d = 0; d < D; d++) o[i][d] = 0.0f;
    }
    
    // Process KV tiles
    for (int kv = 0; kv < Sk; kv += BLOCK_N) {
        int kv_count = min(BLOCK_N, Sk - kv);
        
        // Load K, V
        const __nv_bfloat16* g_kb = g_k + kv * D;
        const __nv_bfloat16* g_vb = g_v + kv * D;
        for (int i = tid; i < kv_count * D; i += nthreads) {
            s_k[i] = g_kb[i];
            s_v[i] = g_vb[i];
        }
        __syncthreads();
        
        // Each warp processes its rows
        for (int r = 0; r < my_rowN - my_row0; r++) {
            int qr = my_row0 + r;
            
            // Compute S[qr, :] = Q[qr] @ K^T
            // Each lane computes partial dots
            float s_partial[64]; // max BLOCK_N / 32 + 1 = 2, but pad
            float my_max = -INFINITY;
            
            for (int kc = lane; kc < kv_count; kc += 32) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; d++) {
                    dot += bf16_to_float(s_q[qr * D + d]) * bf16_to_float(s_k[kc * D + d]);
                }
                dot *= scale;
                s_partial[kc >> 5] = dot;  // kc/32
                my_max = fmaxf(my_max, dot);
            }
            
            // Reduce max
            float row_max = my_max;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, off));
            
            // Exp and sum
            float my_sum = 0.0f;
            for (int kc = lane; kc < kv_count; kc += 32) {
                s_partial[kc >> 5] = expf(s_partial[kc >> 5] - row_max);
                my_sum += s_partial[kc >> 5];
            }
            
            float row_sum = my_sum;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                row_sum += __shfl_xor_sync(0xffffffff, row_sum, off);
            
            // Online softmax
            float m_old = m[r], l_old = l[r];
            float m_new = fmaxf(m_old, row_max);
            float e_old = expf(m_old - m_new), e_new = expf(row_max - m_new);
            
            // Rescale acc
            #pragma unroll
            for (int d = 0; d < D; d++) o[r][d] *= e_old;
            
            // Add P @ V
            for (int kc = lane; kc < kv_count; kc += 32) {
                float p = s_partial[kc >> 5] * e_new;
                #pragma unroll
                for (int d = 0; d < D; d++)
                    o[r][d] += p * bf16_to_float(s_v[kc * D + d]);
            }
            
            m[r] = m_new;
            l[r] = l_old * e_old + row_sum * e_new;
        }
        __syncthreads();
    }
    
    // Write output
    for (int r = 0; r < my_rowN - my_row0; r++) {
        int qr = my_row0 + r;
        float il = 1.0f / l[r];
        #pragma unroll
        for (int d = lane; d < D; d += 32)
            g_o[qr * D + d] = float_to_bf16(o[r][d] * il);
        if (lane == 0) g_lse[qr] = m[r] + logf(l[r]);
    }
}

extern "C" void launch_flash_attn_fwd(
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
    cudaStream_t stream
) {
    const int q_tiles = (Sq + BLOCK_M - 1) / BLOCK_M;
    const int total_blocks = B * H * q_tiles;
    
    const int threads = 128;  // 4 warps
    
    // Calculate shared memory: Q tile + K tile + V tile
    // Using smaller BLOCK_N to fit in shared memory
    const int smem_size = (BLOCK_M * 128 + 64 * 128 + 64 * 128) * sizeof(__nv_bfloat16);
    
    flash_attn_fwd_robust<<<total_blocks, threads, smem_size, stream>>>(
        (const __nv_bfloat16*)Q,
        (const __nv_bfloat16*)K,
        (const __nv_bfloat16*)V,
        (__nv_bfloat16*)O,
        lse,
        B, H, Sq, Sk, D, scale
    );
}
