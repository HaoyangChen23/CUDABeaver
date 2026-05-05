#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <float.h>

#define WARP_SIZE 32
#define NUM_WARPS 4
#define BLOCK_M 64
#define BLOCK_N 64
#define HEAD_DIM 64

// Convert bf16 to float
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

// Convert float to bf16
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float val) {
    return __float2bfloat16(val);
}

// Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Flash Attention forward kernel
// Each block processes BLOCK_M queries
// We iterate over key/value blocks of size BLOCK_N
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM, int NUM_WARPS>
__global__ void flash_attn_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ lse,
    int B, int H, int Sq, int Sk, int D,
    float scale
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Each block handles BLOCK_M queries
    const int q_block_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;  // Combined B*H index
    
    const int batch_idx = bh_idx / H;
    const int head_idx = bh_idx % H;
    
    // Starting query position for this block
    const int q_start = q_block_idx * BLOCK_M;
    
    // Shared memory layout
    // smem_Q: [BLOCK_M, HEAD_DIM] - queries for this block
    // smem_K: [BLOCK_N, HEAD_DIM] - keys for current KV block
    // smem_V: [BLOCK_N, HEAD_DIM] - values for current KV block
    // smem_S: [BLOCK_M, BLOCK_N] - attention scores (partial)
    extern __shared__ char smem[];
    
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem + BLOCK_M * HEAD_DIM * sizeof(__nv_bfloat16));
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem + (BLOCK_M * HEAD_DIM + BLOCK_N * HEAD_DIM) * sizeof(__nv_bfloat16));
    
    // Strides
    const int q_stride = H * Sq * D;
    const int k_stride = H * Sk * D;
    const int v_stride = H * Sk * D;
    const int o_stride = H * Sq * D;
    const int lse_stride = H * Sq;
    
    const int q_batch_offset = batch_idx * q_stride + head_idx * Sq * D;
    const int k_batch_offset = batch_idx * k_stride + head_idx * Sk * D;
    const int v_batch_offset = batch_idx * v_stride + head_idx * Sk * D;
    const int o_batch_offset = batch_idx * o_stride + head_idx * Sq * D;
    const int lse_batch_offset = batch_idx * lse_stride + head_idx * Sq;
    
    // Load Q into shared memory (each thread loads multiple elements)
    // Q layout: [B, H, Sq, D]
    #pragma unroll
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
        int q_row = i / HEAD_DIM;
        int q_col = i % HEAD_DIM;
        int q_idx = q_start + q_row;
        
        if (q_idx < Sq) {
            smem_Q[i] = Q[q_batch_offset + q_idx * D + q_col];
        } else {
            smem_Q[i] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    
    // Each thread handles 1 or more query rows
    // We'll have each warp handle a subset of query rows
    const int rows_per_warp = BLOCK_M / NUM_WARPS;
    const int row_start = warp_id * rows_per_warp;
    
    // Accumulators for O = softmax(S) @ V
    // Each thread maintains partial accumulators for its assigned rows
    float acc[rows_per_warp][HEAD_DIM / WARP_SIZE];  // Split D across lanes
    
    // Running softmax statistics
    float m[rows_per_warp];  // max
    float l[rows_per_warp];  // sum of exp
    
    #pragma unroll
    for (int i = 0; i < rows_per_warp; i++) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < HEAD_DIM / WARP_SIZE; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Iterate over key/value sequence length
    const int num_n_blocks = (Sk + BLOCK_N - 1) / BLOCK_N;
    
    for (int n_block = 0; n_block < num_n_blocks; n_block++) {
        int k_start = n_block * BLOCK_N;
        
        // Load K block into shared memory
        #pragma unroll
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            int k_row = i / HEAD_DIM;
            int k_col = i % HEAD_DIM;
            int k_idx = k_start + k_row;
            
            if (k_idx < Sk) {
                smem_K[i] = K[k_batch_offset + k_idx * D + k_col];
            } else {
                smem_K[i] = __float2bfloat16(0.0f);
            }
        }
        
        // Load V block into shared memory
        #pragma unroll
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            int v_row = i / HEAD_DIM;
            int v_col = i % HEAD_DIM;
            int v_idx = k_start + v_row;
            
            if (v_idx < Sk) {
                smem_V[i] = V[v_batch_offset + v_idx * D + v_col];
            } else {
                smem_V[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        // Compute S = Q @ K^T for this block
        // Each warp handles its rows_per_warp query rows
        // Each thread in warp handles BLOCK_N key columns in chunks
        
        float s[rows_per_warp][BLOCK_N / WARP_SIZE];
        
        #pragma unroll
        for (int i = 0; i < rows_per_warp; i++) {
            int q_row = row_start + i;
            #pragma unroll
            for (int j = 0; j < BLOCK_N / WARP_SIZE; j++) {
                s[i][j] = 0.0f;
            }
            
            // Compute dot product with each key in the block
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                float q_val = bf16_to_float(smem_Q[q_row * HEAD_DIM + d]);
                #pragma unroll
                for (int j = 0; j < BLOCK_N / WARP_SIZE; j++) {
                    int k_col = lane_id + j * WARP_SIZE;
                    float k_val = bf16_to_float(smem_K[k_col * HEAD_DIM + d]);
                    s[i][j] += q_val * k_val;
                }
            }
            
            // Scale
            #pragma unroll
            for (int j = 0; j < BLOCK_N / WARP_SIZE; j++) {
                s[i][j] *= scale;
            }
        }
        
        // Online softmax: find max for each row
        float m_new[rows_per_warp];
        #pragma unroll
        for (int i = 0; i < rows_per_warp; i++) {
            m_new[i] = m[i];
            #pragma unroll
            for (int j = 0; j < BLOCK_N / WARP_SIZE; j++) {
                m_new[i] = fmaxf(m_new[i], s[i][j]);
            }
            m_new[i] = warp_reduce_max(m_new[i]);
        }
        
        // Compute exp(s - m_new) and sum
        float l_new[rows_per_warp];
        #pragma unroll
        for (int i = 0; i < rows_per_warp; i++) {
            l_new[i] = 0.0f;
            #pragma unroll
            for (int j = 0; j < BLOCK_N / WARP_SIZE; j++) {
                int k_idx = k_start + lane_id + j * WARP_SIZE;
                float exp_val = (k_idx < Sk) ? expf(s[i][j] - m_new[i]) : 0.0f;
                s[i][j] = exp_val;  // Store exp for later use
                l_new[i] += exp_val;
            }
            l_new[i] = warp_reduce_sum(l_new[i]);
        }
        
        // Rescale previous accumulator and update running stats
        #pragma unroll
        for (int i = 0; i < rows_per_warp; i++) {
            float alpha = expf(m[i] - m_new[i]);
            l_new[i] = alpha * l[i] + l_new[i];
            
            // Rescale accumulator
            #pragma unroll
            for (int j = 0; j < HEAD_DIM / WARP_SIZE; j++) {
                acc[i][j] *= alpha;
            }
            
            m[i] = m_new[i];
            l[i] = l_new[i];
        }
        
        // Compute contribution to O: s_exp @ V
        // s[i][j] contains exp(s - m_new) for this block
        #pragma unroll
        for (int i = 0; i < rows_per_warp; i++) {
            int q_row = row_start + i;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM / WARP_SIZE; d++) {
                int v_col = lane_id + d * WARP_SIZE;
                #pragma unroll
                for (int j = 0; j < BLOCK_N / WARP_SIZE; j++) {
                    int k_idx_in_block = lane_id + j * WARP_SIZE;
                    float v_val = bf16_to_float(smem_V[k_idx_in_block * HEAD_DIM + v_col]);
                    acc[i][d] += s[i][j] * v_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Normalize and write output
    // O = acc / l
    #pragma unroll
    for (int i = 0; i < rows_per_warp; i++) {
        int q_row = row_start + i;
        int q_idx = q_start + q_row;
        
        if (q_idx < Sq) {
            // Normalize
            float inv_l = 1.0f / l[i];
            
            // Write LSE: log(sum(exp(S))) = log(l) + m
            if (lane_id == 0) {
                lse[lse_batch_offset + q_idx] = logf(l[i]) + m[i];
            }
            
            // Write O
            #pragma unroll
            for (int d = 0; d < HEAD_DIM / WARP_SIZE; d++) {
                int o_col = lane_id + d * WARP_SIZE;
                float o_val = acc[i][d] * inv_l;
                O[o_batch_offset + q_idx * D + o_col] = float_to_bf16(o_val);
            }
        }
    }
}

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
    cudaStream_t stream
) {
    // Grid: (num_q_blocks, B*H)
    // Each block handles BLOCK_M queries
    
    const int num_q_blocks = (Sq + BLOCK_M - 1) / BLOCK_M;
    const int num_bh = B * H;
    
    dim3 grid(num_q_blocks, num_bh);
    dim3 block(NUM_WARPS * WARP_SIZE);  // 128 threads
    
    // Shared memory size
    // Q: BLOCK_M * D
    // K: BLOCK_N * D  
    // V: BLOCK_N * D
    size_t smem_size = (BLOCK_M * D + 2 * BLOCK_N * D) * sizeof(__nv_bfloat16);
    
    flash_attn_fwd_kernel<BLOCK_M, BLOCK_N, HEAD_DIM, NUM_WARPS><<<grid, block, smem_size, stream>>>(
        (const __nv_bfloat16*)Q,
        (const __nv_bfloat16*)K,
        (const __nv_bfloat16*)V,
        (__nv_bfloat16*)O,
        lse,
        B, H, Sq, Sk, D,
        scale
    );
}

} // extern "C"
