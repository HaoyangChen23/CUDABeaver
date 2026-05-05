#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <math.h>

// Helper functions for bf16
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float val) {
    return __float2bfloat16(val);
}

// Tiling configuration
constexpr int kTileSizeQ = 64;   // Number of queries processed per block
constexpr int kTileSizeK = 64;   // Number of keys processed per iteration
constexpr int kHeadDim = 256;    // Fixed head dimension
constexpr int kWarps = 8;        // 8 warps per block
constexpr int kThreads = 256;    // 32 threads per warp * 8 warps

// Shared memory layout
// Q_smem: [kTileSizeQ][kHeadDim] - loaded in chunks
// K_smem: [kTileSizeK][kHeadDim] - loaded in chunks  
// V_smem: [kTileSizeK][kHeadDim] - loaded in chunks

struct SharedStorage {
    __nv_bfloat16 q_smem[kTileSizeQ][kHeadDim];
    __nv_bfloat16 k_smem[kTileSizeK][kHeadDim];
    __nv_bfloat16 v_smem[kTileSizeK][kHeadDim];
};

__device__ __forceinline__ void load_q_tile(
    const __nv_bfloat16* Q,
    __nv_bfloat16* q_smem,
    int b, int h, int q_tile_idx, int D, int Sq, int H
) {
    // Q layout: [B, H, Sq, D]
    const __nv_bfloat16* q_ptr = Q + ((b * H + h) * Sq + q_tile_idx * kTileSizeQ) * D;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Each thread loads multiple elements
    for (int i = tid; i < kTileSizeQ * D; i += num_threads) {
        int row = i / D;
        int col = i % D;
        int q_idx = q_tile_idx * kTileSizeQ + row;
        if (q_idx < Sq) {
            q_smem[row * D + col] = q_ptr[row * D + col];
        } else {
            q_smem[row * D + col] = __float2bfloat16(0.0f);
        }
    }
}

__device__ __forceinline__ void load_kv_tile(
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* k_smem,
    __nv_bfloat16* v_smem,
    int b, int h, int kv_tile_idx, int D, int Sk, int H
) {
    // K, V layout: [B, H, Sk, D]
    const __nv_bfloat16* k_ptr = K + ((b * H + h) * Sk + kv_tile_idx * kTileSizeK) * D;
    const __nv_bfloat16* v_ptr = V + ((b * H + h) * Sk + kv_tile_idx * kTileSizeK) * D;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    for (int i = tid; i < kTileSizeK * D; i += num_threads) {
        int row = i / D;
        int col = i % D;
        int kv_idx = kv_tile_idx * kTileSizeK + row;
        if (kv_idx < Sk) {
            k_smem[row * D + col] = k_ptr[row * D + col];
            v_smem[row * D + col] = v_ptr[row * D + col];
        } else {
            k_smem[row * D + col] = __float2bfloat16(0.0f);
            v_smem[row * D + col] = __float2bfloat16(0.0f);
        }
    }
}

__device__ __forceinline__ float compute_qk_dot(
    const __nv_bfloat16* q_row,
    const __nv_bfloat16* k_row,
    int D
) {
    float sum = 0.0f;
    #pragma unroll 8
    for (int d = 0; d < D; ++d) {
        sum += bf16_to_float(q_row[d]) * bf16_to_float(k_row[d]);
    }
    return sum;
}

__device__ __forceinline__ float compute_pv_dot(
    const float* p_row,
    const __nv_bfloat16* v_col,
    int kTileSizeK
) {
    float sum = 0.0f;
    for (int k = 0; k < kTileSizeK; ++k) {
        sum += p_row[k] * bf16_to_float(v_col[k * kHeadDim]);
    }
    return sum;
}

__global__ void flash_attn_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ lse,
    int B, int H, int Sq, int Sk, int D, float scale
) {
    extern __shared__ char shared_mem[];
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(shared_mem);
    
    int b = blockIdx.z;  // batch
    int h = blockIdx.y;  // head
    int q_tile_idx = blockIdx.x;  // query tile
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Each warp handles 8 queries (64 queries / 8 warps)
    constexpr int kQueriesPerWarp = kTileSizeQ / kWarps;  // 8
    int q_local_start = warp_id * kQueriesPerWarp;
    
    // Registers for this warp's queries
    float qk_max[kQueriesPerWarp];
    float exp_sum[kQueriesPerWarp];
    float o_acc[kQueriesPerWarp][kHeadDim];
    
    // Initialize
    #pragma unroll
    for (int qi = 0; qi < kQueriesPerWarp; ++qi) {
        qk_max[qi] = -CUDART_INF_F;
        exp_sum[qi] = 0.0f;
        #pragma unroll
        for (int d = 0; d < kHeadDim; ++d) {
            o_acc[qi][d] = 0.0f;
        }
    }
    
    // Load Q tile to shared memory
    load_q_tile(Q, &smem->q_smem[0][0], b, h, q_tile_idx, D, Sq, H);
    __syncthreads();
    
    // Each warp loads its queries to registers (as floats for computation)
    float q_reg[kQueriesPerWarp][kHeadDim];
    #pragma unroll
    for (int qi = 0; qi < kQueriesPerWarp; ++qi) {
        int q_row = q_local_start + qi;
        #pragma unroll 8
        for (int d = 0; d < kHeadDim; ++d) {
            q_reg[qi][d] = bf16_to_float(smem->q_smem[q_row][d]);
        }
    }
    
    // Global Q index for causal masking
    int q_global_start = q_tile_idx * kTileSizeQ;
    
    // Iterate over K, V tiles
    int num_kv_tiles = (Sk + kTileSizeK - 1) / kTileSizeK;
    
    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; ++kv_tile_idx) {
        int kv_global_start = kv_tile_idx * kTileSizeK;
        
        // Causal mask: skip tiles where all kv positions are ahead of all queries
        // q_global_start + kTileSizeQ - 1 < kv_global_start means all queries in tile
        // are before all keys in tile (valid), but we need kv <= q
        // If kv_global_start > q_global_start + kTileSizeQ - 1, then all k > all q, skip
        // Actually: valid if kv_idx <= q_idx for some pair
        // Skip if kv_global_start >= q_global_start + kTileSizeQ (all k > all q)
        if (kv_global_start >= q_global_start + kTileSizeQ) {
            continue;
        }
        
        // Load K and V tiles
        load_kv_tile(K, V, &smem->k_smem[0][0], &smem->v_smem[0][0], 
                     b, h, kv_tile_idx, D, Sk, H);
        __syncthreads();
        
        // Compute Q @ K^T for this tile
        float s_reg[kQueriesPerWarp][kTileSizeK];
        
        #pragma unroll
        for (int qi = 0; qi < kQueriesPerWarp; ++qi) {
            int q_global = q_global_start + q_local_start + qi;
            
            #pragma unroll 4
            for (int kj = 0; kj < kTileSizeK; ++kj) {
                int k_global = kv_global_start + kj;
                
                // Causal mask
                if (k_global > q_global) {
                    s_reg[qi][kj] = -CUDART_INF_F;
                } else {
                    // Compute dot product
                    float dot = 0.0f;
                    #pragma unroll 8
                    for (int d = 0; d < kHeadDim; ++d) {
                        dot += q_reg[qi][d] * bf16_to_float(smem->k_smem[kj][d]);
                    }
                    s_reg[qi][kj] = dot * scale;
                }
            }
        }
        
        // Online softmax update
        #pragma unroll
        for (int qi = 0; qi < kQueriesPerWarp; ++qi) {
            // Find max in current tile
            float tile_max = -CUDART_INF_F;
            #pragma unroll 4
            for (int kj = 0; kj < kTileSizeK; ++kj) {
                tile_max = fmaxf(tile_max, s_reg[qi][kj]);
            }
            
            // Check if all are -inf (fully masked)
            if (tile_max == -CUDART_INF_F) {
                continue;  // Skip this tile entirely
            }
            
            float new_max = fmaxf(qk_max[qi], tile_max);
            float exp_scale_old = expf(qk_max[qi] - new_max);
            float exp_scale_new = expf(tile_max - new_max);
            
            // Rescale previous sum and output
            exp_sum[qi] = exp_sum[qi] * exp_scale_old;
            
            #pragma unroll 8
            for (int d = 0; d < kHeadDim; ++d) {
                o_acc[qi][d] *= exp_scale_old;
            }
            
            // Compute exp and sum for this tile
            #pragma unroll 4
            for (int kj = 0; kj < kTileSizeK; ++kj) {
                float exp_val = expf(s_reg[qi][kj] - new_max);
                s_reg[qi][kj] = exp_val;  // Store P value
                exp_sum[qi] += exp_val;
            }
            
            // Update output: o_acc += P @ V
            #pragma unroll 8
            for (int d = 0; d < kHeadDim; ++d) {
                float pv_sum = 0.0f;
                #pragma unroll 4
                for (int kj = 0; kj < kTileSizeK; ++kj) {
                    pv_sum += s_reg[qi][kj] * bf16_to_float(smem->v_smem[kj][d]);
                }
                o_acc[qi][d] += pv_sum * exp_scale_new;
            }
            
            qk_max[qi] = new_max;
        }
        
        __syncthreads();
    }
    
    // Finalize: divide by exp_sum and write output
    #pragma unroll
    for (int qi = 0; qi < kQueriesPerWarp; ++qi) {
        int q_global = q_global_start + q_local_start + qi;
        if (q_global >= Sq) continue;
        
        // Normalize output
        float inv_sum = (exp_sum[qi] > 0.0f) ? (1.0f / exp_sum[qi]) : 0.0f;
        
        #pragma unroll 8
        for (int d = 0; d < kHeadDim; ++d) {
            o_acc[qi][d] *= inv_sum;
        }
        
        // Write O
        int o_offset = ((b * H + h) * Sq + q_global) * D;
        #pragma unroll 8
        for (int d = lane_id; d < kHeadDim; d += 32) {
            O[o_offset + d] = float_to_bf16(o_acc[qi][d]);
        }
        
        // Write LSE (only once per query)
        if (lane_id == 0) {
            // lse = log(sum(exp(s - max))) + max = log(exp_sum) + qk_max
            float lse_val = (exp_sum[qi] > 0.0f) ? (logf(exp_sum[qi]) + qk_max[qi]) : -CUDART_INF_F;
            int lse_idx = ((b * H + h) * Sq + q_global);
            lse[lse_idx] = lse_val;
        }
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
    // Grid: [num_q_tiles, H, B]
    int num_q_tiles = (Sq + kTileSizeQ - 1) / kTileSizeQ;
    
    dim3 grid(num_q_tiles, H, B);
    dim3 block(kThreads);
    
    size_t smem_size = sizeof(SharedStorage);
    
    flash_attn_fwd_kernel<<<grid, block, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q),
        reinterpret_cast<const __nv_bfloat16*>(K),
        reinterpret_cast<const __nv_bfloat16*>(V),
        reinterpret_cast<__nv_bfloat16*>(O),
        lse,
        B, H, Sq, Sk, D, scale
    );
}
