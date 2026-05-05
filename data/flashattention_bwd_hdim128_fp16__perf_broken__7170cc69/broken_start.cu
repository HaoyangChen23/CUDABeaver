#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#define WARP_SIZE 32

// Tile sizes optimized for A100/H100
constexpr int BLOCK_M = 64;   // Q tile rows
constexpr int BLOCK_N = 64;   // K/V tile cols  
constexpr int BLOCK_D = 128;  // Head dim (fixed)
constexpr int NUM_WARPS = 8;
constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE; // 256

// Shared memory layout
// For Q/K/V tiles: BLOCK_M/N x BLOCK_D
// For S/P tiles: BLOCK_M x BLOCK_N

struct SharedMemLayout {
    static constexpr int Q_SIZE = BLOCK_M * BLOCK_D;
    static constexpr int K_SIZE = BLOCK_N * BLOCK_D;
    static constexpr int V_SIZE = BLOCK_N * BLOCK_D;
    static constexpr int DO_SIZE = BLOCK_M * BLOCK_D;
    static constexpr int S_SIZE = BLOCK_M * BLOCK_N;
    static constexpr int DS_SIZE = BLOCK_M * BLOCK_N;
};

__device__ __forceinline__ float2 half2_to_float2(__half2 h) {
    return __half22float2(h);
}

__device__ __forceinline__ float half_to_float(__half h) {
    return __half2float(h);
}

__device__ __forceinline__ __half float_to_half(float f) {
    return __float2half(f);
}

__device__ __forceinline__ __half2 float2_to_half2(float2 f) {
    return __float22half2_rn(f);
}

// Convert fp16 to float
__device__ __forceinline__ void load_fp16_4(const __half* ptr, float4& out) {
    __half2 h0 = *reinterpret_cast<const __half2*>(ptr);
    __half2 h1 = *reinterpret_cast<const __half2*>(ptr + 2);
    float2 f0 = half2_to_float2(h0);
    float2 f1 = half2_to_float2(h1);
    out.x = f0.x; out.y = f0.y;
    out.z = f1.x; out.w = f1.y;
}

// Store float to fp16
__device__ __forceinline__ void store_fp16_4(__half* ptr, float4 val) {
    __half2 h0 = float2_to_half2(make_float2(val.x, val.y));
    __half2 h1 = float2_to_half2(make_float2(val.z, val.w));
    *reinterpret_cast<__half2*>(ptr) = h0;
    *reinterpret_cast<__half2*>(ptr + 2) = h1;
}

// Compute dsoftmax_sum = sum_d(dO * O)
__global__ void compute_dsoftmax_sum_kernel(
    const __half* dO,
    const __half* O,
    float* dsoftmax_sum,
    int B, int H, int S, int D
) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    int b = blockIdx.x;
    int h = blockIdx.y;
    int row = blockIdx.z * BLOCK_M + warp_id * 8 + (lane_id / 4);
    
    if (row >= S) return;
    
    int base_idx = ((b * H + h) * S + row) * D;
    
    float sum = 0.0f;
    // Each thread handles 4 elements, 8 threads per row
    int col_start = (lane_id % 4) * 32;
    
    #pragma unroll 4
    for (int i = 0; i < 8; i++) {
        int col = col_start + i * 4;
        if (col < D) {
            float4 do_val, o_val;
            load_fp16_4(dO + base_idx + col, do_val);
            load_fp16_4(O + base_idx + col, o_val);
            sum += do_val.x * o_val.x;
            sum += do_val.y * o_val.y;
            sum += do_val.z * o_val.z;
            sum += do_val.w * o_val.w;
        }
    }
    
    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        dsoftmax_sum[((b * H + h) * S + row)] = sum;
    }
}

// Main flash attention backward kernel
template<bool IsFirstKTile, bool IsFirstQTile>
__global__ void flash_attn_bwd_kernel(
    const __half* dO,
    const __half* Q,
    const __half* K,
    const __half* V,
    const __half* O,
    const float* lse,
    const float* dsoftmax_sum,
    __half* dQ,
    __half* dK,
    __half* dV,
    int B, int H, int S, int D,
    float scale
) {
    extern __shared__ char smem[];
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    int q_tile = blockIdx.x;  // Q tile index
    int kv_tile = blockIdx.y; // K/V tile index
    
    int q_row_start = q_tile * BLOCK_M;
    int kv_row_start = kv_tile * BLOCK_N;
    
    // Shared memory pointers
    __half* sQ = reinterpret_cast<__half*>(smem);
    __half* sK = sQ + SharedMemLayout::Q_SIZE;
    __half* sV = sK + SharedMemLayout::K_SIZE;
    __half* sDO = sV + SharedMemLayout::V_SIZE;
    float* sS = reinterpret_cast<float*>(sDO + SharedMemLayout::DO_SIZE);
    float* sdS = sS + SharedMemLayout::S_SIZE;
    
    // Zero out dK/dV accumulators if first tile
    if (IsFirstKTile && tid == 0) {
        // Mark as first iteration
    }
    
    int batch_head_offset = ((b * H + h) * S);
    int q_base = batch_head_offset * D + q_row_start * D;
    int kv_base = batch_head_offset * D + kv_row_start * D;
    
    // Load Q tile to shared memory
    // Each thread loads multiple elements
    #pragma unroll
    for (int i = tid; i < BLOCK_M * D; i += NUM_THREADS) {
        int row = i / D;
        int col = i % D;
        int global_row = q_row_start + row;
        if (global_row < S) {
            sQ[row * D + col] = Q[q_base + row * D + col];
        } else {
            sQ[row * D + col] = __float2half(0.0f);
        }
    }
    
    // Load K tile to shared memory
    #pragma unroll
    for (int i = tid; i < BLOCK_N * D; i += NUM_THREADS) {
        int row = i / D;
        int col = i % D;
        int global_row = kv_row_start + row;
        if (global_row < S) {
            sK[row * D + col] = K[kv_base + row * D + col];
        } else {
            sK[row * D + col] = __float2half(0.0f);
        }
    }
    
    // Load V tile to shared memory
    #pragma unroll
    for (int i = tid; i < BLOCK_N * D; i += NUM_THREADS) {
        int row = i / D;
        int col = i % D;
        int global_row = kv_row_start + row;
        if (global_row < S) {
            sV[row * D + col] = V[kv_base + row * D + col];
        } else {
            sV[row * D + col] = __float2half(0.0f);
        }
    }
    
    // Load dO tile to shared memory
    #pragma unroll
    for (int i = tid; i < BLOCK_M * D; i += NUM_THREADS) {
        int row = i / D;
        int col = i % D;
        int global_row = q_row_start + row;
        if (global_row < S) {
            sDO[row * D + col] = dO[q_base + row * D + col];
        } else {
            sDO[row * D + col] = __float2half(0.0f);
        }
    }
    
    __syncthreads();
    
    // Compute S = Q @ K^T * scale
    // Each warp computes a sub-tile of S
    // Warp tile: 16x16 or 8x32 etc.
    constexpr int WARP_TILE_M = 16;
    constexpr int WARP_TILE_N = 16;
    
    int warp_row = warp_id / 2;  // 4 warps vertically
    int warp_col = warp_id % 2;  // 4 warps horizontally (but we have 8 warps, adjust)
    
    // Actually: 8 warps, arrange as 4x2 or 2x4
    // Let's do 4 rows x 2 cols of warp tiles
    // Each warp tile is 16x32 (to cover 64x64 with 8 warps)
    
    // Simpler: each thread computes 4x4 elements using MMA style
    // Or use simple dot products
    
    // Thread-level tile: each thread handles 4 rows x 4 cols of S
    int thread_row = (tid / 8) * 4;  // 32 threads vertically, each handles 4 rows -> 128 rows? No, 64 rows
    // Actually: 256 threads, 64x64 S matrix
    // Each thread computes 1 element with vectorized loads, or 4 elements
    
    // Let's do: each warp computes 16x32 tile, 8 warps cover 64x64
    // Actually 8 warps * 16x32 = 8*512 = 4096, but 64*64=4096. Good.
    
    int warp_m = warp_id / 2;  // 0-3
    int warp_n = warp_id % 2;  // 0-1
    
    // Compute S tile elements
    float local_S[4][4];  // Each thread computes 4x4
    
    // Initialize
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            local_S[i][j] = 0.0f;
        }
    }
    
    // Thread coordinates within warp
    int thread_m = (lane_id / 4) * 4;  // 0,4,8,12,16,20,24,28 -> 8 threads vertically
    int thread_n = (lane_id % 4) * 4;  // 0,4,8,12,16,20,24,28 -> but we only need 0,4,8,12 for 32 width
    
    // Actually: 32 lanes, we need to cover 16x32 per warp
    // 8 threads vertically (each 2 rows), 4 threads horizontally (each 8 cols)? No.
    
    // Simpler approach: each thread computes dot product for 1 element
    // But that's too slow. Let's do vectorized.
    
    // Actually use a simpler scheme: each thread computes 4 elements in a row
    // 256 threads / 64 rows = 4 threads per row, each handles 16 elements
    
    int s_row = tid / 4;      // 0-63
    int s_col_start = (tid % 4) * 16;  // 0,16,32,48
    
    if (s_row < BLOCK_M) {
        float lse_val = 0.0f;
        int global_q_row = q_row_start + s_row;
        if (global_q_row < S) {
            lse_val = lse[batch_head_offset + global_q_row];
        }
        
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int s_col = s_col_start + j;
            if (s_col < BLOCK_N) {
                // Compute dot product Q[s_row] @ K[s_col]
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; d += 4) {
                    float4 q_val, k_val;
                    load_fp16_4(&sQ[s_row * D + d], q_val);
                    load_fp16_4(&sK[s_col * D + d], k_val);
                    dot += q_val.x * k_val.x;
                    dot += q_val.y * k_val.y;
                    dot += q_val.z * k_val.z;
                    dot += q_val.w * k_val.w;
                }
                dot *= scale;
                
                int global_kv_row = kv_row_start + s_col;
                if (global_q_row < S && global_kv_row < S) {
                    float p = expf(dot - lse_val);
                    sS[s_row * BLOCK_N + s_col] = p;
                } else {
                    sS[s_row * BLOCK_N + s_col] = 0.0f;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Compute dV += P^T @ dO
    // P is BLOCK_M x BLOCK_N, dO is BLOCK_M x D
    // dV contribution is BLOCK_N x D
    
    // Each thread computes part of dV tile
    // dV row corresponds to K/V row
    int dv_row = tid / 4;      // 0-63
    int dv_col_start = (tid % 4) * 32;  // 0,32,64,96
    
    float local_dV[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) local_dV[i] = 0.0f;
    
    if (dv_row < BLOCK_N) {
        #pragma unroll
        for (int m = 0; m < BLOCK_M; m++) {
            float p_val = sS[m * BLOCK_N + dv_row];  // P^T element
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                int d = dv_col_start + j;
                float do_val = half_to_float(sDO[m * D + d]);
                local_dV[j] += p_val * do_val;
            }
        }
    }
    
    // Atomic add to global dV
    if (dv_row < BLOCK_N) {
        int global_kv_row = kv_row_start + dv_row;
        if (global_kv_row < S) {
            int dv_idx = ((b * H + h) * S + global_kv_row) * D + dv_col_start;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                atomicAdd(reinterpret_cast<float*>(&dV[dv_idx + j]), local_dV[j]);
            }
        }
    }
    
    // Compute dP = dO @ V^T
    // dP is BLOCK_M x BLOCK_N
    __syncthreads();
    
    float local_dP[4];
    int dp_row = (tid / 16);      // 0-15
    int dp_col_start = (tid % 16) * 4;  // 0,4,8,...,60
    
    if (dp_row < BLOCK_M) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int dp_col = dp_col_start + j;
            if (dp_col < BLOCK_N) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; d += 4) {
                    float4 do_val, v_val;
                    load_fp16_4(&sDO[dp_row * D + d], do_val);
                    load_fp16_4(&sV[dp_col * D + d], v_val);
                    dot += do_val.x * v_val.x;
                    dot += do_val.y * v_val.y;
                    dot += do_val.z * v_val.z;
                    dot += do_val.w * v_val.w;
                }
                local_dP[j] = dot;
            }
        }
    }
    
    // Compute dS = P * (dP - dsoftmax_sum)
    // First get dsoftmax_sum for this Q row
    float dss = 0.0f;
    int global_q_row_for_dss = q_row_start + dp_row;
    if (dp_row < BLOCK_M && global_q_row_for_dss < S) {
        dss = dsoftmax_sum[batch_head_offset + global_q_row_for_dss];
    }
    
    if (dp_row < BLOCK_M) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int dp_col = dp_col_start + j;
            if (dp_col < BLOCK_N) {
                float p_val = sS[dp_row * BLOCK_N + dp_col];
                float ds_val = p_val * (local_dP[j] - dss);
                sdS[dp_row * BLOCK_N + dp_col] = ds_val;
            }
        }
    }
    
    __syncthreads();
    
    // Compute dQ += dS @ K * scale
    int dq_row = tid / 4;
    int dq_col_start = (tid % 4) * 32;
    
    float local_dQ[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) local_dQ[i] = 0.0f;
    
    if (dq_row < BLOCK_M) {
        #pragma unroll
        for (int n = 0; n < BLOCK_N; n++) {
            float ds_val = sdS[dq_row * BLOCK_N + n];
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                int d = dq_col_start + j;
                float k_val = half_to_float(sK[n * D + d]);
                local_dQ[j] += ds_val * k_val;
            }
        }
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            local_dQ[j] *= scale;
        }
    }
    
    // Atomic add to global dQ
    if (dq_row < BLOCK_M) {
        int global_q_row = q_row_start + dq_row;
        if (global_q_row < S) {
            int dq_idx = ((b * H + h) * S + global_q_row) * D + dq_col_start;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                atomicAdd(reinterpret_cast<float*>(&dQ[dq_idx + j]), local_dQ[j]);
            }
        }
    }
    
    // Compute dK += dS^T @ Q * scale
    __syncthreads();
    
    int ddk_row = tid / 4;
    int ddk_col_start = (tid % 4) * 32;
    
    float local_dK[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) local_dK[i] = 0.0f;
    
    if (ddk_row < BLOCK_N) {
        #pragma unroll
        for (int m = 0; m < BLOCK_M; m++) {
            float ds_val = sdS[m * BLOCK_N + ddk_row];  // dS^T
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                int d = ddk_col_start + j;
                float q_val = half_to_float(sQ[m * D + d]);
                local_dK[j] += ds_val * q_val;
            }
        }
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            local_dK[j] *= scale;
        }
    }
    
    // Atomic add to global dK
    if (ddk_row < BLOCK_N) {
        int global_kv_row = kv_row_start + ddk_row;
        if (global_kv_row < S) {
            int dk_idx = ((b * H + h) * S + global_kv_row) * D + ddk_col_start;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                atomicAdd(reinterpret_cast<float*>(&dK[dk_idx + j]), local_dK[j]);
            }
        }
    }
}

// Simple kernel to initialize dQ, dK, dV to zero
__global__ void zero_grads_kernel(__half* dQ, __half* dK, __half* dV, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dQ[idx] = __float2half(0.0f);
        dK[idx] = __float2half(0.0f);
        dV[idx] = __float2half(0.0f);
    }
}

// Convert fp16 atomic adds using float conversion
__device__ float atomicAdd(float* address, float val) {
    return ::atomicAdd(address, val);
}

// Wrapper for half atomic add via float
__device__ void atomicAdd(__half* address, float val) {
    unsigned int* addr_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *addr_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned short target_val = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        __half target_half = reinterpret_cast<__half&>(target_val);
        float new_val = __half2float(target_half) + val;
        unsigned short new_half = __float2half(new_val);
        old = (size_t)address & 2 ? (old & 0xffff) | (new_half << 16) : (old & 0xffff0000) | new_half;
        old = atomicCAS(addr_as_ui, assumed, old);
    } while (assumed != old);
}

// Actually we need proper float accumulation then write back
// Let's use a different approach: accumulate in float shared memory or use float output

// Revised kernel with float accumulation buffers
__global__ void flash_attn_bwd_kernel_v2(
    const __half* dO,
    const __half* Q,
    const __half* K,
    const __half* V,
    const __half* O,
    const float* lse,
    const float* dsoftmax_sum,
    float* dQ_acc,
    float* dK_acc,
    float* dV_acc,
    int B, int H, int S, int D,
    float scale
) {
    extern __shared__ char smem[];
    
    int tid = threadIdx.x;
    
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    int q_tile = blockIdx.x;
    int kv_tile = blockIdx.y;
    
    int q_row_start = q_tile * BLOCK_M;
    int kv_row_start = kv_tile * BLOCK_N;
    
    // Bounds check
    if (q_row_start >= S || kv_row_start >= S) return;
    
    __half* sQ = reinterpret_cast<__half*>(smem);
    __half* sK = sQ + BLOCK_M * BLOCK_D;
    __half* sV = sK + BLOCK_N * BLOCK_D;
    __half* sDO = sV + BLOCK_N * BLOCK_D;
    float* sS = reinterpret_cast<float*>(sDO + BLOCK_M * BLOCK_D);
    float* sdS = sS + BLOCK_M * BLOCK_N;
    
    int batch_head_offset = ((b * H + h) * S);
    
    // Load Q tile
    #pragma unroll
    for (int i = tid; i < BLOCK_M * D; i += NUM_THREADS) {
        int row = i / D;
        int col = i % D;
        int global_row = q_row_start + row;
        if (global_row < S && col < D) {
            sQ[row * D + col] = Q[batch_head_offset * D + global_row * D + col];
        } else {
            sQ[row * D + col] = __float2half(0.0f);
        }
    }
    
    // Load K tile
    #pragma unroll
    for (int i = tid; i < BLOCK_N * D; i += NUM_THREADS) {
        int row = i / D;
        int col = i % D;
        int global_row = kv_row_start + row;
        if (global_row < S && col < D) {
            sK[row * D + col] = K[batch_head_offset * D + global_row * D + col];
        } else {
            sK[row * D + col] = __float2half(0.0f);
        }
    }
    
    // Load V tile
    #pragma unroll
    for (int i = tid; i < BLOCK_N * D; i += NUM_THREADS) {
        int row = i / D;
        int col = i % D;
        int global_row = kv_row_start + row;
        if (global_row < S && col < D) {
            sV[row * D + col] = V[batch_head_offset * D + global_row * D + col];
        } else {
            sV[row * D + col] = __float2half(0.0f);
        }
    }
    
    // Load dO tile
    #pragma unroll
    for (int i = tid; i < BLOCK_M * D; i += NUM_THREADS) {
        int row = i / D;
        int col = i % D;
        int global_row = q_row_start + row;
        if (global_row < S && col < D) {
            sDO[row * D + col] = dO[batch_head_offset * D + global_row * D + col];
        } else {
            sDO[row * D + col] = __float2half(0.0f);
        }
    }
    
    __syncthreads();
    
    // Compute S = Q @ K^T * scale, then P = exp(S - lse)
    int s_row = tid / 4;
    int s_col = (tid % 4) * 16;
    
    if (s_row < BLOCK_M) {
        int global_q_row = q_row_start + s_row;
        float lse_val = (global_q_row < S) ? lse[batch_head_offset + global_q_row] : 0.0f;
        
        #pragma unroll
        for (int j = 0; j < 16 && s_col + j < BLOCK_N; j++) {
            int actual_col = s_col + j;
            int global_kv_row = kv_row_start + actual_col;
            
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; d += 4) {
                float4 q4, k4;
                load_fp16_4(&sQ[s_row * D + d], q4);
                load_fp16_4(&sK[actual_col * D + d], k4);
                dot += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
            }
            dot *= scale;
            
            float p = (global_q_row < S && global_kv_row < S) ? expf(dot - lse_val) : 0.0f;
            sS[s_row * BLOCK_N + actual_col] = p;
        }
    }
    
    __syncthreads();
    
    // Compute dV contribution: P^T @ dO
    // Each thread handles 4 rows of dV (K/V rows) x 32 cols of D
    int dv_row = (tid / 8) * 2;  // 0,2,4,...62
    int dv_col = (tid % 8) * 16; // 0,16,32,48,64,80,96,112
    
    float accum_dV[2][16] = {0};
    
    #pragma unroll
    for (int sub = 0; sub < 2; sub++) {
        int actual_dv_row = dv_row + sub;
        if (actual_dv_row >= BLOCK_N) break;
        
        #pragma unroll
        for (int m = 0; m < BLOCK_M; m++) {
            float p_val = sS[m * BLOCK_N + actual_dv_row];
            #pragma unroll
            for (int j = 0; j < 16 && dv_col + j < D; j++) {
                float do_val = half_to_float(sDO[m * D + dv_col + j]);
                accum_dV[sub][j] += p_val * do_val;
            }
        }
    }
    
    // Write dV
    #pragma unroll
    for (int sub = 0; sub < 2; sub++) {
        int actual_dv_row = dv_row + sub;
        int global_kv_row = kv_row_start + actual_dv_row;
        if (actual_dv_row < BLOCK_N && global_kv_row < S) {
            int base = ((b * H + h) * S + global_kv_row) * D + dv_col;
            #pragma unroll
            for (int j = 0; j < 16 && dv_col + j < D; j++) {
                atomicAdd(&dV_acc[base + j], accum_dV[sub][j]);
            }
        }
    }
    
    // Compute dP = dO @ V^T
    // Then dS = P * (dP - dsoftmax_sum)
    int dp_row = tid / 8;
    int dp_col = (tid % 8) * 8;
    
    float local_dP[8] = {0};
    float dss = 0.0f;
    int global_dp_row = q_row_start + dp_row;
    
    if (dp_row < BLOCK_M) {
        dss = (global_dp_row < S) ? dsoftmax_sum[batch_head_offset + global_dp_row] : 0.0f;
        
        #pragma unroll
        for (int j = 0; j < 8 && dp_col + j < BLOCK_N; j++) {
            int actual_col = dp_col + j;
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; d += 4) {
                float4 do4, v4;
                load_fp16_4(&sDO[dp_row * D + d], do4);
                load_fp16_4(&sV[actual_col * D + d], v4);
                dot += do4.x * v4.x + do4.y * v4.y + do4.z * v4.z + do4.w * v4.w;
            }
            local_dP[j] = dot;
        }
    }
    
    // Compute dS and store
    if (dp_row < BLOCK_M) {
        #pragma unroll
        for (int j = 0; j < 8 && dp_col + j < BLOCK_N; j++) {
            int actual_col = dp_col + j;
            float p_val = sS[dp_row * BLOCK_N + actual_col];
            sdS[dp_row * BLOCK_N + actual_col] = p_val * (local_dP[j] - dss);
        }
    }
    
    __syncthreads();
    
    // Compute dQ += dS @ K * scale
    int dq_row = tid / 4;
    int dq_col = (tid % 4) * 32;
    
    float accum_dQ[32] = {0};
    
    if (dq_row < BLOCK_M) {
        #pragma unroll
        for (int n = 0; n < BLOCK_N; n++) {
            float ds_val = sdS[dq_row * BLOCK_N + n];
            #pragma unroll
            for (int j = 0; j < 32 && dq_col + j < D; j++) {
                float k_val = half_to_float(sK[n * D + dq_col + j]);
                accum_dQ[j] += ds_val * k_val;
            }
        }
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            accum_dQ[j] *= scale;
        }
    }
    
    // Write dQ
    if (dq_row < BLOCK_M) {
        int global_q_row = q_row_start + dq_row;
        if (global_q_row < S) {
            int base = ((b * H + h) * S + global_q_row) * D + dq_col;
            #pragma unroll
            for (int j = 0; j < 32 && dq_col + j < D; j++) {
                atomicAdd(&dQ_acc[base + j], accum_dQ[j]);
            }
        }
    }
    
    // Compute dK += dS^T @ Q * scale
    int dk_row = tid / 4;
    int dk_col = (tid % 4) * 32;
    
    float accum_dK[32] = {0};
    
    if (dk_row < BLOCK_N) {
        #pragma unroll
        for (int m = 0; m < BLOCK_M; m++) {
            float ds_val = sdS[m * BLOCK_N + dk_row];  // dS^T
            #pragma unroll
            for (int j = 0; j < 32 && dk_col + j < D; j++) {
                float q_val = half_to_float(sQ[m * D + dk_col + j]);
                accum_dK[j] += ds_val * q_val;
            }
        }
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            accum_dK[j] *= scale;
        }
    }
    
    // Write dK
    if (dk_row < BLOCK_N) {
        int global_kv_row = kv_row_start + dk_row;
        if (global_kv_row < S) {
            int base = ((b * H + h) * S + global_kv_row) * D + dk_col;
            #pragma unroll
            for (int j = 0; j < 32 && dk_col + j < D; j++) {
                atomicAdd(&dK_acc[base + j], accum_dK[j]);
            }
        }
    }
}

// Convert float accumulators to half
__global__ void convert_accum_to_half(
    const float* dQ_acc,
    const float* dK_acc,
    const float* dV_acc,
    __half* dQ,
    __half* dK,
    __half* dV,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dQ[idx] = __float2half(dQ_acc[idx]);
        dK[idx] = __float2half(dK_acc[idx]);
        dV[idx] = __float2half(dV_acc[idx]);
    }
}

extern "C" {

void launch_flash_attn_bwd(
    const void* dO,
    const void* Q,
    const void* K,
    const void* V,
    const void* O,
    const float* lse,
    void* dQ,
    void* dK,
    void* dV,
    int B,
    int H,
    int Sq,
    int Sk,
    int D,
    float scale,
    cudaStream_t stream
) {
    // Ensure D == 128
    if (D != 128) {
        // Fallback or error - for this task we assume D=128
    }
    
    int total_tokens = B * H * Sq;
    int total_elements = total_tokens * D;
    
    // Allocate float accumulators
    float *dQ_acc, *dK_acc, *dV_acc;
    cudaMalloc(&dQ_acc, total_elements * sizeof(float));
    cudaMalloc(&dK_acc, total_elements * sizeof(float));
    cudaMalloc(&dV_acc, total_elements * sizeof(float));
    
    // Zero accumulators
    cudaMemsetAsync(dQ_acc, 0, total_elements * sizeof(float), stream);
    cudaMemsetAsync(dK_acc, 0, total_elements * sizeof(float), stream);
    cudaMemsetAsync(dV_acc, 0, total_elements * sizeof(float), stream);
    
    // Compute dsoftmax_sum = sum_d(dO * O)
    float* dsoftmax_sum;
    cudaMalloc(&dsoftmax_sum, total_tokens * sizeof(float));
    
    dim3 dss_grid(B, H, (Sq + BLOCK_M - 1) / BLOCK_M);
    dim3 dss_block(NUM_THREADS);
    compute_dsoftmax_sum_kernel<<<dss_grid, dss_block, 0, stream>>>(
        reinterpret_cast<const __half*>(dO),
        reinterpret_cast<const __half*>(O),
        dsoftmax_sum,
        B, H, Sq, D
    );
    
    // Launch main backward kernel
    int num_q_tiles = (Sq + BLOCK_M - 1) / BLOCK_M;
    int num_kv_tiles = (Sk + BLOCK_N - 1) / BLOCK_N;
    
    dim3 grid(num_q_tiles, num_kv_tiles, B * H);
    dim3 block(NUM_THREADS);
    
    size_t smem_size = (BLOCK_M * BLOCK_D + BLOCK_N * BLOCK_D + BLOCK_N * BLOCK_D + 
                       BLOCK_M * BLOCK_D + BLOCK_M * BLOCK_N * 2) * sizeof(__half) + 
                       BLOCK_M * BLOCK_N * 2 * sizeof(float);
    // Actually recalculate properly
    smem_size = (BLOCK_M * D + BLOCK_N * D + BLOCK_N * D + BLOCK_M * D) * sizeof(__half) +
                (BLOCK_M * BLOCK_N * 2) * sizeof(float);
    
    flash_attn_bwd_kernel_v2<<<grid, block, smem_size, stream>>>(
        reinterpret_cast<const __half*>(dO),
        reinterpret_cast<const __half*>(Q),
        reinterpret_cast<const __half*>(K),
        reinterpret_cast<const __half*>(V),
        reinterpret_cast<const __half*>(O),
        lse,
        dsoftmax_sum,
        dQ_acc,
        dK_acc,
        dV_acc,
        B, H, Sq, Sk, D,
        scale
    );
    
    // Convert accumulators to half
    int convert_blocks = (total_elements + 255) / 256;
    convert_accum_to_half<<<convert_blocks, 256, 0, stream>>>(
        dQ_acc, dK_acc, dV_acc,
        reinterpret_cast<__half*>(dQ),
        reinterpret_cast<__half*>(dK),
        reinterpret_cast<__half*>(dV),
        total_elements
    );
    
    // Cleanup
    cudaFree(dQ_acc);
    cudaFree(dK_acc);
    cudaFree(dV_acc);
    cudaFree(dsoftmax_sum);
}

} // extern "C"
