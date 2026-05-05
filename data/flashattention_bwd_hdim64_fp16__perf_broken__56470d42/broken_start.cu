#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Helper: warp shuffle reduction
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block reduce sum using shared memory
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

// Kernel: compute dsoftmax_sum = sum_d(dO * O)
__global__ void compute_dsoftmax_sum_kernel(
    const half* dO, const half* O, float* dsoftmax_sum,
    int B, int H, int S, int D
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx >= S) return;
    
    float sum = 0.0f;
    const half* dO_ptr = dO + ((b * H + h) * S + q_idx) * D;
    const half* O_ptr = O + ((b * H + h) * S + q_idx) * D;
    
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        sum += __half2float(dO_ptr[d]) * __half2float(O_ptr[d]);
    }
    
    dsoftmax_sum[((b * H + h) * S) + q_idx] = sum;
}

// Main backward kernel - tiled implementation
// Tile sizes: Q_TILE=64, KV_TILE=64
#define Q_TILE 64
#define KV_TILE 64

template <int D>
__global__ void flash_attn_bwd_kernel(
    const half* dO, const half* Q, const half* K, const half* V, const half* O,
    const float* lse, const float* dsoftmax_sum,
    half* dQ, half* dK, half* dV,
    int B, int H, int S, float scale
) {
    // Grid: blocks over (kv_tile_idx, batch_head)
    // Each block processes one KV tile and iterates over Q tiles
    
    int kv_tile_idx = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    // Shared memory layout
    // K_tile: [KV_TILE][D], V_tile: [KV_TILE][D]
    // Q_tile: [Q_TILE][D], dO_tile: [Q_TILE][D]
    // S: [Q_TILE][KV_TILE], P: [Q_TILE][KV_TILE]
    // dS: [Q_TILE][KV_TILE]
    
    extern __shared__ char smem[];
    
    half* K_smem = (half*)smem;                           // [KV_TILE][D]
    half* V_smem = K_smem + KV_TILE * D;                  // [KV_TILE][D]
    half* Q_smem = V_smem + KV_TILE * D;                  // [Q_TILE][D]
    half* dO_smem = Q_smem + Q_TILE * D;                  // [Q_TILE][D]
    float* S_smem = (float*)(dO_smem + Q_TILE * D);       // [Q_TILE][KV_TILE]
    float* P_smem = S_smem + Q_TILE * KV_TILE;            // [Q_TILE][KV_TILE]
    float* dS_smem = P_smem + Q_TILE * KV_TILE;           // [Q_TILE][KV_TILE]
    float* dP_smem = dS_smem + Q_TILE * KV_TILE;          // [Q_TILE][KV_TILE]
    float* reduce_smem = dP_smem + Q_TILE * KV_TILE;      // for reductions
    
    // Initialize dK/dV accumulators in registers (thread-local partial sums)
    // Each thread handles a subset of KV rows and D dimensions
    // Distribute KV_TILE rows across warps/threads
    
    float dK_acc[2][8];  // [rows][D/8] - up to 2 rows, 8 elements each for D=64
    float dV_acc[2][8];
    
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            dK_acc[i][j] = 0.0f;
            dV_acc[i][j] = 0.0f;
        }
    }
    
    int kv_start = kv_tile_idx * KV_TILE;
    int kv_end = min(kv_start + KV_TILE, S);
    int kv_len = kv_end - kv_start;
    
    // Load K and V tiles to shared memory
    // Each thread loads multiple elements
    #pragma unroll
    for (int i = tid; i < kv_len * D; i += blockDim.x) {
        int kv_row = i / D;
        int d = i % D;
        int kv_idx = kv_start + kv_row;
        if (kv_idx < S) {
            K_smem[kv_row * D + d] = K[((b * H + h) * S + kv_idx) * D + d];
            V_smem[kv_row * D + d] = V[((b * H + h) * S + kv_idx) * D + d];
        } else {
            K_smem[kv_row * D + d] = __float2half(0.0f);
            V_smem[kv_row * D + d] = __float2half(0.0f);
        }
    }
    
    // Initialize dK/dV output for this tile to zero (will accumulate)
    // We'll write atomically or at the end
    
    __syncthreads();
    
    // Iterate over Q tiles
    for (int q_tile_idx = 0; q_tile_idx < (S + Q_TILE - 1) / Q_TILE; ++q_tile_idx) {
        int q_start = q_tile_idx * Q_TILE;
        int q_end = min(q_start + Q_TILE, S);
        int q_len = q_end - q_start;
        
        // Load Q and dO tiles
        #pragma unroll
        for (int i = tid; i < q_len * D; i += blockDim.x) {
            int q_row = i / D;
            int d = i % D;
            int q_idx = q_start + q_row;
            if (q_idx < S) {
                Q_smem[q_row * D + d] = Q[((b * H + h) * S + q_idx) * D + d];
                dO_smem[q_row * D + d] = dO[((b * H + h) * S + q_idx) * D + d];
            } else {
                Q_smem[q_row * D + d] = __float2half(0.0f);
                dO_smem[q_row * D + d] = __float2half(0.0f);
            }
        }
        __syncthreads();
        
        // Compute S = Q @ K^T * scale
        // Each thread computes a subset of (q_row, kv_col) pairs
        // Use 2D tiling: threadIdx.x maps to (q_idx, kv_idx)
        
        for (int idx = tid; idx < q_len * kv_len; idx += blockDim.x) {
            int q_row = idx / kv_len;
            int kv_col = idx % kv_len;
            
            float s_val = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                s_val += __half2float(Q_smem[q_row * D + d]) * 
                         __half2float(K_smem[kv_col * D + d]);
            }
            s_val *= scale;
            S_smem[q_row * KV_TILE + kv_col] = s_val;
        }
        __syncthreads();
        
        // Compute P = exp(S - lse)
        for (int idx = tid; idx < q_len * kv_len; idx += blockDim.x) {
            int q_row = idx / kv_len;
            int kv_col = idx % kv_len;
            int q_idx = q_start + q_row;
            
            float lse_val = lse[((b * H + h) * S) + q_idx];
            float p_val = expf(S_smem[q_row * KV_TILE + kv_col] - lse_val);
            P_smem[q_row * KV_TILE + kv_col] = p_val;
        }
        __syncthreads();
        
        // Compute dV += P^T @ dO
        // For this KV tile: dV[kv_row, d] += sum_q P[q, kv_row] * dO[q, d]
        // Each thread handles some (kv_row, d) pairs
        
        // Process in groups for better memory coalescing
        int kv_per_thread = (kv_len + blockDim.x - 1) / blockDim.x;
        int my_kv_start = min(tid * kv_per_thread, kv_len);
        int my_kv_end = min((tid + 1) * kv_per_thread, kv_len);
        
        for (int kv_row = my_kv_start; kv_row < my_kv_end; ++kv_row) {
            for (int d = 0; d < D; ++d) {
                float dv_val = 0.0f;
                #pragma unroll
                for (int q_row = 0; q_row < q_len; ++q_row) {
                    dv_val += P_smem[q_row * KV_TILE + kv_row] * 
                              __half2float(dO_smem[q_row * D + d]);
                }
                // Accumulate to dV accumulator
                int acc_idx = d / 8;
                int acc_off = d % 8;
                if (kv_row - my_kv_start < 2 && acc_idx < 8) {
                    dV_acc[kv_row - my_kv_start][acc_off] += dv_val;
                }
            }
        }
        
        // Compute dP = dO @ V^T
        for (int idx = tid; idx < q_len * kv_len; idx += blockDim.x) {
            int q_row = idx / kv_len;
            int kv_col = idx % kv_len;
            
            float dp_val = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                dp_val += __half2float(dO_smem[q_row * D + d]) * 
                          __half2float(V_smem[kv_col * D + d]);
            }
            dP_smem[q_row * KV_TILE + kv_col] = dp_val;
        }
        __syncthreads();
        
        // Compute dS = P * (dP - dsoftmax_sum)
        for (int idx = tid; idx < q_len * kv_len; idx += blockDim.x) {
            int q_row = idx / kv_len;
            int kv_col = idx % kv_len;
            int q_idx = q_start + q_row;
            
            float dss_val = dsoftmax_sum[((b * H + h) * S) + q_idx];
            float ds_val = P_smem[q_row * KV_TILE + kv_col] * 
                          (dP_smem[q_row * KV_TILE + kv_col] - dss_val);
            dS_smem[q_row * KV_TILE + kv_col] = ds_val;
        }
        __syncthreads();
        
        // Compute dQ += dS @ K * scale
        // Write directly to global memory (atomicAdd for accumulation across KV tiles)
        for (int idx = tid; idx < q_len * D; idx += blockDim.x) {
            int q_row = idx / D;
            int d = idx % D;
            int q_idx = q_start + q_row;
            
            float dq_val = 0.0f;
            #pragma unroll
            for (int kv_col = 0; kv_col < kv_len; ++kv_col) {
                dq_val += dS_smem[q_row * KV_TILE + kv_col] * 
                          __half2float(K_smem[kv_col * D + d]);
            }
            dq_val *= scale;
            
            // Atomic add to dQ
            int dq_idx = ((b * H + h) * S + q_idx) * D + d;
            atomicAdd((float*)&((float*)dQ)[dq_idx/1], dq_val);  // Need proper casting
        }
        
        // Compute dK += dS^T @ Q * scale (accumulate to registers)
        for (int kv_row = my_kv_start; kv_row < my_kv_end; ++kv_row) {
            for (int d = 0; d < D; ++d) {
                float dk_val = 0.0f;
                #pragma unroll
                for (int q_row = 0; q_row < q_len; ++q_row) {
                    dk_val += dS_smem[q_row * KV_TILE + kv_row] * 
                              __half2float(Q_smem[q_row * D + d]);
                }
                dk_val *= scale;
                
                int acc_idx = d / 8;
                int acc_off = d % 8;
                if (kv_row - my_kv_start < 2 && acc_idx < 8) {
                    dK_acc[kv_row - my_kv_start][acc_off] += dk_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write accumulated dK and dV to global memory
    // Each thread writes its accumulated rows
    
    int kv_per_thread = (kv_len + blockDim.x - 1) / blockDim.x;
    int my_kv_start = min(tid * kv_per_thread, kv_len);
    int my_kv_end = min((tid + 1) * kv_per_thread, kv_len);
    
    for (int kv_row = my_kv_start; kv_row < my_kv_end; ++kv_row) {
        int kv_idx = kv_start + kv_row;
        if (kv_idx >= S) continue;
        
        for (int d = 0; d < D; ++d) {
            int acc_idx = d / 8;
            int acc_off = d % 8;
            float dk_val = 0.0f, dv_val = 0.0f;
            
            if (kv_row - my_kv_start < 2 && acc_idx < 8) {
                dk_val = dK_acc[kv_row - my_kv_start][acc_off];
                dv_val = dV_acc[kv_row - my_kv_start][acc_off];
            }
            
            // Use shared memory for reduction across threads with same kv_row
            // Simplified: assume each kv_row is handled by one thread
            
            int dk_idx = ((b * H + h) * S + kv_idx) * D + d;
            int dv_idx = ((b * H + h) * S + kv_idx) * D + d;
            
            // Convert to half and store
            dK[dk_idx] = __float2half(dk_val);
            dV[dv_idx] = __float2half(dv_val);
        }
    }
}

// Optimized kernel using proper shared memory and warp-level primitives
template <int D, int Q_TILE_SIZE, int KV_TILE_SIZE>
__global__ void flash_attn_bwd_kernel_v2(
    const half* dO, const half* Q, const half* K, const half* V, const half* O,
    const float* lse, const float* dsoftmax_sum,
    half* dQ, half* dK, half* dV,
    int B, int H, int S, float scale
) {
    // Block handles one KV tile, iterates over all Q tiles
    int kv_tile_idx = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    int kv_start = kv_tile_idx * KV_TILE_SIZE;
    int kv_len = min(KV_TILE_SIZE, S - kv_start);
    if (kv_start >= S) return;
    
    // Shared memory
    extern __shared__ char smem[];
    half* K_smem = (half*)smem;
    half* V_smem = K_smem + KV_TILE_SIZE * D;
    half* Q_smem = V_smem + KV_TILE_SIZE * D;
    half* dO_smem = Q_smem + Q_TILE_SIZE * D;
    float* S_smem = (float*)(dO_smem + Q_TILE_SIZE * D);
    float* P_smem = S_smem + Q_TILE_SIZE * KV_TILE_SIZE;
    float* dS_smem = P_smem + Q_TILE_SIZE * KV_TILE_SIZE;
    float* dP_smem = dS_smem + Q_TILE_SIZE * KV_TILE_SIZE;
    
    // Load K, V tiles
    for (int i = tid; i < kv_len * D; i += blockDim.x) {
        int row = i / D;
        int d = i % D;
        int idx = kv_start + row;
        K_smem[row * D + d] = K[((b * H + h) * S + idx) * D + d];
        V_smem[row * D + d] = V[((b * H + h) * S + idx) * D + d];
    }
    __syncthreads();
    
    // Thread-local accumulators for dK, dV
    // Each warp handles a subset of KV rows
    int kv_rows_per_warp = (kv_len + num_warps - 1) / num_warps;
    int my_kv_start = warp_id * kv_rows_per_warp;
    int my_kv_end = min(my_kv_start + kv_rows_per_warp, kv_len);
    
    float dK_local[8][8];  // [max_rows][D/8] for D=64
    float dV_local[8][8];
    
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            dK_local[i][j] = 0.0f;
            dV_local[i][j] = 0.0f;
        }
    
    // Iterate over Q tiles
    for (int q_tile = 0; q_tile < (S + Q_TILE_SIZE - 1) / Q_TILE_SIZE; ++q_tile) {
        int q_start = q_tile * Q_TILE_SIZE;
        int q_len = min(Q_TILE_SIZE, S - q_start);
        
        // Load Q, dO
        for (int i = tid; i < q_len * D; i += blockDim.x) {
            int row = i / D;
            int d = i % D;
            int idx = q_start + row;
            Q_smem[row * D + d] = Q[((b * H + h) * S + idx) * D + d];
            dO_smem[row * D + d] = dO[((b * H + h) * S + idx) * D + d];
        }
        __syncthreads();
        
        // Compute S = Q @ K^T
        // Each thread computes multiple elements
        for (int q_idx = warp_id; q_idx < q_len; q_idx += num_warps) {
            for (int kv_idx = lane; kv_idx < kv_len; kv_idx += WARP_SIZE) {
                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    s += __half2float(Q_smem[q_idx * D + d]) * 
                         __half2float(K_smem[kv_idx * D + d]);
                }
                S_smem[q_idx * KV_TILE_SIZE + kv_idx] = s * scale;
            }
        }
        __syncthreads();
        
        // Compute P = exp(S - lse)
        for (int q_idx = warp_id; q_idx < q_len; q_idx += num_warps) {
            for (int kv_idx = lane; kv_idx < kv_len; kv_idx += WARP_SIZE) {
                float lse_val = lse[((b * H + h) * S) + q_start + q_idx];
                float p = expf(S_smem[q_idx * KV_TILE_SIZE + kv_idx] - lse_val);
                P_smem[q_idx * KV_TILE_SIZE + kv_idx] = p;
            }
        }
        __syncthreads();
        
        // Compute dP = dO @ V^T
        for (int q_idx = warp_id; q_idx < q_len; q_idx += num_warps) {
            for (int kv_idx = lane; kv_idx < kv_len; kv_idx += WARP_SIZE) {
                float dp = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    dp += __half2float(dO_smem[q_idx * D + d]) * 
                          __half2float(V_smem[kv_idx * D + d]);
                }
                dP_smem[q_idx * KV_TILE_SIZE + kv_idx] = dp;
            }
        }
        __syncthreads();
        
        // Compute dS = P * (dP - dsoftmax_sum)
        for (int q_idx = warp_id; q_idx < q_len; q_idx += num_warps) {
            float dss = dsoftmax_sum[((b * H + h) * S) + q_start + q_idx];
            for (int kv_idx = lane; kv_idx < kv_len; kv_idx += WARP_SIZE) {
                float ds = P_smem[q_idx * KV_TILE_SIZE + kv_idx] * 
                          (dP_smem[q_idx * KV_TILE_SIZE + kv_idx] - dss);
                dS_smem[q_idx * KV_TILE_SIZE + kv_idx] = ds;
            }
        }
        __syncthreads();
        
        // Accumulate dV += P^T @ dO
        // Each warp handles its KV rows
        for (int kv_idx = my_kv_start; kv_idx < my_kv_end; ++kv_idx) {
            int local_kv = kv_idx - my_kv_start;
            if (local_kv >= 8) break;
            
            for (int d = lane; d < D; d += WARP_SIZE) {
                float dv = 0.0f;
                #pragma unroll
                for (int q_idx = 0; q_idx < q_len; ++q_idx) {
                    dv += P_smem[q_idx * KV_TILE_SIZE + kv_idx] * 
                          __half2float(dO_smem[q_idx * D + d]);
                }
                dV_local[local_kv][d / 8] += dv;
            }
        }
        
        // Accumulate dK += dS^T @ Q * scale
        for (int kv_idx = my_kv_start; kv_idx < my_kv_end; ++kv_idx) {
            int local_kv = kv_idx - my_kv_start;
            if (local_kv >= 8) break;
            
            for (int d = lane; d < D; d += WARP_SIZE) {
                float dk = 0.0f;
                #pragma unroll
                for (int q_idx = 0; q_idx < q_len; ++q_idx) {
                    dk += dS_smem[q_idx * KV_TILE_SIZE + kv_idx] * 
                          __half2float(Q_smem[q_idx * D + d]);
                }
                dK_local[local_kv][d / 8] += dk * scale;
            }
        }
        
        // Update dQ += dS @ K * scale (direct global memory update)
        // Use cooperative approach: each warp handles some Q rows
        for (int q_idx = warp_id; q_idx < q_len; q_idx += num_warps) {
            for (int d = lane; d < D; d += WARP_SIZE) {
                float dq = 0.0f;
                #pragma unroll
                for (int kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
                    dq += dS_smem[q_idx * KV_TILE_SIZE + kv_idx] * 
                          __half2float(K_smem[kv_idx * D + d]);
                }
                dq *= scale;
                
                int global_q_idx = q_start + q_idx;
                int dq_idx = ((b * H + h) * S + global_q_idx) * D + d;
                
                // Atomic add since multiple KV tiles contribute to same dQ
                unsigned int* addr = (unsigned int*)&dQ[dq_idx];
                unsigned int old_val, new_val;
                half old_half, new_half;
                float current;
                
                // Simple atomic emulation using CAS loop
                do {
                    old_val = *addr;
                    old_half = __ushort_as_half(old_val);
                    current = __half2float(old_half);
                    new_half = __float2half(current + dq);
                    new_val = __half_as_ushort(new_half);
                } while (atomicCAS(addr, old_val, new_val) != old_val);
            }
        }
        
        __syncthreads();
    }
    
    // Write dK, dV to global memory
    for (int kv_idx = my_kv_start; kv_idx < my_kv_end; ++kv_idx) {
        int local_kv = kv_idx - my_kv_start;
        if (local_kv >= 8) break;
        
        int global_kv_idx = kv_start + kv_idx;
        for (int d = lane; d < D; d += WARP_SIZE) {
            float dk = dK_local[local_kv][d / 8];
            float dv = dV_local[local_kv][d / 8];
            
            int idx = ((b * H + h) * S + global_kv_idx) * D + d;
            dK[idx] = __float2half(dk);
            dV[idx] = __float2half(dv);
        }
    }
}

// Simpler, more robust implementation
template <int D>
__global__ void flash_attn_bwd_simple(
    const half* dO, const half* Q, const half* K, const half* V, const half* O,
    const float* lse, const float* dsoftmax_sum,
    half* dQ, half* dK, half* dV,
    int B, int H, int S, float scale
) {
    // Each block handles one (b, h) and one KV tile
    // Cooperative groups for better reduction
    
    int kv_tile = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;
    
    const int KV_TILE = 64;
    const int Q_TILE = 64;
    
    int kv_start = kv_tile * KV_TILE;
    if (kv_start >= S) return;
    int kv_end = min(kv_start + KV_TILE, S);
    int kv_len = kv_end - kv_start;
    
    // Shared memory for tiles
    __shared__ half K_smem[KV_TILE * D];
    __shared__ half V_smem[KV_TILE * D];
    __shared__ half Q_smem[Q_TILE * D];
    __shared__ half dO_smem[Q_TILE * D];
    __shared__ float S_smem[Q_TILE * KV_TILE];
    __shared__ float P_smem[Q_TILE * KV_TILE];
    __shared__ float dS_smem[Q_TILE * KV_TILE];
    __shared__ float dP_smem[Q_TILE * KV_TILE];
    
    // Load K, V
    for (int i = tid; i < kv_len * D; i += blockDim.x) {
        int row = i / D;
        int d = i % D;
        K_smem[row * D + d] = K[((b * H + h) * S + kv_start + row) * D + d];
        V_smem[row * D + d] = V[((b * H + h) * S + kv_start + row) * D + d];
    }
    
    // Zero-initialize dK, dV accumulators in shared memory for this block
    __shared__ float dK_smem[KV_TILE * D];
    __shared__ float dV_smem[KV_TILE * D];
    
    for (int i = tid; i < kv_len * D; i += blockDim.x) {
        dK_smem[i] = 0.0f;
        dV_smem[i] = 0.0f;
    }
    __syncthreads();
    
    // Process all Q tiles
    for (int q_tile = 0; q_tile < (S + Q_TILE - 1) / Q_TILE; ++q_tile) {
        int q_start = q_tile * Q_TILE;
        int q_len = min(Q_TILE, S - q_start);
        
        // Load Q, dO
        for (int i = tid; i < q_len * D; i += blockDim.x) {
            int row = i / D;
            int d = i % D;
            Q_smem[row * D + d] = Q[((b * H + h) * S + q_start + row) * D + d];
            dO_smem[row * D + d] = dO[((b * H + h) * S + q_start + row) * D + d];
        }
        __syncthreads();
        
        // Compute S = Q @ K^T
        for (int idx = tid; idx < q_len * kv_len; idx += blockDim.x) {
            int q = idx / kv_len;
            int k = idx % kv_len;
            float sum = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                sum += __half2float(Q_smem[q * D + d]) * __half2float(K_smem[k * D + d]);
            }
            S_smem[q * KV_TILE + k] = sum * scale;
        }
        __syncthreads();
        
        // Compute P = softmax(S)
        for (int idx = tid; idx < q_len * kv_len; idx += blockDim.x) {
            int q = idx / kv_len;
            int k = idx % kv_len;
            float lse_val = lse[((b * H + h) * S) + q_start + q];
            P_smem[q * KV_TILE + k] = expf(S_smem[q * KV_TILE + k] - lse_val);
        }
        __syncthreads();
        
        // Compute dP = dO @ V^T
        for (int idx = tid; idx < q_len * kv_len; idx += blockDim.x) {
            int q = idx / kv_len;
            int k = idx % kv_len;
            float sum = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                sum += __half2float(dO_smem[q * D + d]) * __half2float(V_smem[k * D + d]);
            }
            dP_smem[q * KV_TILE + k] = sum;
        }
        __syncthreads();
        
        // Compute dS = P * (dP - dsoftmax_sum)
        for (int idx = tid; idx < q_len * kv_len; idx += blockDim.x) {
            int q = idx / kv_len;
            int k = idx % kv_len;
            float dss = dsoftmax_sum[((b * H + h) * S) + q_start + q];
            dS_smem[q * KV_TILE + k] = P_smem[q * KV_TILE + k] * (dP_smem[q * KV_TILE + k] - dss);
        }
        __syncthreads();
        
        // Accumulate dV += P^T @ dO
        for (int idx = tid; idx < kv_len * D; idx += blockDim.x) {
            int k = idx / D;
            int d = idx % D;
            float sum = 0.0f;
            for (int q = 0; q < q_len; ++q) {
                sum += P_smem[q * KV_TILE + k] * __half2float(dO_smem[q * D + d]);
            }
            atomicAdd(&dV_smem[k * D + d], sum);
        }
        
        // Accumulate dK += dS^T @ Q * scale
        for (int idx = tid; idx < kv_len * D; idx += blockDim.x) {
            int k = idx / D;
            int d = idx % D;
            float sum = 0.0f;
            for (int q = 0; q < q_len; ++q) {
                sum += dS_smem[q * KV_TILE + k] * __half2float(Q_smem[q * D + d]);
            }
            atomicAdd(&dK_smem[k * D + d], sum * scale);
        }
        
        // Update dQ += dS @ K * scale (global atomic)
        for (int idx = tid; idx < q_len * D; idx += blockDim.x) {
            int q = idx / D;
            int d = idx % D;
            float sum = 0.0f;
            for (int k = 0; k < kv_len; ++k) {
                sum += dS_smem[q * KV_TILE + k] * __half2float(K_smem[k * D + d]);
            }
            sum *= scale;
            
            int dq_idx = ((b * H + h) * S + q_start + q) * D + d;
            // Use fp32 atomic add via bit casting
            float current = __half2float(dQ[dq_idx]);
            // Note: This is a race condition without proper atomic, use CUDA's atomic for fp32
            // For simplicity, use global atomicAdd on converted pointer
            atomicAdd((float*)&((float*)dQ)[dq_idx], sum);  // This won't work directly
        }
        
        __syncthreads();
    }
    
    // Write dK, dV
    for (int i = tid; i < kv_len * D; i += blockDim.x) {
        int row = i / D;
        int d = i % D;
        int idx = ((b * H + h) * S + kv_start + row) * D + d;
        dK[idx] = __float2half(dK_smem[row * D + d]);
        dV[idx] = __float2half(dV_smem[row * D + d]);
    }
}

// Final clean implementation with proper fp16 atomics emulation
__device__ __forceinline__ void atomicAddHalf(half* addr, half val) {
    unsigned int* addr_as_ui = (unsigned int*)((char*)addr - ((size_t)addr & 2));
    unsigned int old = *addr_as_ui;
    unsigned int assumed;
    half hsum;
    do {
        assumed = old;
        hsum = __ushort_as_half((size_t)addr & 2 ? (old >> 16) : (old & 0xffff));
        hsum = __float2half(__half2float(hsum) + __half2float(val));
        old = (size_t)addr & 2 ? (old & 0xffff) | (__half_as_ushort(hsum) << 16)
                               : (old & 0xffff0000) | __half_as_ushort(hsum);
        old = atomicCAS(addr_as_ui, assumed, old);
    } while (assumed != old);
}

template <int D>
__global__ void flash_attn_bwd_final(
    const half* dO, const half* Q, const half* K, const half* V, const half* O,
    const float* lse, const float* dsoftmax_sum,
    half* dQ, half* dK, half* dV,
    int B, int H, int S, float scale
) {
    const int KV_TILE = 64;
    const int Q_TILE = 64;
    
    int kv_tile = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;
    
    int kv_start = kv_tile * KV_TILE;
    if (kv_start >= S) return;
    int kv_len = min(KV_TILE, S - kv_start);
    
    // Shared memory
    extern __shared__ char smem[];
    half* K_smem = (half*)smem;
    half* V_smem = K_smem + KV_TILE * D;
    half* Q_smem = V_smem + KV_TILE * D;
    half* dO_smem = Q_smem + Q_TILE * D;
    float* S_smem = (float*)(dO_smem + Q_TILE * D);
    float* P_smem = S_smem + Q_TILE * KV_TILE;
    float* dS_smem = P_smem + Q_TILE * KV_TILE;
    float* dP_smem = dS_smem + Q_TILE * KV_TILE;
    float* dK_smem = dP_smem + Q_TILE * KV_TILE;
    float* dV_smem = dK_smem + KV_TILE * D;
    
    // Load K, V, init dK, dV accumulators
    for (int i = tid; i < kv_len * D; i += blockDim.x) {
        int row = i / D, d = i % D;
        K_smem[row * D + d] = K[((b * H + h) * S + kv_start + row) * D + d];
        V_smem[row * D + d] = V[((b * H + h) * S + kv_start + row) * D + d];
        dK_smem[row * D + d] = 0.0f;
        dV_smem[row * D + d] = 0.0f;
    }
    __syncthreads();
    
    // Process Q tiles
    for (int q_tile = 0; q_tile < (S + Q_TILE - 1) / Q_TILE; ++q_tile) {
        int q_start = q_tile * Q_TILE;
        int q_len = min(Q_TILE, S - q_start);
        
        // Load Q, dO
        for (int i = tid; i < q_len * D; i += blockDim.x) {
            int row = i / D, d = i % D;
            Q_smem[row * D + d] = Q[((b * H + h) * S + q_start + row) * D + d];
            dO_smem[row * D + d] = dO[((b * H + h) * S + q_start + row) * D + d];
        }
        __syncthreads();
        
        // S = Q @ K^T * scale
        for (int idx = tid; idx < q_len * kv_len; idx += blockDim.x) {
            int q = idx / kv_len, k = idx % kv_len;
            float sum = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d)
                sum += __half2float(Q_smem[q * D + d]) * __half2float(K_smem[k * D + d]);
            S_smem[q * KV_TILE + k] = sum * scale;
        }
        __syncthreads();
        
        // P = exp(S - lse)
        for (int idx = tid; idx < q_len * kv_len; idx += blockDim.x) {
            int q = idx / kv_len, k = idx % kv_len;
            float lse_val = lse[((b * H + h) * S) + q_start + q];
            P_smem[q * KV_TILE + k] = expf(S_smem[q * KV_TILE + k] - lse_val);
        }
        __syncthreads();
        
        // dP = dO @ V^T
        for (int idx = tid; idx < q_len * kv_len; idx += blockDim.x) {
            int q = idx / kv_len, k = idx % kv_len;
            float sum = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d)
                sum += __half2float(dO_smem[q * D + d]) * __half2float(V_smem[k * D + d]);
            dP_smem[q * KV_TILE + k] = sum;
        }
        __syncthreads();
        
        // dS = P * (dP - dsoftmax_sum)
        for (int idx = tid; idx < q_len * kv_len; idx += blockDim.x) {
            int q = idx / kv_len, k = idx % kv_len;
            float dss = dsoftmax_sum[((b * H + h) * S) + q_start + q];
            dS_smem[q * KV_TILE + k] = P_smem[q * KV_TILE + k] * (dP_smem[q * KV_TILE + k] - dss);
        }
        __syncthreads();
        
        // dV += P^T @ dO
        for (int idx = tid; idx < kv_len * D; idx += blockDim.x) {
            int k = idx / D, d = idx % D;
            float sum = 0.0f;
            for (int q = 0; q < q_len; ++q)
                sum += P_smem[q * KV_TILE + k] * __half2float(dO_smem[q * D + d]);
            atomicAdd(&dV_smem[k * D + d], sum);
        }
        
        // dK += dS^T @ Q * scale
        for (int idx = tid; idx < kv_len * D; idx += blockDim.x) {
            int k = idx / D, d = idx % D;
            float sum = 0.0f;
            for (int q = 0; q < q_len; ++q)
                sum += dS_smem[q * KV_TILE + k] * __half2float(Q_smem[q * D + d]);
            atomicAdd(&dK_smem[k * D + d], sum * scale);
        }
        
        // dQ += dS @ K * scale (global)
        for (int idx = tid; idx < q_len * D; idx += blockDim.x) {
            int q = idx / D, d = idx % D;
            float sum = 0.0f;
            for (int k = 0; k < kv_len; ++k)
                sum += dS_smem[q * KV_TILE + k] * __half2float(K_smem[k * D + d]);
            sum *= scale;
            
            int dq_idx = ((b * H + h) * S + q_start + q) * D + d;
            // Emulate atomic add on fp16
            unsigned int* base = (unsigned int*)((size_t)&dQ[dq_idx] & ~2);
            unsigned int old, assumed, new_val;
            half current, result;
            do {
                old = *base;
                // Extract correct half
                bool high = ((size_t)&dQ[dq_idx] & 2) != 0;
                unsigned short hval = high ? (old >> 16) : (old & 0xFFFF);
                current = __ushort_as_half(hval);
                result = __float2half(__half2float(current) + sum);
                unsigned short new_hval = __half_as_ushort(result);
                new_val = high ? ((old & 0xFFFF) | (new_hval << 16)) : ((old & 0xFFFF0000) | new_hval);
                assumed = old;
                old = atomicCAS(base, assumed, new_val);
            } while (assumed != old);
        }
        
        __syncthreads();
    }
    
    // Write dK, dV
    for (int i = tid; i < kv_len * D; i += blockDim.x) {
        int row = i / D, d = i % D;
        int idx = ((b * H + h) * S + kv_start + row) * D + d;
        dK[idx] = __float2half(dK_smem[row * D + d]);
        dV[idx] = __float2half(dV_smem[row * D + d]);
    }
}

// Wrapper function
extern "C" void launch_flash_attn_bwd(
    const void* dO, const void* Q, const void* K, const void* V, const void* O,
    const float* lse, void* dQ, void* dK, void* dV,
    int B, int H, int Sq, int Sk, int D, float scale, cudaStream_t stream
) {
    // First compute dsoftmax_sum
    float* dsoftmax_sum;
    cudaMalloc(&dsoftmax_sum, B * H * Sq * sizeof(float));
    
    dim3 grid_dss((Sq + 127) / 128, H, B);
    dim3 block_dss(128);
    compute_dsoftmax_sum_kernel<<<grid_dss, block_dss, 0, stream>>>(
        (const half*)dO, (const half*)O, dsoftmax_sum, B, H, Sq, D
    );
    
    // Main backward kernel
    const int KV_TILE = 64;
    const int Q_TILE = 64;
    
    // Zero dQ for atomic accumulation
    cudaMemsetAsync(dQ, 0, B * H * Sq * D * sizeof(half), stream);
    
    dim3 grid_bwd((Sk + KV_TILE - 1) / KV_TILE, B * H);
    dim3 block_bwd(256);
    
    size_t smem_size = (KV_TILE * D + KV_TILE * D + Q_TILE * D + Q_TILE * D) * sizeof(half) +
                       (Q_TILE * KV_TILE * 4 + KV_TILE * D * 2) * sizeof(float);
    
    flash_attn_bwd_final<64><<<grid_bwd, block_bwd, smem_size, stream>>>(
        (const half*)dO, (const half*)Q, (const half*)K, (const half*)V, (const half*)O,
        lse, dsoftmax_sum, (half*)dQ, (half*)dK, (half*)dV,
        B, H, Sq, scale
    );
    
    cudaFree(dsoftmax_sum);
}
