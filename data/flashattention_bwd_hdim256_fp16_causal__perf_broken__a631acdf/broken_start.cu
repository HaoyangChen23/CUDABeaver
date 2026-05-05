#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 128
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define SMEM_BANKS 32

// Helper to convert float to half
__device__ __forceinline__ half to_half(float x) {
    return __float2half(x);
}

// Helper for half multiply-add
__device__ __forceinline__ half2 hfma2(half2 a, half2 b, half2 c) {
    return __hfma2(a, b, c);
}

// Warp-level reduction sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Compute dsoftmax_sum = sum(dO * O) for each position
__global__ void compute_dsoftmax_sum_kernel(
    const half* __restrict__ dO,
    const half* __restrict__ O,
    float* __restrict__ dsoftmax_sum,
    int B, int H, int S, int D
) {
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b >= B || h >= H || i >= S) return;
    
    float sum = 0.0f;
    int base_idx = ((b * H + h) * S + i) * D;
    
    #pragma unroll 8
    for (int d = threadIdx.x; d < D; d += 32) {
        half do_val = dO[base_idx + d];
        half o_val = O[base_idx + d];
        sum += __half2float(do_val) * __half2float(o_val);
    }
    
    sum = warp_reduce_sum(sum);
    
    if (threadIdx.x == 0) {
        dsoftmax_sum[((b * H + h) * S) + i] = sum;
    }
}

// Main Flash Attention backward kernel
// Processes tiles of Q and KV to compute dQ, dK, dV
template<int D>
__global__ void flash_attn_bwd_kernel(
    const half* __restrict__ dO,
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const half* __restrict__ O,
    const float* __restrict__ lse,
    const float* __restrict__ dsoftmax_sum,
    half* __restrict__ dQ,
    half* __restrict__ dK,
    half* __restrict__ dV,
    int B, int H, int S, float scale
) {
    // Each block processes one (batch, head) and tiles of sequences
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    if (b >= B || h >= H) return;
    
    // Tile indices
    int tile_q = blockIdx.y;  // Q tile index
    int tile_kv = blockIdx.x; // KV tile index
    
    int q_start = tile_q * TILE_M;
    int kv_start = tile_kv * TILE_N;
    
    // Causal masking: skip if kv_start > q_end
    int q_end = min(q_start + TILE_M, S) - 1;
    if (kv_start > q_end) return;
    
    // Shared memory layout
    extern __shared__ char smem[];
    
    // Pointers to shared memory tiles
    half* sQ = (half*)smem;                           // TILE_M x D
    half* sK = sQ + TILE_M * D;                       // TILE_N x D  
    half* sV = sK + TILE_N * D;                       // TILE_N x D
    half* sdO = sV + TILE_N * D;                      // TILE_M x D
    half* sS = sdO + TILE_M * D;                      // TILE_M x TILE_N (scores)
    half* sdS = sS + TILE_M * TILE_N;                 // TILE_M x TILE_N
    
    float* sLSE = (float*)(sdS + TILE_M * TILE_N);    // TILE_M
    float* sdsoftmax = sLSE + TILE_M;                 // TILE_M
    
    // Global memory base indices
    int bh_offset = (b * H + h) * S;
    int bhD_offset = bh_offset * D;
    
    // Load Q tile and dO tile
    #pragma unroll
    for (int i = threadIdx.y; i < TILE_M; i += blockDim.y) {
        int global_i = q_start + i;
        if (global_i < S) {
            int q_idx = bhD_offset + global_i * D;
            int do_idx = q_idx;
            #pragma unroll 4
            for (int d = threadIdx.x; d < D; d += 32) {
                sQ[i * D + d] = Q[q_idx + d];
                sdO[i * D + d] = dO[do_idx + d];
            }
            if (threadIdx.x == 0) {
                sLSE[i] = lse[bh_offset + global_i];
                sdsoftmax[i] = dsoftmax_sum[bh_offset + global_i];
            }
        }
    }
    
    // Load K and V tiles
    #pragma unroll
    for (int j = threadIdx.y; j < TILE_N; j += blockDim.y) {
        int global_j = kv_start + j;
        if (global_j < S) {
            int kv_idx = bhD_offset + global_j * D;
            #pragma unroll 4
            for (int d = threadIdx.x; d < D; d += 32) {
                sK[j * D + d] = K[kv_idx + d];
                sV[j * D + d] = V[kv_idx + d];
            }
        }
    }
    
    __syncthreads();
    
    // Compute S = Q @ K^T * scale
    // Each thread computes a (TILE_M/warps_y) x (TILE_N/warps_x) block
    int warps_y = 4;  // 4 warps in y dimension
    int warps_x = 2;  // 2 warps in x dimension
    
    int warp_id = threadIdx.y / 4;
    int lane_id = threadIdx.x;
    int local_y = threadIdx.y % 4;
    
    // Thread-local accumulation for S computation
    float thread_S[8] = {0};  // 4x2 tile per thread
    
    // Compute attention scores S = Q @ K^T
    #pragma unroll
    for (int d = 0; d < D; d += 8) {
        // Load Q and K fragments
        half q_frag[4];
        half k_frag[2];
        
        #pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            int qi = warp_id * 16 + local_y * 4 + ii;
            if (qi < TILE_M && q_start + qi < S) {
                q_frag[ii] = sQ[qi * D + d + lane_id % 8];
            } else {
                q_frag[ii] = __float2half(0.0f);
            }
        }
        
        #pragma unroll
        for (int jj = 0; jj < 2; jj++) {
            int kj = (lane_id / 8) * 2 + jj;
            if (kj < TILE_N && kv_start + kj < S) {
                k_frag[jj] = sK[kj * D + d + lane_id % 8];
            } else {
                k_frag[jj] = __float2half(0.0f);
            }
        }
        
        // Compute partial dot products
        #pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            #pragma unroll
            for (int jj = 0; jj < 2; jj++) {
                thread_S[ii * 2 + jj] += __half2float(q_frag[ii]) * __half2float(k_frag[jj]);
            }
        }
    }
    
    // Apply scale and causal mask, compute P = exp(S - lse)
    float thread_P[8];
    #pragma unroll
    for (int ii = 0; ii < 4; ii++) {
        int qi = warp_id * 16 + local_y * 4 + ii;
        int global_i = q_start + qi;
        for (int jj = 0; jj < 2; jj++) {
            int kj = (lane_id / 8) * 2 + jj;
            int global_j = kv_start + kj;
            
            float s_val = thread_S[ii * 2 + jj] * scale;
            
            // Causal mask
            if (global_i >= S || global_j >= S || global_j > global_i) {
                thread_P[ii * 2 + jj] = 0.0f;
            } else {
                float lse_val = sLSE[qi];
                thread_P[ii * 2 + jj] = expf(s_val - lse_val);
            }
            
            // Store to shared memory
            if (qi < TILE_M && kj < TILE_N) {
                sS[qi * TILE_N + kj] = __float2half(thread_P[ii * 2 + jj]);
            }
        }
    }
    
    __syncthreads();
    
    // Compute dV += P^T @ dO
    // dV is accumulated in registers, written to global at end
    
    // Compute dP = dO @ V^T
    float thread_dP[8] = {0};
    
    #pragma unroll
    for (int d = 0; d < D; d += 8) {
        half do_frag[4];
        half v_frag[2];
        
        #pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            int qi = warp_id * 16 + local_y * 4 + ii;
            if (qi < TILE_M && q_start + qi < S) {
                do_frag[ii] = sdO[qi * D + d + lane_id % 8];
            } else {
                do_frag[ii] = __float2half(0.0f);
            }
        }
        
        #pragma unroll
        for (int jj = 0; jj < 2; jj++) {
            int kj = (lane_id / 8) * 2 + jj;
            if (kj < TILE_N && kv_start + kj < S) {
                v_frag[jj] = sV[kj * D + d + lane_id % 8];
            } else {
                v_frag[jj] = __float2half(0.0f);
            }
        }
        
        #pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            #pragma unroll
            for (int jj = 0; jj < 2; jj++) {
                thread_dP[ii * 2 + jj] += __half2float(do_frag[ii]) * __half2float(v_frag[jj]);
            }
        }
    }
    
    // Compute dS = P * (dP - dsoftmax_sum)
    #pragma unroll
    for (int ii = 0; ii < 4; ii++) {
        int qi = warp_id * 16 + local_y * 4 + ii;
        int global_i = q_start + qi;
        float dsum = (qi < TILE_M && global_i < S) ? sdsoftmax[qi] : 0.0f;
        
        for (int jj = 0; jj < 2; jj++) {
            int kj = (lane_id / 8) * 2 + jj;
            int global_j = kv_start + kj;
            
            if (qi < TILE_M && kj < TILE_N && global_i < S && global_j < S && global_j <= global_i) {
                float p_val = thread_P[ii * 2 + jj];
                float ds_val = p_val * (thread_dP[ii * 2 + jj] - dsum);
                thread_dP[ii * 2 + jj] = ds_val;  // reuse as dS
                sS[qi * TILE_N + kj] = __float2half(ds_val);
            } else if (qi < TILE_N && kj < TILE_N) {
                sS[qi * TILE_N + kj] = __float2half(0.0f);
            }
        }
    }
    
    __syncthreads();
    
    // Accumulate dV: dV[j] += sum_i P[i,j] * dO[i]
    // Each thread handles part of dV
    float dV_acc[32];  // Accumulate up to D elements
    
    #pragma unroll
    for (int d = 0; d < D; d += 32) {
        dV_acc[d/32] = 0.0f;
    }
    
    // For each position j in KV tile
    int kv_j = threadIdx.y * 8 + threadIdx.x / 4;
    if (kv_j < TILE_N && kv_start + kv_j < S) {
        #pragma unroll
        for (int qi = 0; qi < TILE_M && q_start + qi < S; qi++) {
            int global_i = q_start + qi;
            int global_j = kv_start + kv_j;
            if (global_j <= global_i) {  // causal
                half p_half = sS[qi * TILE_N + kv_j];
                float p_val = __half2float(p_half);
                
                #pragma unroll
                for (int d = 0; d < D; d += 32) {
                    half do_val = sdO[qi * D + d + threadIdx.x % 4 * 8 + (threadIdx.x % 32) / 4];
                    dV_acc[d/32] += p_val * __half2float(do_val);
                }
            }
        }
    }
    
    // Write dV to global memory (atomic add)
    if (kv_j < TILE_N) {
        int global_j = kv_start + kv_j;
        if (global_j < S) {
            int dv_idx = bhD_offset + global_j * D;
            #pragma unroll
            for (int d = 0; d < D; d += 32) {
                int dd = d + threadIdx.x % 4 * 8 + (threadIdx.x % 32) / 4;
                if (dd < D) {
                    half dv_val = __float2half(dV_acc[d/32]);
                    atomicAdd((unsigned short*)&dV[dv_idx + dd], __half_as_ushort(dv_val));
                }
            }
        }
    }
    
    __syncthreads();
    
    // Compute dQ += dS @ K * scale
    float dQ_acc[32] = {0};
    
    int q_i = threadIdx.y * 4 + threadIdx.x / 8;
    if (q_i < TILE_M && q_start + q_i < S) {
        #pragma unroll
        for (int kj = 0; kj < TILE_N && kv_start + kj < S; kj++) {
            int global_i = q_start + q_i;
            int global_j = kv_start + kj;
            if (global_j <= global_i) {  // causal
                half ds_half = sS[q_i * TILE_N + kj];
                float ds_val = __half2float(ds_half);
                
                #pragma unroll
                for (int d = 0; d < D; d += 4) {
                    half k_val = sK[kj * D + d + threadIdx.x % 8];
                    dQ_acc[d/4] += ds_val * __half2float(k_val);
                }
            }
        }
    }
    
    // Write dQ
    if (q_i < TILE_M) {
        int global_i = q_start + q_i;
        if (global_i < S) {
            int dq_idx = bhD_offset + global_i * D;
            #pragma unroll
            for (int d = 0; d < D; d += 4) {
                int dd = d + threadIdx.x % 8;
                if (dd < D) {
                    half dq_val = __float2half(dQ_acc[d/4] * scale);
                    atomicAdd((unsigned short*)&dQ[dq_idx + dd], __half_as_ushort(dq_val));
                }
            }
        }
    }
    
    // Compute dK += dS^T @ Q * scale
    float dK_acc[32] = {0};
    
    kv_j = threadIdx.y * 4 + threadIdx.x / 8;
    if (kv_j < TILE_N && kv_start + kv_j < S) {
        #pragma unroll
        for (int qi = 0; qi < TILE_M && q_start + qi < S; qi++) {
            int global_i = q_start + qi;
            int global_j = kv_start + kv_j;
            if (global_j <= global_i) {  // causal
                half ds_half = sS[qi * TILE_N + kv_j];
                float ds_val = __half2float(ds_half);
                
                #pragma unroll
                for (int d = 0; d < D; d += 4) {
                    half q_val = sQ[qi * D + d + threadIdx.x % 8];
                    dK_acc[d/4] += ds_val * __half2float(q_val);
                }
            }
        }
    }
    
    // Write dK
    if (kv_j < TILE_N) {
        int global_j = kv_start + kv_j;
        if (global_j < S) {
            int dk_idx = bhD_offset + global_j * D;
            #pragma unroll
            for (int d = 0; d < D; d += 4) {
                int dd = d + threadIdx.x % 8;
                if (dd < D) {
                    half dk_val = __float2half(dK_acc[d/4] * scale);
                    atomicAdd((unsigned short*)&dK[dk_idx + dd], __half_as_ushort(dk_val));
                }
            }
        }
    }
}

// Simplified kernel with better memory coalescing for D=256
__global__ void flash_attn_bwd_d256_kernel(
    const half* __restrict__ dO,
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const half* __restrict__ O,
    const float* __restrict__ lse,
    const float* __restrict__ dsoftmax_sum,
    half* __restrict__ dQ,
    half* __restrict__ dK,
    half* __restrict__ dV,
    int B, int H, int S, float scale
) {
    const int D = 256;
    
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    if (b >= B || h >= H) return;
    
    int tile_q = blockIdx.y;
    int tile_kv = blockIdx.x;
    
    int q_start = tile_q * 64;
    int kv_start = tile_kv * 64;
    
    int q_end = min(q_start + 64, S) - 1;
    if (kv_start > q_end) return;
    
    extern __shared__ char smem[];
    
    // Shared memory: Q[64*256] + K[64*256] + V[64*256] + dO[64*256] + S[64*64] + misc
    half* sQ = (half*)smem;
    half* sK = sQ + 64 * 256;
    half* sV = sK + 64 * 256;
    half* sdO = sV + 64 * 256;
    half* sP = sdO + 64 * 256;
    
    float* smisc = (float*)(sP + 64 * 64);
    float* sLSE = smisc;
    float* sdsoft = sLSE + 64;
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane = tid % 32;
    int warp = tid / 32;
    
    int bh_offset = (b * H + h) * S;
    int bhD_offset = bh_offset * D;
    
    // Cooperative loading of Q and dO tiles
    for (int idx = tid; idx < 64 * D; idx += blockDim.x * blockDim.y) {
        int i = idx / D;
        int d = idx % D;
        int global_i = q_start + i;
        if (global_i < S) {
            int base = bhD_offset + global_i * D + d;
            sQ[idx] = Q[base];
            sdO[idx] = dO[base];
        } else {
            sQ[idx] = __float2half(0.0f);
            sdO[idx] = __float2half(0.0f);
        }
    }
    
    // Load LSE and dsoftmax
    if (tid < 64) {
        int global_i = q_start + tid;
        if (global_i < S) {
            sLSE[tid] = lse[bh_offset + global_i];
            sdsoft[tid] = dsoftmax_sum[bh_offset + global_i];
        } else {
            sLSE[tid] = 0.0f;
            sdsoft[tid] = 0.0f;
        }
    }
    
    // Load K and V tiles
    for (int idx = tid; idx < 64 * D; idx += blockDim.x * blockDim.y) {
        int j = idx / D;
        int d = idx % D;
        int global_j = kv_start + j;
        if (global_j < S) {
            int base = bhD_offset + global_j * D + d;
            sK[idx] = K[base];
            sV[idx] = V[base];
        } else {
            sK[idx] = __float2half(0.0f);
            sV[idx] = __float2half(0.0f);
        }
    }
    
    __syncthreads();
    
    // Compute attention scores and P matrix
    // Each warp handles 4x16 block of S
    int qi_base = warp * 4;
    int kj_base = (lane % 16) * 4;
    
    float p_local[4][4];
    float dP_local[4][4];
    
    #pragma unroll
    for (int ii = 0; ii < 4; ii++) {
        #pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            p_local[ii][jj] = 0.0f;
            dP_local[ii][jj] = 0.0f;
        }
    }
    
    // Compute S = Q @ K^T and dP = dO @ V^T
    for (int d = 0; d < D; d += 8) {
        // Load Q and K fragments
        half q_frag[4][8];
        half k_frag[4][8];
        half do_frag[4][8];
        half v_frag[4][8];
        
        #pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            int qi = qi_base + ii;
            #pragma unroll
            for (int dd = 0; dd < 8; dd++) {
                q_frag[ii][dd] = sQ[qi * D + d + dd];
                do_frag[ii][dd] = sdO[qi * D + d + dd];
            }
        }
        
        #pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            int kj = kj_base + jj;
            #pragma unroll
            for (int dd = 0; dd < 8; dd++) {
                k_frag[jj][dd] = sK[kj * D + d + dd];
                v_frag[jj][dd] = sV[kj * D + d + dd];
            }
        }
        
        #pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            #pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                #pragma unroll
                for (int dd = 0; dd < 8; dd++) {
                    p_local[ii][jj] += __half2float(q_frag[ii][dd]) * __half2float(k_frag[jj][dd]);
                    dP_local[ii][jj] += __half2float(do_frag[ii][dd]) * __half2float(v_frag[jj][dd]);
                }
            }
        }
    }
    
    // Apply scale, causal mask, compute P = exp(S - LSE), compute dS
    #pragma unroll
    for (int ii = 0; ii < 4; ii++) {
        int qi = qi_base + ii;
        int global_i = q_start + qi;
        float lse_val = sLSE[qi];
        float dsum = sdsoft[qi];
        
        #pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            int kj = kj_base + jj;
            int global_j = kv_start + kj;
            
            float s_val = p_local[ii][jj] * scale;
            float p_val;
            
            if (global_i >= S || global_j >= S || global_j > global_i) {
                p_val = 0.0f;
            } else {
                p_val = expf(s_val - lse_val);
            }
            
            p_local[ii][jj] = p_val;
            
            // Compute dS = P * (dP - dsoftmax_sum)
            float ds_val = p_val * (dP_local[ii][jj] - dsum);
            dP_local[ii][jj] = ds_val;  // reuse as dS
            
            // Store P to shared memory for dV computation
            if (qi < 64 && kj < 64) {
                sP[qi * 64 + kj] = __float2half(p_val);
            }
        }
    }
    
    __syncthreads();
    
    // Accumulate dV: each thread handles part of dV
    // dV[j, d] += sum_i P[i,j] * dO[i, d]
    int dv_j = warp * 2 + lane / 16;
    int dv_d = (lane % 16) * 16;
    
    if (dv_j < 64) {
        int global_j = kv_start + dv_j;
        
        float dv_acc[16] = {0};
        
        for (int qi = 0; qi < 64 && q_start + qi < S; qi++) {
            int global_i = q_start + qi;
            if (global_j <= global_i && global_j < S) {
                half p_half = sP[qi * 64 + dv_j];
                float p_val = __half2float(p_half);
                
                #pragma unroll
                for (int dd = 0; dd < 16; dd++) {
                    half do_val = sdO[qi * D + dv_d + dd];
                    dv_acc[dd] += p_val * __half2float(do_val);
                }
            }
        }
        
        // Atomic add to global dV
        if (global_j < S) {
            int base = bhD_offset + global_j * D + dv_d;
            #pragma unroll
            for (int dd = 0; dd < 16; dd++) {
                half val = __float2half(dv_acc[dd]);
                atomicAdd((unsigned short*)&dV[base + dd], __half_as_ushort(val));
            }
        }
    }
    
    __syncthreads();
    
    // Compute dQ: dQ[i, d] += sum_j dS[i,j] * K[j, d] * scale
    int dq_i = warp * 2 + lane / 32;
    int dq_d = (lane % 32) * 8;
    
    if (dq_i < 64) {
        int global_i = q_start + dq_i;
        
        float dq_acc[8] = {0};
        
        for (int kj = 0; kj < 64 && kv_start + kj < S; kj++) {
            int global_j = kv_start + kj;
            if (global_j <= global_i && global_i < S) {
                float ds_val = dP_local[dq_i - qi_base][kj - kj_base];
                
                #pragma unroll
                for (int dd = 0; dd < 8; dd++) {
                    half k_val = sK[kj * D + dq_d + dd];
                    dq_acc[dd] += ds_val * __half2float(k_val);
                }
            }
        }
        
        // Atomic add to global dQ
        if (global_i < S) {
            int base = bhD_offset + global_i * D + dq_d;
            #pragma unroll
            for (int dd = 0; dd < 8; dd++) {
                half val = __float2half(dq_acc[dd] * scale);
                atomicAdd((unsigned short*)&dQ[base + dd], __half_as_ushort(val));
            }
        }
    }
    
    // Compute dK: dK[j, d] += sum_i dS[i,j] * Q[i, d] * scale
    int dk_j = warp * 2 + lane / 32;
    int dk_d = (lane % 32) * 8;
    
    if (dk_j < 64) {
        int global_j = kv_start + dk_j;
        
        float dk_acc[8] = {0};
        
        for (int qi = 0; qi < 64 && q_start + qi < S; qi++) {
            int global_i = q_start + qi;
            if (global_j <= global_i && global_j < S) {
                // Need to get dS from correct position
                int ii = qi % 4;
                int jj = dk_j % 4;
                // Simplified: recompute or use stored value
                // For now, use the dP_local if in range
                float ds_val = 0.0f;
                if (qi >= qi_base && qi < qi_base + 4 && dk_j >= kj_base && dk_j < kj_base + 4) {
                    ds_val = dP_local[qi - qi_base][dk_j - kj_base];
                }
                
                #pragma unroll
                for (int dd = 0; dd < 8; dd++) {
                    half q_val = sQ[qi * D + dk_d + dd];
                    dk_acc[dd] += ds_val * __half2float(q_val);
                }
            }
        }
        
        // Atomic add to global dK
        if (global_j < S) {
            int base = bhD_offset + global_j * D + dk_d;
            #pragma unroll
            for (int dd = 0; dd < 8; dd++) {
                half val = __float2half(dk_acc[dd] * scale);
                atomicAdd((unsigned short*)&dK[base + dd], __half_as_ushort(val));
            }
        }
    }
}

// Optimized kernel for D=256 with proper dK computation
__global__ void flash_attn_bwd_d256_v2_kernel(
    const half* __restrict__ dO,
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const half* __restrict__ O,
    const float* __restrict__ lse,
    const float* __restrict__ dsoftmax_sum,
    half* __restrict__ dQ,
    half* __restrict__ dK,
    half* __restrict__ dV,
    int B, int H, int S, float scale
) {
    const int D = 256;
    
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    if (b >= B || h >= H) return;
    
    int tile_q = blockIdx.y;
    int tile_kv = blockIdx.x;
    
    int q_start = tile_q * 64;
    int kv_start = tile_kv * 64;
    
    int q_end = min(q_start + 64, S) - 1;
    if (kv_start > q_end) return;
    
    extern __shared__ char smem[];
    
    half* sQ = (half*)smem;
    half* sK = sQ + 64 * D;
    half* sV = sK + 64 * D;
    half* sdO = sV + 64 * D;
    half* sP = sdO + 64 * D;
    half* sdS = sP + 64 * 64;
    
    float* smisc = (float*)(sdS + 64 * 64);
    float* sLSE = smisc;
    float* sdsoft = sLSE + 64;
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    
    int bh_offset = (b * H + h) * S;
    int bhD_offset = bh_offset * D;
    
    // Load tiles
    for (int idx = tid; idx < 64 * D; idx += blockDim.x * blockDim.y) {
        int i = idx / D;
        int d = idx % D;
        int global_i = q_start + i;
        if (global_i < S) {
            int base = bhD_offset + global_i * D + d;
            sQ[idx] = Q[base];
            sdO[idx] = dO[base];
        } else {
            sQ[idx] = __float2half(0.0f);
            sdO[idx] = __float2half(0.0f);
        }
    }
    
    for (int idx = tid; idx < 64 * D; idx += blockDim.x * blockDim.y) {
        int j = idx / D;
        int d = idx % D;
        int global_j = kv_start + j;
        if (global_j < S) {
            int base = bhD_offset + global_j * D + d;
            sK[idx] = K[base];
            sV[idx] = V[base];
        } else {
            sK[idx] = __float2half(0.0f);
            sV[idx] = __float2half(0.0f);
        }
    }
    
    if (tid < 64) {
        int global_i = q_start + tid;
        if (global_i < S) {
            sLSE[tid] = lse[bh_offset + global_i];
            sdsoft[tid] = dsoftmax_sum[bh_offset + global_i];
        } else {
            sLSE[tid] = 0.0f;
            sdsoft[tid] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Compute S, P, dS in tiles
    int warp = tid / 32;
    int lane = tid % 32;
    
    // 8 warps, each handles 8x8 block of the 64x64 S matrix
    int qi_warp = warp / 2;
    int kj_warp = warp % 2;
    
    for (int qi_sub = 0; qi_sub < 8; qi_sub++) {
        for (int kj_sub = 0; kj_sub < 32; kj_sub += 4) {
            int qi = qi_warp * 8 + qi_sub;
            int kj = kj_warp * 32 + kj_sub + (lane % 4);
            
            if (qi >= 64 || kj >= 64) continue;
            
            int global_i = q_start + qi;
            int global_j = kv_start + kj;
            
            // Compute dot product for S
            float s_val = 0.0f;
            float dp_val = 0.0f;
            
            #pragma unroll
            for (int d = 0; d < D; d += 4) {
                float q0 = __half2float(sQ[qi * D + d + 0]);
                float q1 = __half2float(sQ[qi * D + d + 1]);
                float q2 = __half2float(sQ[qi * D + d + 2]);
                float q3 = __half2float(sQ[qi * D + d + 3]);
                
                float k0 = __half2float(sK[kj * D + d + 0]);
                float k1 = __half2float(sK[kj * D + d + 1]);
                float k2 = __half2float(sK[kj * D + d + 2]);
                float k3 = __half2float(sK[kj * D + d + 3]);
                
                float do0 = __half2float(sdO[qi * D + d + 0]);
                float do1 = __half2float(sdO[qi * D + d + 1]);
                float do2 = __half2float(sdO[qi * D + d + 2]);
                float do3 = __half2float(sdO[qi * D + d + 3]);
                
                float v0 = __half2float(sV[kj * D + d + 0]);
                float v1 = __half2float(sV[kj * D + d + 1]);
                float v2 = __half2float(sV[kj * D + d + 2]);
                float v3 = __half2float(sV[kj * D + d + 3]);
                
                s_val += q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
                dp_val += do0 * v0 + do1 * v1 + do2 * v2 + do3 * v3;
            }
            
            s_val *= scale;
            
            float p_val = 0.0f;
            float ds_val = 0.0f;
            
            if (global_i < S && global_j < S && global_j <= global_i) {
                p_val = expf(s_val - sLSE[qi]);
                ds_val = p_val * (dp_val - sdsoft[qi]);
            }
            
            sP[qi * 64 + kj] = __float2half(p_val);
            sdS[qi * 64 + kj] = __float2half(ds_val);
        }
    }
    
    __syncthreads();
    
    // Compute dV accumulations
    // Each thread handles one (j, d) position
    for (int idx = tid; idx < 64 * D; idx += blockDim.x * blockDim.y) {
        int j = idx / D;
        int d = idx % D;
        int global_j = kv_start + j;
        
        if (global_j >= S) continue;
        
        float dv_val = 0.0f;
        
        for (int qi = 0; qi < 64 && q_start + qi < S; qi++) {
            int global_i = q_start + qi;
            if (global_j <= global_i) {
                float p_val = __half2float(sP[qi * 64 + j]);
                float do_val = __half2float(sdO[qi * D + d]);
                dv_val += p_val * do_val;
            }
        }
        
        if (global_j < S) {
            int base = bhD_offset + global_j * D + d;
            half val = __float2half(dv_val);
            atomicAdd((unsigned short*)&dV[base], __half_as_ushort(val));
        }
    }
    
    // Compute dQ accumulations
    for (int idx = tid; idx < 64 * D; idx += blockDim.x * blockDim.y) {
        int i = idx / D;
        int d = idx % D;
        int global_i = q_start + i;
        
        if (global_i >= S) continue;
        
        float dq_val = 0.0f;
        
        for (int kj = 0; kj < 64 && kv_start + kj < S; kj++) {
            int global_j = kv_start + kj;
            if (global_j <= global_i) {
                float ds_val = __half2float(sdS[i * 64 + kj]);
                float k_val = __half2float(sK[kj * D + d]);
                dq_val += ds_val * k_val;
            }
        }
        
        if (global_i < S) {
            int base = bhD_offset + global_i * D + d;
            half val = __float2half(dq_val * scale);
            atomicAdd((unsigned short*)&dQ[base], __half_as_ushort(val));
        }
    }
    
    // Compute dK accumulations
    for (int idx = tid; idx < 64 * D; idx += blockDim.x * blockDim.y) {
        int j = idx / D;
        int d = idx % D;
        int global_j = kv_start + j;
        
        if (global_j >= S) continue;
        
        float dk_val = 0.0f;
        
        for (int qi = 0; qi < 64 && q_start + qi < S; qi++) {
            int global_i = q_start + qi;
            if (global_j <= global_i) {
                float ds_val = __half2float(sdS[qi * 64 + j]);
                float q_val = __half2float(sQ[qi * D + d]);
                dk_val += ds_val * q_val;
            }
        }
        
        if (global_j < S) {
            int base = bhD_offset + global_j * D + d;
            half val = __float2half(dk_val * scale);
            atomicAdd((unsigned short*)&dK[base], __half_as_ushort(val));
        }
    }
}

// Final optimized version with better parallelism
__global__ void flash_attn_bwd_final_kernel(
    const half* __restrict__ dO,
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const half* __restrict__ O,
    const float* __restrict__ lse,
    const float* __restrict__ dsoftmax_sum,
    half* __restrict__ dQ,
    half* __restrict__ dK,
    half* __restrict__ dV,
    int B, int H, int S, int D, float scale
) {
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    if (b >= B || h >= H) return;
    
    int tile_q = blockIdx.y;
    int tile_kv = blockIdx.x;
    
    const int TILE = 64;
    int q_start = tile_q * TILE;
    int kv_start = tile_kv * TILE;
    
    int q_end = min(q_start + TILE, S) - 1;
    if (kv_start > q_end) return;
    
    extern __shared__ char smem[];
    
    half* sQ = (half*)smem;
    half* sK = sQ + TILE * D;
    half* sV = sK + TILE * D;
    half* sdO = sV + TILE * D;
    float* sP = (float*)(sdO + TILE * D);
    float* sdS = sP + TILE * TILE;
    float* sLSE = sdS + TILE * TILE;
    float* sdsoft = sLSE + TILE;
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int nthreads = blockDim.x * blockDim.y;
    
    int bh_offset = (b * H + h) * S;
    int bhD_offset = bh_offset * D;
    
    // Load Q, dO tiles
    for (int idx = tid; idx < TILE * D; idx += nthreads) {
        int i = idx / D;
        int d = idx % D;
        int gi = q_start + i;
        if (gi < S) {
            int base = bhD_offset + gi * D + d;
            sQ[idx] = Q[base];
            sdO[idx] = dO[base];
        }
    }
    
    // Load K, V tiles
    for (int idx = tid; idx < TILE * D; idx += nthreads) {
        int j = idx / D;
        int d = idx % D;
        int gj = kv_start + j;
        if (gj < S) {
            int base = bhD_offset + gj * D + d;
            sK[idx] = K[base];
            sV[idx] = V[base];
        }
    }
    
    // Load LSE, dsoftmax
    if (tid < TILE) {
        int gi = q_start + tid;
        if (gi < S) {
            sLSE[tid] = lse[bh_offset + gi];
            sdsoft[tid] = dsoftmax_sum[bh_offset + gi];
        } else {
            sLSE[tid] = 0.0f;
            sdsoft[tid] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Compute attention scores P and gradients dS
    // Each thread computes multiple elements
    for (int idx = tid; idx < TILE * TILE; idx += nthreads) {
        int qi = idx / TILE;
        int kj = idx % TILE;
        int gi = q_start + qi;
        int gj = kv_start + kj;
        
        if (gi >= S || gj > gi || gj >= S) {
            sP[idx] = 0.0f;
            sdS[idx] = 0.0f;
            continue;
        }
        
        // Compute S[qi, kj] = sum_d Q[qi, d] * K[kj, d]
        float s_val = 0.0f;
        float dP_val = 0.0f;
        
        #pragma unroll 8
        for (int d = 0; d < D; d++) {
            float q = __half2float(sQ[qi * D + d]);
            float k = __half2float(sK[kj * D + d]);
            float v = __half2float(sV[kj * D + d]);
            float do_val = __half2float(sdO[qi * D + d]);
            s_val += q * k;
            dP_val += do_val * v;
        }
        
        s_val *= scale;
        float p_val = expf(s_val - sLSE[qi]);
        float ds_val = p_val * (dP_val - sdsoft[qi]);
        
        sP[idx] = p_val;
        sdS[idx] = ds_val;
    }
    
    __syncthreads();
    
    // Accumulate dV[j, d] = sum_i P[i, j] * dO[i, d]
    for (int idx = tid; idx < TILE * D; idx += nthreads) {
        int j = idx / D;
        int d = idx % D;
        int gj = kv_start + j;
        if (gj >= S) continue;
        
        float sum = 0.0f;
        for (int qi = 0; qi < TILE && q_start + qi < S; qi++) {
            int gi = q_start + qi;
            if (gj <= gi) {
                sum += sP[qi * TILE + j] * __half2float(sdO[qi * D + d]);
            }
        }
        
        int base = bhD_offset + gj * D + d;
        atomicAdd((unsigned short*)&dV[base], __half_as_ushort(__float2half(sum)));
    }
    
    // Accumulate dQ[i, d] = sum_j dS[i, j] * K[j, d] * scale
    for (int idx = tid; idx < TILE * D; idx += nthreads) {
        int i = idx / D;
        int d = idx % D;
        int gi = q_start + i;
        if (gi >= S) continue;
        
        float sum = 0.0f;
        for (int kj = 0; kj < TILE && kv_start + kj < S; kj++) {
            int gj = kv_start + kj;
            if (gj <= gi) {
                sum += sdS[i * TILE + kj] * __half2float(sK[kj * D + d]);
            }
        }
        
        int base = bhD_offset + gi * D + d;
        atomicAdd((unsigned short*)&dQ[base], __half_as_ushort(__float2half(sum * scale)));
    }
    
    // Accumulate dK[j, d] = sum_i dS[i, j] * Q[i, d] * scale
    for (int idx = tid; idx < TILE * D; idx += nthreads) {
        int j = idx / D;
        int d = idx % D;
        int gj = kv_start + j;
        if (gj >= S) continue;
        
        float sum = 0.0f;
        for (int qi = 0; qi < TILE && q_start + qi < S; qi++) {
            int gi = q_start + qi;
            if (gj <= gi) {
                sum += sdS[qi * TILE + j] * __half2float(sQ[qi * D + d]);
            }
        }
        
        int base = bhD_offset + gj * D + d;
        atomicAdd((unsigned short*)&dK[base], __half_as_ushort(__float2half(sum * scale)));
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
    // Zero output gradients
    size_t grad_size = (size_t)B * H * Sq * D * sizeof(half);
    cudaMemsetAsync(dQ, 0, grad_size, stream);
    cudaMemsetAsync(dK, 0, grad_size, stream);
    cudaMemsetAsync(dV, 0, grad_size, stream);
    
    // Allocate dsoftmax_sum
    float* dsoftmax_sum;
    cudaMallocAsync(&dsoftmax_sum, (size_t)B * H * Sq * sizeof(float), stream);
    
    // Step 1: Compute dsoftmax_sum = sum(dO * O)
    dim3 grid1((Sq + 127) / 128, 1, B * H);
    dim3 block1(32, 4);
    compute_dsoftmax_sum_kernel<<<grid1, block1, 0, stream>>>(
        (const half*)dO, (const half*)O, dsoftmax_sum,
        B, H, Sq, D
    );
    
    // Step 2: Main backward kernel
    const int TILE = 64;
    int n_tiles_q = (Sq + TILE - 1) / TILE;
    int n_tiles_kv = (Sk + TILE - 1) / TILE;
    
    dim3 grid2(n_tiles_kv, n_tiles_q, B * H);
    dim3 block2(32, 8);  // 256 threads
    
    size_t smem_size = (TILE * D * 4 + TILE * TILE * 2 + TILE * 2) * sizeof(half) 
                     + (TILE * TILE * 2 + TILE * 2) * sizeof(float);
    
    // Use appropriate kernel based on D
    if (D == 256) {
        flash_attn_bwd_final_kernel<<<grid2, block2, smem_size, stream>>>(
            (const half*)dO, (const half*)Q, (const half*)K,
            (const half*)V, (const half*)O, lse, dsoftmax_sum,
            (half*)dQ, (half*)dK, (half*)dV,
            B, H, Sq, D, scale
        );
    } else {
        // Generic version (slower but works for any D)
        flash_attn_bwd_final_kernel<<<grid2, block2, smem_size, stream>>>(
            (const half*)dO, (const half*)Q, (const half*)K,
            (const half*)V, (const half*)O, lse, dsoftmax_sum,
            (half*)dQ, (half*)dK, (half*)dV,
            B, H, Sq, D, scale
        );
    }
    
    cudaFreeAsync(dsoftmax_sum, stream);
}

} // extern "C"
