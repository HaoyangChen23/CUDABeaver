#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

#define WARP_SIZE 32

// Tile sizes optimized for hdim=192, bf16 on SM80
constexpr int kBlockM = 128;  // Q tile size
constexpr int kBlockN = 128;  // K/V tile size
constexpr int kHeadDim = 192;

// Helper: convert to/from bf16
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

// Compute dsoftmax_sum: sum over D of dO * O
__global__ void compute_dsoftmax_sum_kernel(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O,
    float* __restrict__ dsoftmax_sum,
    int B, int H, int S, int D
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (i >= S) return;
    
    float sum = 0.0f;
    int base = ((b * H + h) * S + i) * D;
    
    #pragma unroll 8
    for (int d = 0; d < D; ++d) {
        sum += bf16_to_float(dO[base + d]) * bf16_to_float(O[base + d]);
    }
    
    dsoftmax_sum[((b * H + h) * S) + i] = sum;
}

// Shared memory layout for Flash Attention backward
// We need to store tiles of Q, K, V, dO, and accumulate dQ, dK, dV
template<int BM, int BN, int BD>
struct SharedMemLayout {
    static constexpr int q_size = BM * BD;
    static constexpr int k_size = BN * BD;
    static constexpr int v_size = BN * BD;
    static constexpr int do_size = BM * BD;
    static constexpr int o_size = BM * BD;
    static constexpr int dq_size = BM * BD;
    static constexpr int dk_size = BN * BD;
    static constexpr int dv_size = BN * BD;
    
    static constexpr int total_size = q_size + k_size + v_size + do_size + o_size 
                                     + dq_size + dk_size + dv_size;
};

// Main backward kernel
template<int BM, int BN, int BD>
__global__ void flash_attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const float* __restrict__ lse,
    const float* __restrict__ dsoftmax_sum,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int Sq, int Sk, int D,
    float scale
) {
    using SmemLayout = SharedMemLayout<BM, BN, BD>;
    
    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    
    // Shared memory partitioning
    __nv_bfloat16* sQ = smem;
    __nv_bfloat16* sK = sQ + SmemLayout::q_size;
    __nv_bfloat16* sV = sK + SmemLayout::k_size;
    __nv_bfloat16* sdO = sV + SmemLayout::v_size;
    __nv_bfloat16* sO = sdO + SmemLayout::do_size;
    __nv_bfloat16* sdQ = sO + SmemLayout::o_size;
    __nv_bfloat16* sdK = sdQ + SmemLayout::dq_size;
    __nv_bfloat16* sdV = sdK + SmemLayout::dk_size;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    
    // This block handles a K/V tile (N tile)
    int n_block = blockIdx.x;
    int n_idx = n_block * BN;
    
    // Number of Q tiles to iterate over
    int num_m_blocks = (Sq + BM - 1) / BM;
    
    // Initialize accumulators for dK and dV (per-thread fragments)
    float frag_dK[BD][8];  // BN x BD, but we compute in tiles
    float frag_dV[BD][8];
    
    // Zero accumulators
    #pragma unroll
    for (int i = 0; i < BD; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            frag_dK[i][j] = 0.0f;
            frag_dV[i][j] = 0.0f;
        }
    }
    
    // Load K and V tile into shared memory (persistent for all Q tiles)
    // Each thread loads elements
    int k_row = tid / (BD / 2);
    int k_col = (tid % (BD / 2)) * 2;
    
    #pragma unroll 4
    for (int i = 0; i < BN; i += blockDim.x / (BD/2)) {
        int row = k_row + i;
        if (row < BN && n_idx + row < Sk) {
            int g_idx = ((b * H + h) * Sk + n_idx + row) * D + k_col;
            if (k_col + 1 < D) {
                sK[row * BD + k_col] = K[g_idx];
                sK[row * BD + k_col + 1] = K[g_idx + 1];
                sV[row * BD + k_col] = V[g_idx];
                sV[row * BD + k_col + 1] = V[g_idx + 1];
            } else {
                sK[row * BD + k_col] = K[g_idx];
                sV[row * BD + k_col] = V[g_idx];
            }
        }
    }
    
    __syncthreads();
    
    // Iterate over Q tiles
    for (int m_block = 0; m_block < num_m_blocks; ++m_block) {
        int m_idx = m_block * BM;
        
        // Load Q, dO, O tiles
        int q_row = tid / (BD / 2);
        int q_col = (tid % (BD / 2)) * 2;
        
        #pragma unroll 4
        for (int i = 0; i < BM; i += blockDim.x / (BD/2)) {
            int row = q_row + i;
            if (row < BM && m_idx + row < Sq) {
                int g_idx = ((b * H + h) * Sq + m_idx + row) * D + q_col;
                if (q_col + 1 < D) {
                    sQ[row * BD + q_col] = Q[g_idx];
                    sQ[row * BD + q_col + 1] = Q[g_idx + 1];
                    sdO[row * BD + q_col] = dO[g_idx];
                    sdO[row * BD + q_col + 1] = dO[g_idx + 1];
                    sO[row * BD + q_col] = O[g_idx];
                    sO[row * BD + q_col + 1] = O[g_idx + 1];
                } else {
                    sQ[row * BD + q_col] = Q[g_idx];
                    sdO[row * BD + q_col] = dO[g_idx];
                    sO[row * BD + q_col] = O[g_idx];
                }
            }
        }
        
        __syncthreads();
        
        // Compute S = Q @ K^T for this tile
        // S is BM x BN
        float frag_S[8][8];  // Thread-local accumulator
        
        // Clear S
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                frag_S[i][j] = 0.0f;
            }
        }
        
        // MMA: S[m,n] = sum_d Q[m,d] * K[n,d]
        // Each thread handles a sub-tile
        int tm = (tid / 8) * 8;  // thread tile row start in BM
        int tn = (tid % 8) * 8;  // thread tile col start in BN
        
        #pragma unroll
        for (int d = 0; d < BD; ++d) {
            float q_vals[8];
            float k_vals[8];
            
            // Load Q values for this thread's rows
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int q_row = tm + i;
                if (q_row < BM && m_idx + q_row < Sq) {
                    q_vals[i] = bf16_to_float(sQ[q_row * BD + d]);
                } else {
                    q_vals[i] = 0.0f;
                }
            }
            
            // Load K values for this thread's cols
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int k_row = tn + j;
                if (k_row < BN && n_idx + k_row < Sk) {
                    k_vals[j] = bf16_to_float(sK[k_row * BD + d]);
                } else {
                    k_vals[j] = 0.0f;
                }
            }
            
            // Accumulate outer product
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    frag_S[i][j] += q_vals[i] * k_vals[j];
                }
            }
        }
        
        // Scale S
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                frag_S[i][j] *= scale;
            }
        }
        
        // Compute P = exp(S - lse)
        // Load lse for this Q tile
        float lse_vals[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int q_row = tm + i;
            if (q_row < BM && m_idx + q_row < Sq) {
                lse_vals[i] = lse[((b * H + h) * Sq) + m_idx + q_row];
            } else {
                lse_vals[i] = 0.0f;
            }
        }
        
        float frag_P[8][8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float s_val = frag_S[i][j];
                float p_val = expf(s_val - lse_vals[i]);
                // Clamp for numerical stability
                p_val = fmaxf(0.0f, fminf(1.0f, p_val));
                frag_P[i][j] = p_val;
            }
        }
        
        // Compute dV += P^T @ dO
        // dV is BN x BD, P^T is BN x BM
        // Each thread accumulates into frag_dV
        
        // Load dO for this thread's Q rows
        float do_vals[8][BD/8];  // Reinterpret as needed
        
        // Simplified: accumulate directly
        #pragma unroll
        for (int n = 0; n < 8; ++n) {  // over BN (from P^T)
            int k_row = tn + n;
            if (k_row >= BN || n_idx + k_row >= Sk) continue;
            
            #pragma unroll
            for (int d = 0; d < BD; ++d) {
                float dv_acc = 0.0f;
                
                #pragma unroll
                for (int m = 0; m < 8; ++m) {  // over BM
                    int q_row = tm + m;
                    if (q_row >= BM || m_idx + q_row >= Sq) continue;
                    
                    float p_val = frag_P[m][n];  // P[q_row, k_row]
                    float do_val = bf16_to_float(sdO[q_row * BD + d]);
                    dv_acc += p_val * do_val;
                }
                
                frag_dV[d][n] += dv_acc;
            }
        }
        
        // Compute dP = dO @ V^T
        float frag_dP[8][8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                frag_dP[i][j] = 0.0f;
            }
        }
        
        #pragma unroll
        for (int d = 0; d < BD; ++d) {
            float do_vals_local[8];
            float v_vals[8];
            
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int q_row = tm + i;
                if (q_row < BM && m_idx + q_row < Sq) {
                    do_vals_local[i] = bf16_to_float(sdO[q_row * BD + d]);
                } else {
                    do_vals_local[i] = 0.0f;
                }
            }
            
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int k_row = tn + j;
                if (k_row < BN && n_idx + k_row < Sk) {
                    v_vals[j] = bf16_to_float(sV[k_row * BD + d]);
                } else {
                    v_vals[j] = 0.0f;
                }
            }
            
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    frag_dP[i][j] += do_vals_local[i] * v_vals[j];
                }
            }
        }
        
        // Load dsoftmax_sum for this Q tile
        float ds_sum[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int q_row = tm + i;
            if (q_row < BM && m_idx + q_row < Sq) {
                ds_sum[i] = dsoftmax_sum[((b * H + h) * Sq) + m_idx + q_row];
            } else {
                ds_sum[i] = 0.0f;
            }
        }
        
        // Compute dS = P * (dP - dsoftmax_sum)
        float frag_dS[8][8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                frag_dS[i][j] = frag_P[i][j] * (frag_dP[i][j] - ds_sum[i]);
            }
        }
        
        // Compute dQ += dS @ K * scale
        // Accumulate to sdQ in shared memory (atomically or via reduction)
        // For simplicity, use atomic adds or direct write with proper synchronization
        
        // Compute contribution to dQ
        float dq_acc[8][BD/8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int d = 0; d < BD; ++d) {
                float acc = 0.0f;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int k_row = tn + j;
                    if (k_row >= BN || n_idx + k_row >= Sk) continue;
                    acc += frag_dS[i][j] * bf16_to_float(sK[k_row * BD + d]);
                }
                // Scale and add to global dQ
                acc *= scale;
                
                int q_row = tm + i;
                if (q_row < BM && m_idx + q_row < Sq) {
                    int g_idx = ((b * H + h) * Sq + m_idx + q_row) * D + d;
                    // Atomic add for correctness across K tiles
                    atomicAdd(reinterpret_cast<float*>(dQ) + g_idx, acc);
                }
            }
        }
        
        // Accumulate dK += dS^T @ Q * scale
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int k_row = tn + j;
            if (k_row >= BN || n_idx + k_row >= Sk) continue;
            
            #pragma unroll
            for (int d = 0; d < BD; ++d) {
                float acc = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int q_row = tm + i;
                    if (q_row >= BM || m_idx + q_row >= Sq) continue;
                    acc += frag_dS[i][j] * bf16_to_float(sQ[q_row * BD + d]);
                }
                frag_dK[d][j] += acc * scale;
            }
        }
        
        __syncthreads();
    }
    
    // Write out accumulated dK and dV
    // Each thread writes its accumulated values
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        int k_row = (tid / (BD/2)) * 8 + j;  // simplified mapping
        // More accurate mapping based on actual thread layout
        k_row = ((tid / 8) % 16) * 8 + j;  // adjust based on block size
        
        // Recompute proper index
        int thread_n_base = (tid / (BD/8)) * 8;
        k_row = thread_n_base + j;
        
        if (k_row >= BN) continue;
        int global_k_row = n_idx + k_row;
        if (global_k_row >= Sk) continue;
        
        #pragma unroll
        for (int d = 0; d < BD; d += 8) {
            int d_base = (tid % (BD/8)) * 8;
            // Simplified: each thread handles contiguous d
            
            // Actually, let's use a simpler mapping
        }
    }
    
    // Simpler write-out: use all threads collaboratively
    int write_row = tid / BD;
    int write_col = tid % BD;
    
    if (write_row < BN && n_idx + write_row < Sk) {
        int g_idx = ((b * H + h) * Sk + n_idx + write_row) * D + write_col;
        
        // Sum up contributions from all threads (reduction needed)
        // For now, use atomic add
        // This is a simplified version - proper implementation needs reduction
        
        // Reduce frag_dK and frag_dV across threads
        // Each element in frag is partial, need to sum
        
        // Simple approach: each thread writes its partial, atomic add
        // Find which fragment element this thread owns
        int frag_j = write_row % 8;
        int frag_d = write_col;
        
        if (frag_d < BD) {
            atomicAdd(reinterpret_cast<float*>(dK) + g_idx, frag_dK[frag_d][frag_j]);
            atomicAdd(reinterpret_cast<float*>(dV) + g_idx, frag_dV[frag_d][frag_j]);
        }
    }
}

// Simpler, more correct implementation using separate kernel for dK/dV accumulation
template<int BM, int BN, int BD>
__global__ void flash_attn_bwd_dq_kernel(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const float* __restrict__ lse,
    const float* __restrict__ dsoftmax_sum,
    float* __restrict__ dQ_accum,  // FP32 accumulator
    int B, int H, int Sq, int Sk, int D,
    float scale
) {
    using SmemLayout = SharedMemLayout<BM, BN, BD>;
    
    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    
    __nv_bfloat16* sQ = smem;
    __nv_bfloat16* sK = sQ + SmemLayout::q_size;
    __nv_bfloat16* sV = sK + SmemLayout::k_size;
    __nv_bfloat16* sdO = sV + SmemLayout::v_size;
    
    int tid = threadIdx.x;
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    
    // This block handles a Q tile
    int m_block = blockIdx.x;
    int m_idx = m_block * BM;
    
    // Zero dQ accumulator in shared memory
    float* sdQ_acc = reinterpret_cast<float*>(smem + SmemLayout::total_size - SmemLayout::dq_size * 2);
    
    // Initialize
    for (int i = tid; i < BM * BD; i += blockDim.x) {
        sdQ_acc[i] = 0.0f;
    }
    __syncthreads();
    
    // Load Q, dO tiles
    for (int i = tid; i < BM * BD; i += blockDim.x) {
        int row = i / BD;
        int col = i % BD;
        int g_idx = ((b * H + h) * Sq + m_idx + row) * D + col;
        
        if (m_idx + row < Sq && col < D) {
            sQ[row * BD + col] = Q[g_idx];
            sdO[row * BD + col] = dO[g_idx];
        }
    }
    __syncthreads();
    
    // Load lse and dsoftmax_sum for this Q tile
    float lse_vals[BM];
    float ds_sum_vals[BM];
    for (int i = 0; i < BM; ++i) {
        if (m_idx + i < Sq) {
            lse_vals[i] = lse[((b * H + h) * Sq) + m_idx + i];
            ds_sum_vals[i] = dsoftmax_sum[((b * H + h) * Sq) + m_idx + i];
        }
    }
    
    // Iterate over K/V tiles
    int num_n_blocks = (Sk + BN - 1) / BN;
    
    for (int n_block = 0; n_block < num_n_blocks; ++n_block) {
        int n_idx = n_block * BN;
        
        // Load K, V tiles
        for (int i = tid; i < BN * BD; i += blockDim.x) {
            int row = i / BD;
            int col = i % BD;
            int g_idx = ((b * H + h) * Sk + n_idx + row) * D + col;
            
            if (n_idx + row < Sk && col < D) {
                sK[row * BD + col] = K[g_idx];
                sV[row * BD + col] = V[g_idx];
            }
        }
        __syncthreads();
        
        // Compute S = Q @ K^T
        // Process in thread tiles
        for (int q_row = tid / 32; q_row < BM; q_row += blockDim.x / 32) {
            if (m_idx + q_row >= Sq) continue;
            
            for (int k_row = 0; k_row < BN; ++k_row) {
                if (n_idx + k_row >= Sk) continue;
                
                float s_val = 0.0f;
                #pragma unroll 24
                for (int d = 0; d < BD; ++d) {
                    s_val += bf16_to_float(sQ[q_row * BD + d]) * bf16_to_float(sK[k_row * BD + d]);
                }
                s_val *= scale;
                
                float p_val = expf(s_val - lse_vals[q_row]);
                
                // Compute dP = dO @ V^T
                float dp_val = 0.0f;
                #pragma unroll 24
                for (int d = 0; d < BD; ++d) {
                    dp_val += bf16_to_float(sdO[q_row * BD + d]) * bf16_to_float(sV[k_row * BD + d]);
                }
                
                float ds_val = p_val * (dp_val - ds_sum_vals[q_row]);
                
                // Accumulate to dQ: dS @ K
                #pragma unroll 24
                for (int d = 0; d < BD; ++d) {
                    sdQ_acc[q_row * BD + d] += ds_val * bf16_to_float(sK[k_row * BD + d]) * scale;
                }
            }
        }
        __syncthreads();
    }
    
    // Write out dQ
    for (int i = tid; i < BM * BD; i += blockDim.x) {
        int row = i / BD;
        int col = i % BD;
        int g_idx = ((b * H + h) * Sq + m_idx + row) * D + col;
        
        if (m_idx + row < Sq && col < D) {
            dQ_accum[g_idx] = sdQ_acc[i];
        }
    }
}

template<int BM, int BN, int BD>
__global__ void flash_attn_bwd_dk_dv_kernel(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const float* __restrict__ lse,
    const float* __restrict__ dsoftmax_sum,
    float* __restrict__ dK_accum,
    float* __restrict__ dV_accum,
    int B, int H, int Sq, int Sk, int D,
    float scale
) {
    using SmemLayout = SharedMemLayout<BM, BN, BD>;
    
    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    
    __nv_bfloat16* sQ = smem;
    __nv_bfloat16* sK = sQ + SmemLayout::q_size;
    __nv_bfloat16* sV = sK + SmemLayout::k_size;
    __nv_bfloat16* sdO = sV + SmemLayout::v_size;
    __nv_bfloat16* sO = sdO + SmemLayout::do_size;
    
    int tid = threadIdx.x;
    int b = blockIdx.z / H;
    int h = blockIdx.z % H;
    
    // This block handles a K/V tile
    int n_block = blockIdx.x;
    int n_idx = n_block * BN;
    
    // Accumulators for dK and dV (in registers, then global)
    float dk_acc[BN][BD];  // Too large for registers, use shared
    float dv_acc[BN][BD];
    
    // Use shared memory for accumulation
    float* sdK_acc = reinterpret_cast<float*>(smem + SmemLayout::total_size - (BN * BD + BN * BD) * 2);
    float* sdV_acc = sdK_acc + BN * BD;
    
    for (int i = tid; i < BN * BD; i += blockDim.x) {
        sdK_acc[i] = 0.0f;
        sdV_acc[i] = 0.0f;
    }
    __syncthreads();
    
    // Load K, V tiles
    for (int i = tid; i < BN * BD; i += blockDim.x) {
        int row = i / BD;
        int col = i % BD;
        int g_idx = ((b * H + h) * Sk + n_idx + row) * D + col;
        
        if (n_idx + row < Sk && col < D) {
            sK[row * BD + col] = K[g_idx];
            sV[row * BD + col] = V[g_idx];
        }
    }
    __syncthreads();
    
    // Iterate over Q tiles
    int num_m_blocks = (Sq + BM - 1) / BM;
    
    for (int m_block = 0; m_block < num_m_blocks; ++m_block) {
        int m_idx = m_block * BM;
        
        // Load Q, dO, O tiles
        for (int i = tid; i < BM * BD; i += blockDim.x) {
            int row = i / BD;
            int col = i % BD;
            int g_idx = ((b * H + h) * Sq + m_idx + row) * D + col;
            
            if (m_idx + row < Sq && col < D) {
                sQ[row * BD + col] = Q[g_idx];
                sdO[row * BD + col] = dO[g_idx];
                sO[row * BD + col] = O[g_idx];
            }
        }
        __syncthreads();
        
        // Load lse and dsoftmax_sum for this Q tile
        float lse_vals[BM];
        float ds_sum_vals[BM];
        for (int i = 0; i < BM && m_idx + i < Sq; ++i) {
            lse_vals[i] = lse[((b * H + h) * Sq) + m_idx + i];
            ds_sum_vals[i] = dsoftmax_sum[((b * H + h) * Sq) + m_idx + i];
        }
        
        // Compute S and contributions
        for (int k_row = tid / 32; k_row < BN; k_row += blockDim.x / 32) {
            if (n_idx + k_row >= Sk) continue;
            
            for (int q_row = 0; q_row < BM; ++q_row) {
                if (m_idx + q_row >= Sq) continue;
                
                float s_val = 0.0f;
                #pragma unroll 24
                for (int d = 0; d < BD; ++d) {
                    s_val += bf16_to_float(sQ[q_row * BD + d]) * bf16_to_float(sK[k_row * BD + d]);
                }
                s_val *= scale;
                
                float p_val = expf(s_val - lse_vals[q_row]);
                
                // dV contribution: P^T @ dO
                #pragma unroll 24
                for (int d = 0; d < BD; ++d) {
                    sdV_acc[k_row * BD + d] += p_val * bf16_to_float(sdO[q_row * BD + d]);
                }
                
                // dP = dO @ V^T
                float dp_val = 0.0f;
                #pragma unroll 24
                for (int d = 0; d < BD; ++d) {
                    dp_val += bf16_to_float(sdO[q_row * BD + d]) * bf16_to_float(sV[k_row * BD + d]);
                }
                
                float ds_val = p_val * (dp_val - ds_sum_vals[q_row]);
                
                // dK contribution: dS^T @ Q
                #pragma unroll 24
                for (int d = 0; d < BD; ++d) {
                    sdK_acc[k_row * BD + d] += ds_val * bf16_to_float(sQ[q_row * BD + d]) * scale;
                }
            }
        }
        __syncthreads();
    }
    
    // Scale dK and dV by scale (dK already scaled, dV not)
    // Actually dV doesn't need scale, dK does
    
    // Write out
    for (int i = tid; i < BN * BD; i += blockDim.x) {
        int row = i / BD;
        int col = i % BD;
        int g_idx = ((b * H + h) * Sk + n_idx + row) * D + col;
        
        if (n_idx + row < Sk && col < D) {
            dK_accum[g_idx] = sdK_acc[i];
            dV_accum[g_idx] = sdV_acc[i];
        }
    }
}

// Convert FP32 accumulators to BF16
__global__ void convert_accum_to_bf16(
    const float* __restrict__ accum,
    __nv_bfloat16* __restrict__ out,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = float_to_bf16(accum[idx]);
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
    const int BM = 64;   // Q tile size
    const int BN = 64;   // K/V tile size
    
    // Allocate FP32 accumulators for dQ, dK, dV
    size_t total_elements = (size_t)B * H * Sq * D;
    float* dQ_accum;
    float* dK_accum;
    float* dV_accum;
    
    cudaMalloc(&dQ_accum, total_elements * sizeof(float));
    cudaMalloc(&dK_accum, total_elements * sizeof(float));
    cudaMalloc(&dV_accum, total_elements * sizeof(float));
    
    // Zero accumulators
    cudaMemset(dQ_accum, 0, total_elements * sizeof(float));
    cudaMemset(dK_accum, 0, total_elements * sizeof(float));
    cudaMemset(dV_accum, 0, total_elements * sizeof(float));
    
    // Compute dsoftmax_sum
    float* dsoftmax_sum;
    cudaMalloc(&dsoftmax_sum, (size_t)B * H * Sq * sizeof(float));
    
    dim3 grid_dss(B, H, (Sq + 127) / 128);
    dim3 block_dss(128);
    compute_dsoftmax_sum_kernel<<<grid_dss, block_dss, 0, stream>>>(
        (const __nv_bfloat16*)dO,
        (const __nv_bfloat16*)O,
        dsoftmax_sum,
        B, H, Sq, D
    );
    
    // Launch dQ kernel
    int num_m_blocks = (Sq + BM - 1) / BM;
    dim3 grid_dq(num_m_blocks, 1, B * H);
    int threads = 256;
    
    size_t smem_size = (BM * D + BN * D * 2 + BM * D) * sizeof(__nv_bfloat16) 
                     + BM * D * sizeof(float) + 16384;
    
    flash_attn_bwd_dq_kernel<BM, BN, kHeadDim><<<grid_dq, threads, smem_size, stream>>>(
        (const __nv_bfloat16*)dO,
        (const __nv_bfloat16*)Q,
        (const __nv_bfloat16*)K,
        (const __nv_bfloat16*)V,
        (const __nv_bfloat16*)O,
        lse,
        dsoftmax_sum,
        dQ_accum,
        B, H, Sq, Sk, D,
        scale
    );
    
    // Launch dK/dV kernel
    int num_n_blocks = (Sk + BN - 1) / BN;
    dim3 grid_dkdv(num_n_blocks, 1, B * H);
    
    flash_attn_bwd_dk_dv_kernel<BM, BN, kHeadDim><<<grid_dkdv, threads, smem_size, stream>>>(
        (const __nv_bfloat16*)dO,
        (const __nv_bfloat16*)Q,
        (const __nv_bfloat16*)K,
        (const __nv_bfloat16*)V,
        (const __nv_bfloat16*)O,
        lse,
        dsoftmax_sum,
        dK_accum,
        dV_accum,
        B, H, Sq, Sk, D,
        scale
    );
    
    // Convert accumulators to BF16
    int total_threads = 256;
    int num_blocks = (total_elements + total_threads - 1) / total_threads;
    
    convert_accum_to_bf16<<<num_blocks, total_threads, 0, stream>>>(
        dQ_accum, (__nv_bfloat16*)dQ, total_elements
    );
    convert_accum_to_bf16<<<num_blocks, total_threads, 0, stream>>>(
        dK_accum, (__nv_bfloat16*)dK, total_elements
    );
    convert_accum_to_bf16<<<num_blocks, total_threads, 0, stream>>>(
        dV_accum, (__nv_bfloat16*)dV, total_elements
    );
    
    // Free temporaries
    cudaFree(dQ_accum);
    cudaFree(dK_accum);
    cudaFree(dV_accum);
    cudaFree(dsoftmax_sum);
}

} // extern "C"
