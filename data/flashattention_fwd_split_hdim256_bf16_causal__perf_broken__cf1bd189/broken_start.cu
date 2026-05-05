#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <float.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 256

using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

inline __device__ float bf16_to_float(bf16 v) {
    return __bfloat162float(v);
}

inline __device__ bf16 float_to_bf16(float v) {
    return __float2bfloat16(v);
}

inline __device__ float2 bf162_to_float2(bf162 v) {
    return __bfloat1622float2(v);
}

inline __device__ bf162 float2_to_bf162(float2 v) {
    return __float22bfloat162_rn(v);
}

// Load 4 bf16 values (8 bytes) as float4 for vectorized loads
inline __device__ void load_bf16x4(const bf16* ptr, float4& out) {
    uint2 tmp;
    asm volatile("ld.global.ca.v2.b32 {%0, %1}, [%2];" : "=r"(tmp.x), "=r"(tmp.y) : "l"(ptr));
    uint16_t* p16 = reinterpret_cast<uint16_t*>(&tmp);
    out.x = __bfloat162float(*reinterpret_cast<bf16*>(&p16[0]));
    out.y = __bfloat162float(*reinterpret_cast<bf16*>(&p16[1]));
    out.z = __bfloat162float(*reinterpret_cast<bf16*>(&p16[2]));
    out.w = __bfloat162float(*reinterpret_cast<bf16*>(&p16[3]));
}

inline __device__ void store_bf16x4(bf16* ptr, float4 val) {
    uint2 tmp;
    uint16_t* p16 = reinterpret_cast<uint16_t*>(&tmp);
    *reinterpret_cast<bf16*>(&p16[0]) = __float2bfloat16_rn(val.x);
    *reinterpret_cast<bf16*>(&p16[1]) = __float2bfloat16_rn(val.y);
    *reinterpret_cast<bf16*>(&p16[2]) = __float2bfloat16_rn(val.z);
    *reinterpret_cast<bf16*>(&p16[3]) = __float2bfloat16_rn(val.w);
    asm volatile("st.global.wb.v2.b32 [%0], {%1, %2};" :: "l"(ptr), "r"(tmp.x), "r"(tmp.y));
}

// Warp-level reduction for max
inline __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
inline __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
inline __device__ float block_reduce_max(float val, float* shared_mem, int tid) {
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) shared_mem[warp_id] = val;
    __syncthreads();
    
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        val = shared_mem[tid];
    } else {
        val = -INFINITY;
    }
    __syncthreads();
    
    if (tid < WARP_SIZE) {
        val = warp_reduce_max(val);
    }
    return val;
}

inline __device__ float block_reduce_sum(float val, float* shared_mem, int tid) {
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared_mem[warp_id] = val;
    __syncthreads();
    
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        val = shared_mem[tid];
    } else {
        val = 0.0f;
    }
    __syncthreads();
    
    if (tid < WARP_SIZE) {
        val = warp_reduce_sum(val);
    }
    return val;
}

// Compute Q @ K^T for a tile
// Q: [D], K: [D], output: scalar
inline __device__ float compute_qk_dot(const bf16* q_ptr, const bf16* k_ptr, int D, int tid, int num_threads) {
    float sum = 0.0f;
    #pragma unroll
    for (int d = tid; d < D; d += num_threads) {
        float q_val = bf16_to_float(q_ptr[d]);
        float k_val = bf16_to_float(k_ptr[d]);
        sum += q_val * k_val;
    }
    return sum;
}

// Kernel for split-KV flash attention forward
// Each block processes: one query, one head, one batch, and a chunk of K/V
template<int D, int NUM_THREADS>
__global__ void flash_attn_fwd_split_kv_kernel(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    const bf16* __restrict__ V,
    bf16* __restrict__ O_partial,
    float* __restrict__ lse_partial,
    int B, int H, int Sq, int Sk,
    float scale,
    int num_splits,
    int split_idx
) {
    // Each block handles one (b, h, q_idx) and a chunk of K/V
    const int b = blockIdx.x / (H * Sq);
    const int h = (blockIdx.x / Sq) % H;
    const int q_idx = blockIdx.x % Sq;
    
    const int tid = threadIdx.x;
    
    // Determine K/V chunk for this split
    const int chunk_size = (Sk + num_splits - 1) / num_splits;
    const int kv_start = split_idx * chunk_size;
    const int kv_end = min(kv_start + chunk_size, Sk);
    
    // Skip if this query has no valid keys in this chunk (due to causal mask)
    // For causal: q_idx can only attend to j <= q_idx
    // So if kv_start > q_idx, this entire chunk is masked out
    if (kv_start > q_idx) {
        // Write zeros for this partial output
        if (tid == 0) {
            const int partial_idx = ((b * H + h) * Sq + q_idx) * num_splits + split_idx;
            lse_partial[partial_idx] = -INFINITY;
        }
        // Zero out output
        const int o_base = (((b * H + h) * Sq + q_idx) * num_splits + split_idx) * D;
        #pragma unroll
        for (int d = tid; d < D; d += NUM_THREADS) {
            O_partial[o_base + d] = __float2bfloat16_rn(0.0f);
        }
        return;
    }
    
    // Actual end of valid keys for this query (causal)
    const int actual_kv_end = min(kv_end, q_idx + 1);
    const int actual_kv_start = kv_start;
    
    if (actual_kv_start >= actual_kv_end) {
        if (tid == 0) {
            const int partial_idx = ((b * H + h) * Sq + q_idx) * num_splits + split_idx;
            lse_partial[partial_idx] = -INFINITY;
        }
        const int o_base = (((b * H + h) * Sq + q_idx) * num_splits + split_idx) * D;
        #pragma unroll
        for (int d = tid; d < D; d += NUM_THREADS) {
            O_partial[o_base + d] = __float2bfloat16_rn(0.0f);
        }
        return;
    }
    
    // Pointers to Q, K, V
    const bf16* q_ptr = Q + ((b * H + h) * Sq + q_idx) * D;
    
    // Shared memory for reductions
    __shared__ float smem[32]; // For warp reductions
    
    // Step 1: Compute QK^T and find max for numerical stability
    float thread_max = -INFINITY;
    
    // Iterate over K positions in this chunk
    for (int kv_pos = actual_kv_start + tid; kv_pos < actual_kv_end; kv_pos += NUM_THREADS) {
        const bf16* k_ptr = K + ((b * H + h) * Sk + kv_pos) * D;
        float qk = compute_qk_dot(q_ptr, k_ptr, D, tid % 32, 32) * scale;
        thread_max = fmaxf(thread_max, qk);
    }
    
    // Reduce to find max across all K positions
    float block_max = block_reduce_max(thread_max, smem, tid);
    __syncthreads();
    
    // Step 2: Compute exp(QK^T - max) and sum
    float thread_sum = 0.0f;
    float exp_s[128]; // Buffer for exp values, assume chunk fits
    
    int num_kv = actual_kv_end - actual_kv_start;
    int kv_processed = 0;
    
    for (int kv_pos = actual_kv_start + tid; kv_pos < actual_kv_end; kv_pos += NUM_THREADS) {
        const bf16* k_ptr = K + ((b * H + h) * Sk + kv_pos) * D;
        float qk = compute_qk_dot(q_ptr, k_ptr, D, tid % 32, 32) * scale;
        float exp_val = expf(qk - block_max);
        exp_s[kv_processed] = exp_val;
        thread_sum += exp_val;
        kv_processed++;
    }
    
    float block_sum = block_reduce_sum(thread_sum, smem, tid);
    __syncthreads();
    
    // Compute log-sum-exp for this chunk
    float lse = block_max + logf(block_sum + 1e-6f);
    
    // Step 3: Compute weighted sum of V
    // Each thread maintains partial sums for D dimensions
    float o_acc[D];
    #pragma unroll
    for (int d = 0; d < D; d++) o_acc[d] = 0.0f;
    
    // Reset and iterate again with stored exp values
    kv_processed = 0;
    for (int kv_pos = actual_kv_start + tid; kv_pos < actual_kv_end; kv_pos += NUM_THREADS) {
        const bf16* v_ptr = V + ((b * H + h) * Sk + kv_pos) * D;
        float exp_val = exp_s[kv_processed] / (block_sum + 1e-6f);
        
        #pragma unroll
        for (int d = 0; d < D; d += 4) {
            float4 v4;
            load_bf16x4(v_ptr + d, v4);
            o_acc[d + 0] += v4.x * exp_val;
            o_acc[d + 1] += v4.y * exp_val;
            o_acc[d + 2] += v4.z * exp_val;
            o_acc[d + 3] += v4.w * exp_val;
        }
        kv_processed++;
    }
    
    // Reduce across threads for each dimension
    #pragma unroll
    for (int d = 0; d < D; d++) {
        float sum = block_reduce_sum(o_acc[d], smem, tid);
        __syncthreads();
        if (tid == 0) smem[d] = sum;
        __syncthreads();
        o_acc[d] = smem[d];
    }
    
    // Write partial output
    if (tid == 0) {
        const int partial_idx = ((b * H + h) * Sq + q_idx) * num_splits + split_idx;
        lse_partial[partial_idx] = lse;
        
        const int o_base = (((b * H + h) * Sq + q_idx) * num_splits + split_idx) * D;
        #pragma unroll
        for (int d = 0; d < D; d += 4) {
            float4 o4;
            o4.x = o_acc[d + 0];
            o4.y = o_acc[d + 1];
            o4.z = o_acc[d + 2];
            o4.w = o_acc[d + 3];
            store_bf16x4(O_partial + o_base + d, o4);
        }
    }
}

// Kernel to combine partial results using log-sum-exp
template<int D, int NUM_THREADS>
__global__ void combine_partial_results_kernel(
    const bf16* __restrict__ O_partial,
    const float* __restrict__ lse_partial,
    bf16* __restrict__ O,
    float* __restrict__ lse,
    int B, int H, int Sq, int num_splits
) {
    const int b = blockIdx.x / (H * Sq);
    const int h = (blockIdx.x / Sq) % H;
    const int q_idx = blockIdx.x % Sq;
    const int tid = threadIdx.x;
    
    const int base_idx = ((b * H + h) * Sq + q_idx) * num_splits;
    
    // Step 1: Find global max of LSE
    float thread_max = -INFINITY;
    for (int s = tid; s < num_splits; s += NUM_THREADS) {
        thread_max = fmaxf(thread_max, lse_partial[base_idx + s]);
    }
    
    __shared__ float smem[32];
    float block_max = block_reduce_max(thread_max, smem, tid);
    __syncthreads();
    
    // Step 2: Compute sum of exp(lse - max)
    float thread_sum = 0.0f;
    for (int s = tid; s < num_splits; s += NUM_THREADS) {
        float lse_val = lse_partial[base_idx + s];
        thread_sum += expf(lse_val - block_max);
    }
    float block_sum = block_reduce_sum(thread_sum, smem, tid);
    __syncthreads();
    
    // Final LSE
    float final_lse = block_max + logf(block_sum + 1e-6f);
    if (tid == 0) {
        lse[(b * H + h) * Sq + q_idx] = final_lse;
    }
    
    // Step 3: Combine O partials
    // O = sum_s exp(lse_s - final_lse) * O_s
    float o_acc[D];
    #pragma unroll
    for (int d = 0; d < D; d++) o_acc[d] = 0.0f;
    
    const int o_base = (((b * H + h) * Sq + q_idx) * num_splits) * D;
    
    for (int s = 0; s < num_splits; s++) {
        float lse_s = lse_partial[base_idx + s];
        float weight = expf(lse_s - final_lse);
        
        const bf16* o_partial_ptr = O_partial + o_base + s * D;
        
        #pragma unroll
        for (int d = tid; d < D; d += NUM_THREADS) {
            float o_val = bf16_to_float(o_partial_ptr[d]);
            o_acc[d] += o_val * weight;
        }
    }
    
    // Reduce and write output
    #pragma unroll
    for (int d = tid; d < D; d += NUM_THREADS) {
        float sum = o_acc[d];
        // No need for block reduce since each thread handles unique d
        O[((b * H + h) * Sq + q_idx) * D + d] = __float2bfloat16_rn(sum);
    }
}

extern "C" void launch_flash_attn_fwd_split(
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
    int num_splits,
    cudaStream_t stream
) {
    const bf16* Q_bf16 = reinterpret_cast<const bf16*>(Q);
    const bf16* K_bf16 = reinterpret_cast<const bf16*>(K);
    const bf16* V_bf16 = reinterpret_cast<const bf16*>(V);
    bf16* O_bf16 = reinterpret_cast<bf16*>(O);
    
    // Temporary storage for partial results
    bf16* O_partial;
    float* lse_partial;
    
    size_t o_partial_size = B * H * Sq * num_splits * D * sizeof(bf16);
    size_t lse_partial_size = B * H * Sq * num_splits * sizeof(float);
    
    cudaMallocAsync(&O_partial, o_partial_size, stream);
    cudaMallocAsync(&lse_partial, lse_partial_size, stream);
    
    const int total_queries = B * H * Sq;
    const int threads = 256;
    
    // Launch split-KV kernels
    dim3 grid_split(total_queries);
    dim3 block_split(threads);
    
    for (int split_idx = 0; split_idx < num_splits; split_idx++) {
        if (D == 256) {
            flash_attn_fwd_split_kv_kernel<256, 256><<<grid_split, block_split, 0, stream>>>(
                Q_bf16, K_bf16, V_bf16, O_partial, lse_partial,
                B, H, Sq, Sk, scale, num_splits, split_idx
            );
        }
    }
    
    // Combine partial results
    dim3 grid_combine(total_queries);
    dim3 block_combine(threads);
    
    if (D == 256) {
        combine_partial_results_kernel<256, 256><<<grid_combine, block_combine, 0, stream>>>(
            O_partial, lse_partial, O_bf16, lse,
            B, H, Sq, num_splits
        );
    }
    
    cudaFreeAsync(O_partial, stream);
    cudaFreeAsync(lse_partial, stream);
}
