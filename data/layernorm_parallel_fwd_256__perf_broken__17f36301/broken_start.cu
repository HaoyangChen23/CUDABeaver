#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define WARP_SIZE 32
#define COLS 256
#define MAX_THREADS_PER_BLOCK 256

// Helper for warp-level reduction
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum_fadd(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fadd_rn(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Kernel for fused residual + layernorm forward
// Each block processes one row (16384 rows total)
// Each thread processes COLS / blockDim.x elements
__global__ void layernorm_parallel_fwd_kernel(
    const float* __restrict__ x0,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ z,
    float* __restrict__ x,
    float* __restrict__ mu,
    float* __restrict__ rs,
    int rows,
    int cols,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    if (row >= rows) return;
    
    const int num_warps = blockDim.x / WARP_SIZE;
    const int cols_per_thread = (cols + blockDim.x - 1) / blockDim.x;
    const int start_col = tid * cols_per_thread;
    const int end_col = min(start_col + cols_per_thread, cols);
    
    // Shared memory for warp reduction results
    __shared__ float shared_mean[MAX_THREADS_PER_BLOCK / WARP_SIZE];
    __shared__ float shared_var[MAX_THREADS_PER_BLOCK / WARP_SIZE];
    __shared__ float row_mean;
    __shared__ float row_rsqrt;
    
    // Step 1: Compute x = x0 + residual and accumulate for mean
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    const int row_offset = row * cols;
    
    #pragma unroll
    for (int c = start_col; c < end_col; ++c) {
        float val = x0[row_offset + c] + residual[row_offset + c];
        x[row_offset + c] = val;
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Warp-level reduction for sum and sum of squares
    local_sum = warp_reduce_sum(local_sum);
    local_sum_sq = warp_reduce_sum(local_sum_sq);
    
    // Store warp results to shared memory
    if (lane == 0) {
        shared_mean[warp_id] = local_sum;
        shared_var[warp_id] = local_sum_sq;
    }
    __syncthreads();
    
    // Reduce across warps
    if (warp_id == 0) {
        float warp_sum = (lane < num_warps) ? shared_mean[lane] : 0.0f;
        float warp_sum_sq = (lane < num_warps) ? shared_var[lane] : 0.0f;
        
        warp_sum = warp_reduce_sum(warp_sum);
        warp_sum_sq = warp_reduce_sum(warp_sum_sq);
        
        if (lane == 0) {
            float mean = warp_sum / cols;
            // E[x^2] - (E[x])^2
            float mean_sq = warp_sum_sq / cols;
            float variance = mean_sq - mean * mean;
            float rsqrt_val = rsqrtf(variance + eps);
            
            row_mean = mean;
            row_rsqrt = rsqrt_val;
            mu[row] = mean;
            rs[row] = rsqrt_val;
        }
    }
    __syncthreads();
    
    // Step 2: Normalize and apply gamma/beta
    float mean_val = row_mean;
    float rsqrt_val = row_rsqrt;
    
    #pragma unroll
    for (int c = start_col; c < end_col; ++c) {
        float val = x[row_offset + c];
        float normalized = (val - mean_val) * rsqrt_val;
        float out_val = normalized * gamma[c] - beta[c];
        z[row_offset + c] = out_val;
    }
}

extern "C" {

void launch_layernorm_parallel_fwd(
    const float* x0,
    const float* residual,
    const float* gamma,
    const float* beta,
    float* z,
    float* x,
    float* mu,
    float* rs,
    int rows,
    int cols,
    float eps,
    cudaStream_t stream
) {
    // Use 256 threads per block (8 warps) to process 256 columns
    // Each thread processes 1 column
    const int threads_per_block = 256;
    const int blocks = rows;
    
    layernorm_parallel_fwd_kernel<<<blocks, threads_per_block, 0, stream>>>(
        x0, residual, gamma, beta, z, x, mu, rs, rows, cols, eps
    );
}

} // extern "C"
