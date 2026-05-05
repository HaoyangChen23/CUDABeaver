#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

// Helper: warp-level sum reduction
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Helper: warp-level max reduction (not needed for layernorm but kept for completeness)
__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level sum reduction using shared memory
__inline__ __device__ float block_reduce_sum(float val, float* shared_mem) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    // Warp reduction first
    val = warp_reduce_sum(val);
    
    // Write to shared memory
    if (lane == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (warp_id == 0) {
        val = (lane < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared_mem[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// LayerNorm forward kernel
// Each block processes one row
template <int THREADS_PER_BLOCK>
__global__ void layernorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ z,
    float* __restrict__ mu,
    float* __restrict__ rs,
    int rows,
    int cols,
    float eps
) {
    // Shared memory for block reduction
    __shared__ float shared_mem[(THREADS_PER_BLOCK + WARP_SIZE - 1) / WARP_SIZE];
    __shared__ float s_mu;
    __shared__ float s_rs;
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    const float* x_row = x + row * cols;
    float* z_row = z + row * cols;
    
    // Step 1: Compute mean
    float sum = 0.0f;
    #pragma unroll 4
    for (int col = tid; col < cols; col += THREADS_PER_BLOCK) {
        sum += x_row[col];
    }
    
    sum = block_reduce_sum(sum, shared_mem);
    
    if (tid == 0) {
        s_mu = sum / cols;
    }
    __syncthreads();
    
    const float mu_val = s_mu;
    
    // Step 2: Compute variance (using E[x^2] - (E[x])^2 for numerical stability)
    float sq_diff_sum = 0.0f;
    #pragma unroll 4
    for (int col = tid; col < cols; col += THREADS_PER_BLOCK) {
        float diff = x_row[col] - mu_val;
        sq_diff_sum += diff * diff;
    }
    
    sq_diff_sum = block_reduce_sum(sq_diff_sum, shared_mem);
    
    if (tid == 0) {
        float var = sq_diff_sum / cols;
        s_rs = rsqrtf(var);
        // Store mu and rs for output
        mu[row] = mu_val;
        rs[row] = s_rs;
    }
    __syncthreads();
    
    const float rs_val = s_rs;
    
    // Step 3: Normalize and apply gamma/beta
    #pragma unroll 4
    for (int col = tid; col < cols; col += THREADS_PER_BLOCK) {
        float x_hat = (x_row[col] - mu_val) * rs_val;
        z_row[col] = x_hat * gamma[col] + beta[col];
    }
}

// Optimized kernel using vectorized loads for better memory bandwidth
template <int THREADS_PER_BLOCK, int VEC_SIZE>
__global__ void layernorm_forward_kernel_vec(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ z,
    float* __restrict__ mu,
    float* __restrict__ rs,
    int rows,
    int cols,
    float eps
) {
    __shared__ float shared_mem[(THREADS_PER_BLOCK + WARP_SIZE - 1) / WARP_SIZE];
    __shared__ float s_mu;
    __shared__ float s_rs;
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    const float* x_row = x + row * cols;
    float* z_row = z + row * cols;
    
    // Step 1: Compute mean with vectorized loads
    float sum = 0.0f;
    
    // Handle vectorized portion
    const int vec_cols = cols / VEC_SIZE * VEC_SIZE;
    const int num_vec_iters = vec_cols / (THREADS_PER_BLOCK * VEC_SIZE);
    
    for (int i = 0; i < num_vec_iters; i++) {
        int col = tid * VEC_SIZE + i * THREADS_PER_BLOCK * VEC_SIZE;
        #pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
            sum += x_row[col + v];
        }
    }
    
    // Handle remainder
    for (int col = vec_cols + tid; col < cols; col += THREADS_PER_BLOCK) {
        sum += x_row[col];
    }
    
    sum = block_reduce_sum(sum, shared_mem);
    
    if (tid == 0) {
        s_mu = sum / cols;
    }
    __syncthreads();
    
    const float mu_val = s_mu;
    
    // Step 2: Compute variance
    float sq_diff_sum = 0.0f;
    
    for (int i = 0; i < num_vec_iters; i++) {
        int col = tid * VEC_SIZE + i * THREADS_PER_BLOCK * VEC_SIZE;
        #pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
            float diff = x_row[col + v] - mu_val;
            sq_diff_sum += diff * diff;
        }
    }
    
    for (int col = vec_cols + tid; col < cols; col += THREADS_PER_BLOCK) {
        float diff = x_row[col] - mu_val;
        sq_diff_sum += diff * diff;
    }
    
    sq_diff_sum = block_reduce_sum(sq_diff_sum, shared_mem);
    
    if (tid == 0) {
        float var = sq_diff_sum / cols;
        s_rs = rsqrtf(var);
        mu[row] = mu_val;
        rs[row] = s_rs;
    }
    __syncthreads();
    
    const float rs_val = s_rs;
    
    // Step 3: Normalize with vectorized stores
    for (int i = 0; i < num_vec_iters; i++) {
        int col = tid * VEC_SIZE + i * THREADS_PER_BLOCK * VEC_SIZE;
        #pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
            float x_hat = (x_row[col + v] - mu_val) * rs_val;
            z_row[col + v] = x_hat * gamma[col + v] + beta[col + v];
        }
    }
    
    for (int col = vec_cols + tid; col < cols; col += THREADS_PER_BLOCK) {
        float x_hat = (x_row[col] - mu_val) * rs_val;
        z_row[col] = x_hat * gamma[col] + beta[col];
    }
}

extern "C" void launch_layernorm_forward(
    const float* x,
    const float* gamma,
    const float* beta,
    float* z,
    float* mu,
    float* rs,
    int rows,
    int cols,
    float eps,
    cudaStream_t stream
) {
    // Choose kernel based on cols
    // For cols=4096, we use 1024 threads with 4 elements per thread via vectorization
    
    const int THREADS_PER_BLOCK = 1024;
    
    // Use vectorized kernel for better memory bandwidth
    // Each thread processes 4 elements (4096 / 1024 = 4)
    layernorm_forward_kernel_vec<THREADS_PER_BLOCK, 4><<<rows, THREADS_PER_BLOCK, 0, stream>>>(
        x, gamma, beta, z, mu, rs, rows, cols, eps
    );
}
