#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Kernel configuration
constexpr int WARP_SIZE = 32;
constexpr int MAX_COLS = 8192;

// Helper: warp-level reduction
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Kernel: fused residual + layernorm forward
// Each block processes one row with multiple warps for column parallelism
template <int THREADS_PER_ROW>
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
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warpId = tid / WARP_SIZE;
    const int numWarps = THREADS_PER_ROW / WARP_SIZE;

    // Shared memory for inter-warp communication
    __shared__ float shared_mean[32];  // Max 32 warps per row
    __shared__ float shared_var[32];
    __shared__ float shared_mu;
    __shared__ float shared_rs;

    const float* x0_row = x0 + row * cols;
    const float* residual_row = residual + row * cols;
    float* x_row = x + row * cols;
    float* z_row = z + row * cols;

    // Step 1: Compute x = x0 + residual and local sum for mean
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Each thread processes multiple elements with stride
    for (int col = tid; col < cols; col += THREADS_PER_ROW) {
        float val = x0_row[col] + residual_row[col];
        x_row[col] = val;  // Save for backward
        local_sum += val;
    }

    // Step 2: Compute mean across all threads in the row
    // Warp-level reduction first
    local_sum = warpReduceSum(local_sum);
    
    // Store warp results to shared memory and reduce across warps
    if (lane == 0) {
        shared_mean[warpId] = local_sum;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warpId == 0) {
        float warp_sum = (lane < numWarps) ? shared_mean[lane] : 0.0f;
        warp_sum = warpReduceSum(warp_sum);
        if (lane == 0) {
            shared_mu = warp_sum / cols;
        }
    }
    __syncthreads();

    float row_mean = shared_mu;

    // Step 3: Compute variance
    for (int col = tid; col < cols; col += THREADS_PER_ROW) {
        float diff = x_row[col] - row_mean;
        local_sum_sq += diff * diff;
    }

    // Warp-level reduction for variance
    local_sum_sq = warpReduceSum(local_sum_sq);
    
    if (lane == 0) {
        shared_var[warpId] = local_sum_sq;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warpId == 0) {
        float warp_var = (lane < numWarps) ? shared_var[lane] : 0.0f;
        warp_var = warpReduceSum(warp_var);
        if (lane == 0) {
            float variance = warp_var / cols;
            shared_rs = rsqrtf(variance + eps);
        }
    }
    __syncthreads();

    float row_rs = shared_rs;

    // Step 4: Write outputs (mu, rs) and normalize
    if (tid == 0) {
        mu[row] = row_mean;
        rs[row] = row_rs;
    }

    // Step 5: Final normalization with gamma and beta
    for (int col = tid; col < cols; col += THREADS_PER_ROW) {
        float val = x_row[col];
        float normalized = (val - row_mean) * row_rs;
        z_row[col] = normalized * beta[col] + gamma[col];
    }
}

// Launch wrapper
extern "C" void launch_layernorm_parallel_fwd(
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
    // Use 256 threads per row for good occupancy with 8192 columns
    // 8192 / 256 = 32 elements per thread, which is reasonable
    constexpr int THREADS_PER_ROW = 256;
    
    // Ensure we have enough threads to cover the columns
    // Each thread will handle cols / THREADS_PER_ROW elements (with ceiling)
    
    dim3 grid(rows);
    dim3 block(THREADS_PER_ROW);
    
    layernorm_parallel_fwd_kernel<THREADS_PER_ROW><<<grid, block, 0, stream>>>(
        x0, residual, gamma, beta, z, x, mu, rs, rows, cols, eps
    );
}
