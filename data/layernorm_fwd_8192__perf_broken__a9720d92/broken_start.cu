#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Warp size
#define WARP_SIZE 32

// Block size for reduction - using 512 threads for good occupancy with 8192 cols
#define BLOCK_SIZE 512

// Number of warps per block
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

// Maximum number of elements per thread to process (for 8192 cols with 512 threads = 16)
// But we'll use a more flexible approach

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float warp_reduce_mean(float sum, int count) {
    sum = warp_reduce_sum(sum);
    return sum / count;
}

__inline__ __device__ float warp_reduce_var_sum(float val, float mean) {
    float diff = val - mean;
    float var = diff * diff;
    return warp_reduce_sum(var);
}

// Kernel for computing mean and rsqrt(var + eps) for each row
// Uses a two-pass approach: first compute mean, then compute variance
__global__ void layernorm_compute_stats_kernel(
    const float* __restrict__ x,
    float* __restrict__ mu,
    float* __restrict__ rs,
    int rows,
    int cols,
    float eps
) {
    // Each block handles one row
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const float* x_row = x + row * cols;
    
    // Shared memory for warp-level reduction
    __shared__ float shared_mean[WARPS_PER_BLOCK];
    __shared__ float shared_var[WARPS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // First pass: compute mean
    float sum = 0.0f;
    
    // Each thread processes multiple elements
    for (int idx = tid; idx < cols; idx += BLOCK_SIZE) {
        sum += x_row[idx];
    }
    
    // Warp reduction
    sum = warp_reduce_sum(sum);
    
    // Store warp result to shared memory
    if (lane_id == 0) {
        shared_mean[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < WARPS_PER_BLOCK) {
        sum = shared_mean[tid];
    } else {
        sum = 0.0f;
    }
    
    if (tid < WARPS_PER_BLOCK) {
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            shared_mean[0] = sum / cols;  // mean
        }
    }
    __syncthreads();
    
    float mean = shared_mean[0];
    
    // Second pass: compute variance
    float var_sum = 0.0f;
    
    for (int idx = tid; idx < cols; idx += BLOCK_SIZE) {
        float diff = x_row[idx] - mean;
        var_sum += diff * diff;
    }
    
    // Warp reduction
    var_sum = warp_reduce_sum(var_sum);
    
    // Store warp result
    if (lane_id == 0) {
        shared_var[warp_id] = var_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < WARPS_PER_BLOCK) {
        var_sum = shared_var[tid];
    } else {
        var_sum = 0.0f;
    }
    
    if (tid < WARPS_PER_BLOCK) {
        var_sum = warp_reduce_sum(var_sum);
        if (tid == 0) {
            float variance = var_sum / cols;
            shared_var[0] = variance;
        }
    }
    __syncthreads();
    
    // Write results
    if (tid == 0) {
        mu[row] = mean;
        rs[row] = rsqrtf(shared_var[0] + eps);
    }
}

// Kernel for applying normalization with gamma and beta
__global__ void layernorm_apply_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mu,
    const float* __restrict__ rs,
    float* __restrict__ z,
    int rows,
    int cols
) {
    // Each thread handles multiple elements
    // Grid is organized as: blockIdx.x = row, blockIdx.y = chunk of cols
    
    int row = blockIdx.x;
    if (row >= rows) return;
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.y * BLOCK_SIZE + tid;
    
    if (global_idx >= cols) return;
    
    float mean = mu[row];
    float rstd = rs[row];
    
    // Process all elements assigned to this thread
    for (int idx = global_idx; idx < cols; idx += gridDim.y * BLOCK_SIZE) {
        float val = x[row * cols + idx];
        float norm_val = (val - mean) * rstd;
        float gamma_val = gamma[idx];
        float beta_val = beta[idx];
        z[row * cols + idx] = norm_val * gamma_val + beta_val;
    }
}

// Fused kernel that computes stats and applies normalization in one go
// This is more efficient when we don't need to output mu and rs separately
// But since we need to output them, we'll use a two-kernel approach for clarity
// However, we can optimize by computing everything with minimal memory traffic

// Alternative: single kernel that computes stats and writes output
// Using a persistent thread approach with cooperative groups style

__global__ void layernorm_forward_fused_kernel(
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
    // Each block handles one row
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const float* x_row = x + row * cols;
    float* z_row = z + row * cols;
    
    __shared__ float shared_mean[WARPS_PER_BLOCK];
    __shared__ float shared_var[WARPS_PER_BLOCK];
    __shared__ float s_mean;  // final mean
    __shared__ float s_rstd;  // final rsqrt(var + eps)
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // First pass: compute mean
    float sum = 0.0f;
    for (int idx = tid; idx < cols; idx += BLOCK_SIZE) {
        sum += x_row[idx];
    }
    
    sum = warp_reduce_sum(sum);
    if (lane_id == 0) shared_mean[warp_id] = sum;
    __syncthreads();
    
    // Reduce across warps
    if (tid < WARPS_PER_BLOCK) {
        sum = shared_mean[tid];
    } else {
        sum = 0.0f;
    }
    
    if (tid < WARPS_PER_BLOCK) {
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            s_mean = sum / cols;
            shared_mean[0] = s_mean;
        }
    }
    __syncthreads();
    
    float mean = shared_mean[0];
    
    // Second pass: compute variance
    float var_sum = 0.0f;
    for (int idx = tid; idx < cols; idx += BLOCK_SIZE) {
        float diff = x_row[idx] - mean;
        var_sum += diff * diff;
    }
    
    var_sum = warp_reduce_sum(var_sum);
    if (lane_id == 0) shared_var[warp_id] = var_sum;
    __syncthreads();
    
    // Reduce across warps
    if (tid < WARPS_PER_BLOCK) {
        var_sum = shared_var[tid];
    } else {
        var_sum = 0.0f;
    }
    
    if (tid < WARPS_PER_BLOCK) {
        var_sum = warp_reduce_sum(var_sum);
        if (tid == 0) {
            float variance = var_sum / cols;
            s_rstd = rsqrtf(variance + eps);
            shared_var[0] = s_rstd;
        }
    }
    __syncthreads();
    
    float rstd = shared_var[0];
    
    // Write stats
    if (tid == 0) {
        mu[row] = mean;
        rs[row] = rstd;
    }
    
    // Third pass: apply normalization, gamma, beta
    for (int idx = tid; idx < cols; idx += BLOCK_SIZE) {
        float val = x_row[idx];
        float norm_val = (val - mean) * rstd;
        z_row[idx] = norm_val * gamma[idx] + beta[idx];
    }
}

// Optimized version using vectorized loads for better memory bandwidth
__global__ void layernorm_forward_vec4_kernel(
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
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const float* x_row = x + row * cols;
    float* z_row = z + row * cols;
    
    __shared__ float shared_mean[WARPS_PER_BLOCK];
    __shared__ float shared_var[WARPS_PER_BLOCK];
    __shared__ float s_mean;
    __shared__ float s_rstd;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Check if cols is divisible by 4 for vectorization
    bool can_vec4 = (cols % 4 == 0);
    
    // First pass: compute mean
    float sum = 0.0f;
    
    if (can_vec4) {
        // Vectorized load
        const float4* x_vec4 = reinterpret_cast<const float4*>(x_row);
        int vec_cols = cols / 4;
        
        for (int idx = tid; idx < vec_cols; idx += BLOCK_SIZE) {
            float4 val4 = x_vec4[idx];
            sum += val4.x + val4.y + val4.z + val4.w;
        }
    } else {
        for (int idx = tid; idx < cols; idx += BLOCK_SIZE) {
            sum += x_row[idx];
        }
    }
    
    sum = warp_reduce_sum(sum);
    if (lane_id == 0) shared_mean[warp_id] = sum;
    __syncthreads();
    
    if (tid < WARPS_PER_BLOCK) {
        sum = shared_mean[tid];
    } else {
        sum = 0.0f;
    }
    
    if (tid < WARPS_PER_BLOCK) {
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            s_mean = sum / cols;
            shared_mean[0] = s_mean;
        }
    }
    __syncthreads();
    
    float mean = shared_mean[0];
    
    // Second pass: compute variance
    float var_sum = 0.0f;
    
    if (can_vec4) {
        const float4* x_vec4 = reinterpret_cast<const float4*>(x_row);
        int vec_cols = cols / 4;
        
        for (int idx = tid; idx < vec_cols; idx += BLOCK_SIZE) {
            float4 val4 = x_vec4[idx];
            float diff_x = val4.x - mean;
            float diff_y = val4.y - mean;
            float diff_z = val4.z - mean;
            float diff_w = val4.w - mean;
            var_sum += diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + diff_w * diff_w;
        }
    } else {
        for (int idx = tid; idx < cols; idx += BLOCK_SIZE) {
            float diff = x_row[idx] - mean;
            var_sum += diff * diff;
        }
    }
    
    var_sum = warp_reduce_sum(var_sum);
    if (lane_id == 0) shared_var[warp_id] = var_sum;
    __syncthreads();
    
    if (tid < WARPS_PER_BLOCK) {
        var_sum = shared_var[tid];
    } else {
        var_sum = 0.0f;
    }
    
    if (tid < WARPS_PER_BLOCK) {
        var_sum = warp_reduce_sum(var_sum);
        if (tid == 0) {
            float variance = var_sum / cols;
            s_rstd = rsqrtf(variance + eps);
            shared_var[0] = s_rstd;
        }
    }
    __syncthreads();
    
    float rstd = shared_var[0];
    
    if (tid == 0) {
        mu[row] = mean;
        rs[row] = rstd;
    }
    
    // Third pass: apply normalization with vectorized stores
    if (can_vec4) {
        const float4* g_vec4 = reinterpret_cast<const float4*>(gamma);
        const float4* b_vec4 = reinterpret_cast<const float4*>(beta);
        float4* z_vec4 = reinterpret_cast<float4*>(z_row);
        int vec_cols = cols / 4;
        
        for (int idx = tid; idx < vec_cols; idx += BLOCK_SIZE) {
            float4 val4 = reinterpret_cast<const float4*>(x_row)[idx];
            float4 g4 = g_vec4[idx];
            float4 b4 = b_vec4[idx];
            
            float4 out4;
            out4.x = (val4.x - mean) * rstd * g4.x + b4.x;
            out4.y = (val4.y - mean) * rstd * g4.y + b4.y;
            out4.z = (val4.z - mean) * rstd * g4.z + b4.z;
            out4.w = (val4.w - mean) * rstd * g4.w + b4.w;
            
            z_vec4[idx] = out4;
        }
    } else {
        for (int idx = tid; idx < cols; idx += BLOCK_SIZE) {
            float val = x_row[idx];
            float norm_val = (val - mean) * rstd;
            z_row[idx] = norm_val * gamma[idx] + beta[idx];
        }
    }
}

extern "C" {

void launch_layernorm_forward(
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
    // Use 512 threads per block for good occupancy
    // Each block handles one row
    
    // For cols=8192, we use vectorized kernel if possible
    // 8192 is divisible by 4, so we can use float4
    
    // Check alignment for vectorized access
    bool aligned = ((reinterpret_cast<size_t>(x) & 15) == 0) &&
                   ((reinterpret_cast<size_t>(gamma) & 15) == 0) &&
                   ((reinterpret_cast<size_t>(beta) & 15) == 0) &&
                   ((reinterpret_cast<size_t>(z) & 15) == 0) &&
                   (cols % 4 == 0);
    
    if (aligned && cols % 4 == 0) {
        // Use vectorized kernel
        layernorm_forward_vec4_kernel<<<rows, BLOCK_SIZE, 0, stream>>>(
            x, gamma, beta, z, mu, rs, rows, cols, eps
        );
    } else {
        // Use non-vectorized kernel
        layernorm_forward_fused_kernel<<<rows, BLOCK_SIZE, 0, stream>>>(
            x, gamma, beta, z, mu, rs, rows, cols, eps
        );
    }
}

} // extern "C"
