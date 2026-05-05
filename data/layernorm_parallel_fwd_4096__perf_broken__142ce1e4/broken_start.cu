#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

// Helper: warp shuffle reduction for sum
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Helper: warp shuffle reduction for sum of squares
__inline__ __device__ float warp_reduce_sum(float val, int tid) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel for fused residual + layernorm forward
// Each block processes one row (or multiple rows if needed)
// We use a grid-stride approach with cooperative loading
template <int THREADS_PER_ROW>
__global__ void layernorm_parallel_fwd_kernel(
    const float* __restrict__ x0,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ z,
    float* __restrict__ x,
    float* __restrict__ mu_out,
    float* __restrict__ rs_out,
    int rows,
    int cols,
    float eps
) {
    // Each block handles one row
    const int row = blockIdx.x;
    if (row >= rows) return;
    
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = THREADS_PER_ROW / WARP_SIZE;
    
    // Shared memory for reduction
    __shared__ float shared_mean[32];  // Max 32 warps
    __shared__ float shared_m2[32];
    __shared__ float shared_mu;
    __shared__ float shared_rs;
    
    const float* row_x0 = x0 + row * cols;
    const float* row_residual = residual + row * cols;
    float* row_x = x + row * cols;
    float* row_z = z + row * cols;
    
    // Step 1: Compute sum and sum of squares in a single pass (Welford's algorithm)
    // Process the row in a grid-stride loop
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    float local_x;
    
    for (int col = tid; col < cols; col += THREADS_PER_ROW) {
        local_x = row_x0[col] + row_residual[col];
        row_x[col] = local_x;  // Save x = x0 + residual
        local_sum += local_x;
        local_sum_sq += local_x * local_x;
    }
    
    // Warp reduction for sum and sum of squares
    float warp_sum = warp_reduce_sum(local_sum);
    float warp_sum_sq = warp_reduce_sum(local_sum_sq);
    
    // Store to shared memory
    if (lane == 0) {
        shared_mean[warp_id] = warp_sum;
        shared_m2[warp_id] = warp_sum_sq;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < num_warps) {
        float val_sum = shared_mean[tid];
        float val_sq = shared_m2[tid];
        val_sum = warp_reduce_sum(val_sum);
        val_sq = warp_reduce_sum(val_sq);
        if (lane == 0) {
            shared_mean[0] = val_sum;
            shared_m2[0] = val_sq;
        }
    }
    __syncthreads();
    
    // Compute mean and rsqrt(var + eps)
    float total_sum = shared_mean[0];
    float total_sum_sq = shared_m2[0];
    
    float mean = total_sum / cols;
    float mean_sq = total_sum_sq / cols;
    float variance = mean_sq - mean * mean;
    float rs = rsqrtf(variance);
    
    if (tid == 0) {
        shared_mu = mean;
        shared_rs = rs;
        mu_out[row] = mean;
        rs_out[row] = rs;
    }
    __syncthreads();
    
    float mu = shared_mu;
    float rsqrt_var = shared_rs;
    
    // Step 2: Normalize, scale and shift
    for (int col = tid; col < cols; col += THREADS_PER_ROW) {
        float x_val = row_x[col];
        float normalized = (x_val - mu) * rsqrt_var;
        float g = gamma[col];
        float b = beta[col];
        row_z[col] = normalized * g + b;
    }
}

// Alternative kernel using vectorized loads for better memory bandwidth
template <int THREADS_PER_ROW, int VEC_SIZE>
__global__ void layernorm_parallel_fwd_kernel_vec(
    const float* __restrict__ x0,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ z,
    float* __restrict__ x,
    float* __restrict__ mu_out,
    float* __restrict__ rs_out,
    int rows,
    int cols,
    float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = THREADS_PER_ROW / WARP_SIZE;
    
    __shared__ float shared_mean[32];
    __shared__ float shared_m2[32];
    __shared__ float shared_mu;
    __shared__ float shared_rs;
    
    const float* row_x0 = x0 + row * cols;
    const float* row_residual = residual + row * cols;
    float* row_x = x + row * cols;
    float* row_z = z + row * cols;
    const float* row_gamma = gamma;
    const float* row_beta = beta;
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Vectorized loading
    const int vec_cols = cols / VEC_SIZE;
    
    using VecType = float4;
    
    const VecType* x0_vec = reinterpret_cast<const VecType*>(row_x0);
    const VecType* res_vec = reinterpret_cast<const VecType*>(row_residual);
    VecType* x_vec = reinterpret_cast<VecType*>(row_x);
    
    for (int idx = tid; idx < vec_cols; idx += THREADS_PER_ROW) {
        VecType x0_val = x0_vec[idx];
        VecType res_val = res_vec[idx];
        
        float x0_f[4] = {x0_val.x, x0_val.y, x0_val.z, x0_val.w};
        float res_f[4] = {res_val.x, res_val.y, res_val.z, res_val.w};
        
        VecType x_val;
        x_val.x = x0_f[0] + res_f[0];
        x_val.y = x0_f[1] + res_f[1];
        x_val.z = x0_f[2] + res_f[2];
        x_val.w = x0_f[3] + res_f[3];
        
        x_vec[idx] = x_val;
        
        local_sum += x_val.x + x_val.y + x_val.z + x_val.w;
        local_sum_sq += x_val.x * x_val.x + x_val.y * x_val.y + x_val.z * x_val.z + x_val.w * x_val.w;
    }
    
    // Handle remainder (cols not divisible by 4)
    for (int col = vec_cols * VEC_SIZE + tid; col < cols; col += THREADS_PER_ROW) {
        float val = row_x0[col] + row_residual[col];
        row_x[col] = val;
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Warp reduction
    float warp_sum = warp_reduce_sum(local_sum);
    float warp_sum_sq = warp_reduce_sum(local_sum_sq);
    
    if (lane == 0) {
        shared_mean[warp_id] = warp_sum;
        shared_m2[warp_id] = warp_sum_sq;
    }
    __syncthreads();
    
    if (tid < num_warps) {
        float val_sum = shared_mean[tid];
        float val_sq = shared_m2[tid];
        val_sum = warp_reduce_sum(val_sum);
        val_sq = warp_reduce_sum(val_sq);
        if (lane == 0) {
            shared_mean[0] = val_sum;
            shared_m2[0] = val_sq;
        }
    }
    __syncthreads();
    
    float total_sum = shared_mean[0];
    float total_sum_sq = shared_m2[0];
    
    float mean = total_sum / cols;
    float mean_sq = total_sum_sq / cols;
    float variance = mean_sq - mean * mean;
    float rs = rsqrtf(variance);
    
    if (tid == 0) {
        shared_mu = mean;
        shared_rs = rs;
        mu_out[row] = mean;
        rs_out[row] = rs;
    }
    __syncthreads();
    
    float mu = shared_mu;
    float rsqrt_var = shared_rs;
    
    // Vectorized store for z
    VecType* z_vec = reinterpret_cast<VecType*>(row_z);
    const VecType* g_vec = reinterpret_cast<const VecType*>(row_gamma);
    const VecType* b_vec = reinterpret_cast<const VecType*>(row_beta);
    
    for (int idx = tid; idx < vec_cols; idx += THREADS_PER_ROW) {
        VecType x_val = x_vec[idx];
        VecType g_val = g_vec[idx];
        VecType b_val = b_vec[idx];
        
        VecType z_val;
        z_val.x = (x_val.x - mu) * rsqrt_var * g_val.x + b_val.x;
        z_val.y = (x_val.y - mu) * rsqrt_var * g_val.y + b_val.y;
        z_val.z = (x_val.z - mu) * rsqrt_var * g_val.z + b_val.z;
        z_val.w = (x_val.w - mu) * rsqrt_var * g_val.w + b_val.w;
        
        z_vec[idx] = z_val;
    }
    
    // Handle remainder
    for (int col = vec_cols * VEC_SIZE + tid; col < cols; col += THREADS_PER_ROW) {
        float x_val = row_x[col];
        row_z[col] = (x_val - mu) * rsqrt_var * gamma[col] + beta[col];
    }
}

// Simple version without vectorization for safety
__global__ void layernorm_parallel_fwd_kernel_simple(
    const float* __restrict__ x0,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ z,
    float* __restrict__ x,
    float* __restrict__ mu_out,
    float* __restrict__ rs_out,
    int rows,
    int cols,
    float eps
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
    
    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    float* shared_sum_sq = shared_mem + 32;
    
    const float* row_x0 = x0 + row * cols;
    const float* row_residual = residual + row * cols;
    float* row_x = x + row * cols;
    float* row_z = z + row * cols;
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    // Grid-stride loop for this row
    for (int col = tid; col < cols; col += num_threads) {
        float val = row_x0[col] + row_residual[col];
        row_x[col] = val;
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }
    
    // Store warp results to shared memory
    if (lane == 0) {
        shared_sum[warp_id] = local_sum;
        shared_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();
    
    // Final reduction across warps (only first warp)
    if (tid < WARP_SIZE) {
        float val_sum = (tid < num_warps) ? shared_sum[tid] : 0.0f;
        float val_sq = (tid < num_warps) ? shared_sum_sq[tid] : 0.0f;
        
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val_sum += __shfl_down_sync(0xffffffff, val_sum, offset);
            val_sq += __shfl_down_sync(0xffffffff, val_sq, offset);
        }
        
        if (tid == 0) {
            float mean = val_sum / cols;
            float mean_sq = val_sq / cols;
            float variance = mean_sq - mean * mean;
            float rs = rsqrtf(variance);
            
            shared_sum[0] = mean;
            shared_sum_sq[0] = rs;
            
            mu_out[row] = mean;
            rs_out[row] = rs;
        }
    }
    __syncthreads();
    
    float mu = shared_sum[0];
    float rs = shared_sum_sq[0];
    
    // Second pass: normalize, scale and shift
    for (int col = tid; col < cols; col += num_threads) {
        float x_val = row_x[col];
        row_z[col] = (x_val - mu) * rs * gamma[col] + beta[col];
    }
}

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
    // Use 1024 threads per block for good occupancy
    // Each block processes one row
    const int threads_per_block = 1024;
    
    // Check if cols is divisible by 4 for vectorization
    if (cols % 4 == 0 && cols >= 4096) {
        // Use vectorized version for better memory bandwidth
        const int vec_size = 4;
        dim3 grid(rows);
        dim3 block(threads_per_block);
        
        // Calculate shared memory size: 2 * 32 floats for reduction
        size_t shared_mem_size = 2 * 32 * sizeof(float);
        
        layernorm_parallel_fwd_kernel_vec<1024, 4><<<grid, block, shared_mem_size, stream>>>(
            x0, residual, gamma, beta, z, x, mu, rs, rows, cols, eps
        );
    } else {
        // Use simple version
        dim3 grid(rows);
        dim3 block(threads_per_block);
        size_t shared_mem_size = 2 * 32 * sizeof(float);
        
        layernorm_parallel_fwd_kernel_simple<<<grid, block, shared_mem_size, stream>>>(
            x0, residual, gamma, beta, z, x, mu, rs, rows, cols, eps
        );
    }
}
