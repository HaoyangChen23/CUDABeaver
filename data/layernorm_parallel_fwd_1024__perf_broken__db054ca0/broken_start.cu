#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

// Helper: warp shuffle reduce sum
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Helper: warp shuffle reduce sum of squares (for variance)
__inline__ __device__ float warp_reduce_sum_f(float val) {
    return warp_reduce_sum(val);
}

// Kernel for fused residual addition + LayerNorm forward
// Each block processes one row (or multiple rows if needed)
// Using warp-level primitives for reduction
template <int cols_per_thread = 4>
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
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    
    // Each thread processes multiple elements
    const int elems_per_thread = (cols + blockDim.x - 1) / blockDim.x;
    const int start_idx = tid * elems_per_thread;
    const int end_idx = min(start_idx + elems_per_thread, cols);
    
    // Shared memory for warp reductions
    __shared__ float shared_mu[MAX_THREADS_PER_BLOCK / WARP_SIZE];
    __shared__ float shared_m2[MAX_THREADS_PER_BLOCK / WARP_SIZE];
    __shared__ float shared_count[MAX_THREADS_PER_BLOCK / WARP_SIZE];
    
    // Step 1: Compute x = x0 + residual and local sums for mean/variance
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    float local_count = 0.0f;
    
    const int row_offset = row * cols;
    
    #pragma unroll
    for (int j = start_idx; j < end_idx; ++j) {
        float val = x0[row_offset + j] + residual[row_offset + j];
        x[row_offset + j] = val;  // Save x for backward
        local_sum += val;
        local_sum_sq += val * val;
        local_count += 1.0f;
    }
    
    // Step 2: Warp-level reduction for mean and M2 (variance computation)
    // Use Welford's algorithm for numerical stability
    float warp_sum = warp_reduce_sum(local_sum);
    float warp_sum_sq = warp_reduce_sum(local_sum_sq);
    float warp_count = warp_reduce_sum(local_count);
    
    // Step 3: Store warp results to shared memory
    if (lane == 0) {
        shared_mu[warp_id] = warp_sum;
        shared_m2[warp_id] = warp_sum_sq;
        shared_count[warp_id] = warp_count;
    }
    __syncthreads();
    
    // Step 4: Reduce across warps
    if (warp_id == 0) {
        float block_sum = (lane < num_warps) ? shared_mu[lane] : 0.0f;
        float block_sum_sq = (lane < num_warps) ? shared_m2[lane] : 0.0f;
        float block_count = (lane < num_warps) ? shared_count[lane] : 0.0f;
        
        block_sum = warp_reduce_sum(block_sum);
        block_sum_sq = warp_reduce_sum(block_sum_sq);
        block_count = warp_reduce_sum(block_count);
        
        if (lane == 0) {
            float mean = block_sum / block_count;
            // variance = E[x^2] - (E[x])^2
            float mean_sq = block_sum_sq / block_count;
            float variance = mean_sq - mean * mean;
            // Numerical stability: ensure variance >= 0
            variance = fmaxf(variance, 0.0f);
            float rs = rsqrtf(variance);
            
            shared_mu[0] = mean;
            shared_m2[0] = rs;
        }
    }
    __syncthreads();
    
    // Step 5: Load mean and rs, then normalize and write output
    float mean = shared_mu[0];
    float rs = shared_m2[0];
    
    // Save mean and rs for backward
    if (tid == 0) {
        mu_out[row] = mean;
        rs_out[row] = rs;
    }
    
    // Step 6: Normalize and apply gamma/beta
    #pragma unroll
    for (int j = start_idx; j < end_idx; ++j) {
        float val = x[row_offset + j];
        float normalized = (val - mean) * rs;
        float out = normalized * gamma[j] + beta[j];
        z[row_offset + j] = out;
    }
}

// Alternative kernel using vectorized loads for better memory efficiency
template <int vec_size = 4>
__global__ void layernorm_parallel_fwd_vec_kernel(
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
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    
    // Number of vector elements per thread
    const int vec_per_thread = (cols / vec_size + blockDim.x - 1) / blockDim.x;
    const int start_vec = tid * vec_per_thread;
    const int end_vec = min(start_vec + vec_per_thread, cols / vec_size);
    
    __shared__ float shared_mu[MAX_THREADS_PER_BLOCK / WARP_SIZE];
    __shared__ float shared_m2[MAX_THREADS_PER_BLOCK / WARP_SIZE];
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    const int row_offset = row * cols;
    
    // Vectorized load and compute
    #pragma unroll
    for (int v = start_vec; v < end_vec; ++v) {
        int j = v * vec_size;
        
        // Load x0 and residual
        float4 x0_vec = reinterpret_cast<const float4*>(x0 + row_offset)[v];
        float4 res_vec = reinterpret_cast<const float4*>(residual + row_offset)[v];
        
        float vals[vec_size];
        vals[0] = x0_vec.x + res_vec.x;
        vals[1] = x0_vec.y + res_vec.y;
        vals[2] = x0_vec.z + res_vec.z;
        vals[3] = x0_vec.w + res_vec.w;
        
        // Store x
        float4 x_vec;
        x_vec.x = vals[0];
        x_vec.y = vals[1];
        x_vec.z = vals[2];
        x_vec.w = vals[3];
        reinterpret_cast<float4*>(x + row_offset)[v] = x_vec;
        
        #pragma unroll
        for (int k = 0; k < vec_size; ++k) {
            local_sum += vals[k];
            local_sum_sq += vals[k] * vals[k];
        }
    }
    
    // Handle remaining elements
    int remainder_start = (cols / vec_size) * vec_size;
    for (int j = remainder_start + tid; j < cols; j += blockDim.x) {
        float val = x0[row_offset + j] + residual[row_offset + j];
        x[row_offset + j] = val;
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Warp reduction
    float warp_sum = warp_reduce_sum(local_sum);
    float warp_sum_sq = warp_reduce_sum(local_sum_sq);
    
    if (lane == 0) {
        shared_mu[warp_id] = warp_sum;
        shared_m2[warp_id] = warp_sum_sq;
    }
    __syncthreads();
    
    // Reduce across warps
    if (warp_id == 0) {
        float block_sum = (lane < num_warps) ? shared_mu[lane] : 0.0f;
        float block_sum_sq = (lane < num_warps) ? shared_m2[lane] : 0.0f;
        
        block_sum = warp_reduce_sum(block_sum);
        block_sum_sq = warp_reduce_sum(block_sum_sq);
        
        if (lane == 0) {
            float mean = block_sum / cols;
            float mean_sq = block_sum_sq / cols;
            float variance = mean_sq - mean * mean;
            variance = fmaxf(variance, 0.0f);
            float rs = rsqrtf(variance);
            
            shared_mu[0] = mean;
            shared_m2[0] = rs;
        }
    }
    __syncthreads();
    
    float mean = shared_mu[0];
    float rs = shared_m2[0];
    
    if (tid == 0) {
        mu_out[row] = mean;
        rs_out[row] = rs;
    }
    
    // Vectorized store for z
    #pragma unroll
    for (int v = start_vec; v < end_vec; ++v) {
        int j = v * vec_size;
        
        float4 g_vec = reinterpret_cast<const float4*>(gamma + j)[0];
        float4 b_vec = reinterpret_cast<const float4*>(beta + j)[0];
        
        float vals[vec_size];
        vals[0] = reinterpret_cast<float4*>(x + row_offset)[v].x;
        vals[1] = reinterpret_cast<float4*>(x + row_offset)[v].y;
        vals[2] = reinterpret_cast<float4*>(x + row_offset)[v].z;
        vals[3] = reinterpret_cast<float4*>(x + row_offset)[v].w;
        
        float4 z_vec;
        z_vec.x = (vals[0] - mean) * rs * g_vec.x + b_vec.x;
        z_vec.y = (vals[1] - mean) * rs * g_vec.y + b_vec.y;
        z_vec.z = (vals[2] - mean) * rs * g_vec.z + b_vec.z;
        z_vec.w = (vals[3] - mean) * rs * g_vec.w + b_vec.w;
        
        reinterpret_cast<float4*>(z + row_offset)[v] = z_vec;
    }
    
    // Handle remaining elements
    for (int j = remainder_start + tid; j < cols; j += blockDim.x) {
        float val = x[row_offset + j];
        float normalized = (val - mean) * rs;
        z[row_offset + j] = normalized * gamma[j] + beta[j];
    }
}

// Simple kernel that works for any cols, using 1024 threads per block
__global__ void layernorm_parallel_fwd_simple_kernel(
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
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Each thread handles multiple elements with strided access
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    const int row_offset = row * cols;
    
    // First pass: compute x = x0 + residual, and local sums
    for (int j = tid; j < cols; j += num_threads) {
        float val = x0[row_offset + j] + residual[row_offset + j];
        x[row_offset + j] = val;
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Warp reduction
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = (num_threads + WARP_SIZE - 1) / WARP_SIZE;
    
    __shared__ float s_sum[32];  // Max 32 warps for 1024 threads
    __shared__ float s_sum_sq[32];
    __shared__ float s_mean;
    __shared__ float s_rs;
    
    float warp_sum = warp_reduce_sum(local_sum);
    float warp_sum_sq = warp_reduce_sum(local_sum_sq);
    
    if (lane == 0) {
        s_sum[warp_id] = warp_sum;
        s_sum_sq[warp_id] = warp_sum_sq;
    }
    __syncthreads();
    
    // Reduce across warps (only first warp)
    if (warp_id == 0) {
        float val = (lane < num_warps) ? s_sum[lane] : 0.0f;
        float val_sq = (lane < num_warps) ? s_sum_sq[lane] : 0.0f;
        
        val = warp_reduce_sum(val);
        val_sq = warp_reduce_sum(val_sq);
        
        if (lane == 0) {
            float mean = val / cols;
            float mean_sq = val_sq / cols;
            float variance = mean_sq - mean * mean;
            variance = fmaxf(variance, 0.0f);
            float rs = rsqrtf(variance);
            
            s_mean = mean;
            s_rs = rs;
            mu_out[row] = mean;
            rs_out[row] = rs;
        }
    }
    __syncthreads();
    
    float mean = s_mean;
    float rs = s_rs;
    
    // Second pass: normalize and apply gamma/beta
    for (int j = tid; j < cols; j += num_threads) {
        float val = x[row_offset + j];
        float normalized = (val - mean) * rs;
        z[row_offset + j] = normalized * gamma[j] + beta[j];
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
    const int blocks = rows;
    
    // For cols=1024, we can use vectorized loads
    // Check if cols is divisible by 4 for vectorization
    if (cols % 4 == 0 && cols >= 1024) {
        // Use vectorized kernel for better memory bandwidth
        layernorm_parallel_fwd_vec_kernel<4><<<blocks, threads_per_block, 0, stream>>>(
            x0, residual, gamma, beta, z, x, mu, rs, rows, cols, eps
        );
    } else {
        // Use simple kernel for general case
        layernorm_parallel_fwd_simple_kernel<<<blocks, threads_per_block, 0, stream>>>(
            x0, residual, gamma, beta, z, x, mu, rs, rows, cols, eps
        );
    }
}
