#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 256

// Helper warp shuffle reductions
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// LayerNorm forward kernel for cols=1024
// Uses warp-level parallelism with 32 warps per block (1024 threads)
// Each warp processes 32 elements, 32 warps cover 1024 elements
template <int COLS>
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
    // Each block processes one row
    const int row = blockIdx.x;
    if (row >= rows) return;

    // Thread indexing: 32 warps, each warp has 32 threads
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;      // 0-31
    const int lane_id = tid % WARP_SIZE;      // 0-31

    // Each warp processes 32 consecutive elements
    // Warp 0: elements 0-31, Warp 1: 32-63, ..., Warp 31: 992-1023
    const int warp_offset = warp_id * WARP_SIZE;
    
    // Load input for this thread
    const int col = warp_offset + lane_id;
    const float* row_x = x + row * cols;
    
    float val = 0.0f;
    if (col < cols) {
        val = row_x[col];
    }

    // Step 1: Compute mean using warp shuffle
    // First, local sum within warp
    float sum = warp_reduce_sum(val);
    
    // Store warp sums to shared memory and reduce across warps
    __shared__ float shared_sum[32];  // One per warp
    if (lane_id == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();
    
    // Reduce warp sums (only first warp)
    float total_sum = 0.0f;
    if (tid < 32) {
        total_sum = (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared_sum[tid] : 0.0f;
        total_sum = warp_reduce_sum(total_sum);
        if (tid == 0) {
            shared_sum[0] = total_sum / cols;  // Store mean in shared_sum[0]
        }
    }
    __syncthreads();
    
    float mean = shared_sum[0];
    
    // Store mean output
    if (tid == 0) {
        mu[row] = mean;
    }

    // Step 2: Compute variance
    float diff = val - mean;
    float sq_diff = diff * diff;
    float sq_sum = warp_reduce_sum(sq_diff);
    
    // Store warp sums to shared memory
    if (lane_id == 0) {
        shared_sum[warp_id] = sq_sum;
    }
    __syncthreads();
    
    // Reduce across warps
    float total_sq_sum = 0.0f;
    if (tid < 32) {
        total_sq_sum = (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared_sum[tid] : 0.0f;
        total_sq_sum = warp_reduce_sum(total_sq_sum);
        if (tid == 0) {
            float variance = total_sq_sum / cols;
            float rsqrt_val = rsqrtf(variance + eps);
            shared_sum[0] = rsqrt_val;  // Store rs in shared_sum[0]
        }
    }
    __syncthreads();
    
    float rsqrt_var = shared_sum[0];
    
    // Store rs output
    if (tid == 0) {
        rs[row] = rsqrt_var;
    }

    // Step 3: Normalize, scale and shift
    float* row_z = z + row * cols;
    if (col < cols) {
        float normalized = diff * rsqrt_var;
        float scaled = normalized * gamma[col];
        float shifted = scaled + beta[col];
        row_z[col] = shifted;
    }
}

// Alternative kernel using vectorized loads for better memory bandwidth
template <int COLS, int VEC_SIZE>
__global__ void layernorm_forward_vec_kernel(
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
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Each thread processes VEC_SIZE elements
    const int elems_per_thread = VEC_SIZE;
    const int total_vec_elems = cols / VEC_SIZE;
    
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    
    // First pass: compute mean
    #pragma unroll 4
    for (int i = tid; i < total_vec_elems; i += num_threads) {
        int col = i * VEC_SIZE;
        
        // Load 4 floats
        float4 x_vec = reinterpret_cast<const float4*>(x + row * cols)[i];
        
        local_sum += x_vec.x + x_vec.y + x_vec.z + x_vec.w;
    }
    
    // Reduce within warp
    float warp_sum = warp_reduce_sum(local_sum);
    
    __shared__ float shared[64];
    if ((tid & 31) == 0) {
        shared[tid >> 5] = warp_sum;
    }
    __syncthreads();
    
    // Reduce across warps
    if (tid < (num_threads >> 5)) {
        float val = shared[tid];
        val = warp_reduce_sum(val);
        if (tid == 0) shared[0] = val / cols;
    }
    __syncthreads();
    
    float mean = shared[0];
    
    if (tid == 0) {
        mu[row] = mean;
    }
    
    // Second pass: compute variance
    local_sq_sum = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < total_vec_elems; i += num_threads) {
        int col = i * VEC_SIZE;
        
        float4 x_vec = reinterpret_cast<const float4*>(x + row * cols)[i];
        
        float d0 = x_vec.x - mean;
        float d1 = x_vec.y - mean;
        float d2 = x_vec.z - mean;
        float d3 = x_vec.w - mean;
        
        local_sq_sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    
    float warp_sq_sum = warp_reduce_sum(local_sq_sum);
    
    if ((tid & 31) == 0) {
        shared[tid >> 5] = warp_sq_sum;
    }
    __syncthreads();
    
    if (tid < (num_threads >> 5)) {
        float val = shared[tid];
        val = warp_reduce_sum(val);
        if (tid == 0) {
            float variance = val / cols;
            shared[0] = rsqrtf(variance + eps);
        }
    }
    __syncthreads();
    
    float rsqrt_var = shared[0];
    
    if (tid == 0) {
        rs[row] = rsqrt_var;
    }
    
    // Third pass: write output
    #pragma unroll 4
    for (int i = tid; i < total_vec_elems; i += num_threads) {
        int col = i * VEC_SIZE;
        
        float4 x_vec = reinterpret_cast<const float4*>(x + row * cols)[i];
        float4 g_vec = reinterpret_cast<const float4*>(gamma + col)[0];
        float4 b_vec = reinterpret_cast<const float4*>(beta + col)[0];
        
        float d0 = (x_vec.x - mean) * rsqrt_var;
        float d1 = (x_vec.y - mean) * rsqrt_var;
        float d2 = (x_vec.z - mean) * rsqrt_var;
        float d3 = (x_vec.w - mean) * rsqrt_var;
        
        float4 z_vec;
        z_vec.x = d0 * g_vec.x + b_vec.x;
        z_vec.y = d1 * g_vec.y + b_vec.y;
        z_vec.z = d2 * g_vec.z + b_vec.z;
        z_vec.w = d3 * g_vec.w + b_vec.w;
        
        reinterpret_cast<float4*>(z + row * cols)[i] = z_vec;
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
    // For cols=1024, use 256 threads (8 warps) with vectorized loads
    // Each thread loads 4 floats, so 256 threads cover 1024 elements
    const int threads = 256;
    
    // Check if cols is divisible by 4 for vectorized loads
    if (cols % 4 == 0 && cols == 1024) {
        layernorm_forward_vec_kernel<1024, 4><<<rows, threads, 0, stream>>>(
            x, gamma, beta, z, mu, rs, rows, cols, eps
        );
    } else {
        // Fallback to generic kernel
        layernorm_forward_kernel<1024><<<rows, threads, 0, stream>>>(
            x, gamma, beta, z, mu, rs, rows, cols, eps
        );
    }
}

} // extern "C"
