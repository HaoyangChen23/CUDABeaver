#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define WARP_SIZE 32
#define COLS 256
#define THREADS_PER_ROW 256
#define ROWS_PER_BLOCK 4

// Helper for warp-level reduction
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// LayerNorm forward kernel
// Each block processes ROWS_PER_BLOCK rows
// Each row uses THREADS_PER_ROW threads (256 threads = 8 warps)
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
    // Shared memory for inter-warp reduction
    // Need space for: sum (mean), sum of squares (for variance)
    __shared__ float shared_sum[ROWS_PER_BLOCK];
    __shared__ float shared_sq_sum[ROWS_PER_BLOCK];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int row_in_block = blockIdx.y * ROWS_PER_BLOCK + blockIdx.x;
    
    // Each thread processes multiple elements if cols > THREADS_PER_ROW
    // But here cols = 256 and THREADS_PER_ROW = 256, so each thread handles 1 element
    
    if (row_in_block >= rows) return;
    
    // Load gamma and beta for this column (tid corresponds to column)
    float g = (tid < cols) ? gamma[tid] : 0.0f;
    float b = (tid < cols) ? beta[tid] : 0.0f;
    
    // Load input
    float val = 0.0f;
    if (tid < cols) {
        val = x[row_in_block * cols + tid];
    }
    
    // Step 1: Compute mean
    // Warp-level sum
    float warp_sum = warp_reduce_sum(val);
    
    // Store warp sums to shared memory
    __shared__ float warp_sums[THREADS_PER_ROW / WARP_SIZE];
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // First warp reduces warp sums
    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (lane_id < THREADS_PER_ROW / WARP_SIZE) ? warp_sums[lane_id] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane_id == 0) {
            shared_sum[0] = block_sum;
        }
    }
    __syncthreads();
    
    float mean = shared_sum[0] / cols;
    
    // Step 2: Compute variance
    float diff = (tid < cols) ? (val - mean) : 0.0f;
    float sq_diff = diff * diff;
    
    // Warp-level sum of squared diffs
    float warp_sq_sum = warp_reduce_sum(sq_diff);
    
    // Store to shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sq_sum;
    }
    __syncthreads();
    
    // Reduce
    float block_sq_sum = 0.0f;
    if (warp_id == 0) {
        block_sq_sum = (lane_id < THREADS_PER_ROW / WARP_SIZE) ? warp_sums[lane_id] : 0.0f;
        block_sq_sum = warp_reduce_sum(block_sq_sum);
        if (lane_id == 0) {
            shared_sq_sum[0] = block_sq_sum;
        }
    }
    __syncthreads();
    
    float variance = shared_sq_sum[0] / cols;
    float rsqrt_var = rsqrtf(variance + eps);
    
    // Store mu and rs for this row
    if (tid == 0) {
        mu[row_in_block] = mean;
        rs[row_in_block] = rsqrt_var;
    }
    
    // Step 3: Normalize, scale, and shift
    if (tid < cols) {
        float normalized = diff * rsqrt_var;
        z[row_in_block * cols + tid] = normalized * g + b;
    }
}

// Optimized kernel using vectorized loads for better memory bandwidth
__global__ void layernorm_forward_kernel_vec4(
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
    // Each thread processes 4 elements using float4
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int num_threads = blockDim.x;
    const int vec_cols = cols / 4;  // 64
    
    // Shared memory for reduction
    __shared__ float shared_sum[64];
    __shared__ float shared_sq_sum[64];
    
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    float4 local_x[4];  // Store for second pass
    float4 local_g[4];
    float4 local_b[4];
    
    // First pass: compute mean and load data
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * num_threads;
        if (idx < vec_cols) {
            int offset = row * cols + idx * 4;
            float4 xv = *reinterpret_cast<const float4*>(x + offset);
            float4 gv = *reinterpret_cast<const float4*>(gamma + idx * 4);
            float4 bv = *reinterpret_cast<const float4*>(beta + idx * 4);
            
            local_x[i] = xv;
            local_g[i] = gv;
            local_b[i] = bv;
            
            local_sum += xv.x + xv.y + xv.z + xv.w;
        }
    }
    
    // Warp reduction for sum
    float warp_sum = warp_reduce_sum(local_sum);
    
    __shared__ float warp_sums[2];  // 64 threads = 2 warps
    
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Final reduction
    if (tid < 2) {
        float s = (tid < 2) ? warp_sums[tid] : 0.0f;
        s = warp_reduce_sum(s);
        if (tid == 0) {
            shared_sum[0] = s;
        }
    }
    __syncthreads();
    
    float mean = shared_sum[0] / cols;
    
    // Second pass: compute variance
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * num_threads;
        if (idx < vec_cols) {
            float4 xv = local_x[i];
            float dx0 = xv.x - mean;
            float dx1 = xv.y - mean;
            float dx2 = xv.z - mean;
            float dx3 = xv.w - mean;
            
            local_sq_sum += dx0 * dx0 + dx1 * dx1 + dx2 * dx2 + dx3 * dx3;
            
            // Store differences for reuse
            local_x[i].x = dx0;
            local_x[i].y = dx1;
            local_x[i].z = dx2;
            local_x[i].w = dx3;
        }
    }
    
    // Reduce variance
    float warp_sq_sum = warp_reduce_sum(local_sq_sum);
    
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sq_sum;
    }
    __syncthreads();
    
    if (tid < 2) {
        float s = (tid < 2) ? warp_sums[tid] : 0.0f;
        s = warp_reduce_sum(s);
        if (tid == 0) {
            shared_sq_sum[0] = s;
        }
    }
    __syncthreads();
    
    float variance = shared_sq_sum[0] / cols;
    float rsqrt_var = rsqrtf(variance + eps);
    
    if (tid == 0) {
        mu[row] = mean;
        rs[row] = rsqrt_var;
    }
    
    // Third pass: write output
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * num_threads;
        if (idx < vec_cols) {
            int offset = row * cols + idx * 4;
            float4 gv = local_g[i];
            float4 bv = local_b[i];
            
            float4 out;
            out.x = local_x[i].x * rsqrt_var * gv.x + bv.x;
            out.y = local_x[i].y * rsqrt_var * gv.y + bv.y;
            out.z = local_x[i].z * rsqrt_var * gv.z + bv.z;
            out.w = local_x[i].w * rsqrt_var * gv.w + bv.w;
            
            *reinterpret_cast<float4*>(z + offset) = out;
        }
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
    // Use vectorized kernel for better performance with cols=256
    // 64 threads, each handling 4 elements = 256 elements per row
    const int threads = 64;
    const int blocks = rows;
    
    layernorm_forward_kernel_vec4<<<blocks, threads, 0, stream>>>(
        x, gamma, beta, z, mu, rs, rows, cols, eps
    );
}
