#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
constexpr int COLS = 256;

// Warp-level sum reduction using shfl
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level sum reduction using shared memory
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Warp reduce
    val = warp_reduce_sum(val);
    
    // Store to shared memory
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    // Load from shared memory and warp reduce again
    val = (tid < WARPS_PER_BLOCK) ? shared[tid] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

// Kernel for computing dgamma and dbeta (column-wise reductions)
// Each block handles one column, rows are distributed across threads
__global__ void layernorm_bwd_weights_kernel(
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    float* dgamma,
    float* dbeta,
    int rows,
    int cols
) {
    int col = blockIdx.x;
    if (col >= cols) return;
    
    float sum_dgamma = 0.0f;
    float sum_dbeta = 0.0f;
    
    // Each thread processes multiple rows
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        float x_val = x[row * cols + col];
        float mu_val = mu[row];
        float rs_val = rs[row];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + col];
        
        sum_dgamma += dz_val * x_hat;
        sum_dbeta += dz_val;
    }
    
    // Block reduction
    __shared__ float shared_dgamma[WARPS_PER_BLOCK];
    __shared__ float shared_dbeta[WARPS_PER_BLOCK];
    
    sum_dgamma = block_reduce_sum(sum_dgamma, shared_dgamma);
    sum_dbeta = block_reduce_sum(sum_dbeta, shared_dbeta);
    
    if (threadIdx.x == 0) {
        dgamma[col] = sum_dgamma;
        dbeta[col] = sum_dbeta;
    }
}

// Kernel for computing dx (per-row data gradient)
// Each block handles one row, with threads cooperating to compute ds and db
__global__ void layernorm_bwd_data_kernel(
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    const float* gamma,
    float* dx,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    float mu_val = mu[row];
    float rs_val = rs[row];
    
    __shared__ float shared_ds[WARPS_PER_BLOCK];
    __shared__ float shared_db[WARPS_PER_BLOCK];
    
    // First pass: compute ds and db
    float sum_ds = 0.0f;
    float sum_db = 0.0f;
    
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float x_val = x[row * cols + col];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + col];
        float gamma_val = gamma[col];
        float dz_gamma = dz_val * gamma_val;
        
        sum_ds += dz_gamma * x_hat;
        sum_db += dz_gamma;
    }
    
    // Reduce ds and db across block
    sum_ds = block_reduce_sum(sum_ds, shared_ds);
    sum_db = block_reduce_sum(sum_db, shared_db);
    
    // Broadcast ds and db to all threads
    __shared__ float ds_shared;
    __shared__ float db_shared;
    if (threadIdx.x == 0) {
        ds_shared = sum_ds;
        db_shared = sum_db;
    }
    __syncthreads();
    
    float ds = ds_shared;
    float db = db_shared;
    float inv_cols = 1.0f / cols;
    
    // Second pass: compute dx
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float x_val = x[row * cols + col];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + col];
        float gamma_val = gamma[col];
        float dz_gamma = dz_val * gamma_val;
        
        float dx_val = rs_val * (dz_gamma - (db * x_hat + ds) * inv_cols);
        dx[row * cols + col] = dx_val;
    }
}

extern "C" {

void launch_layernorm_backward(
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    const float* gamma,
    float* dx,
    float* dgamma,
    float* dbeta,
    int rows,
    int cols,
    cudaStream_t stream
) {
    // Launch kernel for dgamma and dbeta (column-wise)
    // One block per column, threads handle rows
    int threads_per_block = 256;
    layernorm_bwd_weights_kernel<<<cols, threads_per_block, 0, stream>>>(
        dz, x, mu, rs, dgamma, dbeta, rows, cols
    );
    
    // Launch kernel for dx (row-wise)
    // One block per row, threads handle columns
    layernorm_bwd_data_kernel<<<rows, threads_per_block, 0, stream>>>(
        dz, x, mu, rs, gamma, dx, rows, cols
    );
}

} // extern "C"
