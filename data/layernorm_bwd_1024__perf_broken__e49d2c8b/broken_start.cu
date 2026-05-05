#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int WARP_SIZE = 32;

// Warp reduce sum
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduce sum using warp shuffle
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    if (wid == 0) {
        val = (lane < (blockDim.x / WARP_SIZE)) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// Kernel for computing dgamma and dbeta (column-wise reductions)
// Each block processes one column, threads iterate over rows
__global__ void layernorm_bwd_gamma_beta_kernel(
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
        
        sum_dgamma += dz_val;
        sum_dbeta += dz_val;
    }
    
    // Reduce within block
    __shared__ float shared_dgamma[32];
    __shared__ float shared_dbeta[32];
    
    sum_dgamma = block_reduce_sum(sum_dgamma, shared_dgamma);
    sum_dbeta = block_reduce_sum(sum_dbeta, shared_dbeta);
    
    if (threadIdx.x == 0) {
        dgamma[col] = sum_dgamma;
        dbeta[col] = sum_dbeta;
    }
}

// Kernel for computing dx (per-row computation)
// Each block processes one row, threads iterate over columns
__global__ void layernorm_bwd_dx_kernel(
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
    
    // First pass: compute ds and db for this row
    float sum_ds = 0.0f;  // sum of dz_gamma * x_hat
    float sum_db = 0.0f;  // sum of dz_gamma
    
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float x_val = x[row * cols + col];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + col];
        float gamma_val = gamma[col];
        float dz_gamma = dz_val * gamma_val;
        
        sum_ds += dz_gamma * x_hat;
        sum_db += dz_gamma;
    }
    
    // Reduce ds and db across threads in block
    __shared__ float shared_ds[32];
    __shared__ float shared_db[32];
    
    sum_ds = block_reduce_sum(sum_ds, shared_ds);
    sum_db = block_reduce_sum(sum_db, shared_db);
    
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
        
        float dx_val = rs_val * (dz_gamma - (ds * x_hat + db) * inv_cols);
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
    // One block per column, 256 threads per block
    int threads_gamma = 256;
    layernorm_bwd_gamma_beta_kernel<<<cols, threads_gamma, 0, stream>>>(
        dz, x, mu, rs, dgamma, dbeta, rows, cols
    );
    
    // Launch kernel for dx (row-wise)
    // One block per row, 256 threads per block
    int threads_dx = 256;
    layernorm_bwd_dx_kernel<<<rows, threads_dx, 0, stream>>>(
        dz, x, mu, rs, gamma, dx, rows, cols
    );
}

} // extern "C"
