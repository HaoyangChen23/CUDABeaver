#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
constexpr int COLS = 4096;

// Warp-level reduction using shfl
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        val = (lane < WARPS_PER_BLOCK) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    
    return shared[0];
}

// Kernel for computing per-row statistics (ds, db) and dx
__global__ void layernorm_bwd_dx_kernel(
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    const float* gamma,
    float* dx,
    float* ds_buf,  // Temporary buffer for ds per row
    float* db_buf,  // Temporary buffer for db per row
    int rows,
    int cols
) {
    __shared__ float shared[WARPS_PER_BLOCK];
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    const float mu_row = mu[row];
    const float rs_row = rs[row];
    
    // Compute ds and db for this row
    float ds_local = 0.0f;
    float db_local = 0.0f;
    
    #pragma unroll 4
    for (int j = tid; j < cols; j += BLOCK_SIZE) {
        const float x_val = x[row * cols + j];
        const float x_hat = (x_val - mu_row) * rs_row;
        const float dz_val = dz[row * cols + j];
        const float gamma_j = gamma[j];
        const float dz_gamma = dz_val * gamma_j;
        
        ds_local += dz_gamma * x_hat;
        db_local += dz_gamma;
    }
    
    ds_local = block_reduce_sum(ds_local, shared);
    db_local = block_reduce_sum(db_local, shared);
    
    const float ds_row = ds_local;
    const float db_row = db_local;
    
    // Store ds and db for this row (only thread 0 writes)
    if (tid == 0) {
        ds_buf[row] = ds_row;
        db_buf[row] = db_row;
    }
    
    // Compute dx
    const float inv_cols = 1.0f / cols;
    
    #pragma unroll 4
    for (int j = tid; j < cols; j += BLOCK_SIZE) {
        const float x_val = x[row * cols + j];
        const float x_hat = (x_val - mu_row) * rs_row;
        const float dz_val = dz[row * cols + j];
        const float gamma_j = gamma[j];
        const float dz_gamma = dz_val * gamma_j;
        
        const float dx_val = rs_row * (dz_gamma - (ds_row * x_hat + db_row) * inv_cols);
        dx[row * cols + j] = dx_val;
    }
}

// Kernel for computing dgamma and dbeta (column-wise reduction)
__global__ void layernorm_bwd_dgamma_dbeta_kernel(
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    float* dgamma,
    float* dbeta,
    int rows,
    int cols
) {
    __shared__ float shared_dgamma[WARPS_PER_BLOCK];
    __shared__ float shared_dbeta[WARPS_PER_BLOCK];
    
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int tid = threadIdx.x;
    
    if (col >= cols) return;
    
    float dgamma_local = 0.0f;
    float dbeta_local = 0.0f;
    
    // Each thread processes multiple rows
    #pragma unroll 2
    for (int i = 0; i < rows; i++) {
        const float mu_row = mu[i];
        const float rs_row = rs[i];
        const float x_val = x[i * cols + col];
        const float x_hat = (x_val - mu_row) * rs_row;
        const float dz_val = dz[i * cols + col];
        
        dgamma_local += dz_val * x_hat;
        dbeta_local += dz_val;
    }
    
    // Reduce within warp
    dgamma_local = warp_reduce_sum(dgamma_local);
    dbeta_local = warp_reduce_sum(dbeta_local);
    
    // Write results
    if (tid % WARP_SIZE == 0) {
        const int warp_id = tid / WARP_SIZE;
        shared_dgamma[warp_id] = dgamma_local;
        shared_dbeta[warp_id] = dbeta_local;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < WARPS_PER_BLOCK) {
        dgamma_local = shared_dgamma[tid];
        dbeta_local = shared_dbeta[tid];
    } else {
        dgamma_local = 0.0f;
        dbeta_local = 0.0f;
    }
    
    if (tid < WARP_SIZE) {
        dgamma_local = warp_reduce_sum(dgamma_local);
        dbeta_local = warp_reduce_sum(dbeta_local);
        
        if (tid == 0 && col < cols) {
            dgamma[col] = dgamma_local;
            dbeta[col] = dbeta_local;
        }
    }
}

// Alternative: more efficient column reduction using grid-stride loops
__global__ void layernorm_bwd_dgamma_dbeta_kernel_v2(
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    float* dgamma,
    float* dbeta,
    int rows,
    int cols
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col >= cols) return;
    
    float dgamma_val = 0.0f;
    float dbeta_val = 0.0f;
    
    // Grid-stride loop over rows
    for (int i = 0; i < rows; i++) {
        const float mu_row = mu[i];
        const float rs_row = rs[i];
        const float x_val = x[i * cols + col];
        const float x_hat = (x_val - mu_row) * rs_row;
        const float dz_val = dz[i * cols + col];
        
        dgamma_val += dz_val * x_hat;
        dbeta_val += dz_val;
    }
    
    dgamma[col] = dgamma_val;
    dbeta[col] = dbeta_val;
}

extern "C" void launch_layernorm_backward(
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
    // Allocate temporary buffers for ds and db per row
    float* ds_buf = nullptr;
    float* db_buf = nullptr;
    cudaMallocAsync(&ds_buf, rows * sizeof(float), stream);
    cudaMallocAsync(&db_buf, rows * sizeof(float), stream);
    
    // Launch kernel to compute dx and per-row statistics
    const int num_blocks_dx = rows;
    layernorm_bwd_dx_kernel<<<num_blocks_dx, BLOCK_SIZE, 0, stream>>>(
        dz, x, mu, rs, gamma, dx, ds_buf, db_buf, rows, cols
    );
    
    // Launch kernel to compute dgamma and dbeta
    // Use one thread per column for simplicity and coalesced access
    const int num_blocks_gamma = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    layernorm_bwd_dgamma_dbeta_kernel_v2<<<num_blocks_gamma, BLOCK_SIZE, 0, stream>>>(
        dz, x, mu, rs, dgamma, dbeta, rows, cols
    );
    
    // Free temporary buffers
    cudaFreeAsync(ds_buf, stream);
    cudaFreeAsync(db_buf, stream);
}
