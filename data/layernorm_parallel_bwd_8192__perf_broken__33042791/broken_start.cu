#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int WARP_SIZE = 32;

// Warp reduce sum
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block reduce sum using shared memory
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    
    // Final reduce across warps
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (warp == 0) val = warp_reduce_sum(val);
    
    return val;
}

// Kernel for computing dgamma and dbeta (column-wise reductions)
template <int BLOCK_SIZE>
__global__ void layernorm_parallel_bwd_gamma_beta_kernel(
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    float* dgamma,
    float* dbeta,
    int rows,
    int cols
) {
    const int tid = threadIdx.x;
    const int col = blockIdx.x * BLOCK_SIZE + tid;
    
    float sum_dgamma = 0.0f;
    float sum_dbeta = 0.0f;
    
    if (col < cols) {
        for (int row = 0; row < rows; ++row) {
            float x_val = x[row * cols + col];
            float mu_val = mu[row];
            float rs_val = rs[row];
            float x_hat = (x_val - mu_val) * rs_val;
            float dz_val = dz[row * cols + col];
            
            sum_dgamma += dz_val * x_hat;
            sum_dbeta += dz_val;
        }
        
        dgamma[col] = sum_dgamma;
        dbeta[col] = sum_dbeta;
    }
}

// Kernel for computing dx (per-row computation)
template <int BLOCK_SIZE, int COLS_PER_THREAD>
__global__ void layernorm_parallel_bwd_dx_kernel(
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    const float* gamma,
    float* dx0,
    float* dresidual,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    __shared__ float shared[32];  // For block reduction
    
    if (row >= rows) return;
    
    float mu_val = mu[row];
    float rs_val = rs[row];
    
    // Step 1: Compute ds and db for this row
    float ds_local = 0.0f;
    float db_local = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < COLS_PER_THREAD; ++i) {
        int col = tid + i * BLOCK_SIZE;
        if (col < cols) {
            float x_val = x[row * cols + col];
            float dz_val = dz[row * cols + col];
            float gamma_val = gamma[col];
            float x_hat = (x_val - mu_val) * rs_val;
            float dz_gamma = dz_val * gamma_val;
            
            ds_local += dz_gamma * x_hat;
            db_local += dz_gamma;
        }
    }
    
    ds_local = block_reduce_sum(ds_local, shared);
    db_local = block_reduce_sum(db_local, shared);
    
    __shared__ float ds_shared;
    __shared__ float db_shared;
    
    if (tid == 0) {
        ds_shared = ds_local;
        db_shared = db_local;
    }
    __syncthreads();
    
    float ds = ds_shared;
    float db = db_shared;
    
    // Step 2: Compute dx and write to output
    #pragma unroll
    for (int i = 0; i < COLS_PER_THREAD; ++i) {
        int col = tid + i * BLOCK_SIZE;
        if (col < cols) {
            float x_val = x[row * cols + col];
            float dz_val = dz[row * cols + col];
            float gamma_val = gamma[col];
            float x_hat = (x_val - mu_val) * rs_val;
            float dz_gamma = dz_val * gamma_val;
            
            float dx = rs_val * (dz_gamma - (ds * x_hat - db) / cols);
            
            dx0[row * cols + col] = dx;
            dresidual[row * cols + col] = dx;
        }
    }
}

// Optimized kernel using vectorized loads for dx computation
template <int BLOCK_SIZE>
__global__ void layernorm_parallel_bwd_dx_vec4_kernel(
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    const float* gamma,
    float* dx0,
    float* dresidual,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int vec_cols = cols / 4;
    
    __shared__ float shared[32];
    
    if (row >= rows) return;
    
    float mu_val = mu[row];
    float rs_val = rs[row];
    
    // Cast to float4 for vectorized access
    const float4* dz4 = reinterpret_cast<const float4*>(dz + row * cols);
    const float4* x4 = reinterpret_cast<const float4*>(x + row * cols);
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
    float4* dx04 = reinterpret_cast<float4*>(dx0 + row * cols);
    float4* dresidual4 = reinterpret_cast<float4*>(dresidual + row * cols);
    
    // Step 1: Compute ds and db
    float ds_local = 0.0f;
    float db_local = 0.0f;
    
    for (int idx = tid; idx < vec_cols; idx += BLOCK_SIZE) {
        float4 dz_val4 = dz4[idx];
        float4 x_val4 = x4[idx];
        float4 gamma_val4 = gamma4[idx];
        
        float x_hat0 = (x_val4.x - mu_val) * rs_val;
        float x_hat1 = (x_val4.y - mu_val) * rs_val;
        float x_hat2 = (x_val4.z - mu_val) * rs_val;
        float x_hat3 = (x_val4.w - mu_val) * rs_val;
        
        float dz_gamma0 = dz_val4.x * gamma_val4.x;
        float dz_gamma1 = dz_val4.y * gamma_val4.y;
        float dz_gamma2 = dz_val4.z * gamma_val4.z;
        float dz_gamma3 = dz_val4.w * gamma_val4.w;
        
        ds_local += dz_gamma0 * x_hat0 + dz_gamma1 * x_hat1 + dz_gamma2 * x_hat2 + dz_gamma3 * x_hat3;
        db_local += dz_gamma0 + dz_gamma1 + dz_gamma2 + dz_gamma3;
    }
    
    ds_local = block_reduce_sum(ds_local, shared);
    db_local = block_reduce_sum(db_local, shared);
    
    __shared__ float ds_shared;
    __shared__ float db_shared;
    
    if (tid == 0) {
        ds_shared = ds_local;
        db_shared = db_local;
    }
    __syncthreads();
    
    float ds = ds_shared;
    float db = db_shared;
    
    // Step 2: Compute dx and write output
    for (int idx = tid; idx < vec_cols; idx += BLOCK_SIZE) {
        float4 dz_val4 = dz4[idx];
        float4 x_val4 = x4[idx];
        float4 gamma_val4 = gamma4[idx];
        
        float x_hat0 = (x_val4.x - mu_val) * rs_val;
        float x_hat1 = (x_val4.y - mu_val) * rs_val;
        float x_hat2 = (x_val4.z - mu_val) * rs_val;
        float x_hat3 = (x_val4.w - mu_val) * rs_val;
        
        float dz_gamma0 = dz_val4.x * gamma_val4.x;
        float dz_gamma1 = dz_val4.y * gamma_val4.y;
        float dz_gamma2 = dz_val4.z * gamma_val4.z;
        float dz_gamma3 = dz_val4.w * gamma_val4.w;
        
        float4 dx4;
        dx4.x = rs_val * (dz_gamma0 - (ds * x_hat0 + db) / cols);
        dx4.y = rs_val * (dz_gamma1 - (ds * x_hat1 + db) / cols);
        dx4.z = rs_val * (dz_gamma2 - (ds * x_hat2 + db) / cols);
        dx4.w = rs_val * (dz_gamma3 - (ds * x_hat3 + db) / cols);
        
        dx04[idx] = dx4;
        dresidual4[idx] = dx4;
    }
}

// Fallback kernel for non-divisible cols
template <int BLOCK_SIZE>
__global__ void layernorm_parallel_bwd_dx_general_kernel(
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    const float* gamma,
    float* dx0,
    float* dresidual,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    __shared__ float shared[32];
    
    if (row >= rows) return;
    
    float mu_val = mu[row];
    float rs_val = rs[row];
    
    // Step 1: Compute ds and db
    float ds_local = 0.0f;
    float db_local = 0.0f;
    
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        float x_val = x[row * cols + col];
        float dz_val = dz[row * cols + col];
        float gamma_val = gamma[col];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_gamma = dz_val * gamma_val;
        
        ds_local += dz_gamma * x_hat;
        db_local += dz_gamma;
    }
    
    ds_local = block_reduce_sum(ds_local, shared);
    db_local = block_reduce_sum(db_local, shared);
    
    __shared__ float ds_shared;
    __shared__ float db_shared;
    
    if (tid == 0) {
        ds_shared = ds_local;
        db_shared = db_local;
    }
    __syncthreads();
    
    float ds = ds_shared;
    float db = db_shared;
    
    // Step 2: Compute dx
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        float x_val = x[row * cols + col];
        float dz_val = dz[row * cols + col];
        float gamma_val = gamma[col];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_gamma = dz_val * gamma_val;
        
        float dx = rs_val * (dz_gamma - (ds * x_hat - db) / cols);
        
        dx0[row * cols + col] = dx;
        dresidual[row * cols + col] = dx;
    }
}

extern "C" {

void launch_layernorm_parallel_bwd(
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    const float* gamma,
    float* dx0,
    float* dresidual,
    float* dgamma,
    float* dbeta,
    int rows,
    int cols,
    cudaStream_t stream
) {
    // Kernel 1: Compute dgamma and dbeta (column-wise reduction)
    // Use 256 threads per block for good occupancy
    constexpr int GAMMA_BLOCK_SIZE = 256;
    int gamma_blocks = (cols + GAMMA_BLOCK_SIZE - 1) / GAMMA_BLOCK_SIZE;
    
    layernorm_parallel_bwd_gamma_beta_kernel<GAMMA_BLOCK_SIZE>
        <<<gamma_blocks, GAMMA_BLOCK_SIZE, 0, stream>>>(
        dz, x, mu, rs, dgamma, dbeta, rows, cols
    );
    
    // Kernel 2: Compute dx (per-row computation)
    // Check if cols is divisible by 4 for vectorization
    if (cols % 4 == 0) {
        constexpr int DX_BLOCK_SIZE = 256;
        layernorm_parallel_bwd_dx_vec4_kernel<DX_BLOCK_SIZE>
            <<<rows, DX_BLOCK_SIZE, 0, stream>>>(
            dz, x, mu, rs, gamma, dx0, dresidual, rows, cols
        );
    } else {
        constexpr int DX_BLOCK_SIZE = 256;
        layernorm_parallel_bwd_dx_general_kernel<DX_BLOCK_SIZE>
            <<<rows, DX_BLOCK_SIZE, 0, stream>>>(
            dz, x, mu, rs, gamma, dx0, dresidual, rows, cols
        );
    }
}

} // extern "C"
