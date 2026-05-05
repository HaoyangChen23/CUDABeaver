#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int WARP_SIZE = 32;

// Warp-level sum reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level sum reduction using shared memory
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    // Warp 0 reduces partial sums
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// Kernel for computing dgamma and dbeta (reduction over rows)
// Each block handles a subset of columns, all rows
template <int BLOCK_SIZE>
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
    // Each thread processes one column
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    if (col >= cols) return;
    
    float sum_dgamma = 0.0f;
    float sum_dbeta = 0.0f;
    
    // Iterate over all rows
    for (int row = 0; row < rows; ++row) {
        float x_val = x[row * cols + col];
        float mu_val = mu[row];
        float rs_val = rs[row];
        float dz_val = dz[row * cols + col];
        
        float x_hat = (x_val - mu_val) * rs_val;
        
        sum_dgamma += dz_val * x_hat;
        sum_dbeta += dz_val;
    }
    
    dgamma[col] = sum_dgamma;
    dbeta[col] = sum_dbeta;
}

// Kernel for computing dx (and dx0, dresidual)
// Each block handles one row, threads process columns
template <int BLOCK_SIZE>
__global__ void layernorm_bwd_dx_kernel(
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
    if (row >= rows) return;
    
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
    
    const float mu_val = mu[row];
    const float rs_val = rs[row];
    
    // First pass: compute ds and db for this row
    float ds = 0.0f;
    float db = 0.0f;
    
    // Grid-stride loop over columns
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float x_val = x[row * cols + col];
        float dz_val = dz[row * cols + col];
        float gamma_val = gamma[col];
        
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_gamma = dz_val * gamma_val;
        
        ds += dz_gamma * x_hat;
        db += dz_gamma;
    }
    
    ds = block_reduce_sum<BLOCK_SIZE>(ds, shared);
    db = block_reduce_sum<BLOCK_SIZE>(db, shared);
    
    // Broadcast ds and db to all threads
    __shared__ float ds_shared;
    __shared__ float db_shared;
    if (threadIdx.x == 0) {
        ds_shared = ds;
        db_shared = db;
    }
    __syncthreads();
    
    ds = ds_shared;
    db = db_shared;
    
    const float inv_cols = 1.0f / cols;
    
    // Second pass: compute dx
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float x_val = x[row * cols + col];
        float dz_val = dz[row * cols + col];
        float gamma_val = gamma[col];
        
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_gamma = dz_val * gamma_val;
        
        float dx_val = rs_val * (dz_gamma - (ds * x_hat + db) * inv_cols);
        
        dx0[row * cols + col] = dx_val;
        dresidual[row * cols + col] = dx_val;
    }
}

// Combined kernel that computes everything in one go
// Uses two-phase approach: first compute dgamma/dbeta reductions, then dx
template <int BLOCK_SIZE>
__global__ void layernorm_parallel_bwd_kernel(
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
    int cols
) {
    // Grid: (rows, 1) for dx computation, but we also need column reductions
    
    // Part 1: Each block handles one row for dx computation
    const int row = blockIdx.x;
    if (row >= rows) return;
    
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
    
    const float mu_val = mu[row];
    const float rs_val = rs[row];
    
    // Compute ds and db for this row
    float ds = 0.0f;
    float db = 0.0f;
    
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float x_val = x[row * cols + col];
        float dz_val = dz[row * cols + col];
        float gamma_val = gamma[col];
        
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_gamma = dz_val * gamma_val;
        
        ds += dz_gamma * x_hat;
        db += dz_gamma;
        
        // Also accumulate dgamma and dbeta atomically (will be slow, need better approach)
        // Actually, let's use a separate kernel for dgamma/dbeta
    }
    
    ds = block_reduce_sum<BLOCK_SIZE>(ds, shared);
    db = block_reduce_sum<BLOCK_SIZE>(db, shared);
    
    __shared__ float ds_shared;
    __shared__ float db_shared;
    if (threadIdx.x == 0) {
        ds_shared = ds;
        db_shared = db;
    }
    __syncthreads();
    
    ds = ds_shared;
    db = db_shared;
    
    const float inv_cols = 1.0f / cols;
    
    // Compute dx
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float x_val = x[row * cols + col];
        float dz_val = dz[row * cols + col];
        float gamma_val = gamma[col];
        
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_gamma = dz_val * gamma_val;
        
        float dx_val = rs_val * (dz_gamma - (ds * x_hat + db) * inv_cols);
        
        dx0[row * cols + col] = dx_val;
        dresidual[row * cols + col] = dx_val;
    }
}

// Optimized kernel using shared memory for dgamma/dbeta accumulation
// Each block processes multiple rows, threads within block handle different columns
template <int BLOCK_SIZE, int COLS_PER_THREAD>
__global__ void layernorm_bwd_dx_optimized_kernel(
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
    if (row >= rows) return;
    
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
    
    const float mu_val = mu[row];
    const float rs_val = rs[row];
    
    // Compute ds and db
    float ds = 0.0f;
    float db = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < COLS_PER_THREAD; ++i) {
        int col = threadIdx.x + i * BLOCK_SIZE;
        if (col < cols) {
            float x_val = x[row * cols + col];
            float dz_val = dz[row * cols + col];
            float gamma_val = gamma[col];
            
            float x_hat = (x_val - mu_val) * rs_val;
            float dz_gamma = dz_val * gamma_val;
            
            ds += dz_gamma * x_hat;
            db += dz_gamma;
        }
    }
    
    ds = block_reduce_sum<BLOCK_SIZE>(ds, shared);
    db = block_reduce_sum<BLOCK_SIZE>(db, shared);
    
    __shared__ float ds_shared;
    __shared__ float db_shared;
    if (threadIdx.x == 0) {
        ds_shared = ds;
        db_shared = db;
    }
    __syncthreads();
    
    ds = ds_shared;
    db = db_shared;
    
    const float inv_cols = 1.0f / cols;
    
    // Compute dx
    #pragma unroll
    for (int i = 0; i < COLS_PER_THREAD; ++i) {
        int col = threadIdx.x + i * BLOCK_SIZE;
        if (col < cols) {
            float x_val = x[row * cols + col];
            float dz_val = dz[row * cols + col];
            float gamma_val = gamma[col];
            
            float x_hat = (x_val - mu_val) * rs_val;
            float dz_gamma = dz_val * gamma_val;
            
            float dx_val = rs_val * (dz_gamma - (ds * x_hat + db) * inv_cols);
            
            dx0[row * cols + col] = dx_val;
            dresidual[row * cols + col] = dx_val;
        }
    }
}

// Kernel for dgamma/dbeta using atomic operations with block-level aggregation
template <int BLOCK_SIZE>
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
    // Each block handles a chunk of rows and all columns
    // Use shared memory for local accumulation, then atomic add to global
    
    const int row_start = blockIdx.x * BLOCK_SIZE;
    const int row_end = min(row_start + BLOCK_SIZE, rows);
    
    // Each thread handles a subset of columns
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float sum_dgamma = 0.0f;
        float sum_dbeta = 0.0f;
        
        for (int row = row_start; row < row_end; ++row) {
            float x_val = x[row * cols + col];
            float mu_val = mu[row];
            float rs_val = rs[row];
            float dz_val = dz[row * cols + col];
            
            float x_hat = (x_val - mu_val) * rs_val;
            
            sum_dgamma += dz_val * x_hat;
            sum_dbeta += dz_val;
        }
        
        // Atomic add to global memory
        atomicAdd(&dgamma[col], sum_dgamma);
        atomicAdd(&dbeta[col], sum_dbeta);
    }
}

// Optimized version: each block handles one row, uses warp shuffle for dgamma/dbeta
template <int BLOCK_SIZE>
__global__ void layernorm_bwd_all_in_one_kernel(
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
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    
    __shared__ float shared_ds[BLOCK_SIZE / WARP_SIZE];
    __shared__ float shared_db[BLOCK_SIZE / WARP_SIZE];
    
    const float mu_val = mu[row];
    const float rs_val = rs[row];
    
    // Compute ds and db for this row
    float ds = 0.0f;
    float db = 0.0f;
    
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float x_val = x[row * cols + col];
        float dz_val = dz[row * cols + col];
        float gamma_val = gamma[col];
        
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_gamma = dz_val * gamma_val;
        
        ds += dz_gamma * x_hat;
        db += dz_gamma;
    }
    
    ds = block_reduce_sum<BLOCK_SIZE>(ds, shared_ds);
    db = block_reduce_sum<BLOCK_SIZE>(db, shared_db);
    
    __shared__ float ds_shared;
    __shared__ float db_shared;
    if (threadIdx.x == 0) {
        ds_shared = ds;
        db_shared = db;
    }
    __syncthreads();
    
    ds = ds_shared;
    db = db_shared;
    
    const float inv_cols = 1.0f / cols;
    
    // Compute dx and also accumulate dgamma/dbeta contributions
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float x_val = x[row * cols + col];
        float dz_val = dz[row * cols + col];
        float gamma_val = gamma[col];
        
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_gamma = dz_val * gamma_val;
        
        float dx_val = rs_val * (dz_gamma - (ds * x_hat + db) * inv_cols);
        
        dx0[row * cols + col] = dx_val;
        dresidual[row * cols + col] = dx_val;
        
        // Accumulate dgamma and dbeta using atomic operations
        // Each thread handles unique columns, no conflict
        atomicAdd(&dgamma[col], dz_val * x_hat);
        atomicAdd(&dbeta[col], dz_val);
    }
}

// Final optimized version using separate passes
// Pass 1: Compute dx, dx0, dresidual (one kernel)
// Pass 2: Compute dgamma, dbeta (one kernel with atomic adds)

template <int BLOCK_SIZE>
__global__ void compute_dx_kernel(
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
    if (row >= rows) return;
    
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
    
    const float mu_val = mu[row];
    const float rs_val = rs[row];
    
    // Compute ds and db
    float ds = 0.0f;
    float db = 0.0f;
    
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float x_val = x[row * cols + col];
        float dz_val = dz[row * cols + col];
        float gamma_val = gamma[col];
        
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_gamma = dz_val * gamma_val;
        
        ds += dz_gamma * x_hat;
        db += dz_gamma;
    }
    
    ds = block_reduce_sum<BLOCK_SIZE>(ds, shared);
    db = block_reduce_sum<BLOCK_SIZE>(db, shared);
    
    __shared__ float ds_shared;
    __shared__ float db_shared;
    if (threadIdx.x == 0) {
        ds_shared = ds;
        db_shared = db;
    }
    __syncthreads();
    
    ds = ds_shared;
    db = db_shared;
    
    const float inv_cols = 1.0f / cols;
    
    // Compute and write dx
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float x_val = x[row * cols + col];
        float dz_val = dz[row * cols + col];
        float gamma_val = gamma[col];
        
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_gamma = dz_val * gamma_val;
        
        float dx_val = rs_val * (dz_gamma - (ds * x_hat + db) * inv_cols);
        
        dx0[row * cols + col] = dx_val;
        dresidual[row * cols + col] = dx_val;
    }
}

// Grid-stride kernel for dgamma/dbeta - each block handles multiple rows
template <int BLOCK_SIZE>
__global__ void compute_dgamma_dbeta_kernel(
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    float* dgamma,
    float* dbeta,
    int rows,
    int cols
) {
    // Each thread handles one column, iterates over all rows
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (col >= cols) return;
    
    float sum_dgamma = 0.0f;
    float sum_dbeta = 0.0f;
    
    // Grid-stride over rows for better memory coalescing
    for (int row = 0; row < rows; ++row) {
        float x_val = x[row * cols + col];
        float mu_val = mu[row];
        float rs_val = rs[row];
        float dz_val = dz[row * cols + col];
        
        float x_hat = (x_val - mu_val) * rs_val;
        
        sum_dgamma += dz_val * x_hat;
        sum_dbeta += dz_val;
    }
    
    dgamma[col] = sum_dgamma;
    dbeta[col] = sum_dbeta;
}

// Optimized: use vectorized loads for better bandwidth
template <int BLOCK_SIZE>
__global__ void compute_dx_kernel_vec4(
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
    if (row >= rows) return;
    
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
    
    const float mu_val = mu[row];
    const float rs_val = rs[row];
    
    // Compute ds and db with vectorized loads
    float ds = 0.0f;
    float db = 0.0f;
    
    // Process 4 elements at a time
    const int vec_cols = cols / 4 * 4;
    
    for (int col = threadIdx.x * 4; col < vec_cols; col += BLOCK_SIZE * 4) {
        float4 x4 = reinterpret_cast<const float4*>(x + row * cols)[col / 4];
        float4 dz4 = reinterpret_cast<const float4*>(dz + row * cols)[col / 4];
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float gamma_val = gamma[col + i];
            float x_hat = ((&x4.x)[i] - mu_val) * rs_val;
            float dz_gamma = (&dz4.x)[i] * gamma_val;
            
            ds += dz_gamma * x_hat;
            db += dz_gamma;
        }
    }
    
    // Handle remaining elements
    for (int col = vec_cols + threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float x_val = x[row * cols + col];
        float dz_val = dz[row * cols + col];
        float gamma_val = gamma[col];
        
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_gamma = dz_val * gamma_val;
        
        ds += dz_gamma * x_hat;
        db += dz_gamma;
    }
    
    ds = block_reduce_sum<BLOCK_SIZE>(ds, shared);
    db = block_reduce_sum<BLOCK_SIZE>(db, shared);
    
    __shared__ float ds_shared;
    __shared__ float db_shared;
    if (threadIdx.x == 0) {
        ds_shared = ds;
        db_shared = db;
    }
    __syncthreads();
    
    ds = ds_shared;
    db = db_shared;
    
    const float inv_cols = 1.0f / cols;
    
    // Write dx with vectorized stores
    for (int col = threadIdx.x * 4; col < vec_cols; col += BLOCK_SIZE * 4) {
        float4 x4 = reinterpret_cast<const float4*>(x + row * cols)[col / 4];
        float4 dz4 = reinterpret_cast<const float4*>(dz + row * cols)[col / 4];
        
        float4 dx4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float gamma_val = gamma[col + i];
            float x_hat = ((&x4.x)[i] - mu_val) * rs_val;
            float dz_gamma = (&dz4.x)[i] * gamma_val;
            
            float dx_val = rs_val * (dz_gamma - (ds * x_hat + db) * inv_cols);
            (&dx4.x)[i] = dx_val;
        }
        
        reinterpret_cast<float4*>(dx0 + row * cols)[col / 4] = dx4;
        reinterpret_cast<float4*>(dresidual + row * cols)[col / 4] = dx4;
    }
    
    // Handle remaining elements
    for (int col = vec_cols + threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float x_val = x[row * cols + col];
        float dz_val = dz[row * cols + col];
        float gamma_val = gamma[col];
        
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_gamma = dz_val * gamma_val;
        
        float dx_val = rs_val * (dz_gamma - (ds * x_hat + db) * inv_cols);
        
        dx0[row * cols + col] = dx_val;
        dresidual[row * cols + col] = dx_val;
    }
}

// Vectorized version for dgamma/dbeta
template <int BLOCK_SIZE>
__global__ void compute_dgamma_dbeta_kernel_vec4(
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    float* dgamma,
    float* dbeta,
    int rows,
    int cols
) {
    const int col_start = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 4;
    if (col_start >= cols) return;
    
    float sum_dgamma[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum_dbeta[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for (int row = 0; row < rows; ++row) {
        float4 x4 = reinterpret_cast<const float4*>(x + row * cols)[col_start / 4];
        float4 dz4 = reinterpret_cast<const float4*>(dz + row * cols)[col_start / 4];
        float mu_val = mu[row];
        float rs_val = rs[row];
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float x_hat = ((&x4.x)[i] - mu_val) * rs_val;
            sum_dgamma[i] += (&dz4.x)[i] * x_hat;
            sum_dbeta[i] += (&dz4.x)[i];
        }
    }
    
    #pragma unroll
    for (int i = 0; i < 4 && (col_start + i) < cols; ++i) {
        dgamma[col_start + i] = sum_dgamma[i];
        dbeta[col_start + i] = sum_dbeta[i];
    }
}

extern "C" void launch_layernorm_parallel_bwd(
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
    constexpr int BLOCK_SIZE = 256;
    
    // Zero out dgamma and dbeta first (since we accumulate into them)
    cudaMemsetAsync(dgamma, 0, cols * sizeof(float), stream);
    cudaMemsetAsync(dbeta, 0, cols * sizeof(float), stream);
    
    // Check if cols is divisible by 4 for vectorization
    if (cols % 4 == 0) {
        // Use vectorized kernels
        dim3 grid_dx(rows);
        dim3 block_dx(BLOCK_SIZE);
        compute_dx_kernel_vec4<BLOCK_SIZE><<<grid_dx, block_dx, 0, stream>>>(
            dz, x, mu, rs, gamma, dx0, dresidual, rows, cols
        );
        
        // For dgamma/dbeta, each thread handles 4 columns
        int num_col_threads = (cols + 3) / 4;
        dim3 grid_gb((num_col_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
        compute_dgamma_dbeta_kernel_vec4<BLOCK_SIZE><<<grid_gb, block_dx, 0, stream>>>(
            dz, x, mu, rs, dgamma, dbeta, rows, cols
        );
    } else {
        // Use non-vectorized kernels
        dim3 grid_dx(rows);
        dim3 block_dx(BLOCK_SIZE);
        compute_dx_kernel<BLOCK_SIZE><<<grid_dx, block_dx, 0, stream>>>(
            dz, x, mu, rs, gamma, dx0, dresidual, rows, cols
        );
        
        dim3 grid_gb((cols + BLOCK_SIZE - 1) / BLOCK_SIZE);
        compute_dgamma_dbeta_kernel<BLOCK_SIZE><<<grid_gb, block_dx, 0, stream>>>(
            dz, x, mu, rs, dgamma, dbeta, rows, cols
        );
    }
}
