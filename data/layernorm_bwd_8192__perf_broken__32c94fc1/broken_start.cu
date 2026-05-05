#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int WARP_SIZE = 32;

// Warp-level sum reduction
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level sum reduction using shared memory
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    // Warp reduce
    val = warp_reduce_sum(val);
    
    // First thread of each warp writes to shared memory
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (warp_id == 0) {
        val = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// Kernel for computing per-row statistics (ds, db) and dx
// Also computes partial sums for dgamma and dbeta
template <int BLOCK_SIZE, int COLS_PER_THREAD>
__global__ void layernorm_backward_kernel(
    const float* __restrict__ dz,
    const float* __restrict__ x,
    const float* __restrict__ mu,
    const float* __restrict__ rs,
    const float* __restrict__ gamma,
    float* __restrict__ dx,
    float* __restrict__ dgamma_partial,
    float* __restrict__ dbeta_partial,
    int rows,
    int cols,
    int num_row_tiles  // Number of row groups for dgamma/dbeta reduction
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    if (row >= rows) return;
    
    const float mu_val = mu[row];
    const float rs_val = rs[row];
    
    // Shared memory for block reduction
    __shared__ float shared_ds[BLOCK_SIZE / WARP_SIZE];
    __shared__ float shared_db[BLOCK_SIZE / WARP_SIZE];
    
    // Accumulators for ds and db
    float ds = 0.0f;
    float db = 0.0f;
    
    // First pass: compute x_hat, dz_gamma, and accumulate ds, db
    // Also compute partial dgamma and dbeta contributions
    float local_dgamma[COLS_PER_THREAD];
    float local_dbeta[COLS_PER_THREAD];
    
    #pragma unroll
    for (int i = 0; i < COLS_PER_THREAD; i++) {
        local_dgamma[i] = 0.0f;
        local_dbeta[i] = 0.0f;
    }
    
    // Process columns in tiles
    const int cols_per_block = BLOCK_SIZE * COLS_PER_THREAD;
    const int num_iters = (cols + cols_per_block - 1) / cols_per_block;
    
    for (int iter = 0; iter < num_iters; iter++) {
        int col_base = iter * cols_per_block + tid * COLS_PER_THREAD;
        
        float local_x_hat[COLS_PER_THREAD];
        float local_dz_gamma[COLS_PER_THREAD];
        
        #pragma unroll
        for (int i = 0; i < COLS_PER_THREAD; i++) {
            int col = col_base + i;
            if (col < cols) {
                float x_val = x[row * cols + col];
                float x_hat = (x_val - mu_val) * rs_val;
                local_x_hat[i] = x_hat;
                
                float dz_val = dz[row * cols + col];
                float gamma_val = gamma[col];
                float dz_gamma = dz_val * gamma_val;
                local_dz_gamma[i] = dz_gamma;
                
                ds += dz_gamma * x_hat;
                db += dz_gamma;
                
                // Accumulate for dgamma and dbeta
                local_dgamma[i] += dz_val * x_hat;
                local_dbeta[i] += dz_val;
            }
        }
        
        // Store x_hat and dz_gamma for second pass (we recompute to save memory)
        // Actually, let's just recompute in second pass to avoid extra memory
    }
    
    // Reduce ds and db across block
    ds = block_reduce_sum(ds, shared_ds);
    db = block_reduce_sum(db, shared_db);
    
    // Broadcast ds and db to all threads
    __shared__ float shared_ds_final;
    __shared__ float shared_db_final;
    if (tid == 0) {
        shared_ds_final = ds;
        shared_db_final = db;
    }
    __syncthreads();
    
    ds = shared_ds_final;
    db = shared_db_final;
    
    const float inv_cols = 1.0f / cols;
    
    // Second pass: compute dx and write partial dgamma/dbeta
    for (int iter = 0; iter < num_iters; iter++) {
        int col_base = iter * cols_per_block + tid * COLS_PER_THREAD;
        
        #pragma unroll
        for (int i = 0; i < COLS_PER_THREAD; i++) {
            int col = col_base + i;
            if (col < cols) {
                // Recompute x_hat and dz_gamma
                float x_val = x[row * cols + col];
                float x_hat = (x_val - mu_val) * rs_val;
                float dz_val = dz[row * cols + col];
                float gamma_val = gamma[col];
                float dz_gamma = dz_val * gamma_val;
                
                // dx formula
                float term = (ds * x_hat - db) * inv_cols;
                dx[row * cols + col] = rs_val * (dz_gamma - term);
            }
        }
    }
    
    // Write partial dgamma and dbeta
    // Each row writes its contribution to a separate slice
    int partial_offset = row * cols;
    for (int iter = 0; iter < num_iters; iter++) {
        int col_base = iter * cols_per_block + tid * COLS_PER_THREAD;
        
        #pragma unroll
        for (int i = 0; i < COLS_PER_THREAD; i++) {
            int col = col_base + i;
            if (col < cols) {
                dgamma_partial[partial_offset + col] = local_dgamma[i];
                dbeta_partial[partial_offset + col] = local_dbeta[i];
            }
        }
    }
}

// Kernel to reduce partial dgamma and dbeta across rows
template <int BLOCK_SIZE>
__global__ void reduce_dgamma_dbeta_kernel(
    const float* __restrict__ dgamma_partial,
    const float* __restrict__ dbeta_partial,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows,
    int cols
) {
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (col >= cols) return;
    
    float sum_dgamma = 0.0f;
    float sum_dbeta = 0.0f;
    
    for (int row = 0; row < rows; row++) {
        sum_dgamma += dgamma_partial[row * cols + col];
        sum_dbeta += dbeta_partial[row * cols + col];
    }
    
    dgamma[col] = sum_dgamma;
    dbeta[col] = sum_dbeta;
}

// Optimized reduction using tree-based approach
template <int BLOCK_SIZE, int ROWS_PER_THREAD>
__global__ void reduce_dgamma_dbeta_optimized_kernel(
    const float* __restrict__ dgamma_partial,
    const float* __restrict__ dbeta_partial,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows,
    int cols
) {
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (col >= cols) return;
    
    float sum_dgamma = 0.0f;
    float sum_dbeta = 0.0f;
    
    // Each thread handles multiple rows with stride
    for (int row = 0; row < rows; row++) {
        sum_dgamma += dgamma_partial[row * cols + col];
        sum_dbeta += dbeta_partial[row * cols + col];
    }
    
    dgamma[col] = sum_dgamma;
    dbeta[col] = sum_dbeta;
}

// Fused kernel: compute dx and accumulate dgamma/dbeta using warp-level parallelism
// This version uses a different strategy: process rows in groups and use shared memory for dgamma/dbeta
template <int BLOCK_SIZE, int WARPS_PER_BLOCK, int COLS_PER_WARP>
__global__ void layernorm_backward_fused_kernel(
    const float* __restrict__ dz,
    const float* __restrict__ x,
    const float* __restrict__ mu,
    const float* __restrict__ rs,
    const float* __restrict__ gamma,
    float* __restrict__ dx,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows,
    int cols
) {
    // Shared memory for dgamma and dbeta accumulation
    // Each block handles a subset of columns
    extern __shared__ float shared_mem[];
    float* shared_dgamma = shared_mem;
    float* shared_dbeta = shared_mem + cols;  // This won't work - need dynamic sizing
    
    // Actually, use a different approach: each block handles COLS_PER_BLOCK columns
}

// Main kernel with two-phase approach optimized for large cols=8192
template <int BLOCK_SIZE>
__global__ void layernorm_backward_main_kernel(
    const float* __restrict__ dz,
    const float* __restrict__ x,
    const float* __restrict__ mu,
    const float* __restrict__ rs,
    const float* __restrict__ gamma,
    float* __restrict__ dx,
    float* __restrict__ dgamma_scratch,
    float* __restrict__ dbeta_scratch,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    const float mu_val = mu[row];
    const float rs_val = rs[row];
    
    // Use registers for accumulation
    float ds = 0.0f;
    float db = 0.0f;
    
    // First pass: compute ds and db
    // Each thread handles multiple columns
    const int cols_per_thread = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Align to 4 for better memory access
    for (int c = tid; c < cols; c += BLOCK_SIZE) {
        float x_val = x[row * cols + c];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + c];
        float dz_gamma = dz_val * gamma[c];
        
        ds += dz_gamma * x_hat;
        db += dz_gamma;
    }
    
    // Warp reduction
    ds = warp_reduce_sum(ds);
    db = warp_reduce_sum(db);
    
    // Block reduction using shared memory
    __shared__ float shared_ds[BLOCK_SIZE / WARP_SIZE];
    __shared__ float shared_db[BLOCK_SIZE / WARP_SIZE];
    
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    if (lane == 0) {
        shared_ds[warp_id] = ds;
        shared_db[warp_id] = db;
    }
    __syncthreads();
    
    // Final reduction
    if (tid < BLOCK_SIZE / WARP_SIZE) {
        ds = shared_ds[tid];
        db = shared_db[tid];
    } else {
        ds = 0.0f;
        db = 0.0f;
    }
    __syncthreads();
    
    if (tid < BLOCK_SIZE / WARP_SIZE) {
        ds = warp_reduce_sum(ds);
        if (tid == 0) {
            shared_ds[0] = ds;
            shared_db[0] = db;
        }
    }
    __syncthreads();
    
    ds = shared_ds[0];
    db = shared_db[0];
    
    const float inv_cols = 1.0f / cols;
    
    // Second pass: compute dx and write dgamma/dbeta scratch
    for (int c = tid; c < cols; c += BLOCK_SIZE) {
        float x_val = x[row * cols + c];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + c];
        float gamma_val = gamma[c];
        float dz_gamma = dz_val * gamma_val;
        
        // Compute dx
        float term = (ds * x_hat - db) * inv_cols;
        dx[row * cols + c] = rs_val * (dz_gamma - term);
        
        // Write scratch space for dgamma/dbeta
        dgamma_scratch[row * cols + c] = dz_val * x_hat;
        dbeta_scratch[row * cols + c] = dz_val;
    }
}

// Optimized version with vectorized loads for cols=8192
template <int BLOCK_SIZE>
__global__ void layernorm_backward_vec4_kernel(
    const float* __restrict__ dz,
    const float* __restrict__ x,
    const float* __restrict__ mu,
    const float* __restrict__ rs,
    const float* __restrict__ gamma,
    float* __restrict__ dx,
    float* __restrict__ dgamma_scratch,
    float* __restrict__ dbeta_scratch,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    const float mu_val = mu[row];
    const float rs_val = rs[row];
    
    float ds = 0.0f;
    float db = 0.0f;
    
    // Vectorized loads for better memory bandwidth
    const int vec_cols = cols / 4;
    
    // Cast to float4 for vectorized access
    const float4* x_vec = reinterpret_cast<const float4*>(x + row * cols);
    const float4* dz_vec = reinterpret_cast<const float4*>(dz + row * cols);
    const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
    float4* dx_vec = reinterpret_cast<float4*>(dx + row * cols);
    float4* dgamma_vec = reinterpret_cast<float4*>(dgamma_scratch + row * cols);
    float4* dbeta_vec = reinterpret_cast<float4*>(dbeta_scratch + row * cols);
    
    // First pass with vectorized loads
    for (int c = tid; c < vec_cols; c += BLOCK_SIZE) {
        float4 x4 = x_vec[c];
        float4 dz4 = dz_vec[c];
        float4 g4 = gamma_vec[c];
        
        // Process 4 elements
        float x_hat[4], dz_gamma[4];
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float xv = reinterpret_cast<const float*>(&x4)[i];
            x_hat[i] = (xv - mu_val) * rs_val;
            float dzv = reinterpret_cast<const float*>(&dz4)[i];
            float gv = reinterpret_cast<const float*>(&g4)[i];
            dz_gamma[i] = dzv * gv;
            ds += dz_gamma[i] * x_hat[i];
            db += dz_gamma[i];
        }
    }
    
    // Handle remaining elements
    for (int c = vec_cols * 4 + tid; c < cols; c += BLOCK_SIZE) {
        float x_val = x[row * cols + c];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + c];
        float dz_gamma = dz_val * gamma[c];
        ds += dz_gamma * x_hat;
        db += dz_gamma;
    }
    
    // Block reduction
    __shared__ float shared_ds[BLOCK_SIZE];
    __shared__ float shared_db[BLOCK_SIZE];
    
    shared_ds[tid] = ds;
    shared_db[tid] = db;
    __syncthreads();
    
    // Tree reduction
    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_ds[tid] += shared_ds[tid + stride];
            shared_db[tid] += shared_db[tid + stride];
        }
        __syncthreads();
    }
    
    ds = shared_ds[0];
    db = shared_db[0];
    
    const float inv_cols = 1.0f / cols;
    
    // Second pass: compute dx and write scratch
    for (int c = tid; c < vec_cols; c += BLOCK_SIZE) {
        float4 x4 = x_vec[c];
        float4 dz4 = dz_vec[c];
        float4 g4 = gamma_vec[c];
        
        float4 dx4, dg4, db4;
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float xv = reinterpret_cast<const float*>(&x4)[i];
            float x_hat = (xv - mu_val) * rs_val;
            float dzv = reinterpret_cast<const float*>(&dz4)[i];
            float gv = reinterpret_cast<const float*>(&g4)[i];
            float dz_gamma = dzv * gv;
            
            float term = (ds * x_hat - db) * inv_cols;
            float dxv = rs_val * (dz_gamma - term);
            
            reinterpret_cast<float*>(&dx4)[i] = dxv;
            reinterpret_cast<float*>(&dg4)[i] = dzv * x_hat;
            reinterpret_cast<float*>(&db4)[i] = dzv;
        }
        
        dx_vec[c] = dx4;
        dgamma_vec[c] = dg4;
        dbeta_vec[c] = db4;
    }
    
    // Handle remaining
    for (int c = vec_cols * 4 + tid; c < cols; c += BLOCK_SIZE) {
        float x_val = x[row * cols + c];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + c];
        float gamma_val = gamma[c];
        float dz_gamma = dz_val * gamma_val;
        
        float term = (ds * x_hat - db) * inv_cols;
        dx[row * cols + c] = rs_val * (dz_gamma - term);
        dgamma_scratch[row * cols + c] = dz_val * x_hat;
        dbeta_scratch[row * cols + c] = dz_val;
    }
}

// Final reduction kernel for dgamma and dbeta
template <int BLOCK_SIZE, int ROWS_PER_THREAD>
__global__ void final_reduce_kernel(
    const float* __restrict__ dgamma_scratch,
    const float* __restrict__ dbeta_scratch,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows,
    int cols
) {
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (col >= cols) return;
    
    float sum_dgamma = 0.0f;
    float sum_dbeta = 0.0f;
    
    // Each thread sums over all rows
    for (int row = 0; row < rows; row++) {
        sum_dgamma += dgamma_scratch[row * cols + col];
        sum_dbeta += dbeta_scratch[row * cols + col];
    }
    
    dgamma[col] = sum_dgamma;
    dbeta[col] = sum_dbeta;
}

// Vectorized final reduction
template <int BLOCK_SIZE>
__global__ void final_reduce_vec4_kernel(
    const float* __restrict__ dgamma_scratch,
    const float* __restrict__ dbeta_scratch,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows,
    int cols
) {
    const int col = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 4;
    if (col >= cols) return;
    
    float4 sum_dgamma = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum_dbeta = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    for (int row = 0; row < rows; row++) {
        float4 dg = reinterpret_cast<const float4*>(dgamma_scratch + row * cols)[col / 4];
        float4 db = reinterpret_cast<const float4*>(dbeta_scratch + row * cols)[col / 4];
        
        sum_dgamma.x += dg.x;
        sum_dgamma.y += dg.y;
        sum_dgamma.z += dg.z;
        sum_dgamma.w += dg.w;
        
        sum_dbeta.x += db.x;
        sum_dbeta.y += db.y;
        sum_dbeta.z += db.z;
        sum_dbeta.w += db.w;
    }
    
    reinterpret_cast<float4*>(dgamma)[col / 4] = sum_dgamma;
    reinterpret_cast<float4*>(dbeta)[col / 4] = sum_dbeta;
}

// Scratch space size: rows * cols * 2 * sizeof(float)
// For rows=16384, cols=8192: 16384 * 8192 * 8 = 1GB - too large!

// Need a different approach: process in column tiles
// Each block handles a tile of columns for all rows

template <int BLOCK_SIZE, int COL_TILE>
__global__ void layernorm_backward_tiled_kernel(
    const float* __restrict__ dz,
    const float* __restrict__ x,
    const float* __restrict__ mu,
    const float* __restrict__ rs,
    const float* __restrict__ gamma,
    float* __restrict__ dx,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows,
    int cols
) {
    // Each block handles COL_TILE columns
    const int col_tile_id = blockIdx.x;
    const int col_start = col_tile_id * COL_TILE;
    const int col_end = min(col_start + COL_TILE, cols);
    const int tid = threadIdx.x;
    
    // Shared memory for this tile's gamma
    __shared__ float shared_gamma[COL_TILE];
    __shared__ float shared_dgamma[COL_TILE];
    __shared__ float shared_dbeta[COL_TILE];
    
    // Load gamma for this tile
    for (int c = tid; c < COL_TILE && col_start + c < cols; c += BLOCK_SIZE) {
        shared_gamma[c] = gamma[col_start + c];
        shared_dgamma[c] = 0.0f;
        shared_dbeta[c] = 0.0f;
    }
    __syncthreads();
    
    // Process each row
    for (int row = 0; row < rows; row++) {
        const float mu_val = mu[row];
        const float rs_val = rs[row];
        
        // Compute ds and db for this row (locally, then reduce)
        // Actually, we need global ds and db per row, so this approach doesn't work well
        
        // Alternative: each thread computes partial sums for its columns
        float ds_local = 0.0f;
        float db_local = 0.0f;
        
        // First compute local contributions
        for (int c = tid; c < COL_TILE && col_start + c < cols; c += BLOCK_SIZE) {
            int global_col = col_start + c;
            float x_val = x[row * cols + global_col];
            float x_hat = (x_val - mu_val) * rs_val;
            float dz_val = dz[row * cols + global_col];
            float dz_gamma = dz_val * shared_gamma[c];
            
            ds_local += dz_gamma * x_hat;
            db_local += dz_gamma;
        }
        
        // Need to reduce ds and db across all blocks for this row
        // This requires global synchronization...
    }
}

// Best approach: Two kernel launches
// Kernel 1: For each row, compute dx and write partial dgamma/dbeta to scratch
// Kernel 2: Reduce scratch across rows for each column

// But scratch is too big. Alternative: use atomicAdd or segmented reduction

// Actually, let's use a hybrid: process in row batches
// Each batch computes partial dgamma/dbeta, accumulate to output

template <int BLOCK_SIZE>
__global__ void layernorm_backward_batch_kernel(
    const float* __restrict__ dz,
    const float* __restrict__ x,
    const float* __restrict__ mu,
    const float* __restrict__ rs,
    const float* __restrict__ gamma,
    float* __restrict__ dx,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows,
    int cols,
    int row_batch_size  // Number of rows this kernel processes
) {
    const int row_batch = blockIdx.y;  // Which batch of rows
    const int row = row_batch * row_batch_size + blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    const float mu_val = mu[row];
    const float rs_val = rs[row];
    
    // Accumulate ds and db
    float ds = 0.0f;
    float db = 0.0f;
    
    // First pass
    for (int c = tid; c < cols; c += BLOCK_SIZE) {
        float x_val = x[row * cols + c];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + c];
        float dz_gamma = dz_val * gamma[c];
        
        ds += dz_gamma * x_hat;
        db += dz_gamma;
    }
    
    // Block reduction
    __shared__ float shared_ds[BLOCK_SIZE];
    __shared__ float shared_db[BLOCK_SIZE];
    
    shared_ds[tid] = ds;
    shared_db[tid] = db;
    __syncthreads();
    
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_ds[tid] += shared_ds[tid + stride];
            shared_db[tid] += shared_db[tid + stride];
        }
        __syncthreads();
    }
    
    ds = shared_ds[0];
    db = shared_db[0];
    
    const float inv_cols = 1.0f / cols;
    
    // Second pass: compute dx and atomicAdd to dgamma/dbeta
    for (int c = tid; c < cols; c += BLOCK_SIZE) {
        float x_val = x[row * cols + c];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + c];
        float gamma_val = gamma[c];
        float dz_gamma = dz_val * gamma_val;
        
        float term = (ds * x_hat - db) * inv_cols;
        dx[row * cols + c] = rs_val * (dz_gamma - term);
        
        // Atomic add for dgamma and dbeta
        float dg = dz_val * x_hat;
        atomicAdd(&dgamma[c], dg);
        atomicAdd(&dbeta[c], dz_val);
    }
}

// Optimized version with warp-level atomics and better memory coalescing
template <int BLOCK_SIZE, int WARP_SIZE_ = 32>
__global__ void layernorm_backward_optimized_kernel(
    const float* __restrict__ dz,
    const float* __restrict__ x,
    const float* __restrict__ mu,
    const float* __restrict__ rs,
    const float* __restrict__ gamma,
    float* __restrict__ dx,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE_;
    const int warp_id = tid / WARP_SIZE_;
    
    if (row >= rows) return;
    
    const float mu_val = mu[row];
    const float rs_val = rs[row];
    
    // Use warp shuffle for reduction
    float ds = 0.0f;
    float db = 0.0f;
    
    // Vectorized processing
    const int vec_cols = cols / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x + row * cols);
    const float4* dz_vec = reinterpret_cast<const float4*>(dz + row * cols);
    const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
    float4* dx_vec = reinterpret_cast<float4*>(dx + row * cols);
    
    // First pass with vectorized loads
    for (int c = tid; c < vec_cols; c += BLOCK_SIZE) {
        float4 x4 = x_vec[c];
        float4 dz4 = dz_vec[c];
        float4 g4 = gamma_vec[c];
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float xv = reinterpret_cast<const float*>(&x4)[i];
            float x_hat = (xv - mu_val) * rs_val;
            float dzv = reinterpret_cast<const float*>(&dz4)[i];
            float gv = reinterpret_cast<const float*>(&g4)[i];
            float dz_gamma = dzv * gv;
            ds += dz_gamma * x_hat;
            db += dz_gamma;
        }
    }
    
    // Remainder
    for (int c = vec_cols * 4 + tid; c < cols; c += BLOCK_SIZE) {
        float x_val = x[row * cols + c];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + c];
        float dz_gamma = dz_val * gamma[c];
        ds += dz_gamma * x_hat;
        db += dz_gamma;
    }
    
    // Warp reduction
    ds = warp_reduce_sum(ds);
    db = warp_reduce_sum(db);
    
    // Broadcast to all threads in warp
    ds = __shfl_sync(0xffffffff, ds, 0);
    db = __shfl_sync(0xffffffff, db, 0);
    
    const float inv_cols = 1.0f / cols;
    
    // Second pass: compute dx and accumulate dgamma/dbeta
    // Use shared memory buffer for atomic coalescing
    __shared__ float shared_dgamma[BLOCK_SIZE * 4];  // Buffer for atomics
    __shared__ float shared_dbeta[BLOCK_SIZE * 4];
    
    // Actually, direct atomicAdd is fine for this scale
    
    for (int c = tid; c < vec_cols; c += BLOCK_SIZE) {
        float4 x4 = x_vec[c];
        float4 dz4 = dz_vec[c];
        float4 g4 = gamma_vec[c];
        
        float4 dx4;
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float xv = reinterpret_cast<const float*>(&x4)[i];
            float x_hat = (xv - mu_val) * rs_val;
            float dzv = reinterpret_cast<const float*>(&dz4)[i];
            float gv = reinterpret_cast<const float*>(&g4)[i];
            float dz_gamma = dzv * gv;
            
            float term = (ds * x_hat - db) * inv_cols;
            float dxv = rs_val * (dz_gamma - term);
            
            reinterpret_cast<float*>(&dx4)[i] = dxv;
        }
        
        dx_vec[c] = dx4;
        
        // Atomic adds for dgamma and dbeta
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int global_col = c * 4 + i;
            float dzv = reinterpret_cast<const float*>(&dz4)[i];
            float xv = reinterpret_cast<const float*>(&x4)[i];
            float x_hat = (xv - mu_val) * rs_val;
            atomicAdd(&dgamma[global_col], dzv * x_hat);
            atomicAdd(&dbeta[global_col], dzv);
        }
    }
    
    // Remainder
    for (int c = vec_cols * 4 + tid; c < cols; c += BLOCK_SIZE) {
        float x_val = x[row * cols + c];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + c];
        float gamma_val = gamma[c];
        float dz_gamma = dz_val * gamma_val;
        
        float term = (ds * x_hat - db) * inv_cols;
        dx[row * cols + c] = rs_val * (dz_gamma - term);
        
        atomicAdd(&dgamma[c], dz_val * x_hat);
        atomicAdd(&dbeta[c], dz_val);
    }
}

// Most optimized: use two-phase with scratch buffer, but process in chunks
// Actually atomicAdd should be fine for 16384 rows * 8192 cols with good distribution

// Let's verify: 16384 * 8192 = 134M atomic adds per output
// Each column gets 16384 atomic adds, which is manageable

// Final optimized kernel
template <int BLOCK_SIZE>
__global__ void layernorm_backward_final_kernel(
    const float* __restrict__ dz,
    const float* __restrict__ x,
    const float* __restrict__ mu,
    const float* __restrict__ rs,
    const float* __restrict__ gamma,
    float* __restrict__ dx,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    const float mu_val = mu[row];
    const float rs_val = rs[row];
    
    float ds = 0.0f;
    float db = 0.0f;
    
    // Coalesced memory access - each thread handles strided columns
    for (int c = tid; c < cols; c += BLOCK_SIZE) {
        float x_val = x[row * cols + c];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + c];
        float dz_gamma = dz_val * gamma[c];
        
        ds += dz_gamma * x_hat;
        db += dz_gamma;
    }
    
    // Block reduction
    __shared__ float shared_ds[BLOCK_SIZE];
    __shared__ float shared_db[BLOCK_SIZE];
    
    shared_ds[tid] = ds;
    shared_db[tid] = db;
    __syncthreads();
    
    // Tree reduction
    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_ds[tid] += shared_ds[tid + stride];
            shared_db[tid] += shared_db[tid + stride];
        }
        __syncthreads();
    }
    
    ds = shared_ds[0];
    db = shared_db[0];
    
    const float inv_cols = 1.0f / cols;
    
    // Compute dx and atomicAdd for dgamma/dbeta
    for (int c = tid; c < cols; c += BLOCK_SIZE) {
        float x_val = x[row * cols + c];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + c];
        float gamma_val = gamma[c];
        float dz_gamma = dz_val * gamma_val;
        
        float term = (ds * x_hat - db) * inv_cols;
        dx[row * cols + c] = rs_val * (dz_gamma - term);
        
        // Atomic adds - coalesced access pattern
        atomicAdd(&dgamma[c], dz_val * x_hat);
        atomicAdd(&dbeta[c], dz_val);
    }
}

// Vectorized version for better memory bandwidth
template <int BLOCK_SIZE>
__global__ void layernorm_backward_vec_kernel(
    const float* __restrict__ dz,
    const float* __restrict__ x,
    const float* __restrict__ mu,
    const float* __restrict__ rs,
    const float* __restrict__ gamma,
    float* __restrict__ dx,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    const float mu_val = mu[row];
    const float rs_val = rs[row];
    
    float ds = 0.0f;
    float db = 0.0f;
    
    // Process 4 elements at a time
    const int vec_size = 4;
    const int vec_cols = cols / vec_size;
    
    // Ensure alignment
    const float4* x_vec = reinterpret_cast<const float4*>(x + row * cols);
    const float4* dz_vec = reinterpret_cast<const float4*>(dz + row * cols);
    const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
    float4* dx_vec = reinterpret_cast<float4*>(dx + row * cols);
    
    // First pass - vectorized
    for (int c = tid; c < vec_cols; c += BLOCK_SIZE) {
        float4 x4 = x_vec[c];
        float4 dz4 = dz_vec[c];
        float4 g4 = gamma_vec[c];
        
        float4 x_hat4, dz_gamma4;
        
        x_hat4.x = (x4.x - mu_val) * rs_val;
        x_hat4.y = (x4.y - mu_val) * rs_val;
        x_hat4.z = (x4.z - mu_val) * rs_val;
        x_hat4.w = (x4.w - mu_val) * rs_val;
        
        dz_gamma4.x = dz4.x * g4.x;
        dz_gamma4.y = dz4.y * g4.y;
        dz_gamma4.z = dz4.z * g4.z;
        dz_gamma4.w = dz4.w * g4.w;
        
        ds += dz_gamma4.x * x_hat4.x;
        ds += dz_gamma4.y * x_hat4.y;
        ds += dz_gamma4.z * x_hat4.z;
        ds += dz_gamma4.w * x_hat4.w;
        
        db += dz_gamma4.x;
        db += dz_gamma4.y;
        db += dz_gamma4.z;
        db += dz_gamma4.w;
    }
    
    // Handle remainder
    for (int c = vec_cols * vec_size + tid; c < cols; c += BLOCK_SIZE) {
        float x_val = x[row * cols + c];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + c];
        float dz_gamma = dz_val * gamma[c];
        ds += dz_gamma * x_hat;
        db += dz_gamma;
    }
    
    // Block reduction
    __shared__ float shared_ds[BLOCK_SIZE];
    __shared__ float shared_db[BLOCK_SIZE];
    
    shared_ds[tid] = ds;
    shared_db[tid] = db;
    __syncthreads();
    
    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_ds[tid] += shared_ds[tid + stride];
            shared_db[tid] += shared_db[tid + stride];
        }
        __syncthreads();
    }
    
    ds = shared_ds[0];
    db = shared_db[0];
    
    const float inv_cols = 1.0f / cols;
    
    // Second pass - vectorized
    for (int c = tid; c < vec_cols; c += BLOCK_SIZE) {
        float4 x4 = x_vec[c];
        float4 dz4 = dz_vec[c];
        float4 g4 = gamma_vec[c];
        
        float4 x_hat4, dz_gamma4;
        float4 dx4;
        
        x_hat4.x = (x4.x - mu_val) * rs_val;
        x_hat4.y = (x4.y - mu_val) * rs_val;
        x_hat4.z = (x4.z - mu_val) * rs_val;
        x_hat4.w = (x4.w - mu_val) * rs_val;
        
        dz_gamma4.x = dz4.x * g4.x;
        dz_gamma4.y = dz4.y * g4.y;
        dz_gamma4.z = dz4.z * g4.z;
        dz_gamma4.w = dz4.w * g4.w;
        
        float term_x = (ds * x_hat4.x + db) * inv_cols;
        float term_y = (ds * x_hat4.y + db) * inv_cols;
        float term_z = (ds * x_hat4.z + db) * inv_cols;
        float term_w = (ds * x_hat4.w + db) * inv_cols;
        
        dx4.x = rs_val * (dz_gamma4.x - term_x);
        dx4.y = rs_val * (dz_gamma4.y - term_y);
        dx4.z = rs_val * (dz_gamma4.z - term_z);
        dx4.w = rs_val * (dz_gamma4.w - term_w);
        
        dx_vec[c] = dx4;
        
        // Atomic adds
        atomicAdd(&dgamma[c * 4 + 0], dz4.x * x_hat4.x);
        atomicAdd(&dgamma[c * 4 + 1], dz4.y * x_hat4.y);
        atomicAdd(&dgamma[c * 4 + 2], dz4.z * x_hat4.z);
        atomicAdd(&dgamma[c * 4 + 3], dz4.w * x_hat4.w);
        
        atomicAdd(&dbeta[c * 4 + 0], dz4.x);
        atomicAdd(&dbeta[c * 4 + 1], dz4.y);
        atomicAdd(&dbeta[c * 4 + 2], dz4.z);
        atomicAdd(&dbeta[c * 4 + 3], dz4.w);
    }
    
    // Handle remainder
    for (int c = vec_cols * vec_size + tid; c < cols; c += BLOCK_SIZE) {
        float x_val = x[row * cols + c];
        float x_hat = (x_val - mu_val) * rs_val;
        float dz_val = dz[row * cols + c];
        float gamma_val = gamma[c];
        float dz_gamma = dz_val * gamma_val;
        
        float term = (ds * x_hat - db) * inv_cols;
        dx[row * cols + c] = rs_val * (dz_gamma - term);
        
        atomicAdd(&dgamma[c], dz_val * x_hat);
        atomicAdd(&dbeta[c], dz_val);
    }
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
    // Zero out dgamma and dbeta first
    cudaMemsetAsync(dgamma, 0, cols * sizeof(float), stream);
    cudaMemsetAsync(dbeta, 0, cols * sizeof(float), stream);
    
    // Choose block size based on cols
    // For cols=8192, we want enough threads to cover the columns with good occupancy
    
    constexpr int BLOCK_SIZE = 256;
    
    // Use vectorized kernel for better performance
    // Check if cols is divisible by 4
    if (cols % 4 == 0) {
        layernorm_backward_vec_kernel<BLOCK_SIZE><<<rows, BLOCK_SIZE, 0, stream>>>(
            dz, x, mu, rs, gamma, dx, dgamma, dbeta, rows, cols
        );
    } else {
        layernorm_backward_final_kernel<BLOCK_SIZE><<<rows, BLOCK_SIZE, 0, stream>>>(
            dz, x, mu, rs, gamma, dx, dgamma, dbeta, rows, cols
        );
    }
}
