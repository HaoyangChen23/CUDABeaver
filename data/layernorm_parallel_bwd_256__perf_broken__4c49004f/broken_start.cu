#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_COLS = 256; // cols = 256
constexpr int THREADS_PER_BLOCK = 256; // 1 warp per col group, 8 warps = 256 threads

// Warp-level reduction using shuffle
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for a single value per warp
__inline__ __device__ float block_reduce_sum(float val, float* shared_mem) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane_id == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (warp_id == 0) {
        val = (lane_id < blockDim.x / WARP_SIZE) ? shared_mem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// Kernel for computing dgamma and dbeta (column-wise reductions)
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
    // Each thread handles one column
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col >= cols) return;
    
    float sum_dz = 0.0f;
    float sum_dz_xhat = 0.0f;
    
    const float gamma_col = 1.0f; // gamma[col] for x_hat computation
    
    for (int row = 0; row < rows; ++row) {
        const float dz_val = dz[row * cols + col];
        const float x_val = x[row * cols + col];
        const float mu_row = mu[row];
        const float rs_row = rs[row];
        
        const float x_hat = (x_val - mu_row) * rs_row;
        
        sum_dz += dz_val;
        sum_dz_xhat += dz_val * x_hat;
    }
    
    dgamma[col] = sum_dz_xhat;
    dbeta[col] = sum_dz;
}

// Kernel for computing dx (per-row processing)
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
    // Each block handles one row, threads process columns in parallel
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    __shared__ float shared_ds[32];
    __shared__ float shared_db[32];
    __shared__ float s_gamma[BLOCK_COLS];
    
    // Load gamma into shared memory
    for (int i = tid; i < cols; i += blockDim.x) {
        s_gamma[i] = gamma[i];
    }
    __syncthreads();
    
    const float mu_row = mu[row];
    const float rs_row = rs[row];
    
    // First pass: compute dz * gamma and accumulate ds and db
    float local_ds = 0.0f;
    float local_db = 0.0f;
    
    // Each thread handles multiple columns
    for (int col = tid; col < cols; col += blockDim.x) {
        const int idx = row * cols + col;
        const float dz_val = dz[idx];
        const float x_val = x[idx];
        const float x_hat = (x_val - mu_row) * rs_row;
        const float dz_gamma = dz_val * s_gamma[col];
        
        local_ds += dz_gamma * x_hat;
        local_db += dz_gamma;
    }
    
    // Reduce ds and db across block
    local_ds = block_reduce_sum(local_ds, shared_ds);
    local_db = block_reduce_sum(local_db, shared_db);
    
    __syncthreads();
    
    // Broadcast ds and db to all threads
    const float ds = shared_ds[0];
    const float db = shared_db[0];
    const float inv_cols = 1.0f / float(cols);
    
    // Second pass: compute dx
    for (int col = tid; col < cols; col += blockDim.x) {
        const int idx = row * cols + col;
        const float dz_val = dz[idx];
        const float x_val = x[idx];
        const float x_hat = (x_val - mu_row) * rs_row;
        const float dz_gamma = dz_val * s_gamma[col];
        
        const float dx_val = rs_row * (dz_gamma - (ds * x_hat - db) * inv_cols);
        
        dx0[idx] = dx_val;
        dresidual[idx] = dx_val;
    }
}

// Fused kernel that does everything in one go
__global__ void layernorm_parallel_bwd_fused_kernel(
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
    // Grid: multiple blocks per row for columns, multiple rows
    // Each block handles a portion of columns for a set of rows
    
    const int row_stride = gridDim.y;
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x * blockDim.x + threadIdx.x;
    const int col_stride = gridDim.x * blockDim.x;
    
    // Accumulators for dgamma and dbeta (only needed once globally)
    // We use atomic adds for dgamma/dbeta
    
    for (int row = row_start; row < rows; row += row_stride) {
        if (col_start >= cols) continue;
        
        const float mu_row = mu[row];
        const float rs_row = rs[row];
        
        // Load and compute x_hat, accumulate for dgamma/dbeta
        float local_dgamma = 0.0f;
        float local_dbeta = 0.0f;
        
        // First: compute per-row statistics ds and db
        // We need to compute these before we can compute dx
        
        // This approach doesn't work well for fused - let's use a different strategy
    }
}

// Optimized kernel: each block handles one row
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
    // Shared memory layout: gamma[cols], partial sums for ds/db reduction
    extern __shared__ float shared_mem[];
    
    float* s_gamma = shared_mem;
    float* s_ds = &shared_mem[cols];     // size: blockDim.x / 32
    float* s_db = &shared_mem[cols + blockDim.x / 32]; // size: blockDim.x / 32
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    if (row >= rows) return;
    
    // Load gamma into shared memory
    for (int i = tid; i < cols; i += blockDim.x) {
        s_gamma[i] = gamma[i];
    }
    __syncthreads();
    
    const float mu_row = mu[row];
    const float rs_row = rs[row];
    
    // Compute dz * gamma and accumulate ds, db
    float thread_ds = 0.0f;
    float thread_db = 0.0f;
    
    // Also accumulate dgamma and dbeta contributions
    float thread_dgamma = 0.0f;
    float thread_dbeta = 0.0f;
    
    for (int col = tid; col < cols; col += blockDim.x) {
        const int idx = row * cols + col;
        const float dz_val = dz[idx];
        const float x_val = x[idx];
        const float x_hat = (x_val - mu_row) * rs_row;
        const float dz_gamma = dz_val * s_gamma[col];
        
        thread_ds += dz_gamma * x_hat;
        thread_db += dz_gamma;
        
        thread_dgamma += dz_val * x_hat;
        thread_dbeta += dz_val;
    }
    
    // Warp reduction for ds and db
    thread_ds = warp_reduce_sum(thread_ds);
    thread_db = warp_reduce_sum(thread_db);
    
    if (lane_id == 0) {
        s_ds[warp_id] = thread_ds;
        s_db[warp_id] = thread_db;
    }
    __syncthreads();
    
    // Block reduction for ds and db
    if (warp_id == 0) {
        float warp_ds = (lane_id < blockDim.x / WARP_SIZE) ? s_ds[lane_id] : 0.0f;
        float warp_db = (lane_id < blockDim.x / WARP_SIZE) ? s_db[lane_id] : 0.0f;
        
        warp_ds = warp_reduce_sum(warp_ds);
        warp_db = warp_reduce_sum(warp_db);
        
        if (lane_id == 0) {
            s_ds[0] = warp_ds;
            s_db[0] = warp_db;
        }
    }
    __syncthreads();
    
    const float ds = s_ds[0];
    const float db = s_db[0];
    const float inv_cols = 1.0f / float(cols);
    
    // Compute and write dx, also atomically accumulate dgamma and dbeta
    for (int col = tid; col < cols; col += blockDim.x) {
        const int idx = row * cols + col;
        const float dz_val = dz[idx];
        const float x_val = x[idx];
        const float x_hat = (x_val - mu_row) * rs_row;
        const float dz_gamma = dz_val * s_gamma[col];
        
        const float dx_val = rs_row * (dz_gamma - (ds * x_hat - db) * inv_cols);
        
        dx0[idx] = dx_val;
        dresidual[idx] = dx_val;
    }
    
    // Atomically add to dgamma and dbeta
    for (int col = tid; col < cols; col += blockDim.x) {
        atomicAdd(&dgamma[col], thread_dgamma);
        atomicAdd(&dbeta[col], thread_dbeta);
    }
}

// Alternative: two-pass approach without atomics in the main kernel
__global__ void layernorm_bwd_part1_kernel(
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    const float* gamma,
    float* dx0,
    float* dresidual,
    float* row_ds,
    float* row_db,
    int rows,
    int cols
) {
    // Shared memory: gamma[cols], reduction workspace
    extern __shared__ float shared_mem[];
    
    float* s_gamma = shared_mem;
    float* s_red = &shared_mem[cols]; // for reduction
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    // Load gamma
    for (int i = tid; i < cols; i += blockDim.x) {
        s_gamma[i] = gamma[i];
    }
    __syncthreads();
    
    const float mu_row = mu[row];
    const float rs_row = rs[row];
    
    // Compute dz*gamma and accumulate ds, db
    float thread_ds = 0.0f;
    float thread_db = 0.0f;
    
    for (int col = tid; col < cols; col += blockDim.x) {
        const int idx = row * cols + col;
        const float dz_val = dz[idx];
        const float x_val = x[idx];
        const float x_hat = (x_val - mu_row) * rs_row;
        const float dz_gamma = dz_val * s_gamma[col];
        
        thread_ds += dz_gamma * x_hat;
        thread_db += dz_gamma;
    }
    
    // Block reduction
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    thread_ds = warp_reduce_sum(thread_ds);
    thread_db = warp_reduce_sum(thread_db);
    
    if (lane_id == 0) {
        s_red[warp_id] = thread_ds;
        s_red[warp_id + blockDim.x / WARP_SIZE] = thread_db;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float val_ds = (lane_id < blockDim.x / WARP_SIZE) ? s_red[lane_id] : 0.0f;
        float val_db = (lane_id < blockDim.x / WARP_SIZE) ? s_red[lane_id + blockDim.x / WARP_SIZE] : 0.0f;
        
        val_ds = warp_reduce_sum(val_ds);
        val_db = warp_reduce_sum(val_db);
        
        if (lane_id == 0) {
            row_ds[row] = val_ds;
            row_db[row] = val_db;
        }
    }
}

__global__ void layernorm_bwd_part2_kernel(
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    const float* gamma,
    const float* row_ds,
    const float* row_db,
    float* dx0,
    float* dresidual,
    int rows,
    int cols
) {
    extern __shared__ float shared_mem[];
    
    float* s_gamma = shared_mem;
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    // Load gamma
    for (int i = tid; i < cols; i += blockDim.x) {
        s_gamma[i] = gamma[i];
    }
    __syncthreads();
    
    const float mu_row = mu[row];
    const float rs_row = rs[row];
    const float ds = row_ds[row];
    const float db = row_db[row];
    const float inv_cols = 1.0f / float(cols);
    
    for (int col = tid; col < cols; col += blockDim.x) {
        const int idx = row * cols + col;
        const float dz_val = dz[idx];
        const float x_val = x[idx];
        const float x_hat = (x_val - mu_row) * rs_row;
        const float dz_gamma = dz_val * s_gamma[col];
        
        const float dx_val = rs_row * (dz_gamma - (ds * x_hat - db) * inv_cols);
        
        dx0[idx] = dx_val;
        dresidual[idx] = dx_val;
    }
}

// Final optimized version: single kernel per row with proper shared memory
__global__ void layernorm_parallel_bwd_final_kernel(
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
    // Dynamic shared memory: gamma[256] + ds reduction[8] + db reduction[8] = 272 floats
    extern __shared__ float smem[];
    
    float* s_gamma = smem;                    // [cols]
    float* s_ds_warp = &smem[cols];           // [num_warps]
    float* s_db_warp = &smem[cols + 8];       // [num_warps] - assuming 8 warps
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;  // tid / 32
    const int lane_id = tid & 31;  // tid % 32
    
    if (row >= rows) return;
    
    // Cooperative load of gamma
    const int num_threads = blockDim.x;
    #pragma unroll 4
    for (int i = tid; i < cols; i += num_threads) {
        s_gamma[i] = gamma[i];
    }
    __syncthreads();
    
    const float mu_row = mu[row];
    const float rs_row = rs[row];
    
    // Accumulate ds, db, dgamma, dbeta
    float acc_ds = 0.0f;
    float acc_db = 0.0f;
    float acc_dgamma = 0.0f;
    float acc_dbeta = 0.0f;
    
    // Process all columns
    #pragma unroll 4
    for (int col = tid; col < cols; col += num_threads) {
        const int idx = row * cols + col;
        const float dz_val = dz[idx];
        const float x_val = x[idx];
        const float x_hat = (x_val - mu_row) * rs_row;
        const float dz_gamma = dz_val * s_gamma[col];
        
        acc_ds += dz_gamma * x_hat;
        acc_db += dz_gamma;
        acc_dgamma += dz_val * x_hat;
        acc_dbeta += dz_val;
    }
    
    // Warp-level reduction
    acc_ds = warp_reduce_sum(acc_ds);
    acc_db = warp_reduce_sum(acc_db);
    
    // Write warp results to shared memory
    if (lane_id == 0) {
        s_ds_warp[warp_id] = acc_ds;
        s_db_warp[warp_id] = acc_db;
    }
    __syncthreads();
    
    // Final reduction across warps (warp 0 only)
    float final_ds, final_db;
    if (warp_id == 0) {
        float val_ds = (lane_id < (num_threads >> 5)) ? s_ds_warp[lane_id] : 0.0f;
        float val_db = (lane_id < (num_threads >> 5)) ? s_db_warp[lane_id] : 0.0f;
        
        val_ds = warp_reduce_sum(val_ds);
        val_db = warp_reduce_sum(val_db);
        
        if (lane_id == 0) {
            s_ds_warp[0] = val_ds;
            s_db_warp[0] = val_db;
        }
        final_ds = val_ds;
        final_db = val_db;
    }
    __syncthreads();
    
    final_ds = s_ds_warp[0];
    final_db = s_db_warp[0];
    
    const float inv_cols = 1.0f / float(cols);
    
    // Compute and store dx, atomically accumulate dgamma/dbeta
    #pragma unroll 4
    for (int col = tid; col < cols; col += num_threads) {
        const int idx = row * cols + col;
        const float dz_val = dz[idx];
        const float x_val = x[idx];
        const float x_hat = (x_val - mu_row) * rs_row;
        const float dz_gamma = dz_val * s_gamma[col];
        
        const float dx_val = rs_row * (dz_gamma - (final_ds * x_hat + final_db) * inv_cols);
        
        dx0[idx] = dx_val;
        dresidual[idx] = dx_val;
    }
    
    // Atomic updates for dgamma and dbeta
    #pragma unroll 4
    for (int col = tid; col < cols; col += num_threads) {
        atomicAdd(&dgamma[col], acc_dgamma);
        atomicAdd(&dbeta[col], acc_dbeta);
    }
}

// Even better: avoid atomics by using separate reduction kernel for dgamma/dbeta
// But for simplicity and given the problem size, let's use the atomic version with proper initialization

__global__ void init_zeros_kernel(float* dgamma, float* dbeta, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cols) {
        dgamma[idx] = 0.0f;
        dbeta[idx] = 0.0f;
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
    // Initialize dgamma and dbeta to zero
    const int init_threads = 256;
    const int init_blocks = (cols + init_threads - 1) / init_threads;
    init_zeros_kernel<<<init_blocks, init_threads, 0, stream>>>(dgamma, dbeta, cols);
    
    // Main kernel: one block per row
    const int threads = 256; // 8 warps
    const int blocks = rows;
    
    // Shared memory: gamma[256] + ds[8] + db[8] = 272 floats = 1088 bytes
    const int smem_size = (cols + 16) * sizeof(float);
    
    layernorm_parallel_bwd_final_kernel<<<blocks, threads, smem_size, stream>>>(
        dz, x, mu, rs, gamma,
        dx0, dresidual, dgamma, dbeta,
        rows, cols
    );
}

} // extern "C"
