#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
constexpr int COLS = 1024;

// Warp-level sum reduction using shuffle
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level max reduction using shuffle (not used here but kept for completeness)
__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Kernel for computing dgamma, dbeta (reduction over rows) and dx (per-element)
// We use a two-level approach: first compute per-row partials, then reduce
template <int COLS_PER_THREAD>
__global__ void layernorm_parallel_bwd_kernel(
    const float* __restrict__ dz,
    const float* __restrict__ x,
    const float* __restrict__ mu,
    const float* __restrict__ rs,
    const float* __restrict__ gamma,
    float* __restrict__ dx0,
    float* __restrict__ dresidual,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows,
    int cols
) {
    // Each block processes one row
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Shared memory for warp-level reductions within the block
    // Layout: [num_warps] for ds_sum, [num_warps] for db_sum
    __shared__ float shared_ds[WARPS_PER_BLOCK];
    __shared__ float shared_db[WARPS_PER_BLOCK];
    __shared__ float shared_dg[WARPS_PER_BLOCK * COLS]; // For dgamma reduction across warps
    __shared__ float shared_db_col[WARPS_PER_BLOCK * COLS]; // For dbeta reduction across warps

    const float row_mu = mu[row];
    const float row_rs = rs[row];

    // Step 1: Compute x_hat and local contributions
    // Each thread processes COLS_PER_THREAD elements
    float local_ds = 0.0f;  // sum of dz * gamma * x_hat
    float local_db = 0.0f;  // sum of dz * gamma

    // For dgamma/dbeta: we need per-column sums
    // Use shared memory to accumulate per-column values from each warp
    
    // Initialize shared memory for column reductions
    // Each warp handles a subset of columns
    for (int c = tid; c < cols; c += blockDim.x) {
        shared_dg[warp_id * cols + c] = 0.0f;
        shared_db_col[warp_id * cols + c] = 0.0f;
    }
    __syncthreads();

    // Process elements in strided fashion for coalesced memory access
    for (int c = tid; c < cols; c += blockDim.x) {
        const float dz_val = dz[row * cols + c];
        const float x_val = x[row * cols + c];
        const float gamma_c = gamma[c];
        
        const float x_hat = (x_val - row_mu) * row_rs;
        
        const float dz_gamma = dz_val * gamma_c;
        
        // Accumulate for ds and db (scalar reductions)
        local_ds += dz_gamma * x_hat;
        local_db += dz_gamma;
        
        // Accumulate for dgamma and dbeta (column-wise reductions)
        // Use atomic or warp-level reduction? Better: store and reduce later
        // Actually, let's use warp shuffle for column reduction within warp first
        float dg_contrib = dz_val * x_hat;
        float db_contrib = dz_val;
        
        // Reduce within warp for this column
        // Since each thread in warp handles different columns, we need different approach
        // Instead, accumulate in shared memory per warp
        atomicAdd(&shared_dg[warp_id * cols + c], dg_contrib);
        atomicAdd(&shared_db_col[warp_id * cols + c], db_contrib);
    }

    // Warp-level reduction for ds and db
    local_ds = warp_reduce_sum(local_ds);
    local_db = warp_reduce_sum(local_db);

    // Store warp results to shared memory
    if (lane_id == 0) {
        shared_ds[warp_id] = local_ds;
        shared_db[warp_id] = local_db;
    }
    __syncthreads();

    // Reduce across warps for ds and db
    if (tid < num_warps) {
        float ds_val = shared_ds[tid];
        float db_val = shared_db[tid];
        
        ds_val = warp_reduce_sum(ds_val);
        db_val = warp_reduce_sum(db_val);
        
        if (tid == 0) {
            shared_ds[0] = ds_val;
            shared_db[0] = db_val;
        }
    }
    __syncthreads();

    const float ds = shared_ds[0];
    const float db = shared_db[0];

    // Step 2: Compute dx and write output
    // Also finalize dgamma and dbeta reduction across warps
    // First, reduce dgamma and dbeta across warps in shared memory
    if (tid < cols) {
        float dg_sum = 0.0f;
        float db_sum = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            dg_sum += shared_dg[w * cols + tid];
            db_sum += shared_db_col[w * cols + tid];
        }
        // Write to global memory using atomicAdd (since multiple blocks per column)
        atomicAdd(&dgamma[tid], dg_sum);
        atomicAdd(&dbeta[tid], db_sum);
    }

    // Compute dx for each element
    const float inv_cols = 1.0f / cols;
    for (int c = tid; c < cols; c += blockDim.x) {
        const float dz_val = dz[row * cols + c];
        const float x_val = x[row * cols + c];
        const float gamma_c = gamma[c];
        
        const float x_hat = (x_val - row_mu) * row_rs;
        const float dz_gamma = dz_val * gamma_c;
        
        // dx = rs * (dz*gamma - (ds*x_hat + db) / cols)
        const float dx_val = row_rs * (dz_gamma - (ds * x_hat + db) * inv_cols);
        
        dx0[row * cols + c] = dx_val;
        dresidual[row * cols + c] = dx_val;
    }
}

// Optimized kernel using vectorized loads for better memory throughput
template <int VEC_SIZE>
__global__ void layernorm_parallel_bwd_kernel_vec(
    const float* __restrict__ dz,
    const float* __restrict__ x,
    const float* __restrict__ mu,
    const float* __restrict__ rs,
    const float* __restrict__ gamma,
    float* __restrict__ dx0,
    float* __restrict__ dresidual,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    __shared__ float shared_ds[WARPS_PER_BLOCK];
    __shared__ float shared_db[WARPS_PER_BLOCK];
    __shared__ float shared_dg[WARPS_PER_BLOCK * COLS];
    __shared__ float shared_db_col[WARPS_PER_BLOCK * COLS];

    const float row_mu = mu[row];
    const float row_rs = rs[row];

    // Initialize shared memory
    for (int c = tid; c < cols; c += blockDim.x) {
        shared_dg[warp_id * cols + c] = 0.0f;
        shared_db_col[warp_id * cols + c] = 0.0f;
    }
    __syncthreads();

    float local_ds = 0.0f;
    float local_db = 0.0f;

    // Vectorized processing
    const int vec_cols = cols / VEC_SIZE;
    
    // Use float4 for vectorized loads when VEC_SIZE == 4
    if constexpr (VEC_SIZE == 4) {
        const float4* dz_vec = reinterpret_cast<const float4*>(dz + row * cols);
        const float4* x_vec = reinterpret_cast<const float4*>(x + row * cols);
        const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);

        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 dz4 = dz_vec[i];
            float4 x4 = x_vec[i];
            float4 g4 = gamma_vec[i];

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float dz_val = (j == 0) ? dz4.x : (j == 1) ? dz4.y : (j == 2) ? dz4.z : dz4.w;
                float x_val = (j == 0) ? x4.x : (j == 1) ? x4.y : (j == 2) ? x4.z : x4.w;
                float gamma_c = (j == 0) ? g4.x : (j == 1) ? g4.y : (j == 2) ? g4.z : g4.w;
                
                const float x_hat = (x_val - row_mu) * row_rs;
                const float dz_gamma = dz_val * gamma_c;
                
                local_ds += dz_gamma * x_hat;
                local_db += dz_gamma;
                
                int c = i * 4 + j;
                atomicAdd(&shared_dg[warp_id * cols + c], dz_val * x_hat);
                atomicAdd(&shared_db_col[warp_id * cols + c], dz_val);
            }
        }
    } else {
        // Scalar fallback
        for (int c = tid; c < cols; c += blockDim.x) {
            const float dz_val = dz[row * cols + c];
            const float x_val = x[row * cols + c];
            const float gamma_c = gamma[c];
            
            const float x_hat = (x_val - row_mu) * row_rs;
            const float dz_gamma = dz_val * gamma_c;
            
            local_ds += dz_gamma * x_hat;
            local_db += dz_gamma;
            
            atomicAdd(&shared_dg[warp_id * cols + c], dz_val * x_hat);
            atomicAdd(&shared_db_col[warp_id * cols + c], dz_val);
        }
    }

    // Warp reduction
    local_ds = warp_reduce_sum(local_ds);
    local_db = warp_reduce_sum(local_db);

    if (lane_id == 0) {
        shared_ds[warp_id] = local_ds;
        shared_db[warp_id] = local_db;
    }
    __syncthreads();

    if (tid < num_warps) {
        float ds_val = shared_ds[tid];
        float db_val = shared_db[tid];
        ds_val = warp_reduce_sum(ds_val);
        db_val = warp_reduce_sum(db_val);
        if (tid == 0) {
            shared_ds[0] = ds_val;
            shared_db[0] = db_val;
        }
    }
    __syncthreads();

    const float ds = shared_ds[0];
    const float db = shared_db[0];

    // Write dgamma, dbeta
    if (tid < cols) {
        float dg_sum = 0.0f;
        float db_sum = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            dg_sum += shared_dg[w * cols + tid];
            db_sum += shared_db_col[w * cols + tid];
        }
        atomicAdd(&dgamma[tid], dg_sum);
        atomicAdd(&dbeta[tid], db_sum);
    }

    // Compute and write dx
    const float inv_cols = 1.0f / cols;
    
    if constexpr (VEC_SIZE == 4) {
        const float4* dz_vec = reinterpret_cast<const float4*>(dz + row * cols);
        const float4* x_vec = reinterpret_cast<const float4*>(x + row * cols);
        const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
        float4* dx0_vec = reinterpret_cast<float4*>(dx0 + row * cols);
        float4* dres_vec = reinterpret_cast<float4*>(dresidual + row * cols);

        for (int i = tid; i < vec_cols; i += blockDim.x) {
            float4 dz4 = dz_vec[i];
            float4 x4 = x_vec[i];
            float4 g4 = gamma_vec[i];
            float4 dx4;

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float dz_val = (j == 0) ? dz4.x : (j == 1) ? dz4.y : (j == 2) ? dz4.z : dz4.w;
                float x_val = (j == 0) ? x4.x : (j == 1) ? x4.y : (j == 2) ? x4.z : x4.w;
                float gamma_c = (j == 0) ? g4.x : (j == 1) ? g4.y : (j == 2) ? g4.z : g4.w;
                
                const float x_hat = (x_val - row_mu) * row_rs;
                const float dz_gamma = dz_val * gamma_c;
                const float dx_val = row_rs * (dz_gamma - (ds * x_hat + db) * inv_cols);
                
                if (j == 0) dx4.x = dx_val;
                else if (j == 1) dx4.y = dx_val;
                else if (j == 2) dx4.z = dx_val;
                else dx4.w = dx_val;
            }
            dx0_vec[i] = dx4;
            dres_vec[i] = dx4;
        }
    } else {
        for (int c = tid; c < cols; c += blockDim.x) {
            const float dz_val = dz[row * cols + c];
            const float x_val = x[row * cols + c];
            const float gamma_c = gamma[c];
            
            const float x_hat = (x_val - row_mu) * row_rs;
            const float dz_gamma = dz_val * gamma_c;
            const float dx_val = row_rs * (dz_gamma - (ds * x_hat + db) * inv_cols);
            
            dx0[row * cols + c] = dx_val;
            dresidual[row * cols + c] = dx_val;
        }
    }
}

// Simple, reliable kernel without vectorization
__global__ void layernorm_parallel_bwd_kernel_simple(
    const float* __restrict__ dz,
    const float* __restrict__ x,
    const float* __restrict__ mu,
    const float* __restrict__ rs,
    const float* __restrict__ gamma,
    float* __restrict__ dx0,
    float* __restrict__ dresidual,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    __shared__ float shared_ds[WARPS_PER_BLOCK];
    __shared__ float shared_db[WARPS_PER_BLOCK];

    const float row_mu = mu[row];
    const float row_rs = rs[row];

    // First pass: compute ds and db for this row
    float local_ds = 0.0f;
    float local_db = 0.0f;

    for (int c = tid; c < cols; c += blockDim.x) {
        const float dz_val = dz[row * cols + c];
        const float x_val = x[row * cols + c];
        const float gamma_c = gamma[c];
        
        const float x_hat = (x_val - row_mu) * row_rs;
        const float dz_gamma = dz_val * gamma_c;
        
        local_ds += dz_gamma * x_hat;
        local_db += dz_gamma;
    }

    // Warp-level reduction
    local_ds = warp_reduce_sum(local_ds);
    local_db = warp_reduce_sum(local_db);

    if (lane_id == 0) {
        shared_ds[warp_id] = local_ds;
        shared_db[warp_id] = local_db;
    }
    __syncthreads();

    // Cross-warp reduction
    if (tid < num_warps) {
        float ds_val = shared_ds[tid];
        float db_val = shared_db[tid];
        ds_val = warp_reduce_sum(ds_val);
        db_val = warp_reduce_sum(db_val);
        if (tid == 0) {
            shared_ds[0] = ds_val;
            shared_db[0] = db_val;
        }
    }
    __syncthreads();

    const float ds = shared_ds[0];
    const float db = shared_db[0];
    const float inv_cols = 1.0f / cols;

    // Second pass: compute dx and accumulate dgamma/dbeta
    for (int c = tid; c < cols; c += blockDim.x) {
        const float dz_val = dz[row * cols + c];
        const float x_val = x[row * cols + c];
        const float gamma_c = gamma[c];
        
        const float x_hat = (x_val - row_mu) * row_rs;
        const float dz_gamma = dz_val * gamma_c;
        
        // dx computation
        const float dx_val = row_rs * (dz_gamma - (ds * x_hat + db) * inv_cols);
        
        dx0[row * cols + c] = dx_val;
        dresidual[row * cols + c] = dx_val;
        
        // Accumulate dgamma and dbeta using atomic operations
        atomicAdd(&dgamma[c], dz_val * x_hat);
        atomicAdd(&dbeta[c], dz_val);
    }
}

// Multi-block column reduction kernel for dgamma and dbeta
__global__ void init_zeros(float* ptr, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ptr[idx] = 0.0f;
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
    const int block_size_init = 256;
    const int num_blocks_init = (cols + block_size_init - 1) / block_size_init;
    init_zeros<<<num_blocks_init, block_size_init, 0, stream>>>(dgamma, cols);
    init_zeros<<<num_blocks_init, block_size_init, 0, stream>>>(dbeta, cols);

    // Launch main kernel - one block per row
    const int block_size = 256;
    const int num_blocks = rows;
    
    // Use simple kernel for reliability
    layernorm_parallel_bwd_kernel_simple<<<num_blocks, block_size, 0, stream>>>(
        dz, x, mu, rs, gamma, dx0, dresidual, dgamma, dbeta, rows, cols
    );
}

} // extern "C"
