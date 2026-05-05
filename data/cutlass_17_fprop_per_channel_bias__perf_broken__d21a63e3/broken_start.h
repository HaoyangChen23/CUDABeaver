#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Conv2D Forward Propagation with per-channel bias and ReLU
// output = max(0, alpha * Conv2d(input, filter) + bias)

// Block dimensions: 16x16 threads = 256 threads per block
// Each thread computes one output element (one K channel)
// Grid covers: N * P * Q output positions, with K handled in blocks

#define BLOCK_SIZE 16

__global__ void Conv2dFpropBiasReluKernel(
    __half const *input, __half const *filter,
    float const *bias, float *output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int P, int Q,
    float alpha)
{
    // Block dimensions: (BLOCK_SIZE, BLOCK_SIZE)
    // Grid covers (N*P*Q, K) in a tiled fashion
    
    int tid_x = blockIdx.x * BLOCK_SIZE + threadIdx.x; // covers output spatial (n,p,q) flattened
    int tid_y = blockIdx.y * BLOCK_SIZE + threadIdx.y; // covers K channel
    
    // Total output spatial positions
    int NPQ = N * P * Q;
    
    if (tid_x >= NPQ || tid_y >= K) return;
    
    // Decode tid_x into n, p, q
    int n = tid_x / (P * Q);
    int pq = tid_x % (P * Q);
    int p = pq / Q;
    int q = pq % Q;
    
    int k = tid_y;
    
    // Compute convolution for this output position
    float accum = 0.0f;
    
    for (int r = 0; r < R; ++r) {
        int h = p * stride_h + r - pad_h;
        if (h < 0 || h >= H) continue;
        
        for (int s = 0; s < S; ++s) {
            int w = q * stride_w + s - pad_w;
            if (w < 0 || w >= W) continue;
            
            // Input pointer for this (n,h,w)
            int input_base = ((n * H + h) * W + w) * C;
            // Filter pointer for this (k,r,s)
            int filter_base = ((k * R + r) * S + s) * C;
            
            // Inner loop over C
            #pragma unroll 4
            for (int c = 0; c < C; ++c) {
                float in_val = __half2float(input[input_base + c]);
                float f_val = __half2float(filter[filter_base + c]);
                accum += in_val * f_val;
            }
        }
    }
    
    // Add bias, scale by alpha, apply ReLU
    accum = alpha * accum + bias[k];
    accum = fmaxf(0.0f, accum);
    
    // Write output: NHWC layout for output is (N, P, Q, K)
    int output_idx = ((n * P + p) * Q + q) * K + k;
    output[output_idx] = accum;
}

// Optimized kernel using shared memory for input tiles
// This is more efficient for larger channel counts

#define TILE_SIZE 16
#define SMEM_C_TILE 32  // Process C in tiles of this size

__global__ void Conv2dFpropBiasReluKernelOptimized(
    __half const *input, __half const *filter,
    float const *bias, float *output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int P, int Q,
    float alpha)
{
    // Each block computes a TILE_SIZE x TILE_SIZE tile of output
    // Thread (tx, ty) computes output at (block_start_x + tx, block_start_y + ty)
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int block_x = blockIdx.x * TILE_SIZE;
    int block_y = blockIdx.y * TILE_SIZE;
    
    int tid_x = block_x + tx;
    int tid_y = block_y + ty;
    
    int NPQ = N * P * Q;
    
    if (tid_x >= NPQ || tid_y >= K) return;
    
    // Decode output position
    int n = tid_x / (P * Q);
    int pq = tid_x % (P * Q);
    int p = pq / Q;
    int q = pq % Q;
    int k = tid_y;
    
    float accum = 0.0f;
    
    // Precompute filter offsets for this k
    // We'll iterate over r, s, and tile over C
    
    for (int c_tile = 0; c_tile < C; c_tile += SMEM_C_TILE) {
        int c_end = min(c_tile + SMEM_C_TILE, C);
        
        // Compute partial convolution for this C tile
        for (int r = 0; r < R; ++r) {
            int h = p * stride_h + r - pad_h;
            if (h < 0 || h >= H) continue;
            
            for (int s = 0; s < S; ++s) {
                int w = q * stride_w + s - pad_w;
                if (w < 0 || w >= W) continue;
                
                int input_base = ((n * H + h) * W + w) * C;
                int filter_base = ((k * R + r) * S + s) * C;
                
                for (int c = c_tile; c < c_end; ++c) {
                    float in_val = __half2float(input[input_base + c]);
                    float f_val = __half2float(filter[filter_base + c]);
                    accum += in_val * f_val;
                }
            }
        }
    }
    
    // Apply bias, alpha, ReLU
    accum = alpha * accum + bias[k];
    accum = fmaxf(0.0f, accum);
    
    int output_idx = ((n * P + p) * Q + q) * K + k;
    output[output_idx] = accum;
}

// Even more optimized: process multiple K channels per thread to improve memory coalescing
#define K_UNROLL 4

__global__ void Conv2dFpropBiasReluKernelV2(
    __half const *input, __half const *filter,
    float const *bias, float *output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int P, int Q,
    float alpha)
{
    // Each thread processes K_UNROLL consecutive K channels
    // This improves filter memory coalescing (consecutive threads access consecutive K)
    
    int tx = threadIdx.x;  // 0 to TILE_SIZE-1, covers spatial dimension
    int ty = threadIdx.y;  // 0 to TILE_SIZE/K_UNROLL - 1, covers K dimension
    
    int block_x = blockIdx.x * TILE_SIZE;
    int block_y = blockIdx.y * TILE_SIZE;  // K is blocked by TILE_SIZE
    
    int tid_x = block_x + tx;
    int k_base = block_y + ty * K_UNROLL;
    
    int NPQ = N * P * Q;
    
    if (tid_x >= NPQ) return;
    
    // Decode output position
    int n = tid_x / (P * Q);
    int pq = tid_x % (P * Q);
    int p = pq / Q;
    int q = pq % Q;
    
    // Initialize accumulators for K_UNROLL channels
    float accum[K_UNROLL];
    #pragma unroll
    for (int i = 0; i < K_UNROLL; ++i) {
        accum[i] = 0.0f;
    }
    
    // Convolution
    for (int r = 0; r < R; ++r) {
        int h = p * stride_h + r - pad_h;
        if (h < 0 || h >= H) continue;
        
        for (int s = 0; s < S; ++s) {
            int w = q * stride_w + s - pad_w;
            if (w < 0 || w >= W) continue;
            
            int input_base = ((n * H + h) * W + w) * C;
            
            for (int c = 0; c < C; ++c) {
                float in_val = __half2float(input[input_base + c]);
                
                // Load K_UNROLL filter values
                #pragma unroll
                for (int i = 0; i < K_UNROLL; ++i) {
                    int k = k_base + i;
                    if (k < K) {
                        int filter_idx = ((k * R + r) * S + s) * C + c;
                        float f_val = __half2float(filter[filter_idx]);
                        accum[i] += in_val * f_val;
                    }
                }
            }
        }
    }
    
    // Apply bias, alpha, ReLU and write output
    #pragma unroll
    for (int i = 0; i < K_UNROLL; ++i) {
        int k = k_base + i;
        if (k < K) {
            float result = alpha * accum[i] + bias[k];
            result = fmaxf(0.0f, result);
            
            int output_idx = ((n * P + p) * Q + q) * K + k;
            output[output_idx] = result;
        }
    }
}

// Host wrapper function
cudaError_t Conv2dFpropBiasRelu(
    __half const *input, __half const *filter,
    float const *bias, float *output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    float alpha)
{
    // Calculate output dimensions
    int P = (H + 2 * pad_h - R) / stride_h + 1;
    int Q = (W + 2 * pad_w - S) / stride_w + 1;
    
    int NPQ = N * P * Q;
    
    // Choose kernel based on problem size
    // For small K, use simple kernel. For larger K, use unrolled version.
    
    cudaError_t err;
    
    if (K <= 64) {
        // Simple kernel: each thread handles one K
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((NPQ + BLOCK_SIZE - 1) / BLOCK_SIZE, (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        Conv2dFpropBiasReluKernel<<<grid, block>>>(
            input, filter, bias, output,
            N, C, H, W, K, R, S,
            pad_h, pad_w, stride_h, stride_w,
            P, Q, alpha);
    } else {
        // Optimized kernel with K unrolling
        const int TILE_SIZE_V2 = 16;
        const int K_UNROLL_VAL = 4;
        
        dim3 block(TILE_SIZE_V2, TILE_SIZE_V2 / K_UNROLL_VAL);
        dim3 grid((NPQ + TILE_SIZE_V2 - 1) / TILE_SIZE_V2, 
                  (K + TILE_SIZE_V2 - 1) / TILE_SIZE_V2);
        
        Conv2dFpropBiasReluKernelV2<<<grid, block>>>(
            input, filter, bias, output,
            N, C, H, W, K, R, S,
            pad_h, pad_w, stride_h, stride_w,
            P, Q, alpha);
    }
    
    err = cudaGetLastError();
    return err;
}
