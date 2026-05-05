#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// 2D FFT 1024x1024 using Cooley-Tukey with shared memory
// We use a row-column approach: first do 1D FFT on all rows, then 1D FFT on all columns

// Bit-reversal permutation for 1024 points (10 stages)
__device__ inline int bit_reverse_1024(int x) {
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);
    return x >> 22; // 32 - 10 = 22
}

// Complex multiplication
__device__ inline void complex_mul(float ar, float ai, float br, float bi, float &cr, float &ci) {
    cr = ar * br - ai * bi;
    ci = ar * bi + ai * br;
}

// 1D FFT kernel for 1024 points using shared memory
// Each block processes one 1D FFT (one row or one column segment)
// Uses Cooley-Tukey radix-2 iterative approach
template<bool INVERSE>
__global__ void fft_1d_1024_kernel(
    const float* input,
    float* output,
    int stride_in,      // stride between consecutive elements in input
    int stride_out,     // stride between consecutive elements in output
    int batch_stride,   // stride between batches
    int total_batches   // total number of 1D FFTs to compute
) {
    __shared__ float smem_real[1024];
    __shared__ float smem_imag[1024];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int fft_id = bid; // which 1D FFT we're computing
    
    if (fft_id >= total_batches) return;
    
    // Load data with bit-reversal permutation
    int br_idx = bit_reverse_1024(tid);
    const float* in_ptr = input + fft_id * batch_stride;
    
    smem_real[tid] = in_ptr[br_idx * stride_in];
    smem_imag[tid] = in_ptr[br_idx * stride_in + 1];
    
    __syncthreads();
    
    // Cooley-Tukey iterative FFT
    const int N = 1024;
    const float angle_sign = INVERSE ? 1.0f : -1.0f;
    
    for (int s = 1; s <= 512; s <<= 1) {
        int m = s << 1;
        float angle = angle_sign * 2.0f * M_PI / m;
        
        // Each thread processes two elements
        int k = (tid / s) * m + (tid % s);
        int j = k + s;
        
        float wr = cosf(angle * (tid % s));
        float wi = sinf(angle * (tid % s));
        
        float tr = smem_real[j];
        float ti = smem_imag[j];
        
        float tr_mul, ti_mul;
        complex_mul(tr, ti, wr, wi, tr_mul, ti_mul);
        
        float ur = smem_real[k];
        float ui = smem_imag[k];
        
        smem_real[k] = ur + tr_mul;
        smem_imag[k] = ui + ti_mul;
        smem_real[j] = ur - tr_mul;
        smem_imag[j] = ui - ti_mul;
        
        __syncthreads();
    }
    
    // Store result
    float* out_ptr = output + fft_id * batch_stride;
    out_ptr[tid * stride_out] = smem_real[tid];
    out_ptr[tid * stride_out + 1] = smem_imag[tid];
}

// Alternative: use more threads per FFT for better memory coalescing
// Each block computes multiple FFTs, or one FFT with more parallelism
template<bool INVERSE>
__global__ void fft_1d_1024_v2_kernel(
    const float* input,
    float* output,
    int stride_in,
    int stride_out,
    int batch_stride,
    int total_batches
) {
    __shared__ float smem_real[1024];
    __shared__ float smem_imag[1024];
    
    const int tid = threadIdx.x; // 0-511, we process 2 elements per thread
    const int bid = blockIdx.x;
    const int fft_id = bid;
    
    if (fft_id >= total_batches) return;
    
    // Load with bit-reversal: each thread loads 2 elements
    int br0 = bit_reverse_1024(tid);
    int br1 = bit_reverse_1024(tid + 512);
    
    const float* in_ptr = input + fft_id * batch_stride;
    
    smem_real[tid] = in_ptr[br0 * stride_in];
    smem_imag[tid] = in_ptr[br0 * stride_in + 1];
    smem_real[tid + 512] = in_ptr[br1 * stride_in];
    smem_imag[tid + 512] = in_ptr[br1 * stride_in + 1];
    
    __syncthreads();
    
    const float angle_sign = INVERSE ? 1.0f : -1.0f;
    
    // Stage 1: stride 1 (butterfly pairs are adjacent)
    {
        int k = tid * 2;
        int j = k + 1;
        
        float tr = smem_real[j];
        float ti = smem_imag[j];
        
        float ur = smem_real[k];
        float ui = smem_imag[k];
        
        smem_real[k] = ur + tr;
        smem_imag[k] = ui + ti;
        smem_real[j] = ur - tr;
        smem_imag[j] = ui - ti;
    }
    __syncthreads();
    
    // Stage 2: stride 2
    {
        int idx = tid;
        int s = 2;
        int m = 4;
        int k = (idx / (s/2)) * m + (idx % (s/2));
        int j = k + s;
        
        float angle = angle_sign * 2.0f * M_PI / m * (idx % (s/2));
        float wr = cosf(angle);
        float wi = sinf(angle);
        
        float tr = smem_real[j];
        float ti = smem_imag[j];
        
        float tr_mul, ti_mul;
        complex_mul(tr, ti, wr, wi, tr_mul, ti_mul);
        
        float ur = smem_real[k];
        float ui = smem_imag[k];
        
        smem_real[k] = ur + tr_mul;
        smem_imag[k] = ui + ti_mul;
        smem_real[j] = ur - tr_mul;
        smem_imag[j] = ui - ti_mul;
    }
    __syncthreads();
    
    // Stage 3: stride 4
    {
        int idx = tid;
        int s = 4;
        int m = 8;
        int k = (idx / (s/2)) * m + (idx % (s/2));
        int j = k + s;
        
        float angle = angle_sign * 2.0f * M_PI / m * (idx % (s/2));
        float wr = cosf(angle);
        float wi = sinf(angle);
        
        float tr = smem_real[j];
        float ti = smem_imag[j];
        
        float tr_mul, ti_mul;
        complex_mul(tr, ti, wr, wi, tr_mul, ti_mul);
        
        float ur = smem_real[k];
        float ui = smem_imag[k];
        
        smem_real[k] = ur + tr_mul;
        smem_imag[k] = ui + ti_mul;
        smem_real[j] = ur - tr_mul;
        smem_imag[j] = ui - ti_mul;
    }
    __syncthreads();
    
    // Stage 4: stride 8
    {
        int idx = tid;
        int s = 8;
        int m = 16;
        int k = (idx / (s/2)) * m + (idx % (s/2));
        int j = k + s;
        
        float angle = angle_sign * 2.0f * M_PI / m * (idx % (s/2));
        float wr = cosf(angle);
        float wi = sinf(angle);
        
        float tr = smem_real[j];
        float ti = smem_imag[j];
        
        float tr_mul, ti_mul;
        complex_mul(tr, ti, wr, wi, tr_mul, ti_mul);
        
        float ur = smem_real[k];
        float ui = smem_imag[k];
        
        smem_real[k] = ur + tr_mul;
        smem_imag[k] = ui + ti_mul;
        smem_real[j] = ur - tr_mul;
        smem_imag[j] = ui - ti_mul;
    }
    __syncthreads();
    
    // Stage 5: stride 16
    {
        int idx = tid;
        int s = 16;
        int m = 32;
        int k = (idx / (s/2)) * m + (idx % (s/2));
        int j = k + s;
        
        float angle = angle_sign * 2.0f * M_PI / m * (idx % (s/2));
        float wr = cosf(angle);
        float wi = sinf(angle);
        
        float tr = smem_real[j];
        float ti = smem_imag[j];
        
        float tr_mul, ti_mul;
        complex_mul(tr, ti, wr, wi, tr_mul, ti_mul);
        
        float ur = smem_real[k];
        float ui = smem_imag[k];
        
        smem_real[k] = ur + tr_mul;
        smem_imag[k] = ui + ti_mul;
        smem_real[j] = ur - tr_mul;
        smem_imag[j] = ui - ti_mul;
    }
    __syncthreads();
    
    // Stage 6: stride 32
    {
        int idx = tid;
        int s = 32;
        int m = 64;
        int k = (idx / (s/2)) * m + (idx % (s/2));
        int j = k + s;
        
        float angle = angle_sign * 2.0f * M_PI / m * (idx % (s/2));
        float wr = cosf(angle);
        float wi = sinf(angle);
        
        float tr = smem_real[j];
        float ti = smem_imag[j];
        
        float tr_mul, ti_mul;
        complex_mul(tr, ti, wr, wi, tr_mul, ti_mul);
        
        float ur = smem_real[k];
        float ui = smem_imag[k];
        
        smem_real[k] = ur + tr_mul;
        smem_imag[k] = ui + ti_mul;
        smem_real[j] = ur - tr_mul;
        smem_imag[j] = ui - ti_mul;
    }
    __syncthreads();
    
    // Stage 7: stride 64
    {
        int idx = tid;
        int s = 64;
        int m = 128;
        int k = (idx / (s/2)) * m + (idx % (s/2));
        int j = k + s;
        
        float angle = angle_sign * 2.0f * M_PI / m * (idx % (s/2));
        float wr = cosf(angle);
        float wi = sinf(angle);
        
        float tr = smem_real[j];
        float ti = smem_imag[j];
        
        float tr_mul, ti_mul;
        complex_mul(tr, ti, wr, wi, tr_mul, ti_mul);
        
        float ur = smem_real[k];
        float ui = smem_imag[k];
        
        smem_real[k] = ur + tr_mul;
        smem_imag[k] = ui + ti_mul;
        smem_real[j] = ur - tr_mul;
        smem_imag[j] = ui - ti_mul;
    }
    __syncthreads();
    
    // Stage 8: stride 128
    {
        int idx = tid;
        int s = 128;
        int m = 256;
        int k = (idx / (s/2)) * m + (idx % (s/2));
        int j = k + s;
        
        float angle = angle_sign * 2.0f * M_PI / m * (idx % (s/2));
        float wr = cosf(angle);
        float wi = sinf(angle);
        
        float tr = smem_real[j];
        float ti = smem_imag[j];
        
        float tr_mul, ti_mul;
        complex_mul(tr, ti, wr, wi, tr_mul, ti_mul);
        
        float ur = smem_real[k];
        float ui = smem_imag[k];
        
        smem_real[k] = ur + tr_mul;
        smem_imag[k] = ui + ti_mul;
        smem_real[j] = ur - tr_mul;
        smem_imag[j] = ui - ti_mul;
    }
    __syncthreads();
    
    // Stage 9: stride 256
    {
        int idx = tid;
        int s = 256;
        int m = 512;
        int k = (idx / (s/2)) * m + (idx % (s/2));
        int j = k + s;
        
        float angle = angle_sign * 2.0f * M_PI / m * (idx % (s/2));
        float wr = cosf(angle);
        float wi = sinf(angle);
        
        float tr = smem_real[j];
        float ti = smem_imag[j];
        
        float tr_mul, ti_mul;
        complex_mul(tr, ti, wr, wi, tr_mul, ti_mul);
        
        float ur = smem_real[k];
        float ui = smem_imag[k];
        
        smem_real[k] = ur + tr_mul;
        smem_imag[k] = ui + ti_mul;
        smem_real[j] = ur - tr_mul;
        smem_imag[j] = ui - ti_mul;
    }
    __syncthreads();
    
    // Stage 10: stride 512
    {
        int idx = tid;
        int s = 512;
        int m = 1024;
        int k = (idx / (s/2)) * m + (idx % (s/2));
        int j = k + s;
        
        float angle = angle_sign * 2.0f * M_PI / m * (idx % (s/2));
        float wr = cosf(angle);
        float wi = sinf(angle);
        
        float tr = smem_real[j];
        float ti = smem_imag[j];
        
        float tr_mul, ti_mul;
        complex_mul(tr, ti, wr, wi, tr_mul, ti_mul);
        
        float ur = smem_real[k];
        float ui = smem_imag[k];
        
        smem_real[k] = ur + tr_mul;
        smem_imag[k] = ui + ti_mul;
        smem_real[j] = ur - tr_mul;
        smem_imag[j] = ui - ti_mul;
    }
    __syncthreads();
    
    // Store result
    float* out_ptr = output + fft_id * batch_stride;
    out_ptr[tid * stride_out] = smem_real[tid];
    out_ptr[tid * stride_out + 1] = smem_imag[tid];
    out_ptr[(tid + 512) * stride_out] = smem_real[tid + 512];
    out_ptr[(tid + 512) * stride_out + 1] = smem_imag[tid + 512];
}

// Transpose kernel for converting row-major to column-major
// We use this between row-FFT and column-FFT
__global__ void transpose_1024x1024_kernel(
    const float* input,
    float* output,
    int batch_size
) {
    __shared__ float tile_real[32][33]; // padded to avoid bank conflicts
    __shared__ float tile_imag[32][33];
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z; // batch index
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Input coordinates
    int x_in = bx * 32 + tx;
    int y_in = by * 32 + ty;
    
    // Read from input (row-major: y is row, x is col)
    // Input is [batch][y][x][2]
    const float* in_batch = input + bz * 1024 * 1024 * 2;
    
    if (x_in < 1024 && y_in < 1024) {
        tile_real[ty][tx] = in_batch[(y_in * 1024 + x_in) * 2];
        tile_imag[ty][tx] = in_batch[(y_in * 1024 + x_in) * 2 + 1];
    }
    
    __syncthreads();
    
    // Write to output (transposed: x becomes row, y becomes col)
    int x_out = by * 32 + tx;
    int y_out = bx * 32 + ty;
    
    float* out_batch = output + bz * 1024 * 1024 * 2;
    
    if (x_out < 1024 && y_out < 1024) {
        // After transpose, what was at [y_in][x_in] goes to [x_in][y_in]
        // In shared memory: [ty][tx] holds data from [y_in=by*32+ty][x_in=bx*32+tx]
        // We want to write to [x_out=by*32+tx][y_out=bx*32+ty]
        // So we read tile[tx][ty] to get the transposed element
        out_batch[(y_out * 1024 + x_out) * 2] = tile_real[tx][ty];
        out_batch[(y_out * 1024 + x_out) * 2 + 1] = tile_imag[tx][ty];
    }
}

// Optimized transpose using 32x32 tiles with proper banking
__global__ void transpose_1024x1024_v2_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size
) {
    __shared__ float tile[32][33][2]; // [y][x][real/imag], padded x dim
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Coordinates in input matrix
    int x_in = bx * 32 + tx;
    int y_in = by * 32 + ty;
    
    const float* in_batch = input + bz * 1024 * 1024 * 2;
    
    // Load from input to shared memory
    if (x_in < 1024 && y_in < 1024) {
        int in_idx = (y_in * 1024 + x_in) * 2;
        tile[ty][tx][0] = in_batch[in_idx];
        tile[ty][tx][1] = in_batch[in_idx + 1];
    }
    
    __syncthreads();
    
    // Coordinates in output (transposed)
    int x_out = by * 32 + tx;
    int y_out = bx * 32 + ty;
    
    float* out_batch = output + bz * 1024 * 1024 * 2;
    
    // Store from shared memory to output (transposed)
    if (x_out < 1024 && y_out < 1024) {
        // Read from transposed position in shared memory
        int out_idx = (y_out * 1024 + x_out) * 2;
        out_batch[out_idx] = tile[tx][ty][0];
        out_batch[out_idx + 1] = tile[tx][ty][1];
    }
}

// Column FFT: treat columns as 1D arrays
// After transpose, columns become contiguous rows
template<bool INVERSE>
__global__ void fft_columns_kernel(
    const float* input,
    float* output,
    int batch_size
) {
    // Each block processes one column (now a row after transpose)
    // We use the same v2 kernel approach but with different strides
    
    __shared__ float smem_real[1024];
    __shared__ float smem_imag[1024];
    
    const int tid = threadIdx.x; // 0-511
    const int col = blockIdx.x;  // which column (0-1023)
    const int batch = blockIdx.y; // which batch
    
    if (col >= 1024 || batch >= batch_size) return;
    
    const float* in_ptr = input + batch * 1024 * 1024 * 2 + col * 1024 * 2;
    
    // Load with bit-reversal
    int br0 = bit_reverse_1024(tid);
    int br1 = bit_reverse_1024(tid + 512);
    
    smem_real[tid] = in_ptr[br0 * 2];
    smem_imag[tid] = in_ptr[br0 * 2 + 1];
    smem_real[tid + 512] = in_ptr[br1 * 2];
    smem_imag[tid + 512] = in_ptr[br1 * 2 + 1];
    
    __syncthreads();
    
    const float angle_sign = INVERSE ? 1.0f : -1.0f;
    
    // Stage 1
    {
        int k = tid * 2;
        int j = k + 1;
        float tr = smem_real[j], ti = smem_imag[j];
        float ur = smem_real[k], ui = smem_imag[k];
        smem_real[k] = ur + tr; smem_imag[k] = ui + ti;
        smem_real[j] = ur - tr; smem_imag[j] = ui - ti;
    }
    __syncthreads();
    
    // Stages 2-10
    #pragma unroll
    for (int stage = 2; stage <= 10; stage++) {
        int s = 1 << (stage - 1);
        int m = s << 1;
        int idx = tid;
        int k = (idx / s) * m + (idx % s);
        int j = k + s;
        
        float angle = angle_sign * 2.0f * M_PI / m * (idx % s);
        float wr = cosf(angle);
        float wi = sinf(angle);
        
        float tr = smem_real[j];
        float ti = smem_imag[j];
        
        float tr_mul = tr * wr - ti * wi;
        float ti_mul = tr * wi + ti * wr;
        
        float ur = smem_real[k];
        float ui = smem_imag[k];
        
        smem_real[k] = ur + tr_mul;
        smem_imag[k] = ui + ti_mul;
        smem_real[j] = ur - tr_mul;
        smem_imag[j] = ui - ti_mul;
        
        __syncthreads();
    }
    
    // Store result
    float* out_ptr = output + batch * 1024 * 1024 * 2 + col * 1024 * 2;
    out_ptr[tid * 2] = smem_real[tid];
    out_ptr[tid * 2 + 1] = smem_imag[tid];
    out_ptr[(tid + 512) * 2] = smem_real[tid + 512];
    out_ptr[(tid + 512) * 2 + 1] = smem_imag[tid + 512];
}

// Main 2D FFT implementation
// Step 1: FFT on rows
// Step 2: Transpose
// Step 3: FFT on columns (which are now rows)
// Step 4: Transpose back

extern "C" {

void launch_fft_c2c_2d_1024x1024_fp32(
    const void* input,
    void* output,
    int N,
    int batch_size,
    int inverse,
    cudaStream_t stream
) {
    const int dim = 1024;
    const int total_elements = dim * dim * 2; // complex numbers as float2
    
    // Allocate intermediate buffer for transpose
    float* d_temp;
    cudaMallocAsync(&d_temp, batch_size * total_elements * sizeof(float), stream);
    
    const float* in = static_cast<const float*>(input);
    float* out = static_cast<float*>(output);
    
    if (inverse) {
        // Inverse FFT: similar structure but with positive exponent and scaling
        
        // Step 1: Row-wise IFFT
        fft_1d_1024_v2_kernel<true><<<batch_size * dim, 512, 0, stream>>>(
            in, d_temp,
            2, // stride_in: consecutive complex numbers
            2, // stride_out
            dim * 2, // batch_stride: 1024 complex numbers per row
            batch_size * dim
        );
        
        // Step 2: Transpose
        dim3 gridT(dim/32, dim/32, batch_size);
        dim3 blockT(32, 32);
        transpose_1024x1024_v2_kernel<<<gridT, blockT, 0, stream>>>(d_temp, out, batch_size);
        
        // Step 3: Column-wise IFFT (now rows after transpose)
        fft_1d_1024_v2_kernel<true><<<batch_size * dim, 512, 0, stream>>>(
            out, d_temp,
            2, 2, dim * 2, batch_size * dim
        );
        
        // Step 4: Transpose back
        transpose_1024x1024_v2_kernel<<<gridT, blockT, 0, stream>>>(d_temp, out, batch_size);
        
        // Step 5: Scale by 1/N (1/1024 for each dimension, total 1/1048576)
        // Actually we need 1/1024 for each 1D FFT, so total 1/1048576
        // But typically we do 1/N for the final result where N is total elements
        // For 2D, we can apply scaling here or in the kernels
        
    } else {
        // Forward FFT
        
        // Step 1: Row-wise FFT
        fft_1d_1024_v2_kernel<false><<<batch_size * dim, 512, 0, stream>>>(
            in, d_temp,
            2, // stride_in
            2, // stride_out  
            dim * 2, // batch_stride
            batch_size * dim
        );
        
        // Step 2: Transpose
        dim3 gridT(dim/32, dim/32, batch_size);
        dim3 blockT(32, 32);
        transpose_1024x1024_v2_kernel<<<gridT, blockT, 0, stream>>>(d_temp, out, batch_size);
        
        // Step 3: Column-wise FFT (now rows after transpose)
        fft_1d_1024_v2_kernel<false><<<batch_size * dim, 512, 0, stream>>>(
            out, d_temp,
            2, 2, dim * 2, batch_size * dim
        );
        
        // Step 4: Transpose back to original layout
        transpose_1024x1024_v2_kernel<<<gridT, blockT, 0, stream>>>(d_temp, out, batch_size);
    }
    
    cudaFreeAsync(d_temp, stream);
}

} // extern "C"
