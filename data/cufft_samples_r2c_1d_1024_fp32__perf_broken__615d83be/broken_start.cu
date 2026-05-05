#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Constants for FFT size 1024
#define FFT_SIZE 1024
#define HALF_SIZE 513  // 1024/2 + 1
#define LOG2_SIZE 10

// Twiddle factors precomputed for each stage
// Using shared memory for twiddle factors to avoid recomputation

// Bit reversal permutation for 1024 elements
__device__ inline int bit_reversal_10(int n) {
    // Reverse 10 bits
    n = ((n >> 1) & 0x55555555) | ((n & 0x55555555) << 1);
    n = ((n >> 2) & 0x33333333) | ((n & 0x33333333) << 2);
    n = ((n >> 4) & 0x0F0F0F0F) | ((n & 0x0F0F0F0F) << 4);
    n = ((n >> 8) & 0x00FF00FF) | ((n & 0x00FF00FF) << 8);
    return n >> 22;  // Keep only lower 10 bits
}

// Complex multiply
__device__ inline void complex_mul(float& r_out, float& i_out, float r1, float i1, float r2, float i2) {
    r_out = r1 * r2 - i1 * i2;
    i_out = r1 * i2 + i1 * r2;
}

// Complex multiply with conjugate (for inverse, but we only do forward here)
__device__ inline void complex_mul_conj(float& r_out, float& i_out, float r1, float i1, float r2, float i2) {
    r_out = r1 * r2 + i1 * i2;
    i_out = -r1 * i2 + i1 * r2;
}

// Butterfly operation for Cooley-Tukey FFT
// In-place DFT-2: (a, b) -> (a + w*b, a - w*b) where w is twiddle factor
__device__ inline void butterfly(float& a_r, float& a_i, float& b_r, float& b_i, 
                                  float w_r, float w_i) {
    float t_r = b_r * w_r - b_i * w_i;
    float t_i = b_r * w_i + b_i * w_r;
    float new_a_r = a_r + t_r;
    float new_a_i = a_i + t_i;
    float new_b_r = a_r - t_r;
    float new_b_i = a_i - t_i;
    a_r = new_a_r;
    a_i = new_a_i;
    b_r = new_b_r;
    b_i = new_b_i;
}

// Stockham-style FFT kernel for 1D R2C
// Each block processes one batch element
// Uses shared memory for the entire FFT
__global__ void fft_r2c_1d_1024_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int N,
                                        int batch_size) {
    // Shared memory: 1024 complex numbers = 2048 floats
    // Layout: interleaved real/imag
    __shared__ float smem[2048];
    
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * N;
    
    // Load data with bit-reversal permutation
    // For R2C, we load real data and zero imaginary part
    #pragma unroll 8
    for (int i = threadIdx.x; i < FFT_SIZE; i += blockDim.x) {
        int rev_i = bit_reversal_10(i);
        smem[2 * i] = batch_input[rev_i];      // real
        smem[2 * i + 1] = 0.0f;                // imag
    }
    
    __syncthreads();
    
    // Cooley-Tukey FFT with 10 stages
    // Stage s: groups of size 2^s, butterflies within groups
    // Twiddle factor: exp(-2*pi*i * (j * 2^s) / N) for j-th element in group
    
    for (int s = 0; s < LOG2_SIZE; s++) {
        int m = 1 << (s + 1);  // group size
        int half_m = m >> 1;    // half group size
        
        // Each thread processes multiple butterflies
        // Total butterflies per stage: N/2 = 512
        // Each butterfly: (a, b) with stride half_m
        
        for (int idx = threadIdx.x; idx < (FFT_SIZE >> 1); idx += blockDim.x) {
            // Find which group and position within group
            int group = idx / half_m;
            int j = idx % half_m;
            
            int a_idx = group * m + j;
            int b_idx = a_idx + half_m;
            
            // Compute twiddle factor W_N^(j * 2^s) = exp(-2*pi*i * j * 2^s / N)
            // = exp(-2*pi*i * j / m)
            float angle = -2.0f * M_PI * (float)j / (float)m;
            float w_r = cosf(angle);
            float w_i = sinf(angle);
            
            float a_r = smem[2 * a_idx];
            float a_i = smem[2 * a_idx + 1];
            float b_r = smem[2 * b_idx];
            float b_i = smem[2 * b_idx + 1];
            
            butterfly(a_r, a_i, b_r, b_i, w_r, w_i);
            
            smem[2 * a_idx] = a_r;
            smem[2 * a_idx + 1] = a_i;
            smem[2 * b_idx] = b_r;
            smem[2 * b_idx + 1] = b_i;
        }
        
        __syncthreads();
    }
    
    // Write output: first HALF_SIZE complex numbers (0 to 512)
    // Output format: interleaved real/imag
    float* batch_output = output + batch_idx * HALF_SIZE * 2;
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < HALF_SIZE; i += blockDim.x) {
        batch_output[2 * i] = smem[2 * i];       // real
        batch_output[2 * i + 1] = smem[2 * i + 1]; // imag
    }
}

// Optimized version using more threads and better memory access
__global__ void fft_r2c_1d_1024_kernel_optimized(const float* __restrict__ input,
                                                  float* __restrict__ output,
                                                  int N,
                                                  int batch_size) {
    // Use 512 threads per block, each handles 2 elements
    __shared__ float smem[2048];
    
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * N;
    
    // Each thread loads 2 elements with bit-reversal
    int tid = threadIdx.x;
    int lane = tid & 31;  // lane in warp
    int warp_id = tid >> 5;
    
    // Load: each thread handles 2 elements (1024 elements / 512 threads = 2 per thread)
    #pragma unroll 2
    for (int k = 0; k < 2; k++) {
        int i = tid + k * 512;
        if (i < FFT_SIZE) {
            int rev_i = bit_reversal_10(i);
            smem[2 * i] = batch_input[rev_i];
            smem[2 * i + 1] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // FFT stages
    for (int s = 0; s < LOG2_SIZE; s++) {
        int m = 1 << (s + 1);
        int half_m = m >> 1;
        
        // Each thread processes butterflies
        // Total butterflies: 512
        for (int idx = tid; idx < (FFT_SIZE >> 1); idx += blockDim.x) {
            int group = idx / half_m;
            int j = idx % half_m;
            
            int a_idx = group * m + j;
            int b_idx = a_idx + half_m;
            
            float angle = -2.0f * M_PI * (float)j / (float)m;
            float w_r = cosf(angle);
            float w_i = sinf(angle);
            
            float a_r = smem[2 * a_idx];
            float a_i = smem[2 * a_idx + 1];
            float b_r = smem[2 * b_idx];
            float b_i = smem[2 * b_idx + 1];
            
            float t_r = b_r * w_r - b_i * w_i;
            float t_i = b_r * w_i + b_i * w_r;
            
            smem[2 * a_idx] = a_r + t_r;
            smem[2 * a_idx + 1] = a_i + t_i;
            smem[2 * b_idx] = a_r - t_r;
            smem[2 * b_idx + 1] = a_i - t_i;
        }
        
        __syncthreads();
    }
    
    // Write output (first 513 complex values)
    float* batch_output = output + batch_idx * HALF_SIZE * 2;
    
    #pragma unroll 2
    for (int k = 0; k < 2; k++) {
        int i = tid + k * 512;
        if (i < HALF_SIZE) {
            batch_output[2 * i] = smem[2 * i];
            batch_output[2 * i + 1] = smem[2 * i + 1];
        }
    }
}

// Even more optimized: use warp shuffle and reduce synchronization
// Stockham autosort algorithm - no bit reversal needed
__global__ void fft_r2c_1d_1024_kernel_stockham(const float* __restrict__ input,
                                                 float* __restrict__ output,
                                                 int N,
                                                 int batch_size) {
    // Stockham FFT: uses two buffers and avoids bit-reversal
    // But we need bit-reversal for output, so we stick to Cooley-Tukey
    
    __shared__ float smem[2048];
    
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * N;
    
    // Load with bit-reversal
    for (int i = threadIdx.x; i < FFT_SIZE; i += blockDim.x) {
        int rev_i = bit_reversal_10(i);
        smem[2 * i] = batch_input[rev_i];
        smem[2 * i + 1] = 0.0f;
    }
    
    __syncthreads();
    
    // FFT stages - unrolled for better performance
    // Stage 0: groups of 2
    // Stage 1: groups of 4
    // ... up to groups of 1024
    
    // Stage 0 (m=2)
    for (int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        int a_idx = 2 * idx;
        int b_idx = a_idx + 1;
        
        float a_r = smem[2 * a_idx];
        float a_i = smem[2 * a_idx + 1];
        float b_r = smem[2 * b_idx];
        float b_i = smem[2 * b_idx + 1];
        
        // W = 1 for stage 0 (j=0 always)
        smem[2 * a_idx] = a_r + b_r;
        smem[2 * a_idx + 1] = a_i + b_i;
        smem[2 * b_idx] = a_r - b_r;
        smem[2 * b_idx + 1] = a_i - b_i;
    }
    __syncthreads();
    
    // Stage 1 (m=4)
    for (int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        int group = idx / 2;
        int j = idx % 2;
        int a_idx = group * 4 + j;
        int b_idx = a_idx + 2;
        
        float angle = -M_PI * j;  // -2*pi*j/4 = -pi*j/2, but j in {0,1}, so 0 or -pi/2? No wait...
        // Actually: angle = -2*pi * j / 4 = -pi * j / 2
        // For j=0: 0, for j=1: -pi/2
        angle = -M_PI * 0.5f * j;
        float w_r = cosf(angle);
        float w_i = sinf(angle);
        
        float a_r = smem[2 * a_idx];
        float a_i = smem[2 * a_idx + 1];
        float b_r = smem[2 * b_idx];
        float b_i = smem[2 * b_idx + 1];
        
        float t_r = b_r * w_r - b_i * w_i;
        float t_i = b_r * w_i + b_i * w_r;
        
        smem[2 * a_idx] = a_r + t_r;
        smem[2 * a_idx + 1] = a_i + t_i;
        smem[2 * b_idx] = a_r - t_r;
        smem[2 * b_idx + 1] = a_i - t_i;
    }
    __syncthreads();
    
    // Remaining stages using loop
    for (int s = 2; s < LOG2_SIZE; s++) {
        int m = 1 << (s + 1);
        int half_m = m >> 1;
        
        for (int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
            int group = idx / half_m;
            int j = idx % half_m;
            
            int a_idx = group * m + j;
            int b_idx = a_idx + half_m;
            
            float angle = -2.0f * M_PI * (float)j / (float)m;
            float w_r = cosf(angle);
            float w_i = sinf(angle);
            
            float a_r = smem[2 * a_idx];
            float a_i = smem[2 * a_idx + 1];
            float b_r = smem[2 * b_idx];
            float b_i = smem[2 * b_idx + 1];
            
            float t_r = b_r * w_r - b_i * w_i;
            float t_i = b_r * w_i + b_i * w_r;
            
            smem[2 * a_idx] = a_r + t_r;
            smem[2 * a_idx + 1] = a_i + t_i;
            smem[2 * b_idx] = a_r - t_r;
            smem[2 * b_idx + 1] = a_i - t_i;
        }
        __syncthreads();
    }
    
    // Write output
    float* batch_output = output + batch_idx * HALF_SIZE * 2;
    for (int i = threadIdx.x; i < HALF_SIZE; i += blockDim.x) {
        batch_output[2 * i] = smem[2 * i];
        batch_output[2 * i + 1] = smem[2 * i + 1];
    }
}

// Main kernel using constant memory for twiddle factors
__constant__ float c_twiddle_r[512];
__constant__ float c_twiddle_i[512];

__global__ void fft_r2c_1d_1024_kernel_const(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              int N,
                                              int batch_size) {
    __shared__ float smem[2048];
    
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * N;
    
    // Load with bit-reversal
    for (int i = threadIdx.x; i < FFT_SIZE; i += blockDim.x) {
        int rev_i = bit_reversal_10(i);
        smem[2 * i] = batch_input[rev_i];
        smem[2 * i + 1] = 0.0f;
    }
    
    __syncthreads();
    
    // FFT stages
    for (int s = 0; s < LOG2_SIZE; s++) {
        int m = 1 << (s + 1);
        int half_m = m >> 1;
        int twiddle_stride = FFT_SIZE / m;
        
        for (int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
            int group = idx / half_m;
            int j = idx % half_m;
            
            int a_idx = group * m + j;
            int b_idx = a_idx + half_m;
            
            // Get twiddle factor from constant memory
            int twiddle_idx = j * twiddle_stride;
            float w_r = c_twiddle_r[twiddle_idx];
            float w_i = c_twiddle_i[twiddle_idx];
            
            float a_r = smem[2 * a_idx];
            float a_i = smem[2 * a_idx + 1];
            float b_r = smem[2 * b_idx];
            float b_i = smem[2 * b_idx + 1];
            
            float t_r = b_r * w_r - b_i * w_i;
            float t_i = b_r * w_i + b_i * w_r;
            
            smem[2 * a_idx] = a_r + t_r;
            smem[2 * a_idx + 1] = a_i + t_i;
            smem[2 * b_idx] = a_r - t_r;
            smem[2 * b_idx + 1] = a_i - t_i;
        }
        __syncthreads();
    }
    
    // Write output
    float* batch_output = output + batch_idx * HALF_SIZE * 2;
    for (int i = threadIdx.x; i < HALF_SIZE; i += blockDim.x) {
        batch_output[2 * i] = smem[2 * i];
        batch_output[2 * i + 1] = smem[2 * i + 1];
    }
}

// Simple, reliable implementation
__global__ void fft_r2c_1d_1024_kernel_simple(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               int N,
                                               int batch_size) {
    __shared__ float smem[2048];
    
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * N;
    
    // Load with bit-reversal - each thread loads multiple elements
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    for (int i = tid; i < FFT_SIZE; i += num_threads) {
        int rev_i = bit_reversal_10(i);
        smem[2 * i] = batch_input[rev_i];
        smem[2 * i + 1] = 0.0f;
    }
    
    __syncthreads();
    
    // FFT stages
    for (int s = 0; s < LOG2_SIZE; s++) {
        int m = 1 << (s + 1);
        int half_m = m >> 1;
        
        // Each thread does multiple butterflies
        for (int idx = tid; idx < 512; idx += num_threads) {
            int group = idx / half_m;
            int j = idx % half_m;
            
            int a_idx = group * m + j;
            int b_idx = a_idx + half_m;
            
            float angle = -2.0f * M_PI * (float)j / (float)m;
            float w_r = cosf(angle);
            float w_i = sinf(angle);
            
            float a_r = smem[2 * a_idx];
            float a_i = smem[2 * a_idx + 1];
            float b_r = smem[2 * b_idx];
            float b_i = smem[2 * b_idx + 1];
            
            float t_r = b_r * w_r - b_i * w_i;
            float t_i = b_r * w_i + b_i * w_r;
            
            smem[2 * a_idx] = a_r + t_r;
            smem[2 * a_idx + 1] = a_i + t_i;
            smem[2 * b_idx] = a_r - t_r;
            smem[2 * b_idx + 1] = a_i - t_i;
        }
        
        __syncthreads();
    }
    
    // Write output - first 513 complex values
    float* batch_output = output + batch_idx * HALF_SIZE * 2;
    
    for (int i = tid; i < HALF_SIZE; i += num_threads) {
        batch_output[2 * i] = smem[2 * i];
        batch_output[2 * i + 1] = smem[2 * i + 1];
    }
}

extern "C" {

void launch_fft_r2c_1d_1024_fp32(
    const void* input,
    void* output,
    int N,
    int batch_size,
    int inverse,
    cudaStream_t stream) {
    
    // We only implement forward FFT (inverse == 0)
    // N should be 1024, batch_size should be 1024
    
    const float* input_f = (const float*)input;
    float* output_f = (float*)output;
    
    // Use 256 threads per block for better occupancy
    // Each block processes one batch element
    int threads_per_block = 256;
    int num_blocks = batch_size;
    
    fft_r2c_1d_1024_kernel_simple<<<num_blocks, threads_per_block, 0, stream>>>(
        input_f, output_f, N, batch_size);
}

} // extern "C"
