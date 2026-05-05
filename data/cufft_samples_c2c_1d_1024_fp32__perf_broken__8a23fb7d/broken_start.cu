#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// 1D Complex-to-Complex FFT for N=1024 using Cooley-Tukey radix-2 algorithm
// Uses shared memory for efficient data movement and computation

#define N 1024
#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_THREAD 4  // 1024 / 256 = 4

// Complex multiply: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
__device__ __forceinline__ void complex_mul(float ar, float ai, float br, float bi, float& or, float& oi) {
    or = ar * br - ai * bi;
    oi = ar * bi + ai * br;
}

// Complex add: (a+bi) + (c+di) = (a+c) + (b+d)i
__device__ __forceinline__ void complex_add(float ar, float ai, float br, float bi, float& or, float& oi) {
    or = ar + br;
    oi = ai + bi;
}

// Complex sub: (a+bi) - (c+di) = (a-c) + (b-d)i
__device__ __forceinline__ void complex_sub(float ar, float ai, float br, float bi, float& or, float& oi) {
    or = ar - br;
    oi = ai - bi;
}

// Bit reverse index for 10 bits (since 1024 = 2^10)
__device__ __forceinline__ int bit_reverse(int x) {
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);
    return x >> 22;  // Shift right by 32-10 = 22 bits
}

__global__ void fft_c2c_1d_1024_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int inverse
) {
    // Shared memory for the FFT: 1024 complex numbers = 2048 floats
    // Use double buffering with two arrays
    __shared__ float smem_real[1024];
    __shared__ float smem_imag[1024];
    
    const int tid = threadIdx.x;
    const int batch = blockIdx.x;
    
    if (batch >= batch_size) return;
    
    // Each thread loads 4 elements (1024 / 256 = 4)
    // Load with bit-reversal permutation
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = tid + i * THREADS_PER_BLOCK;
        int rev_idx = bit_reverse(idx);
        // Input is interleaved real/imag
        smem_real[rev_idx] = input[(batch * N + idx) * 2 + 0];
        smem_imag[rev_idx] = input[(batch * N + idx) * 2 + 1];
    }
    
    __syncthreads();
    
    // Cooley-Tukey iterative FFT
    // N = 1024 = 2^10, so we need 10 stages
    const float angle_sign = inverse ? 1.0f : -1.0f;
    
    // Iterative FFT stages
    for (int stage = 1; stage <= 10; stage++) {
        int butterfly_width = 1 << stage;           // 2, 4, 8, ..., 1024
        int half_width = butterfly_width >> 1;      // 1, 2, 4, ..., 512
        int num_groups = N >> stage;                // 512, 256, ..., 1
        
        // Each thread processes ELEMENTS_PER_THREAD butterflies
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int global_idx = tid + i * THREADS_PER_BLOCK;
            
            // Determine which butterfly this element belongs to
            int group = global_idx >> (stage - 1);  // Which group of butterflies
            int pos_in_group = global_idx & (half_width - 1);  // Position within half-butterfly
            
            // Indices for the butterfly
            int idx0 = (group << stage) + pos_in_group;
            int idx1 = idx0 + half_width;
            
            // Load values
            float a_r = smem_real[idx0];
            float a_i = smem_imag[idx0];
            float b_r = smem_real[idx1];
            float b_i = smem_imag[idx1];
            
            // Compute twiddle factor: exp(angle_sign * -2*pi*i * k / butterfly_width)
            // where k = pos_in_group
            float angle = angle_sign * 2.0f * M_PI * pos_in_group / butterfly_width;
            float w_r = cosf(angle);
            float w_i = sinf(angle);
            
            // Multiply b by twiddle factor
            float t_r, t_i;
            complex_mul(b_r, b_i, w_r, w_i, t_r, t_i);
            
            // Butterfly: a + t, a - t
            float out0_r, out0_i, out1_r, out1_i;
            complex_add(a_r, a_i, t_r, t_i, out0_r, out0_i);
            complex_sub(a_r, a_i, t_r, t_i, out1_r, out1_i);
            
            // Store back
            smem_real[idx0] = out0_r;
            smem_imag[idx0] = out0_i;
            smem_real[idx1] = out1_r;
            smem_imag[idx1] = out1_i;
        }
        
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = tid + i * THREADS_PER_BLOCK;
        output[(batch * N + idx) * 2 + 0] = smem_real[idx];
        output[(batch * N + idx) * 2 + 1] = smem_imag[idx];
    }
}

extern "C" {

void launch_fft_c2c_1d_1024_fp32(
    const void* input,
    void* output,
    int N_arg,
    int batch_size,
    int inverse,
    cudaStream_t stream
) {
    // N_arg should be 1024, but we use compile-time constant for optimization
    const int threads = THREADS_PER_BLOCK;
    const int blocks = batch_size;
    
    fft_c2c_1d_1024_kernel<<<blocks, threads, 0, stream>>>(
        (const float*)input,
        (float*)output,
        batch_size,
        inverse
    );
}

} // extern "C"
