#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Complex double2 operations
__device__ inline double2 make_double2(double x, double y) {
    double2 res;
    res.x = x;
    res.y = y;
    return res;
}

__device__ inline double2 complex_add(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}

__device__ inline double2 complex_sub(double2 a, double2 b) {
    return make_double2(a.x - b.x, a.y - b.y);
}

__device__ inline double2 complex_mul(double2 a, double2 b) {
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Bit-reverse permutation for indices 0 to 1023
__device__ inline int bit_reverse_10(int x) {
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);
    return x >> 22;  // For 10-bit reversal
}

// Shared memory for in-place FFT
// We need 1024 complex numbers = 1024 * 2 * 8 = 16384 bytes per batch
// Using float2/double2 for complex

#define N 1024
#define LOG2N 10

// Cooley-Tukey iterative FFT with shared memory
__global__ void fft_c2c_1d_1024_fp64_kernel(
    const double* __restrict__ input,
    double* __restrict__ output,
    int batch_size,
    int inverse
) {
    // Each block handles one batch
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    // Shared memory: 1024 complex numbers
    __shared__ double2 s_data[N];
    
    int tid = threadIdx.x;
    
    // Load data with bit-reversal permutation
    int rev_idx = bit_reverse_10(tid);
    int in_offset = batch * N * 2 + rev_idx * 2;
    s_data[tid] = make_double2(input[in_offset], input[in_offset + 1]);
    
    __syncthreads();
    
    // Iterative Cooley-Tukey FFT
    double sign = inverse ? 1.0 : -1.0;
    
    // Butterfly stages
    for (int stage = 1; stage <= LOG2N; stage++) {
        int butterfly_width = 1 << stage;           // 2, 4, 8, ..., 1024
        int half_width = butterfly_width >> 1;      // 1, 2, 4, ..., 512
        
        // Each thread processes multiple butterflies
        // We have 1024 threads, each butterfly needs 2 elements
        // Number of butterflies = N / butterfly_width = 1024 / butterfly_width
        
        int butterflies_per_thread = (N / butterfly_width + blockDim.x - 1) / blockDim.x;
        
        for (int b = 0; b < butterflies_per_thread; b++) {
            int butterfly = tid + b * blockDim.x;
            if (butterfly >= (N / butterfly_width)) continue;
            
            int base = butterfly * butterfly_width;
            
            for (int j = 0; j < half_width; j++) {
                int idx0 = base + j;
                int idx1 = idx0 + half_width;
                
                // Twiddle factor: exp(sign * 2*pi*i*j/butterfly_width)
                double angle = sign * 2.0 * M_PI * j / butterfly_width;
                double2 w = make_double2(cos(angle), sin(angle));
                
                double2 a = s_data[idx0];
                double2 b_val = s_data[idx1];
                
                double2 wb = complex_mul(w, b_val);
                
                s_data[idx0] = complex_add(a, wb);
                s_data[idx1] = complex_sub(a, wb);
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    int out_offset = batch * N * 2 + tid * 2;
    double2 val = s_data[tid];
    
    // Scale for inverse transform
    if (inverse) {
        val.x /= N;
        val.y /= N;
    }
    
    output[out_offset] = val.x;
    output[out_offset + 1] = val.y;
}

// Optimized version with better thread utilization
__global__ void fft_c2c_1d_1024_fp64_kernel_v2(
    const double* __restrict__ input,
    double* __restrict__ output,
    int batch_size,
    int inverse
) {
    __shared__ double2 s_data[N];
    
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    int tid = threadIdx.x;
    int lane = tid & 31;  // lane within warp
    int warp = tid >> 5;  // warp id
    
    // Bit-reverse load: 1024 threads, each loads one element
    int rev_idx = bit_reverse_10(tid);
    int in_offset = batch * N * 2 + rev_idx * 2;
    s_data[tid] = make_double2(input[in_offset], input[in_offset + 1]);
    
    __syncthreads();
    
    double sign = inverse ? 1.0 : -1.0;
    
    // Stage 1: butterfly_width = 2, half_width = 1
    // Each thread does N/2 = 512 butterflies? No, we need to parallelize
    
    // Better approach: each thread handles multiple elements in stages
    // For stage s with butterfly_width = 2^s, we have 1024/2^s butterflies
    // Each butterfly has 2^s elements, we process 2^(s-1) pairs
    
    // Stage 1: butterfly_width=2, 512 butterflies, 2 threads per butterfly? 
    // Actually let's do: for each stage, threads cooperate on butterflies
    
    // Alternative: use 512 threads, each handles 2 elements
    // But we have 1024 threads, so let's use all
    
    // Actually, let's use a different approach: each thread processes
    // one butterfly per stage, with appropriate indexing
    
    for (int stage = 1; stage <= LOG2N; stage++) {
        int butterfly_width = 1 << stage;
        int half_width = butterfly_width >> 1;
        int num_butterflies = N >> stage;  // N / butterfly_width
        
        // Each thread handles multiple butterflies
        // Total work: num_butterflies * half_width = N/2 twiddle multiplications per stage
        
        // Distribute work across 1024 threads
        int total_ops = num_butterflies * half_width;  // = N/2 = 512 per stage
        // Wait, that's wrong. Let me recalculate.
        // Each butterfly needs half_width complex muls and adds
        // Total complex ops: num_butterflies * half_width
        
        // Actually for Cooley-Tukey: we have N/2 butterflies per stage (radix-2)
        // No wait, that's not right either.
        
        // Correct: for butterfly_width = 2^stage, we have N / 2^stage butterflies
        // Each butterfly combines 2^stage elements pairwise
        // We need to do half_width = 2^(stage-1) operations per butterfly
        
        // Total parallel operations: (N / butterfly_width) * half_width = N/2 per stage
        
        // With 1024 threads, each thread does nothing or we need to recalculate
        
        // Let me use a simpler approach: 512 threads active, each does one radix-2 butterfly
        // But we have 1024 threads...
        
        // Actually use: threadIdx.x from 0 to 511 for actual work
        // Or: each thread does multiple operations
        
        // Simpler: use 512 threads, but we have 1024. Let's use 512 active.
        
        if (tid < 512) {
            // For stage s: butterfly has 2^s elements
            // Butterfly index: tid / half_width? No...
            
            // In Cooley-Tukey, at stage s, element i is paired with i + 2^(s-1)
            // if the s-th bit of i is 0
            
            // Let me use the standard iterative approach:
            // for i from 0 to N-1 step 2^s:
            //   for j from 0 to 2^(s-1)-1:
            //     idx0 = i + j
            //     idx1 = i + j + 2^(s-1)
            
            // Map tid to (i, j) pairs
            // i = (tid / half_width) * butterfly_width
            // j = tid % half_width
            
            int i = (tid / half_width) * butterfly_width;
            int j = tid % half_width;
            
            int idx0 = i + j;
            int idx1 = idx0 + half_width;
            
            double angle = sign * 2.0 * M_PI * j / butterfly_width;
            double2 w = make_double2(cos(angle), sin(angle));
            
            double2 a = s_data[idx0];
            double2 b_val = s_data[idx1];
            
            double2 wb = complex_mul(w, b_val);
            
            s_data[idx0] = complex_add(a, wb);
            s_data[idx1] = complex_sub(a, wb);
        }
        
        __syncthreads();
    }
    
    // Write output
    int out_offset = batch * N * 2 + tid * 2;
    double2 val = s_data[tid];
    
    if (inverse) {
        val.x /= N;
        val.y /= N;
    }
    
    output[out_offset] = val.x;
    output[out_offset + 1] = val.y;
}

// Further optimized - use warp shuffle where possible
// But for simplicity and correctness, use shared memory version

// Final optimized kernel with proper work distribution
__global__ void fft_c2c_1d_1024_fp64_kernel_final(
    const double* __restrict__ input,
    double* __restrict__ output,
    int batch_size,
    int inverse
) {
    __shared__ double2 s_data[N];
    
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    int tid = threadIdx.x;
    
    // Bit-reverse load
    int rev_idx = bit_reverse_10(tid);
    int in_offset = batch * N * 2 + rev_idx * 2;
    s_data[tid] = make_double2(input[in_offset], input[in_offset + 1]);
    
    __syncthreads();
    
    double sign = inverse ? 1.0 : -1.0;
    
    // FFT stages - using all 1024 threads
    // Key insight: at early stages, we have few butterflies but many elements per butterfly
    // At late stages, we have many butterflies but few elements per butterfly
    
    for (int stage = 1; stage <= LOG2N; stage++) {
        int butterfly_width = 1 << stage;
        int half_width = butterfly_width >> 1;
        
        // Each thread computes its position in the butterfly structure
        // For butterfly computation: we need to pair elements (i, i+half_width)
        // where i ranges over appropriate indices
        
        // The pattern: for each group of butterfly_width consecutive elements,
        // we do half_width butterfly operations
        
        // Thread tid can participate in multiple ways:
        // Method: thread tid handles element indices where (tid % butterfly_width) < half_width
        // and the group is determined by tid / butterfly_width
        
        int group = tid >> stage;           // tid / butterfly_width
        int offset = tid & (half_width - 1); // tid % half_width (for stage >= 1)
        
        // Actually simpler: use the fact that 1024 = 2^10
        // At stage s, we have 2^(10-s) groups, each with 2^s elements
        // We need to do 2^(s-1) operations per group
        
        // Total operations: 2^(10-s) * 2^(s-1) = 2^9 = 512 per stage
        
        // So we only need 512 threads! But we have 1024.
        // Let's use tid < 512 for computation, tid >= 512 idle
        
        // Actually, let's use a different mapping to use all threads
        // when possible, or just use 512 threads efficiently
        
        // Simplest correct approach: 512 threads do work
        if (tid < 512) {
            // Compute which butterfly this thread handles
            // At stage s: we have 512 butterflies total (each combining 2 elements)
            // Wait no, that's for stage 1. At stage s we have 1024/2 = 512 radix-2 butterflies
            
            // Actually at each stage we always have N/2 = 512 "butterfly operations"
            // where each operation combines 2 elements with a twiddle
            
            int butterfly = tid;
            int group_size = butterfly_width;
            int num_groups = N / group_size;
            
            int group_id = butterfly / half_width;  // which group of size butterfly_width
            int pos_in_group = butterfly % half_width;  // position within first half
            
            int idx0 = group_id * group_size + pos_in_group;
            int idx1 = idx0 + half_width;
            
            double angle = sign * 2.0 * M_PI * pos_in_group / butterfly_width;
            double2 w = make_double2(cos(angle), sin(angle));
            
            double2 a = s_data[idx0];
            double2 b_val = s_data[idx1];
            
            double2 wb = complex_mul(w, b_val);
            
            s_data[idx0] = complex_add(a, wb);
            s_data[idx1] = complex_sub(a, wb);
        }
        
        __syncthreads();
    }
    
    // Write output
    int out_offset = batch * N * 2 + tid * 2;
    double2 val = s_data[tid];
    
    if (inverse) {
        val.x /= N;
        val.y /= N;
    }
    
    output[out_offset] = val.x;
    output[out_offset + 1] = val.y;
}

// Most optimized: use register-based FFT with shared memory for communication
// Stockham autosort algorithm - avoids bit reversal

__global__ void fft_c2c_1d_1024_fp64_stockham(
    const double* __restrict__ input,
    double* __restrict__ output,
    int batch_size,
    int inverse
) {
    // Stockham algorithm: two buffers in shared memory
    __shared__ double2 s_data[2][N];
    
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    int tid = threadIdx.x;
    
    // Load directly (no bit reversal needed in Stockham)
    int in_offset = batch * N * 2 + tid * 2;
    s_data[0][tid] = make_double2(input[in_offset], input[in_offset + 1]);
    
    __syncthreads();
    
    double sign = inverse ? 1.0 : -1.0;
    int src = 0, dst = 1;
    
    for (int stage = 0; stage < LOG2N; stage++) {
        int butterfly_width = 2 << stage;  // 2, 4, 8, ..., 1024
        int half_width = butterfly_width >> 1;
        
        // Each thread handles N / butterfly_width butterflies
        // = 1024 / (2^(stage+1)) = 2^(9-stage) butterflies
        
        int butterflies_per_thread = (N / butterfly_width + blockDim.x - 1) / blockDim.x;
        
        for (int b = 0; b < butterflies_per_thread; b++) {
            int butterfly = tid + b * blockDim.x;
            if (butterfly >= N / butterfly_width) continue;
            
            // In Stockham, butterflies are arranged differently
            // For butterfly j at stage s:
            // Input indices: j and j + N/2 (for first stage), but general pattern is complex
            
            // Actually let me use a cleaner indexing
            // At stage s, we combine pairs separated by stride = N / butterfly_width * half_width? 
            
            // Standard Stockham indexing:
            // stride = N / butterfly_width (number of butterflies)
            // Actually for DIT Stockham: we read from src at indices...
            
            // Let me use a simpler known approach:
            // For butterfly k at stage s with width 2^(s+1):
            // top = k, bottom = k + 2^s, but with stride adjustments
            
            // Actually, let's use the correct Stockham formula:
            // At stage s (0-indexed), we have L = 2^(s+1) point DFTs
            // r = N / L = number of DFTs
            // For DFT number m (0 to r-1), we combine elements at:
            //   m + j*r and m + j*r + r*L/2 for j = 0 to L/2-1
            
            // Wait, that's getting messy. Let me use the working iterative approach
            // but with ping-pong buffers.
            
            // Actually, let me just implement the working Cooley-Tukey with ping-pong
            // to avoid the bit-reversal
            
            // For stage s in 0..9 (butterfly_width = 2^(s+1)):
            // We process N/2 pairs total
            
            // Index mapping for ping-pong:
            // butterfly k processes elements at positions based on bit patterns
            
            // Simpler: use the same 512-thread approach but with ping-pong
        }
        
        // Let me restart with a cleaner Stockham implementation
        
        // Actually, use the fact that we can compute indices directly
        // For DIF Stockham (decimation in frequency):
        
        // At each stage, we read from src and write to dst
        // The computation is the same butterfly as Cooley-Tukey
        // But the indexing is: for output position p, where did it come from?
        
        // Let me use a working approach: just do Cooley-Tukey with explicit bit reversal at end
        
        __syncthreads();
        // swap
        int tmp = src; src = dst; dst = tmp;
    }
    
    // For now, fall back to the working kernel
    // (this Stockham kernel is incomplete - use the final version below)
}

// Complete working kernel - Cooley-Tukey with bit reversal
__global__ void fft_c2c_1d_1024_fp64_complete(
    const double* __restrict__ input,
    double* __restrict__ output,
    int batch_size,
    int inverse
) {
    __shared__ double2 s_data[N];
    
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    int tid = threadIdx.x;
    
    // Load with bit reversal
    int rev_idx = bit_reverse_10(tid);
    int in_offset = batch * N * 2 + rev_idx * 2;
    s_data[tid] = make_double2(input[in_offset], input[in_offset + 1]);
    
    __syncthreads();
    
    double sign = inverse ? 1.0 : -1.0;
    
    // 10 stages of FFT
    for (int stage = 1; stage <= LOG2N; stage++) {
        int butterfly_width = 1 << stage;
        int half_width = butterfly_width >> 1;
        
        // 512 active threads
        if (tid < 512) {
            // Map tid to butterfly and position within butterfly
            // Total butterflies at this stage: N / butterfly_width
            // Each butterfly needs half_width operations
            
            // We have 512 threads, need to cover all operations
            // Operations per stage: (N / butterfly_width) * half_width = N/2 = 512
            
            // So thread tid handles operation number tid
            int op = tid;
            
            // Which butterfly?
            int butterfly = op / half_width;
            // Position within butterfly?
            int j = op % half_width;
            
            int base = butterfly * butterfly_width;
            int idx0 = base + j;
            int idx1 = idx0 + half_width;
            
            double angle = sign * 2.0 * M_PI * j / butterfly_width;
            double2 w = make_double2(cos(angle), sin(angle));
            
            double2 a = s_data[idx0];
            double2 b = s_data[idx1];
            
            double2 wb = complex_mul(w, b);
            
            s_data[idx0] = complex_add(a, wb);
            s_data[idx1] = complex_sub(a, wb);
        }
        
        __syncthreads();
    }
    
    // Store result
    int out_offset = batch * N * 2 + tid * 2;
    double2 val = s_data[tid];
    
    if (inverse) {
        val.x *= (1.0 / N);
        val.y *= (1.0 / N);
    }
    
    output[out_offset] = val.x;
    output[out_offset + 1] = val.y;
}

// Optimized version with loop unrolling and fewer syncthreads opportunities
// Actually keep it simple for correctness

extern "C" {

void launch_fft_c2c_1d_1024_fp64(
    const void* input,
    void* output,
    int N_param,
    int batch_size,
    int inverse,
    cudaStream_t stream
) {
    // N_param should be 1024, but we use compile-time constant
    // batch_size = 256
    
    int threads = 1024;  // One thread per element
    int blocks = batch_size;
    
    fft_c2c_1d_1024_fp64_complete<<<blocks, threads, 0, stream>>>(
        (const double*)input,
        (double*)output,
        batch_size,
        inverse
    );
}

} // extern "C"
