#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define PI 3.14159265358979323846f

// Complex multiply: (a+ib)*(c+id) = (ac-bd) + i(ad+bc)
__device__ __forceinline__ void cmul(float ar, float ai, float br, float bi, float& or_out, float& oi_out) {
    or_out = ar * br - ai * bi;
    oi_out = ar * bi + ai * br;
}

// Complex add
__device__ __forceinline__ void cadd(float ar, float ai, float br, float bi, float& or_out, float& oi_out) {
    or_out = ar + br;
    oi_out = ai + bi;
}

// 1D Cooley-Tukey FFT (radix-2, iterative, in-place)
// n must be power of 2
__device__ void fft1d_inplace(float* real, float* imag, int n, bool inverse) {
    // Bit reversal permutation
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            float tr = real[i], ti = imag[i];
            real[i] = real[j];
            imag[i] = imag[j];
            real[j] = tr;
            imag[j] = ti;
        }
    }

    // Cooley-Tukey iterations
    float sign = inverse ? 1.0f : -1.0f;
    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2 * PI / len * sign;
        float wpr = cosf(ang);
        float wpi = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float wr = 1.0f;
            float wi = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                int idx1 = i + k;
                int idx2 = i + k + len / 2;
                float tr, ti;
                cmul(wr, wi, real[idx2], imag[idx2], tr, ti);
                float ur = real[idx1];
                float ui = imag[idx1];
                real[idx1] = ur + tr;
                imag[idx1] = ui + ti;
                real[idx2] = ur - tr;
                imag[idx2] = ui - ti;
                // Update twiddle factor
                float twr = wr * wpr - wi * wpi;
                float twi = wr * wpi + wi * wpr;
                wr = twr;
                wi = twi;
            }
        }
    }

    // Scale for inverse
    if (inverse) {
        float scale = 1.0f / n;
        for (int i = 0; i < n; i++) {
            real[i] *= scale;
            imag[i] *= scale;
        }
    }
}

// Shared memory layout for 64x64 plane: 2 * 64 * 64 floats
// We process FFTs along each dimension

__global__ void fft3d_64x64x64_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int N,
                                      int batch_size,
                                      int inverse) {
    // Dimensions: 64x64x64
    const int DIM = 64;
    const int DIM2 = DIM * DIM;      // 4096
    const int DIM3 = DIM * DIM * DIM; // 262144
    
    // Shared memory for in-place FFT along each dimension
    // Layout: [real][imag] or interleaved - using separate arrays for simplicity
    __shared__ float s_real[64];
    __shared__ float s_imag[64];
    
    // Process each batch
    for (int b = blockIdx.z; b < batch_size; b += gridDim.z) {
        int batch_offset = b * DIM3 * 2; // *2 for complex interleaved
        
        // Phase 1: FFT along X dimension (contiguous in memory: real, imag pairs)
        // Each thread block handles a (Y, Z) plane
        // Grid: (X_blocks, Y, Z) but we use 2D grid for Y,Z
        // Actually: process X-FFT for all Y,Z
        
        int tid = threadIdx.x;
        int nthreads = blockDim.x; // Should be 64
        
        // For X-FFT: each warp/block processes one (y,z) line
        // We need to cover 64*64 = 4096 lines
        int yz_per_block = (DIM2 + gridDim.x - 1) / gridDim.x;
        int yz_start = blockIdx.x * yz_per_block;
        int yz_end = min(yz_start + yz_per_block, DIM2);
        
        for (int yz = yz_start + tid / 64; yz < yz_end; yz += (nthreads / 64)) {
            if (yz >= DIM2) break;
            int y = yz / DIM;
            int z = yz % DIM;
            int local_tid = tid % 64;
            
            // Load X line (64 elements) - interleaved real/imag
            if (local_tid < DIM) {
                int idx = batch_offset + ((z * DIM + y) * DIM + local_tid) * 2;
                s_real[local_tid] = input[idx];
                s_imag[local_tid] = input[idx + 1];
            }
            __syncthreads();
            
            // FFT along X (64 points)
            // Need all 64 threads to participate
            if (local_tid == 0) {
                fft1d_inplace(s_real, s_imag, DIM, inverse != 0);
            }
            __syncthreads();
            
            // Store back to output (temporary, or we can use output as workspace)
            if (local_tid < DIM) {
                int idx = batch_offset + ((z * DIM + y) * DIM + local_tid) * 2;
                output[idx] = s_real[local_tid];
                output[idx + 1] = s_imag[local_tid];
            }
            __syncthreads();
        }
        
        // Ensure all X-FFTs complete
        __threadfence();
        
        // Phase 2: FFT along Y dimension
        // Y is stride DIM in memory (after X)
        // Grid: blocks for (X, Z) planes
        
        // Reconfigure: each block handles multiple (x,z) lines for Y-FFT
        int xz_per_block = (DIM2 + gridDim.y - 1) / gridDim.y;
        int xz_start = blockIdx.y * xz_per_block;
        int xz_end = min(xz_start + xz_per_block, DIM2);
        
        for (int xz = xz_start + tid / 64; xz < xz_end; xz += (nthreads / 64)) {
            if (xz >= DIM2) break;
            int x = xz / DIM;
            int z = xz % DIM;
            int local_tid = tid % 64;
            
            // Load Y line (64 elements at stride DIM)
            if (local_tid < DIM) {
                // Read from output (after X-FFT)
                int idx = batch_offset + ((z * DIM + local_tid) * DIM + x) * 2;
                s_real[local_tid] = output[idx];
                s_imag[local_tid] = output[idx + 1];
            }
            __syncthreads();
            
            // FFT along Y
            if (local_tid == 0) {
                fft1d_inplace(s_real, s_imag, DIM, inverse != 0);
            }
            __syncthreads();
            
            // Store back
            if (local_tid < DIM) {
                int idx = batch_offset + ((z * DIM + local_tid) * DIM + x) * 2;
                output[idx] = s_real[local_tid];
                output[idx + 1] = s_imag[local_tid];
            }
            __syncthreads();
        }
        
        __threadfence();
        
        // Phase 3: FFT along Z dimension
        // Z is stride DIM*DIM = 4096 in memory
        
        int xy_per_block = (DIM2 + gridDim.z - 1) / gridDim.z;
        int xy_start = blockIdx.z * xy_per_block; // But we use blockIdx.z for batch...
        // Actually we need different grid configuration
        
        // For simplicity, use a different approach: serialize with atomic or use more blocks
        // Since batch=1, we can use blockIdx.x for X, blockIdx.y for Y, process Z
        
        // Let me restructure: use 3D grid properly
        // Actually, let's use a simpler approach with full serialization per batch
    }
}

// Simpler approach: single kernel with proper grid for 3D FFT
__global__ void fft3d_kernel_v2(const float* __restrict__ input,
                                float* __restrict__ output,
                                int batch_size,
                                int inverse) {
    const int DIM = 64;
    const int DIM2 = DIM * DIM;
    const int DIM3 = DIM * DIM * DIM;
    
    __shared__ float s_real[64];
    __shared__ float s_imag[64];
    
    int tid = threadIdx.x;
    bool is_inverse = (inverse != 0);
    
    for (int b = 0; b < batch_size; b++) {
        int batch_offset = b * DIM3 * 2;
        
        // Each block handles one line in one dimension
        // Grid should be sized to cover all lines in all dimensions across phases
        
        // We'll do 3 phases, each with appropriate grid
        
        // For now, use this kernel with proper 3D grid launch
        int line_id = blockIdx.x; // 0 to 4095 for X or Y lines
        
        // Phase 1: X-FFT (contiguous)
        // Each of 4096 (y,z) lines
        if (line_id < DIM2) {
            int y = line_id / DIM;
            int z = line_id % DIM;
            
            // Load
            if (tid < DIM) {
                int idx = batch_offset + ((z * DIM + y) * DIM + tid) * 2;
                s_real[tid] = input[idx];
                s_imag[tid] = input[idx + 1];
            }
            __syncthreads();
            
            // FFT
            if (tid == 0) {
                fft1d_inplace(s_real, s_imag, DIM, is_inverse);
            }
            __syncthreads();
            
            // Store to output
            if (tid < DIM) {
                int idx = batch_offset + ((z * DIM + y) * DIM + tid) * 2;
                output[idx] = s_real[tid];
                output[idx + 1] = s_imag[tid];
            }
        }
        
        __threadfence();
        __syncthreads();
        
        // Phase 2: Y-FFT (stride DIM)
        if (line_id < DIM2) {
            int x = line_id / DIM;
            int z = line_id % DIM;
            
            // Load from output (X-FFT result)
            if (tid < DIM) {
                int idx = batch_offset + ((z * DIM + tid) * DIM + x) * 2;
                s_real[tid] = output[idx];
                s_imag[tid] = output[idx + 1];
            }
            __syncthreads();
            
            if (tid == 0) {
                fft1d_inplace(s_real, s_imag, DIM, is_inverse);
            }
            __syncthreads();
            
            if (tid < DIM) {
                int idx = batch_offset + ((z * DIM + tid) * DIM + x) * 2;
                output[idx] = s_real[tid];
                output[idx + 1] = s_imag[tid];
            }
        }
        
        __threadfence();
        __syncthreads();
        
        // Phase 3: Z-FFT (stride DIM*DIM)
        if (line_id < DIM2) {
            int x = line_id / DIM;
            int y = line_id % DIM;
            
            // Load
            if (tid < DIM) {
                int idx = batch_offset + ((tid * DIM + y) * DIM + x) * 2;
                s_real[tid] = output[idx];
                s_imag[tid] = output[idx + 1];
            }
            __syncthreads();
            
            if (tid == 0) {
                fft1d_inplace(s_real, s_imag, DIM, is_inverse);
            }
            __syncthreads();
            
            // Final store
            if (tid < DIM) {
                int idx = batch_offset + ((tid * DIM + y) * DIM + x) * 2;
                output[idx] = s_real[tid];
                output[idx + 1] = s_imag[tid];
            }
        }
        
        __threadfence();
    }
}

// Proper 3-phase kernel with separate launches for synchronization
__global__ void fft1d_x_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int batch_size,
                               int inverse) {
    const int DIM = 64;
    const int DIM2 = DIM * DIM;
    const int DIM3 = DIM * DIM * DIM;
    
    __shared__ float s_real[64];
    __shared__ float s_imag[64];
    
    int tid = threadIdx.x;
    int line_id = blockIdx.x;
    bool is_inverse = (inverse != 0);
    
    for (int b = 0; b < batch_size; b++) {
        int batch_offset = b * DIM3 * 2;
        
        if (line_id >= DIM2) return;
        
        int y = line_id / DIM;
        int z = line_id % DIM;
        
        // Load X line
        if (tid < DIM) {
            int idx = batch_offset + ((z * DIM + y) * DIM + tid) * 2;
            s_real[tid] = input[idx];
            s_imag[tid] = input[idx + 1];
        }
        __syncthreads();
        
        // FFT X
        if (tid == 0) {
            fft1d_inplace(s_real, s_imag, DIM, is_inverse);
        }
        __syncthreads();
        
        // Store
        if (tid < DIM) {
            int idx = batch_offset + ((z * DIM + y) * DIM + tid) * 2;
            output[idx] = s_real[tid];
            output[idx + 1] = s_imag[tid];
        }
    }
}

__global__ void fft1d_y_kernel(float* __restrict__ data,
                               int batch_size,
                               int inverse) {
    const int DIM = 64;
    const int DIM2 = DIM * DIM;
    const int DIM3 = DIM * DIM * DIM;
    
    __shared__ float s_real[64];
    __shared__ float s_imag[64];
    
    int tid = threadIdx.x;
    int line_id = blockIdx.x;
    bool is_inverse = (inverse != 0);
    
    for (int b = 0; b < batch_size; b++) {
        int batch_offset = b * DIM3 * 2;
        
        if (line_id >= DIM2) return;
        
        int x = line_id / DIM;
        int z = line_id % DIM;
        
        // Load Y line (stride DIM)
        if (tid < DIM) {
            int idx = batch_offset + ((z * DIM + tid) * DIM + x) * 2;
            s_real[tid] = data[idx];
            s_imag[tid] = data[idx + 1];
        }
        __syncthreads();
        
        // FFT Y
        if (tid == 0) {
            fft1d_inplace(s_real, s_imag, DIM, is_inverse);
        }
        __syncthreads();
        
        // Store
        if (tid < DIM) {
            int idx = batch_offset + ((z * DIM + tid) * DIM + x) * 2;
            data[idx] = s_real[tid];
            data[idx + 1] = s_imag[tid];
        }
    }
}

__global__ void fft1d_z_kernel(float* __restrict__ data,
                               int batch_size,
                               int inverse) {
    const int DIM = 64;
    const int DIM2 = DIM * DIM;
    const int DIM3 = DIM * DIM * DIM;
    
    __shared__ float s_real[64];
    __shared__ float s_imag[64];
    
    int tid = threadIdx.x;
    int line_id = blockIdx.x;
    bool is_inverse = (inverse != 0);
    
    for (int b = 0; b < batch_size; b++) {
        int batch_offset = b * DIM3 * 2;
        
        if (line_id >= DIM2) return;
        
        int x = line_id / DIM;
        int y = line_id % DIM;
        
        // Load Z line (stride DIM*DIM)
        if (tid < DIM) {
            int idx = batch_offset + ((tid * DIM + y) * DIM + x) * 2;
            s_real[tid] = data[idx];
            s_imag[tid] = data[idx + 1];
        }
        __syncthreads();
        
        // FFT Z
        if (tid == 0) {
            fft1d_inplace(s_real, s_imag, DIM, is_inverse);
        }
        __syncthreads();
        
        // Store
        if (tid < DIM) {
            int idx = batch_offset + ((tid * DIM + y) * DIM + x) * 2;
            data[idx] = s_real[tid];
            data[idx + 1] = s_imag[tid];
        }
    }
}

extern "C" {

void launch_fft_c2c_3d_64x64x64_fp32(
    const void* input,
    void* output,
    int N,
    int batch_size,
    int inverse,
    cudaStream_t stream
) {
    const int DIM = 64;
    const int DIM2 = DIM * DIM; // 4096 lines per dimension
    
    // Use 64 threads per block (one per element in FFT)
    const int threads = 64;
    const int blocks = DIM2; // 4096 blocks to process all lines
    
    // Three separate kernel launches for proper synchronization between dimensions
    
    // Phase 1: FFT along X (contiguous)
    fft1d_x_kernel<<<blocks, threads, 0, stream>>>(
        (const float*)input,
        (float*)output,
        batch_size,
        inverse
    );
    
    // Phase 2: FFT along Y (stride DIM)
    fft1d_y_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        batch_size,
        inverse
    );
    
    // Phase 3: FFT along Z (stride DIM*DIM)
    fft1d_z_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        batch_size,
        inverse
    );
}

} // extern "C"
