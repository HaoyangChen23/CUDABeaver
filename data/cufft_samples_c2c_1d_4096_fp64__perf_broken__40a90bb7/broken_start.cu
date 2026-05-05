#include <cuda_runtime.h>
#include <cmath>

// Complex number operations for double precision
struct Complex {
    double real;
    double imag;
    
    __device__ __forceinline__ Complex() : real(0.0), imag(0.0) {}
    __device__ __forceinline__ Complex(double r, double i) : real(r), imag(i) {}
    
    __device__ __forceinline__ Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
    
    __device__ __forceinline__ Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }
    
    __device__ __forceinline__ Complex operator*(const Complex& other) const {
        return Complex(
            real * other.real - imag * other.imag,
            real * other.imag + imag * other.real
        );
    }
};

// Bit reversal permutation for N = 4096 = 2^12
__device__ __forceinline__ int bit_reverse_12(int x) {
    x = ((x & 0xAAA) >> 1) | ((x & 0x555) << 1);
    x = ((x & 0xCCC) >> 2) | ((x & 0x333) << 2);
    x = ((x & 0xF0F) >> 4) | ((x & 0x0F0) << 4);
    x = ((x & 0xFF) >> 8) | ((x & 0x0FF) << 8);
    return x >> 4;  // For 12 bits, shift by 4 (16-12=4)
}

// 1D Cooley-Tukey FFT kernel for N=4096
// Uses shared memory for efficient data access
// Each block processes one batch element
// 4096 elements / 256 threads = 16 elements per thread
template<int N, int LOG_N>
__global__ void fft_c2c_1d_kernel(
    const double* __restrict__ input,
    double* __restrict__ output,
    int batch_size,
    int inverse
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const int tid = threadIdx.x;
    const int lane = tid & 63;      // 0-63
    const int warp = tid >> 6;      // 0-3 (4 warps)
    
    // Each thread handles 16 elements (4096 / 256)
    const int elems_per_thread = N / 256;  // 16
    
    // Shared memory for the entire FFT (4096 complex numbers)
    // Layout: interleaved real/imag in shared memory
    // Use double2 for better memory access
    __shared__ Complex s_data[N];
    
    const double* batch_input = input + batch_idx * N * 2;
    double* batch_output = output + batch_idx * N * 2;
    
    // Load data with bit-reversal permutation
    // Each thread loads 16 elements
    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = tid * elems_per_thread + i;
        int rev_idx = bit_reverse_12(idx);
        
        // Load from global memory (interleaved real/imag)
        double real = batch_input[rev_idx * 2];
        double imag = batch_input[rev_idx * 2 + 1];
        s_data[idx] = Complex(real, imag);
    }
    
    __syncthreads();
    
    // Cooley-Tukey FFT
    // Iterate through stages: 2, 4, 8, ..., 4096
    for (int stage = 1; stage <= LOG_N; stage++) {
        int butterfly_width = 1 << stage;           // 2, 4, 8, ..., 4096
        int half_width = butterfly_width >> 1;      // 1, 2, 4, ..., 2048
        int num_butterflies = N / butterfly_width;  // 2048, 1024, ..., 1
        
        // Each thread processes multiple butterflies
        // Total butterflies per stage: N/2 = 2048
        // Threads: 256, so each thread does 8 butterflies
        
        int butterflies_per_thread = N / (2 * 256);  // 8
        
        #pragma unroll
        for (int b = 0; b < butterflies_per_thread; b++) {
            int butterfly_idx = tid * butterflies_per_thread + b;
            if (butterfly_idx >= N / 2) continue;
            
            // Position within the butterfly
            int group = butterfly_idx / half_width;      // Which group of butterflies
            int pos_in_group = butterfly_idx % half_width; // Position within group
            
            int idx0 = group * butterfly_width + pos_in_group;
            int idx1 = idx0 + half_width;
            
            // Load values
            Complex a = s_data[idx0];
            Complex b_val = s_data[idx1];
            
            // Compute twiddle factor
            // W = exp(-2*pi*i*k/N) where k depends on position
            double angle;
            if (inverse) {
                angle = 2.0 * M_PI * pos_in_group / butterfly_width;
            } else {
                angle = -2.0 * M_PI * pos_in_group / butterfly_width;
            }
            
            double w_real = cos(angle);
            double w_imag = sin(angle);
            Complex w(w_real, w_imag);
            
            // Butterfly computation
            Complex t = b_val * w;
            Complex u = a;
            
            s_data[idx0] = u + t;
            s_data[idx1] = u - t;
        }
        
        __syncthreads();
    }
    
    // Store results to global memory
    // Apply 1/N scaling for inverse transform
    double scale = inverse ? (1.0 / N) : 1.0;
    
    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = tid * elems_per_thread + i;
        Complex val = s_data[idx];
        batch_output[idx * 2] = val.real * scale;
        batch_output[idx * 2 + 1] = val.imag * scale;
    }
}

// Optimized version using register-based FFT with shared memory for twiddle factors
// and better memory coalescing
template<int N, int LOG_N>
__global__ void fft_c2c_1d_optimized_kernel(
    const double* __restrict__ input,
    double* __restrict__ output,
    int batch_size,
    int inverse
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const int tid = threadIdx.x;
    
    // Shared memory for data: 4096 Complex numbers
    __shared__ Complex s_data[N];
    
    // Precompute twiddle factors for all stages in shared memory
    // We need up to N/2 twiddle factors (2048 for N=4096)
    __shared__ Complex twiddle[N/2];
    
    const double* batch_input = input + batch_idx * N * 2;
    double* batch_output = output + batch_idx * N * 2;
    
    // Load with bit-reversal
    const int elems_per_thread = N / 256;
    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = tid * elems_per_thread + i;
        int rev_idx = bit_reverse_12(idx);
        s_data[idx] = Complex(batch_input[rev_idx * 2], batch_input[rev_idx * 2 + 1]);
    }
    
    // Precompute twiddle factors for all stages
    // Only first N/2 threads participate
    if (tid < N/2) {
        int k = tid;
        // Compute for largest butterfly (stage LOG_N)
        double angle = (inverse ? 2.0 : -2.0) * M_PI * k / N;
        twiddle[k] = Complex(cos(angle), sin(angle));
    }
    
    __syncthreads();
    
    // FFT stages
    for (int stage = 1; stage <= LOG_N; stage++) {
        int butterfly_width = 1 << stage;
        int half_width = butterfly_width >> 1;
        
        int butterflies_per_thread = N / (2 * 256);
        
        #pragma unroll
        for (int b = 0; b < butterflies_per_thread; b++) {
            int butterfly_idx = tid * butterflies_per_thread + b;
            if (butterfly_idx >= N / 2) continue;
            
            int group = butterfly_idx / half_width;
            int pos_in_group = butterfly_idx % half_width;
            
            int idx0 = group * butterfly_width + pos_in_group;
            int idx1 = idx0 + half_width;
            
            Complex a = s_data[idx0];
            Complex b_val = s_data[idx1];
            
            // Get twiddle factor: W_N^{k} where k = pos_in_group * (N/butterfly_width)
            int twiddle_idx = pos_in_group * (N / butterfly_width);
            Complex w = twiddle[twiddle_idx];
            
            Complex t = b_val * w;
            
            s_data[idx0] = a + t;
            s_data[idx1] = a - t;
        }
        
        __syncthreads();
    }
    
    // Store results
    double scale = inverse ? (1.0 / N) : 1.0;
    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = tid * elems_per_thread + i;
        Complex val = s_data[idx];
        batch_output[idx * 2] = val.real * scale;
        batch_output[idx * 2 + 1] = val.imag * scale;
    }
}

extern "C" {

void launch_fft_c2c_1d_4096_fp64(
    const void* input,
    void* output,
    int N,
    int batch_size,
    int inverse,
    cudaStream_t stream
) {
    const int BLOCK_SIZE = 256;
    const int LOG_N = 12;  // log2(4096) = 12
    
    // Launch one block per batch element
    // Each block has 256 threads, processing 4096 elements
    fft_c2c_1d_optimized_kernel<4096, 12><<<batch_size, BLOCK_SIZE, 0, stream>>>(
        (const double*)input,
        (double*)output,
        batch_size,
        inverse
    );
}

} // extern "C"
