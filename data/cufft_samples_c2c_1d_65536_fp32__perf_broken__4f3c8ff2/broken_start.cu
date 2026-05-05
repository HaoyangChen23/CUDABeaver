#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Constants for FFT of size 65536 = 2^16
#define N 65536
#define LOG2N 16
#define BATCH_SIZE 16

// Maximum threads per block
#define MAX_THREADS 256

// Shared memory for butterfly operations
// Each thread processes 2 complex numbers (4 floats) per stage
// We need N/2 complex numbers per batch element in shared memory
// But we process in tiles to fit shared memory

// Use 2048 elements (4096 floats) per block - fits in 48KB shared memory
// This means 1024 complex numbers per block
#define TILE_SIZE 2048  // Number of floats (1024 complex numbers)
#define COMPLEX_PER_TILE (TILE_SIZE / 2)

// Twiddle factor lookup table in constant memory
__constant__ float d_twiddle_real[N];
__constant__ float d_twiddle_imag[N];

// Initialize twiddle factors
__global__ void init_twiddle_kernel(int n, float* real_out, float* imag_out, int inverse) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float angle = -2.0f * M_PI * idx / n;  // negative for forward FFT
        if (inverse) angle = -angle;
        real_out[idx] = cosf(angle);
        imag_out[idx] = sinf(angle);
    }
}

// Bit reversal permutation
__device__ inline int bit_reverse(int x, int bits) {
    int result = 0;
    for (int i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// Complex multiplication
__device__ inline void complex_mul(float a_r, float a_i, float b_r, float b_i, float& out_r, float& out_i) {
    out_r = a_r * b_r - a_i * b_i;
    out_i = a_r * b_i + a_i * b_r;
}

// Radix-2 butterfly: in-place DIT FFT butterfly
// (a, b) -> (a + w*b, a - w*b) where w is twiddle factor
__device__ inline void butterfly(float& a_r, float& a_i, float& b_r, float& b_i, float w_r, float w_i) {
    // t = b * w
    float t_r = b_r * w_r - b_i * w_i;
    float t_i = b_r * w_i + b_i * w_r;
    
    // b = a - t
    b_r = a_r - t_r;
    b_i = a_i - t_i;
    
    // a = a + t
    a_r = a_r + t_r;
    a_i = a_i + t_i;
}

// Single 65536-point FFT using shared memory with tiling
// Each block handles one batch element
// We process the FFT in stages, using shared memory for the current tile

__global__ void fft_c2c_1d_65536_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n,
    int batch_size,
    int inverse
) {
    // Shared memory: TILE_SIZE floats = TILE_SIZE/2 complex numbers
    extern __shared__ float s_data[];
    
    const int batch = blockIdx.y;  // Which batch element
    const int tid = threadIdx.x;   // Thread within block
    
    if (batch >= batch_size) return;
    
    // Each thread handles 2 complex numbers (4 floats) for better memory coalescing
    // So we need TILE_SIZE/4 threads to load TILE_SIZE/2 complex numbers
    const int complex_per_block = TILE_SIZE / 2;
    const int threads_needed = complex_per_block / 2;  // Each thread handles 2 complex
    
    // We process 65536 elements in tiles
    // Number of tiles: 65536 / (TILE_SIZE/2) = 65536 / 1024 = 64 tiles
    
    const int num_tiles = N / complex_per_block;  // 64 tiles
    
    // Global input/output pointer for this batch
    const float* g_in = input + batch * N * 2;
    float* g_out = output + batch * N * 2;
    
    // Register storage for this thread's 2 complex numbers
    float reg_r[2], reg_i[2];
    int reg_idx[2];
    
    // We need to perform 16 stages of radix-2 butterflies
    // Stages 0-5 can be done within a tile (local to shared memory)
    // Stages 6-15 require cross-tile communication
    
    // First, let's do bit-reversal permutation and load data
    // We process in tiles, but need to handle the full FFT
    
    // Alternative: Use global memory with coalesced access and perform 
    // Cooley-Tukey FFT with iterative stages
    
    // Simpler approach: Stockham auto-sort FFT that doesn't need bit reversal
    
    // We'll use a different strategy: process the entire FFT in global memory
    // with careful coalescing, using registers for butterfly computation
    
    // Each thread processes multiple elements strided across the array
    
    const int total_complex = N;  // 65536
    const int stride = blockDim.x * gridDim.x;  // Not used, we use 1 block per batch
    
    // Actually, let's use 1 block per batch with many threads
    // 256 threads * 256 = 65536, so each thread handles 256 complex numbers
    
    const int elem_per_thread = N / blockDim.x;  // 65536 / 256 = 256
    
    // Load data into registers with bit-reversal permutation
    // Each thread loads elem_per_thread complex numbers
    
    // Use global memory FFT with register caching
    
    // Index calculation: thread tid handles elements [tid*elem_per_thread, (tid+1)*elem_per_thread)
    const int my_start = tid * elem_per_thread;
    
    // Load from global memory (with bit reversal for first stage)
    // Actually, let's use iterative FFT without explicit bit reversal using Stockham
    
    // Stockham FFT: at each stage, we reorder so output is naturally ordered
    
    // We'll use shared memory for the current stage's data
    
    // Simpler: Just do the full iterative FFT in shared memory if it fits
    // 65536 * 2 * 4 bytes = 512 KB, too big for shared memory
    
    // Must use tiling or register-based approach
    
    // Let's use a register-based approach with global memory for the full array
    // Each thread keeps its 256 complex numbers in registers (512 floats)
    // That's too much (2048 bytes per thread, 256 threads = 512KB registers)
    
    // Better: Use shared memory for current working set, stream through
    
    // Final approach: Use 64 warps (2048 threads) per batch, but that's too many
    // Actually use 256 threads, each handling 256 elements = 65536 total
    
    // Use shared memory as cache for butterfly partners
    
    // Re-thinking: Use 1D FFT with 256 threads, each doing 256-point sub-FFT
    // Then combine using twiddle factors (Cooley-Tukey decomposition)
    
    // Actually, let's implement a straightforward iterative FFT with global memory
    // and rely on L2 cache. Use 256 threads, each handling butterflies at different offsets
    
    // Each stage has N/2 butterflies
    // With 256 threads, each thread does N/512 = 128 butterflies per stage
    
    // Use shared memory for the current tile of 2048 floats (1024 complex)
    
    // New strategy: Process 1024 elements at a time in shared memory
    // Do 10 stages locally (1024 = 2^10), then write back
    // Then do remaining 6 stages with twiddle multiplication
    
    // Actually 65536 = 64 * 1024, so we can do:
    // - 64 parallel 1024-point FFTs (stages 0-9)
    // - Then 6 stages of combining with twiddle factors (mixed-radix)
    
    // Let's implement this mixed-radix approach
    
    const int sub_fft_size = 1024;  // 2^10
    const int num_sub_ffts = N / sub_fft_size;  // 64
    
    // Each thread block handles one batch
    // We need to do 64 1024-point FFTs
    
    // Stage 1: Do 1024-point FFTs on each of 64 segments
    // Use 256 threads, each thread handles 4 complex numbers per sub-FFT
    
    const int sub_fft_per_thread = sub_fft_size / blockDim.x;  // 4
    
    // For each of the 64 sub-FFTs
    for (int sub = 0; sub < num_sub_ffts; sub++) {
        // Load 1024 complex numbers into shared memory
        // 256 threads, each loads 4 complex numbers (8 floats)
        
        int base_idx = sub * sub_fft_size * 2;  // float index
        
        // Cooperative load
        for (int i = 0; i < sub_fft_per_thread; i++) {
            int local_idx = tid * sub_fft_per_thread + i;
            int global_idx = base_idx + local_idx * 2;
            
            if (local_idx < sub_fft_size) {
                s_data[local_idx * 2] = g_in[global_idx];
                s_data[local_idx * 2 + 1] = g_in[global_idx + 1];
            }
        }
        
        __syncthreads();
        
        // Do 10 stages of radix-2 FFT on 1024 points
        // Using Stockham algorithm (no bit reversal needed)
        
        for (int stage = 0; stage < 10; stage++) {
            int butterfly_width = 1 << stage;  // 1, 2, 4, ..., 512
            int group_size = butterfly_width * 2;  // 2, 4, 8, ..., 1024
            
            // Number of butterflies: 1024/2 = 512 per stage
            // Each thread does 512/256 = 2 butterflies
            
            const int butterflies_per_thread = sub_fft_size / 2 / blockDim.x;  // 2
            
            for (int b = 0; b < butterflies_per_thread; b++) {
                int butterfly_idx = tid * butterflies_per_thread + b;
                
                int group = butterfly_idx / butterfly_width;
                int pos_in_group = butterfly_idx % butterfly_width;
                
                int idx0 = group * group_size + pos_in_group;
                int idx1 = idx0 + butterfly_width;
                
                // Load from shared memory
                float a_r = s_data[idx0 * 2];
                float a_i = s_data[idx0 * 2 + 1];
                float b_r = s_data[idx1 * 2];
                float b_i = s_data[idx1 * 2 + 1];
                
                // Twiddle factor index
                int twiddle_idx = pos_in_group * (N / group_size);  // Scale for full FFT
                
                // Actually for sub-FFT, use local twiddle
                int local_twiddle = pos_in_group * (sub_fft_size / group_size);
                
                float w_r = d_twiddle_real[local_twiddle * (N / sub_fft_size)];  // Scale
                float w_i = d_twiddle_imag[local_twiddle * (N / sub_fft_size)];
                
                if (inverse) w_i = -w_i;
                
                butterfly(a_r, a_i, b_r, b_i, w_r, w_i);
                
                // Write back
                s_data[idx0 * 2] = a_r;
                s_data[idx0 * 2 + 1] = a_i;
                s_data[idx1 * 2] = b_r;
                s_data[idx1 * 2 + 1] = b_i;
            }
            
            __syncthreads();
        }
        
        // Store back to global memory
        for (int i = 0; i < sub_fft_per_thread; i++) {
            int local_idx = tid * sub_fft_per_thread + i;
            int global_idx = base_idx + local_idx * 2;
            
            if (local_idx < sub_fft_size) {
                g_out[global_idx] = s_data[local_idx * 2];
                g_out[global_idx + 1] = s_data[local_idx * 2 + 1];
            }
        }
        
        __syncthreads();
    }
    
    // Now do the remaining 6 stages (combining the 64 sub-FFTs)
    // This is the "twiddle" stage of Cooley-Tukey
    
    // We need to do butterflies between sub-FFTs
    // Stage 10: butterfly width = 1024, groups of 2048, across 2 sub-FFTs
    // ...
    // Stage 15: butterfly width = 32768, full N
    
    for (int stage = 10; stage < LOG2N; stage++) {
        int butterfly_width = 1 << stage;
        int group_size = butterfly_width * 2;
        
        // Number of butterflies per stage: N/2 = 32768
        // Each thread does 32768/256 = 128 butterflies
        
        const int butterflies_per_thread = N / 2 / blockDim.x;  // 128
        
        for (int b = 0; b < butterflies_per_thread; b++) {
            int butterfly_idx = tid * butterflies_per_thread + b;
            
            int group = butterfly_idx / butterfly_width;
            int pos_in_group = butterfly_idx % butterfly_width;
            
            int idx0 = group * group_size + pos_in_group;
            int idx1 = idx0 + butterfly_width;
            
            // Load from global memory
            float a_r = g_out[idx0 * 2];
            float a_i = g_out[idx0 * 2 + 1];
            float b_r = g_out[idx1 * 2];
            float b_i = g_out[idx1 * 2 + 1];
            
            // Twiddle factor
            int twiddle_idx = pos_in_group * (N / group_size);
            
            float w_r = d_twiddle_real[twiddle_idx];
            float w_i = d_twiddle_imag[twiddle_idx];
            if (inverse) w_i = -w_i;
            
            butterfly(a_r, a_i, b_r, b_i, w_r, w_i);
            
            // Write back
            g_out[idx0 * 2] = a_r;
            g_out[idx0 * 2 + 1] = a_i;
            g_out[idx1 * 2] = b_r;
            g_out[idx1 * 2 + 1] = b_i;
        }
        
        __syncthreads();
    }
    
    // Handle scaling for inverse FFT
    if (inverse) {
        __syncthreads();
        const int total_floats = N * 2;
        const int floats_per_thread = total_floats / blockDim.x;  // 512
        
        for (int i = 0; i < floats_per_thread; i++) {
            int idx = tid * floats_per_thread + i;
            if (idx < total_floats) {
                g_out[idx] = g_out[idx] / N;
            }
        }
    }
}

// Alternative: More efficient implementation using proper Cooley-Tukey
// with explicit bit-reversal or Stockham auto-sort

// Improved kernel using Stockham algorithm for first stages in shared memory
// and global memory for cross-tile stages

__global__ void fft_c2c_1d_65536_kernel_v2(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n,
    int batch_size,
    int inverse
) {
    extern __shared__ float s_mem[];
    
    const int batch = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;  // 256
    
    if (batch >= batch_size) return;
    
    const float* g_in = input + batch * N * 2;
    float* g_out = output + batch * N * 2;
    
    // We process 4096 complex numbers (8192 floats) per tile in shared memory
    // That's 2048 * 2 = 4096 complex, but we need power of 2
    // Actually use 2048 complex = 4096 floats
    
    const int tile_complex = 2048;  // 2^11
    const int num_tiles = N / tile_complex;  // 32 tiles
    
    // Each tile does 11 stages in shared memory
    // Remaining 5 stages done in global memory
    
    // Process each tile
    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_offset = tile * tile_complex * 2;
        
        // Load tile into shared memory
        // 256 threads, each loads 8 complex numbers (16 floats)
        for (int i = 0; i < tile_complex / nt; i++) {
            int local_idx = tid * (tile_complex / nt) + i;
            int global_idx = tile_offset + local_idx * 2;
            
            s_mem[local_idx * 2] = g_in[global_idx];
            s_mem[local_idx * 2 + 1] = g_in[global_idx + 1];
        }
        
        __syncthreads();
        
        // 11 stages of radix-2 FFT in shared memory
        for (int stage = 0; stage < 11; stage++) {
            int bw = 1 << stage;  // butterfly width
            int gs = bw * 2;      // group size
            
            // 2048/2 = 1024 butterflies, 256 threads -> 4 per thread
            int bpt = tile_complex / 2 / nt;  // 4
            
            for (int b = 0; b < bpt; b++) {
                int bid = tid * bpt + b;
                int grp = bid / bw;
                int pos = bid % bw;
                
                int i0 = grp * gs + pos;
                int i1 = i0 + bw;
                
                float ar = s_mem[i0 * 2];
                float ai = s_mem[i0 * 2 + 1];
                float br = s_mem[i1 * 2];
                float bi = s_mem[i1 * 2 + 1];
                
                // Local twiddle for this stage within tile
                int tw = pos * (tile_complex / gs);
                // Scale to full FFT twiddle table
                int tw_full = tw * (N / tile_complex);
                
                float wr = d_twiddle_real[tw_full];
                float wi = d_twiddle_imag[tw_full];
                if (inverse) wi = -wi;
                
                butterfly(ar, ai, br, bi, wr, wi);
                
                s_mem[i0 * 2] = ar;
                s_mem[i0 * 2 + 1] = ai;
                s_mem[i1 * 2] = br;
                s_mem[i1 * 2 + 1] = bi;
            }
            __syncthreads();
        }
        
        // Store back
        for (int i = 0; i < tile_complex / nt; i++) {
            int local_idx = tid * (tile_complex / nt) + i;
            int global_idx = tile_offset + local_idx * 2;
            
            g_out[global_idx] = s_mem[local_idx * 2];
            g_out[global_idx + 1] = s_mem[local_idx * 2 + 1];
        }
        
        __syncthreads();
    }
    
    // Remaining 5 stages (11 to 15) in global memory
    // These combine the 32 tiles
    
    for (int stage = 11; stage < LOG2N; stage++) {
        int bw = 1 << stage;
        int gs = bw * 2;
        
        // N/2 = 32768 butterflies, 256 threads -> 128 per thread
        int bpt = N / 2 / nt;  // 128
        
        for (int b = 0; b < bpt; b++) {
            int bid = tid * bpt + b;
            int grp = bid / bw;
            int pos = bid % bw;
            
            int i0 = grp * gs + pos;
            int i1 = i0 + bw;
            
            float ar = g_out[i0 * 2];
            float ai = g_out[i0 * 2 + 1];
            float br = g_out[i1 * 2];
            float bi = g_out[i1 * 2 + 1];
            
            int tw = pos * (N / gs);
            
            float wr = d_twiddle_real[tw];
            float wi = d_twiddle_imag[tw];
            if (inverse) wi = -wi;
            
            butterfly(ar, ai, br, bi, wr, wi);
            
            g_out[i0 * 2] = ar;
            g_out[i0 * 2 + 1] = ai;
            g_out[i1 * 2] = br;
            g_out[i1 * 2 + 1] = bi;
        }
        __syncthreads();
    }
    
    // Scale for inverse
    if (inverse) {
        __syncthreads();
        int fpt = N * 2 / nt;  // 512 floats per thread
        for (int i = 0; i < fpt; i++) {
            int idx = tid * fpt + i;
            if (idx < N * 2) {
                g_out[idx] = g_out[idx] / N;
            }
        }
    }
}

// Even better: Use full shared memory for 4096 complex (8192 floats) = 32KB
// and do 12 stages in shared memory, only 4 in global

__global__ void fft_c2c_1d_65536_kernel_v3(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n,
    int batch_size,
    int inverse
) {
    extern __shared__ float s_mem[];
    
    const int batch = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    
    if (batch >= batch_size) return;
    
    const float* g_in = input + batch * N * 2;
    float* g_out = output + batch * N * 2;
    
    // Use 4096 complex numbers = 8192 floats = 32KB shared memory
    const int tile_complex = 4096;  // 2^12
    const int num_tiles = N / tile_complex;  // 16 tiles
    
    // 12 stages in shared memory, 4 in global
    
    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_offset = tile * tile_complex * 2;
        
        // Load: 4096/256 = 16 complex per thread
        for (int i = 0; i < tile_complex / nt; i++) {
            int local_idx = tid * (tile_complex / nt) + i;
            int global_idx = tile_offset + local_idx * 2;
            s_mem[local_idx * 2] = g_in[global_idx];
            s_mem[local_idx * 2 + 1] = g_in[global_idx + 1];
        }
        __syncthreads();
        
        // 12 stages
        for (int stage = 0; stage < 12; stage++) {
            int bw = 1 << stage;
            int gs = bw * 2;
            int bpt = tile_complex / 2 / nt;  // 8 butterflies per thread
            
            for (int b = 0; b < bpt; b++) {
                int bid = tid * bpt + b;
                int grp = bid / bw;
                int pos = bid % bw;
                
                int i0 = grp * gs + pos;
                int i1 = i0 + bw;
                
                float ar = s_mem[i0 * 2], ai = s_mem[i0 * 2 + 1];
                float br = s_mem[i1 * 2], bi = s_mem[i1 * 2 + 1];
                
                int tw = pos * (tile_complex / gs);
                int tw_full = tw * (N / tile_complex);
                
                float wr = d_twiddle_real[tw_full];
                float wi = d_twiddle_imag[tw_full];
                if (inverse) wi = -wi;
                
                butterfly(ar, ai, br, bi, wr, wi);
                
                s_mem[i0 * 2] = ar; s_mem[i0 * 2 + 1] = ai;
                s_mem[i1 * 2] = br; s_mem[i1 * 2 + 1] = bi;
            }
            __syncthreads();
        }
        
        // Store
        for (int i = 0; i < tile_complex / nt; i++) {
            int local_idx = tid * (tile_complex / nt) + i;
            int global_idx = tile_offset + local_idx * 2;
            g_out[global_idx] = s_mem[local_idx * 2];
            g_out[global_idx + 1] = s_mem[local_idx * 2 + 1];
        }
        __syncthreads();
    }
    
    // 4 remaining stages in global memory
    for (int stage = 12; stage < LOG2N; stage++) {
        int bw = 1 << stage;
        int gs = bw * 2;
        int bpt = N / 2 / nt;  // 128
        
        for (int b = 0; b < bpt; b++) {
            int bid = tid * bpt + b;
            int grp = bid / bw;
            int pos = bid % bw;
            
            int i0 = grp * gs + pos;
            int i1 = i0 + bw;
            
            float ar = g_out[i0 * 2], ai = g_out[i0 * 2 + 1];
            float br = g_out[i1 * 2], bi = g_out[i1 * 2 + 1];
            
            int tw = pos * (N / gs);
            
            float wr = d_twiddle_real[tw];
            float wi = d_twiddle_imag[tw];
            if (inverse) wi = -wi;
            
            butterfly(ar, ai, br, bi, wr, wi);
            
            g_out[i0 * 2] = ar; g_out[i0 * 2 + 1] = ai;
            g_out[i1 * 2] = br; g_out[i1 * 2 + 1] = bi;
        }
        __syncthreads();
    }
    
    if (inverse) {
        __syncthreads();
        int fpt = N * 2 / nt;
        for (int i = 0; i < fpt; i++) {
            int idx = tid * fpt + i;
            if (idx < N * 2) g_out[idx] /= N;
        }
    }
}

// Host function to initialize twiddle factors
static bool twiddle_initialized = false;

void init_twiddle_factors(cudaStream_t stream) {
    if (twiddle_initialized) return;
    
    float* h_real = new float[N];
    float* h_imag = new float[N];
    
    for (int i = 0; i < N; i++) {
        float angle = -2.0f * M_PI * i / N;
        h_real[i] = cosf(angle);
        h_imag[i] = sinf(angle);
    }
    
    cudaMemcpyToSymbolAsync(d_twiddle_real, h_real, N * sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(d_twiddle_imag, h_imag, N * sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    
    delete[] h_real;
    delete[] h_imag;
    
    twiddle_initialized = true;
}

extern "C" {

void launch_fft_c2c_1d_65536_fp32(
    const void* input,
    void* output,
    int N_arg,
    int batch_size,
    int inverse,
    cudaStream_t stream
) {
    // Initialize twiddle factors on first call
    init_twiddle_factors(stream);
    
    // Use v3 kernel with 4096-complex tiles
    const int tile_complex = 4096;
    const int shared_mem = tile_complex * 2 * sizeof(float);  // 32768 bytes
    
    dim3 grid(batch_size);
    dim3 block(256);  // 256 threads
    
    fft_c2c_1d_65536_kernel_v3<<<grid, block, shared_mem, stream>>>(
        (const float*)input,
        (float*)output,
        N_arg,
        batch_size,
        inverse
    );
}

} // extern "C"
