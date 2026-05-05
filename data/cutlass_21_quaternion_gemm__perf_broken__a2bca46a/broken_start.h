#pragma once
#include <cuda_runtime.h>

// Quaternion multiplication using Hamilton product
// a = (ax, ay, az, aw), b = (bx, by, bz, bw)
// result.x = a.w*b.x + b.w*a.x + a.y*b.z - a.z*b.y
// result.y = a.w*b.y + b.w*a.y + a.z*b.x - a.x*b.z
// result.z = a.w*b.z + b.w*a.z + a.x*b.y - a.y*b.x
// result.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
__device__ inline void quat_mul(const float* a, const float* b, float* out) {
    float ax = a[0], ay = a[1], az = a[2], aw = a[3];
    float bx = b[0], by = b[1], bz = b[2], bw = b[3];
    
    out[0] = aw * bx + bw * ax + ay * bz - az * by;
    out[1] = aw * by + bw * ay + az * bx - ax * bz;
    out[2] = aw * bz + bw * az + ax * by - ay * bx;
    out[3] = aw * bw - ax * bx - ay * by - az * bz;
}

// Quaternion addition: out = a + b
__device__ inline void quat_add(const float* a, const float* b, float* out) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    out[3] = a[3] + b[3];
}

// Quaternion scalar multiply: out = s * a
__device__ inline void quat_scale(float s, const float* a, float* out) {
    out[0] = s * a[0];
    out[1] = s * a[1];
    out[2] = s * a[2];
    out[3] = s * a[3];
}

// Each thread computes one quaternion element of C (4 floats)
// A is row-major, B is column-major, C is row-major
__global__ void quaternion_gemm_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float beta,
    float* __restrict__ C, int ldc) {
    
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row of C
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col of C
    
    if (i >= M || j >= N) return;
    
    // Accumulator for quaternion result
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float temp[4];
    float a_quat[4];
    float b_quat[4];
    
    // Compute dot product over K dimension
    for (int k = 0; k < K; ++k) {
        // Load A[i, k] - row-major: A[(i * lda + k) * 4]
        const float* a_ptr = A + ((size_t)i * lda + k) * 4;
        a_quat[0] = a_ptr[0];
        a_quat[1] = a_ptr[1];
        a_quat[2] = a_ptr[2];
        a_quat[3] = a_ptr[3];
        
        // Load B[k, j] - column-major: B[(k + j * ldb) * 4]
        const float* b_ptr = B + ((size_t)k + (size_t)j * ldb) * 4;
        b_quat[0] = b_ptr[0];
        b_quat[1] = b_ptr[1];
        b_quat[2] = b_ptr[2];
        b_quat[3] = b_ptr[3];
        
        // temp = a_quat * b_quat (Hamilton product)
        quat_mul(a_quat, b_quat, temp);
        
        // acc += temp
        acc[0] += temp[0];
        acc[1] += temp[1];
        acc[2] += temp[2];
        acc[3] += temp[3];
    }
    
    // Apply alpha: acc = alpha * acc
    quat_scale(alpha, acc, acc);
    
    // Load original C[i, j] and apply beta
    float* c_ptr = C + ((size_t)i * ldc + j) * 4;
    float c_orig[4];
    c_orig[0] = c_ptr[0];
    c_orig[1] = c_ptr[1];
    c_orig[2] = c_ptr[2];
    c_orig[3] = c_ptr[3];
    
    float beta_scaled[4];
    quat_scale(beta, c_orig, beta_scaled);
    
    // result = acc + beta * C
    float result[4];
    quat_add(acc, beta_scaled, result);
    
    // Store result
    c_ptr[0] = result[0];
    c_ptr[1] = result[1];
    c_ptr[2] = result[2];
    c_ptr[3] = result[3];
}

cudaError_t QuaternionGemm(
    int M, int N, int K,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float *C, int ldc) {
    
    // Use 16x16 thread blocks for good occupancy
    // Each thread computes one quaternion (4 floats) of the output
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);
    
    quaternion_gemm_kernel<<<gridSize, blockSize>>>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    
    return cudaGetLastError();
}
