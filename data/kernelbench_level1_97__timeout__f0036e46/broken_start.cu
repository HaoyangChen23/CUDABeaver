import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void sdpa_kernel(const float* Q, const float* K, const float* V, float* out, int B, int H, int N, int D) {
    int b = blockIdx.x / (H * N);
    int h = (blockIdx.x % (H * N)) / N;
    int i = blockIdx.x % N;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    int chunk_size = D / num_threads;
    int start_d = tid * chunk_size;
    
    __shared__ float scores[512];
    __shared__ float s_max[256];
    __shared__ float s_sum[256];
    
    // Compute raw scores
    for (int j = tid; j < N; j += num_threads) {
        float sum = 0.0f;
        int q_offset = (b * H * N * D) + (h * N * D) + (i * D);
        int k_offset = (b * H * N * D) + (h * N * D) + (j * D);
        for (int d = 0; d < D; d++) {
            sum += Q[q_offset + d] * K[k_offset + d];
        }
        scores[j] = sum / sqrtf((float)D);
    }
    __syncthreads();
    
    // Find max
    float max_val = -1e30f;
    for (int j = tid; j < N; j += num_threads) {
        if (scores[j] > max_val) max_val = scores[j];
    }
    s_max[tid] = max_val;
    __syncthreads();
    
    // Reduction for max
    if (tid < 128) s_max[tid] = fmaxf(s_max[tid], s_max[tid+128]);
    __syncthreads();
    if (tid < 64) s_max[tid] = fmaxf(s_max[tid], s_max[tid+64]);
    __syncthreads();
    if (tid < 32) s_max[tid] = fmaxf(s_max[tid], s_max[tid+32]);
    __syncthreads();
    max_val = s_max[0];
    __syncthreads();
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int j = tid; j < N; j += num_threads) {
        float val = expf(scores[j] - max_val);
        scores[j] = val;
        sum_exp += val;
    }
    s_sum[tid] = sum_exp;
    __syncthreads();
    
    // Reduction for sum
    if (tid < 128) s_sum[tid] += s_sum[tid+128];
    __syncthreads();
    if (tid < 64) s_sum[tid] += s_sum[tid+64];
    __syncthreads();
    if (tid < 32) s_sum[tid] += s_sum[tid+32];
    __syncthreads();
    sum_exp = s_sum[0];
    __syncthreads();
    
    // Normalize
    for (int j = tid; j < N; j += num_threads) {
        scores[j] /= sum_exp;
    }
    __syncthreads();
    
    // Compute output
    for (int d = start_d; d < start_d + chunk_size; d++) {
        if (d < D) {
            float out_val = 0.0f;
            for (int j = 0; j < N; j++) {
                int v_offset = (b * H * N * D) + (h * N * D) + (j * D) + d;
                out_val += scores[j] * V[v_offset];
            }
            int out_offset = (b * H * N * D) + (h * N * D) + (i * D) + d;
            out[out_offset] = out_val;
        }
    }
}

torch::Tensor sdpa_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    
    auto B = Q.size(0);
    auto H = Q.size(1);
    auto N = Q.size(2);
    auto D = Q.size(3);
    
    auto out = torch::empty_like(Q);
    
    int blocks = B * H * N;
    int threads = 256;
    
    sdpa_kernel<<<blocks, threads>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        out.data_ptr<float>(),
        B, H, N, D
    );
    return out;
}
"""

cpp_source = "torch::Tensor sdpa_cuda(torch::Tensor Q, torch::Tensor K, torch::Tensor V);"

sdpa_op = load_inline(
    name="sdpa_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["sdpa_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-use_fast_math"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.sdpa = sdpa_op

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self.sdpa.sdpa_cuda(Q, K, V)