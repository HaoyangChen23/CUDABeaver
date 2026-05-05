import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_scale_bn_kernel(
    const float* y,
    const float* scale,
    const float* bn_weight,
    const float* bn_bias,
    float* out,
    int N, int C, float eps
) {
    int c = blockIdx.x;
    if (c >= C) return;

    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    int tid = threadIdx.x;
    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;

    float sc = scale[c];
    float w = bn_weight[c];
    float b = bn_bias[c];

    for (int i = tid; i < N; i += 256) {
        float val = y[i * C + c] * sc;
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    s_sum[tid] = thread_sum;
    s_sum_sq[tid] = thread_sum_sq;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / N;
    float var = s_sum_sq[0] / N - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float inv_std = rsqrtf(var + eps);

    if (tid == 0) {
        s_mean = mean;
        s_inv_std = inv_std;
    }
    __syncthreads();

    mean = s_mean;
    inv_std = s_inv_std;

    for (int i = tid; i < N; i += 256) {
        float val = y[i * C + c] * sc;
        out[i * C + c] = ((val - mean) * inv_std) * w + b;
    }
}

torch::Tensor fused_scale_bn_cuda(torch::Tensor y, torch::Tensor scale, torch::Tensor bn_weight, torch::Tensor bn_bias, float eps) {
    auto N = y.size(0);
    auto C = y.size(1);
    auto out = torch::empty_like(y);
    
    const int block_size = 256;
    const int num_blocks = C;
    
    fused_scale_bn_kernel<<<num_blocks, block_size>>>(
        y.data_ptr<float>(),
        scale.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, eps
    );
    
    return out;
}
"""

cpp_source = "torch::Tensor fused_scale_bn_cuda(torch::Tensor y, torch::Tensor scale, torch::Tensor bn_weight, torch::Tensor bn_bias, float eps);"

fused_op = load_inline(
    name="fused_scale_bn",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_scale_bn_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.eps = eps

    def forward(self, x):
        x = self.gemm(x)
        x = fused_op.fused_scale_bn_cuda(x, self.scale, self.bn.weight, self.bn.bias, self.eps)
        return x