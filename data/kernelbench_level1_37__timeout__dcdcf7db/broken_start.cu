import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cpp_source = r"""
torch::Tensor frobenius_norm_normalize_cuda(torch::Tensor x);
"""

_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__global__ void partial_sum_squares_kernel(
    const float* __restrict__ x,
    float* __restrict__ partial,
    int64_t n
) {
    extern __shared__ float sdata[];
    float sum = 0.0f;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        float v = x[i];
        sum += v * v;
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

__global__ void final_reduce_kernel(
    const float* __restrict__ partial,
    float* __restrict__ out_sum,
    int64_t n
) {
    extern __shared__ float sdata[];
    float sum = 0.0f;

    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        sum += partial[i];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_sum[0] = sdata[0];
    }
}

__global__ void normalize_kernel(
    const float* __restrict__ x,
    const float* __restrict__ sumsq,
    float* __restrict__ out,
    int64_t n
) {
    float norm = sqrtf(sumsq[0]);
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        out[i] = x[i] / norm;
    }
}

torch::Tensor frobenius_norm_normalize_cuda(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_FLOAT(x);
    CHECK_CONTIGUOUS(x);

    auto n = static_cast<int64_t>(x.numel());
    auto out = torch::empty_like(x);

    if (n == 0) {
        return out;
    }

    constexpr int threads = 256;
    int64_t max_blocks = 4096;
    int64_t blocks64 = (n + threads - 1) / threads;
    int blocks = static_cast<int>(blocks64 < max_blocks ? blocks64 : max_blocks);
    if (blocks < 1) blocks = 1;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto partial = torch::empty({blocks}, opts);
    auto sumsq = torch::empty({1}, opts);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    partial_sum_squares_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(
        x.data_ptr<float>(),
        partial.data_ptr<float>(),
        n
    );

    final_reduce_kernel<<<1, threads, threads * sizeof(float), stream>>>(
        partial.data_ptr<float>(),
        sumsq.data_ptr<float>(),
        blocks
    );

    normalize_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        sumsq.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_ext = None
if torch.cuda.is_available():
    _ext = load_inline(
        name="frobenius_norm_normalize_ext",
        cpp_sources=_cpp_source,
        cuda_sources=_cuda_source,
        functions=["frobenius_norm_normalize_cuda"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._ext = _ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.float32:
            if not x.is_contiguous():
                x = x.contiguous()
            return self._ext.frobenius_norm_normalize_cuda(x)
        norm = torch.norm(x, p='fro')
        return x / norm