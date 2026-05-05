import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GELU activation
gelu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float gelu_func(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float c = 0.044715f;
    
    float x3 = x * x * x;
    float inner = x + c * x3;
    float tanh_arg = sqrt_2_over_pi * inner;
    float tanh_val = tanhf(tanh_arg);
    
    return 0.5f * x * (1.0f + tanh_val);
}

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = gelu_func(input[idx]);
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    int size = input.numel();
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

gelu_cpp_source = (
    "torch::Tensor gelu_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for GELU
gelu_op = load_inline(
    name="gelu_custom",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_cuda_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs a GELU activation using custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_cuda = gelu_op.gelu_cuda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return self.gelu_cuda(x)