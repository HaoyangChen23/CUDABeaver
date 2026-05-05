import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for min reduction over a dimension
min_reduce_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

template <int BLOCK_SIZE>
__global__ void min_reduce_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                   int batch_size, int dim1, int dim2, int reduce_dim) {
    // reduce_dim: 0=batch, 1=dim1, 2=dim2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float sdata[BLOCK_SIZE];
    
    if (reduce_dim == 1) {
        // Reduce over dim1: input[batch, dim1, dim2] -> output[batch, dim2]
        int total = batch_size * dim2;
        if (idx >= total) return;
        
        int b = idx / dim2;
        int d2 = idx % dim2;
        
        float min_val = FLT_MAX;
        for (int d1 = 0; d1 < dim1; d1++) {
            float val = input[b * dim1 * dim2 + d1 * dim2 + d2];
            min_val = fminf(min_val, val);
        }
        output[idx] = min_val;
    }
    else if (reduce_dim == 2) {
        // Reduce over dim2: input[batch, dim1, dim2] -> output[batch, dim1]
        int total = batch_size * dim1;
        if (idx >= total) return;
        
        int b = idx / dim1;
        int d1 = idx % dim1;
        
        float min_val = FLT_MAX;
        for (int d2 = 0; d2 < dim2; d2++) {
            float val = input[b * dim1 * dim2 + d1 * dim2 + d2];
            min_val = fminf(min_val, val);
        }
        output[idx] = min_val;
    }
    else if (reduce_dim == 0) {
        // Reduce over batch: input[batch, dim1, dim2] -> output[dim1, dim2]
        int total = dim1 * dim2;
        if (idx >= total) return;
        
        int d1 = idx / dim2;
        int d2 = idx % dim2;
        
        float min_val = FLT_MAX;
        for (int b = 0; b < batch_size; b++) {
            float val = input[b * dim1 * dim2 + d1 * dim2 + d2];
            min_val = fminf(min_val, val);
        }
        output[idx] = min_val;
    }
}

torch::Tensor min_reduce_cuda(torch::Tensor input, int dim) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int dim1 = sizes[1];
    int dim2 = sizes[2];
    
    torch::Tensor output;
    int total_elements;
    
    if (dim == 0) {
        output = torch::empty({dim1, dim2}, input.options());
        total_elements = dim1 * dim2;
    } else if (dim == 1) {
        output = torch::empty({batch_size, dim2}, input.options());
        total_elements = batch_size * dim2;
    } else { // dim == 2
        output = torch::empty({batch_size, dim1}, input.options());
        total_elements = batch_size * dim1;
    }
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    min_reduce_kernel<256><<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(),
        batch_size, dim1, dim2, dim
    );
    
    return output;
}
"""

min_reduce_cpp_source = "torch::Tensor min_reduce_cuda(torch::Tensor input, int dim);"

# Compile the inline CUDA code for min reduction
min_reduce = load_inline(
    name="min_reduce",
    cpp_sources=min_reduce_cpp_source,
    cuda_sources=min_reduce_source,
    functions=["min_reduce_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over a specific dimension using custom CUDA kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.min_reduce = min_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies min reduction over the specified dimension to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        if not x.is_cuda:
            x = x.cuda()
        return self.min_reduce.min_reduce_cuda(x, self.dim)


def get_inputs():
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]


def get_init_inputs():
    return [1]