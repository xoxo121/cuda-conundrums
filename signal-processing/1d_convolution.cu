// 1D Convolution

// Implement a program that performs a 1D convolution operation. Given an input array and a kernel (filter), compute the convolved output. The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied where it fully overlaps with the input.

// The input consists of two arrays:

// input: A 1D array of 32-bit floating-point numbers.
// kernel: A 1D array of 32-bit floating-point numbers representing the convolution kernel.
// The output should be written to the output array, which will have a size of input_size - kernel_size + 1.


// Constraints
// 1 ≤ input_size ≤ 1,500,000
// 1 ≤ kernel_size ≤ 2047
// kernel_size ≤ input_size



#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* __restrict__ input,
                                      const float* __restrict__ kernel,
                                      float* __restrict__ output,
                                      int input_size,
                                      int kernel_size) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int output_size = input_size - kernel_size + 1;

    if (idx >= output_size) {
        return;
    }

    const float* in = input + idx;
    float sum = 0.0f;
    for (int j = 0; j < kernel_size; ++j) {
        sum = fmaf(in[j], kernel[j], sum);
    }
    output[idx] = sum;
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}
