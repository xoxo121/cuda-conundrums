// Softmax

// Write a program that computes the softmax function for an array of 32-bit floating-point numbers on a GPU.

// Your solution should handle potential overflow issues by using the "max trick". 
// Subtract the maximum value of the input array from each element before exponentiation.

// Constraints
// 1 ≤ N ≤ 500,000

#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
