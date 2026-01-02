// Count Array Element

// Write a GPU program that counts the number of elements with the integer value k in an array of 32-bit integers. The program should count the number of elements with k in an array. You are given an input array input of length N and integer k.

// Implementation Requirements
// Use only native features (external libraries are not permitted)
// The solve function signature must remain unchanged
// The final result must be stored in the output variable

// Constraints
// 1 ≤ N ≤ 100,000,000
// 1 ≤ input[i], k ≤ 100,000


#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N){
        if(input[i] == K){
            // Use atomicAdd to safely increment the global counter 
            // across multiple threads/blocks to race condition.
            atomicAdd(output, 1);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}
