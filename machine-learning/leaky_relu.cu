// Leaky ReLU

// Implement a program that performs the leaky ReLU activation function on a vector of floating-point numbers.
 

// Implementation Requirements
// External libraries are not permitted
// The solve function signature must remain unchanged
// The final result must be stored in vector output
// Use alpha = 0.01 as the leaky coefficient

// Constraints
// 1 ≤ N ≤ 100,000,000
// -1000.0 ≤ input[i] ≤ 1000.0


#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        float val = input[i];
        output[i] = (val > 0.0f) ? val : 0.01f * val;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
