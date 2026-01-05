// Dot Product

// Implement a GPU program that computes the dot product of two vectors containing 32-bit floating point numbers. The dot product is the sum of the products of the corresponding elements of two vectors.

// Constraints
// A and B have identical lengths
// 1 ≤ N ≤ 100,000,000


#include <cuda_runtime.h>

__global__ void dot_product_kernel(const float* A, const float* B, float* result, int N){
    extern __shared__ float shared_data[];

    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute partial dot product
    float tmp = 0.0f;
    while (i < N){
        tmp += A[i] * B[i];
        i += blockDim.x * gridDim.x;
    }
    shared_data[t] = tmp;
    __syncthreads();

    // Reduce within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if(t < s){
            shared_data[t] += shared_data[t + s];
        }
        __syncthreads();
    }

    // Final reduction
    if(t == 0){
        atomicAdd(result, shared_data[0]);
    }
}
// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {

    cudaMemset(result, 0, sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 1024) blocksPerGrid = 1024;

    // Allocate shared memory
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    dot_product_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(A, B, result, N);
}
