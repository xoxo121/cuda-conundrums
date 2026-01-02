// Swish-Gated Linear Unit

// Implement the Swish-Gated Linear Unit (SWiGLU) activation function forward pass for 1D input vectors. Given an input tensor of shape [N] where N is the number of elements, compute the output using the elementwise formula. The input and output tensor must be of type float32.
 
// Implementation Requirements
// Use only native features (external libraries are not permitted)
// The solve function signature must remain unchanged
// The final result must be stored in the output tensor

// Constraints
// 1 ≤ N ≤ 100,000
// N is an even number
// -100.0 ≤ input values ≤ 100.0


#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < halfN) {
        float x = input[i];
        float g = input[i + halfN];

        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float swish_x = x * sigmoid_x;

        output[i] = swish_x * g;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
