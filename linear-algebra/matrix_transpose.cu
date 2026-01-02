// Matrix Transpose

// Write a program that transposes a matrix of 32-bit floating point numbers on a GPU. The transpose of a matrix switches its rows and columns. 
//Given a matrix A of dimensions MxN, the transpose of A will have dimensions NxM. All matrices are stored in row-major format.

// Implementation Requirements
// Use only native features (external libraries are not permitted)
// The solve function signature must remain unchanged
// The final result must be stored in the matrix output

// Constraints
// 1 ≤ rows, cols ≤ 8192
// Input matrix dimensions: rows × cols
// Output matrix dimensions: cols × rows

#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < cols && y < rows){
        int inp_idx = y*cols + x;
        int out_idx = x*rows + y;

        output[out_idx] = input[inp_idx];

    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
