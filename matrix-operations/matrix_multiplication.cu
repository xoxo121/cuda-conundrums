// Matrix Multiplication

// Write a program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix A of dimensions 
// MxN and matrix B of dimensions NxK, compute the product matrix C = AxB, which will have dimensions MxK. 
// All matrices are stored in row-major format.

// Implementation Requirements
// Use only native features (external libraries are not permitted)
// The solve function signature must remain unchanged
// The final result must be stored in matrix C


// Constraints
// 1 ≤ M, N, K ≤ 8192
// Performance is measured with M = 8192, N = 6144, K = 4096


#include <cuda_runtime.h>

#define TILE_SIZE 32

// GPU kernel for matrix multiplication using tiling and shared memory
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {

    // Shared memory tiles for sections of A and B
    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE];

    // Accumulator for the computed element of C
    float sum = 0.0;

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute global indices for row and column in the result matrix
    int col = blockDim.x * blockIdx.x + tx;
    int row = blockDim.y * blockIdx.y + ty;

    // Loop over all tiles needed to compute C[row][col]
    for (int t = 0; t < ((N + TILE_SIZE - 1) / TILE_SIZE); ++t) {

        // Load a tile of A into shared memory if within bounds
        if (row < M && t * TILE_SIZE + tx < N) {
            s_a[ty][tx] = A[row * N + (t * TILE_SIZE + tx)];
        } else {
            s_a[ty][tx] = 0.0;
        }

        // Load a tile of B into shared memory if within bounds
        if (col < K && t * TILE_SIZE + ty < N) {
            s_b[ty][tx] = B[(t * TILE_SIZE + ty) * K + col];
        } else {
            s_b[ty][tx] = 0.0;
        }

        // Wait until all threads have loaded their data
        __syncthreads();

        // Compute partial sums for the current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_a[ty][k] * s_b[k][tx];
        }

        // Wait before loading the next tile
        __syncthreads();
    }

    // Write the computed element to the output matrix
    if (row < M && col < K) {
        C[col + row * K] = sum;
    }
}

// A, B, and C are pointers to memory on the GPU (device memory)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((K + TILE_SIZE - 1) / TILE_SIZE,
                       (M + TILE_SIZE - 1) / TILE_SIZE);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    
    cudaDeviceSynchronize();
}
