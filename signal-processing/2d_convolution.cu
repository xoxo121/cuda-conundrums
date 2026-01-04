// 2D Convolution

// Write a program that performs a 2D convolution operation on the GPU. Given an input matrix and a kernel (filter), compute the convolved output. The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied where it fully overlaps with the input.

// The input consists of:

// input: A 2D matrix of 32-bit floating-point numbers, represented as a 1D array in row-major order.
// kernel: A 2D kernel (filter) of 32-bit floating-point numbers, also represented as a 1D array in row-major order.
// The output should be written to the output matrix (also a 1D array in row-major order). The output matrix will have dimensions:

// output_rows = input_rows - kernel_rows + 1
// output_cols = input_cols - kernel_cols + 1
 
// Constraints
// 1 ≤ input_rows, input_cols ≤ 3072
// 1 ≤ kernel_rows, kernel_cols ≤ 31
// kernel_rows ≤ input_rows
// kernel_cols ≤ input_cols


#include <cuda_runtime.h>

#define MAX_K 31
#define TILE 16
#define SHARED_DIM (TILE + MAX_K - 1)

__constant__ float c_kernel[MAX_K * MAX_K];

__global__ void conv2d_tiled_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                    int in_h, int in_w, int k_h, int k_w) {
    
    __shared__ float s_tile[SHARED_DIM][SHARED_DIM];

    int out_h = in_h - k_h + 1;
    int out_w = in_w - k_w + 1;

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // Load tile + halo into shared memory
    for (int y = threadIdx.y; y < TILE + k_h - 1; y += TILE) {
        for (int x = threadIdx.x; x < TILE + k_w - 1; x += TILE) {
            int iy = blockIdx.y * TILE + y;
            int ix = blockIdx.x * TILE + x;

            if (iy < in_h && ix < in_w)
                s_tile[y][x] = input[iy * in_w + ix];
            else
                s_tile[y][x] = 0.0f;
        }
    }

    __syncthreads();

    if (row < out_h && col < out_w) {
        float val = 0.0f;
        #pragma unroll
        for (int i = 0; i < k_h; ++i) {
            for (int j = 0; j < k_w; ++j) {
                val += s_tile[threadIdx.y + i][threadIdx.x + j] * c_kernel[i * k_w + j];
            }
        }
        output[row * out_w + col] = val;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, 
                      int in_h, int in_w, int k_h, int k_w) {
    
    int out_h = in_h - k_h + 1;
    int out_w = in_w - k_w + 1;

    if (out_h <= 0 || out_w <= 0) return;

    cudaMemcpyToSymbol(c_kernel, kernel, k_h * k_w * sizeof(float));

    dim3 block(TILE, TILE);
    dim3 grid((out_w + TILE - 1) / TILE, (out_h + TILE - 1) / TILE);

    conv2d_tiled_kernel<<<grid, block>>>(input, output, in_h, in_w, k_h, k_w);
    
    cudaDeviceSynchronize();
}