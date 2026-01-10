// Histogramming

// Write a GPU program that computes the histogram of an array of 32-bit integers. The histogram should count the number of occurrences of each integer value in the range [0, num_bins). You are given an input array input of length N and the number of bins num_bins.

// The result should be an array of integers of length num_bins, where each element represents the count of occurrences of its corresponding index in the input array.

// Constraints
// 1 ≤ N ≤ 100,000,000
// 0 ≤ input[i] < num_bins
// 1 ≤ num_bins ≤ 1024

#include <cuda_runtime.h>

__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins) {
    extern __shared__ int s_hist[];

    // Initialize shared memory
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute local histogram in shared memory
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        int val = input[i];
        // Ensure the value is within bounds
        if (val >= 0 && val < num_bins) {
            atomicAdd(&s_hist[val], 1);
        }
    }
    __syncthreads();

    //  Write results back to global memory
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        if (s_hist[i] > 0) {
            atomicAdd(&histogram[i], s_hist[i]);
        }
    }
}

extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    cudaMemset(histogram, 0, num_bins * sizeof(int));

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid > 1024) blocks_per_grid = 1024; 

    size_t shared_mem_size = num_bins * sizeof(int);

    histogram_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        input, histogram, N, num_bins
    );

    cudaDeviceSynchronize();
}