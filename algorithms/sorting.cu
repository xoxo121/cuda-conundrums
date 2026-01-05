// Sorting

// Write a program that sorts an array of 32-bit floating-point numbers in ascending order. You are free to choose any sorting algorithm.

// Constraints
// 1 ≤ N ≤ 1,000,000


//it leads to out of bounds without shared memory

#include <cuda_runtime.h>
#include <float.h>

__global__ void bitonic_sort( float* data, int N, int j, int k){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int ixj = i^j;

    if (ixj > i){
        float a = (i < N) ? data[i] : FLT_MAX;
        float b = (ixj < N) ? data[ixj] : FLT_MAX;

        bool asc = (i & k) == 0;

        if (asc){
            if (a > b){
                if (ixj < N) data[i] = b;
                if (i < N) data[ixj] = a;
            }
        }
        else{
            if (a < b){
                if (ixj < N) data[i] = b;
                if (i < N) data[ixj] = a;
            }
        }
    }
}


extern "C" void solve(float* data, int N) {
    if (N <= 1) return;

    // Find the next power of 2 greater than or equal to N
    int n_pow2 = 1;
    while (n_pow2 < N) n_pow2 <<= 1;

    int threads = 256;
    int blocks = (n_pow2 / 2 + threads - 1) / threads;

    // Major step
    for (int k = 2; k <= n_pow2; k <<= 1) {
        // Minor step
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<blocks, threads>>>(data, N, j, k);
            cudaDeviceSynchronize();
        }
    }
}