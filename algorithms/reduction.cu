// Reduction

// Write a GPU program that performs parallel reduction on an array of 32-bit floating point numbers to compute their sum. The program should take an input array and produce a single output value containing the sum of all elements.

// Implementation Requirements
// Use only GPU native features (external libraries are not permitted)
// The solve function signature must remain unchanged
// The final result must be stored in the output variable

// Constraints
// 1 ≤ N ≤ 100,000,000
// -1000.0 ≤ input[i] ≤ 1000.0
// The final sum will always fit within a 32-bit float

#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Warp-level reduction uses shuffle instructions to sum values
__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduction_kernel(const float* input, float* output, int N){
    // One partial sum per warp (max 1024 threads/block => 32 warps).
    __shared__ float warp_sums[32];

    float sum = 0.0f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    // Grid-stride loop
    while (i < N){
        sum += input[i];
        i += grid_size;
    }

    // reduce within each warp
    sum = warpReduce(sum);

    // write the result of each warp to shared memory
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // lane 0 of each warp writes that warp's sum to shared memory.
    if (lane == 0) warp_sums[wid] = sum;
    __syncthreads();

    // have only warp 0 reduce the per-warp sums.
    if (wid == 0){
        // First (blockDim/32) lanes load a warp sum; other lanes contribute 0.
        sum = (threadIdx.x < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        sum = warpReduce(sum);

        if (lane == 0){
            // one value per block remains. Atomically add it into the single
            // global output to combine results from all blocks.
            atomicAdd(output, sum);
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    cudaMemset(output, 0, sizeof(float));

    if (N <= 0) return;

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 2048) blocksPerGrid = 2048;

    reduction_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}
