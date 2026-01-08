#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int hash = OFFSET_BASIS;

    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }

    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    // 1. Calculate the unique index for this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        unsigned int current_val = (unsigned int)input[i];

        // 3. Apply the hash R times iteratively
        for (int r = 0; r < R; r++) {
            // Note: fnv1a_hash takes an int, so we cast current_val 
            // to treat its bits as an integer input.
            current_val = fnv1a_hash((int)current_val);
        }

        // 4. Store the final result
        output[i] = current_val;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    // Calculate enough blocks to cover all N elements
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    
    // Synchronize to ensure the GPU finishes before returning to the caller
    cudaDeviceSynchronize();
}