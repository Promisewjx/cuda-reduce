#include <cuda_runtime.h>
#include <stdio.h>

__global__ void block_reduce_kernel(const float* input, float* output, int N) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < N) val = input[idx];
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}


void launch_block_reduce(const float* d_input, float* d_output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t shared_bytes = threads * sizeof(float);

    block_reduce_kernel<<<blocks, threads, shared_bytes>>>(d_input, d_output, N);
}
