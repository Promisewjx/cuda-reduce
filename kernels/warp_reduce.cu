#include <cuda_runtime.h>
#include <stdio.h>

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__global__ void warp_reduce_kernel(const float* input, float* output) {
    int tid = threadIdx.x;
    float val = input[tid];
    float sum = warp_reduce_sum(val);
    if (tid == 0) output[0] = sum;
}

void launch_warp_reduce(const float* d_input, float* d_output, int N) {
    warp_reduce_kernel<<<1, N>>>(d_input, d_output);
}
