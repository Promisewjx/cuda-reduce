// include/reduce_kernels.h
#include <cuda_runtime.h>

// template <typename Op>
// __global__ void grid_reduce_kernel_op(const float* input, float* output, int N, Op op) {
//     extern __shared__ float sdata[];

//     int tid = threadIdx.x;
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     float val = op.identity();  // 初始值
//     if (idx < N) val = input[idx];
//     sdata[tid] = val;
//     __syncthreads();

//     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             sdata[tid] = op(sdata[tid], sdata[tid + s]);
//         }
//         __syncthreads();
//     }

//     if (tid == 0)
//         output[blockIdx.x] = sdata[0];
// }

// template <typename Op>
// void launch_grid_reduce_op(const float* d_input, float* d_output, int N, Op op) {
//     int blockSize = 256;
//     int gridSize = (N + blockSize - 1) / blockSize;

//     float* temp_output;
//     cudaMalloc(&temp_output, gridSize * sizeof(float));

//     grid_reduce_kernel_op<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, temp_output, N, op);
//     grid_reduce_kernel_op<<<1, blockSize, blockSize * sizeof(float)>>>(temp_output, d_output, gridSize, op);

//     cudaFree(temp_output);
// }
