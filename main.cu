#include <stdio.h>
#include <cuda_runtime.h>
#include "include/reduce_kernels.h"

int main() {
    const int N = 1000;
    float* h_input = new float[N];
    for (int i = 0; i < N; ++i) h_input[i] = i;

    float h_output = 0.0f;
    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, sizeof(float) * N, cudaMemcpyHostToDevice);

    printf("== Warp-Level Reduce ==\n");
    launch_warp_reduce(d_input, d_output, 32);
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: %.1f\n", h_output);  // 0+1+...+31 = 496

    // 只有第一个块
    printf("== Block-Level Reduce ==\n");
    launch_block_reduce(d_input, d_output, N);
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: %.1f\n", h_output);

    printf("== Grid-Level Reduce (Sum) ==\n");
    launch_grid_reduce_op(d_input, d_output, N, SumOp());
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Sum Result: %.1f\n", h_output);

    printf("== Grid-Level Reduce (Max) ==\n");
    launch_grid_reduce_op(d_input, d_output, N, MaxOp());
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Max Result: %.1f\n", h_output);

    printf("== Grid-Level Reduce (Product) ==\n");
    launch_grid_reduce_op(d_input, d_output, N, ProdOp());
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Product Result: %.1f\n", h_output);

    cudaFree(d_input); cudaFree(d_output); delete[] h_input;
    return 0;
}
