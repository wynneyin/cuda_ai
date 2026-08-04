#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void softmax_v0(float* input, float* output, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float* x = input + row * N;   // 本行输入的起始指针
    float* y = output + row * N;  // 本行输出的起始指针

    // 找最大值（数值稳定）
    float max_val = -INFINITY;
    for (int i = 0; i < N; i++)
        max_val = max(max_val, x[i]);

    // 计算指数和
    float sum = 0.0f;
    for (int i = 0; i < N; i++)
        sum += expf(x[i] - max_val);

    // 归一化
    for (int i = 0; i < N; i++)
        y[i] = expf(x[i] - max_val) / sum;
}

int main() {
    const int M = 256;  // 256行，刚好填满2个block
    const int N = 8;    // 每行8个元素

    float h_input[M * N];
    for (int i = 0; i < M * N; i++)
        h_input[i] = (float)(i % N) - N / 2.0f;  // 每行值为 -4,-3,-2,-1,0,1,2,3
    float h_output[M * N] = {};

    // 分配 device 内存
    float *d_input, *d_output;
    cudaMalloc(&d_input,  M * N * sizeof(float));
    cudaMalloc(&d_output, M * N * sizeof(float));

    // 拷贝输入到 GPU
    cudaMemcpy(d_input, h_input, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel：每个线程处理一行，共 M 个线程
    int BLOCK_SIZE = 128;
    dim3 block(BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    softmax_v0<<<grid, block>>>(d_input, d_output, M, N);

    // 拷贝结果回 CPU
    cudaMemcpy(h_output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < M; i++) {
        float row_sum = 0.0f;
        printf("row %d: ", i);
        for (int j = 0; j < N; j++) {
            printf("%.4f ", h_output[i * N + j]);
            row_sum += h_output[i * N + j];
        }
        printf("(sum=%.4f)\n", row_sum);  // 应该等于1
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
