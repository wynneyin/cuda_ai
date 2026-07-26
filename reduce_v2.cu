#include <cuda_runtime.h>
#include "reduce.h"

// v2: shared mem + 反转步长顺序寻址（s = blockDim/2 → 1）
// 修复 v1 bank conflict：相邻线程访问相邻地址，32 个线程映射到 32 个不同 bank
__global__ void reduce_v2(float* input, float* output, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, smem[0]);
}

extern "C" void solve(const float* input, float* output, int N) {
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    cudaMemset(output, 0, sizeof(float));
    reduce_v2<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        const_cast<float*>(input), output, N);
    cudaDeviceSynchronize();
}
