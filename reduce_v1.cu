#include <cuda_runtime.h>
#include "reduce.h"

// v1: shared mem + 连续线程寻址（index = tid * 2 * s）
// 修复 v0 warp divergence：活跃线程从低 tid 连续排列，整个 warp 要么全活跃要么全不活跃
// 残留问题: stride 从小到大 → 访问 smem[0], smem[2], smem[4]... → bank conflict
__global__ void reduce_v1(float* input, float* output, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + tid;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = tid * 2 * s;
        if (index < blockDim.x) {
            smem[index] += smem[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, smem[0]);
}

extern "C" void solve(const float* input, float* output, int N) {
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    cudaMemset(output, 0, sizeof(float));
    reduce_v1<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        const_cast<float*>(input), output, N);
    cudaDeviceSynchronize();
}
