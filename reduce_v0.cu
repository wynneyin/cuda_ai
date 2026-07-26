#include <cuda_runtime.h>
#include "reduce.h"

// v0: shared mem + 交错寻址，tid % (2*step) == 0 做活跃线程选择
// 问题: 同一 warp 内部分线程走 if、部分不走 → warp divergence
__global__ void reduce_v0(float* input, float* output, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (int step = 1; step < blockDim.x; step *= 2) {
        if (tid % (2 * step) == 0) {
            if (tid + step < blockDim.x) {
                smem[tid] += smem[tid + step];
            }
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, smem[0]);
}

extern "C" void solve(const float* input, float* output, int N) {
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    cudaMemset(output, 0, sizeof(float));
    reduce_v0<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        const_cast<float*>(input), output, N);
    cudaDeviceSynchronize();
}
