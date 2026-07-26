#include <cuda_runtime.h>
#include "reduce.h"

// v3 解决线程 idle 有一半的线程只负责加载数据，可以在加载数据的时候让他们做一次加法
__global__ void reduce_v3(float* input, float* output, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = (blockDim.x*2) * blockIdx.x + tid;

    int val=0.0f;
    if(gid<n){
        val+=input[gid];
    }
    if(gid+blockDim.x<n){
        val+=input[gid+blockDim.x];
    }
    smem[tid] = val;
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
    reduce_v3<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        const_cast<float*>(input), output, N);
    cudaDeviceSynchronize();
}
