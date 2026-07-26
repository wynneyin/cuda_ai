#include <cuda_runtime.h>
#include "reduce.h"

__device__ void warpReduce(volatile float* smem, int tid) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
}

// v4 当 step <=32 的时候只有一个 Warp 在工作，可以直接展开循环，省去了__syncthreads()的开销
__global__ void reduce_v4(float* input, float* output, int n) {
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

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

     // 最后一个 Warp 内的规约，无需 __syncthreads()
    if (tid < 32) {
        warpReduce(smem, tid);
    }

    if (tid == 0) atomicAdd(output, smem[0]);
}

extern "C" void solve(const float* input, float* output, int N) {
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    cudaMemset(output, 0, sizeof(float));
    reduce_v4<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        const_cast<float*>(input), output, N);
    cudaDeviceSynchronize();
}
