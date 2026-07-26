#include <cuda_runtime.h>
#include "reduce.h"


// v5 完全循环展开
template <unsigned int BLOCK_SIZE>
__global__ void reduce_v5(float* input, float* output, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = (BLOCK_SIZE*2) * blockIdx.x + tid;

    float val=0.0f;
    if(gid<n){
        val+=input[gid];
    }
    if(gid+blockDim.x<n){
        val+=input[gid+blockDim.x];
    }
    smem[tid] = val;
    __syncthreads();


    // 编译期展开：BLOCK_SIZE 已知，不满足的分支会被编译器直接删除
    if (BLOCK_SIZE >= 512) { if (tid < 256) smem[tid] += smem[tid + 256]; __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (tid <  64) smem[tid] += smem[tid +  64]; __syncthreads(); }

    if(tid < 32) {
        volatile float* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) atomicAdd(output, smem[0]);
}

extern "C" void solve(const float* input, float* output, int N) {
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    cudaMemset(output, 0, sizeof(float));
    switch (blockSize) {
    case 512: reduce_v5<512><<<gridSize, 512, 512*sizeof(float)>>>(const_cast<float*>(input), output, N); break;
    case 256: reduce_v5<256><<<gridSize, 256, 256*sizeof(float)>>>(const_cast<float*>(input), output, N); break;
    case 128: reduce_v5<128><<<gridSize, 128, 128*sizeof(float)>>>(const_cast<float*>(input), output, N); break;
}

    cudaDeviceSynchronize();
}
