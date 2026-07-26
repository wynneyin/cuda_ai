#include <cuda_runtime.h>
#include "reduce.h"

// warp 内规约辅助函数
__device__ float warpReduceSum(float val){
    for(int offset=16;offset>0;offset>>=1){
        val+=__shfl_down_sync(0xffffffff,val,offset); //__shfl_down_sync 将 32 个值规约为 1 个值
    }
    return val;
}


// v6 版本 warp 内的 warp shuffle 代替 share memeory，两级规约算法，256 个线程 8 个 warp 每个 warp 内先规约，写入 share mem 然后再 load 回来在 warp 0 内规约
__global__ void reduce_v6(float* input, float* output, int n) {
    int tid=threadIdx.x;
    int gid=blockIdx.x*(blockDim.x*2)+tid;
    int lane =tid%32;  // warp 内的线程 id
    int wid=tid/32;     // warp id

    float val=0.0f;
    if(gid<n){  // 每个线程处理两个元素
        val+=input[gid]; 
    }
    if(gid+blockDim.x<n){
        val+=input[gid+blockDim.x];
    }

    // warp 内规约
    val=warpReduceSum(val);
    // 将每个 warp 的规约结果写入 share memory
    __shared__ float smem[32]; // 8 个 warp 每个 warp
    if(lane==0){
        smem[wid]=val;
    }
    __syncthreads();

    // warp 0 间规约
    int num_warps=(blockDim.x)/32;
    if(wid==0){
        val=(lane<num_warps)?smem[lane]:0.0f;
        val=warpReduceSum(val);
    }

    if (tid == 0) atomicAdd(output, val);
}

extern "C" void solve(const float* input, float* output, int N) {
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    cudaMemset(output, 0, sizeof(float));
    reduce_v6<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        const_cast<float*>(input), output, N);
    cudaDeviceSynchronize();
}
