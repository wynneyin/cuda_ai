#include <cuda_runtime.h>
#include "reduce.h"

// v7 float4 向量化加载 + Grid Stride Loop + warp 内的 warp shuffle 代替 share memeory，两级规约算法，256 个线程 8 个 warp 每个 warp 内先规约，写入 share mem 然后再 load 回来在 warp 0 内规约
__global__ void reduce_v7(float* input, float* output, int n) {
    int tid=threadIdx.x;
    int gid=blockIdx.x*(blockDim.x*2)+tid;
    int lane =tid%32;  // warp 内的线程 id
    int wid=tid/32;     // warp id

    float4* input4= reinterpret_cast<float4*>(input); // 将输入数组转换为 float4 指针
    int n4=n/4; // 计算 float4 元素的数量   
    
    float val=0.0f;
    // Grid Stride Loop：每个线程以 gridDim.x * blockDim.x 为步长迭代
    for (int idx = blockIdx.x * blockDim.x + tid;
         idx < n4;
         idx += gridDim.x * blockDim.x)
    {
        float4 data = input4[idx];
        val += data.x + data.y + data.z + data.w;
    }   


    // 处理 n 不是 4 的倍数时的尾部元素
    int tail_start = n4 * 4;
    for (int idx = tail_start + blockIdx.x * blockDim.x + tid;
         idx < n;
         idx += gridDim.x * blockDim.x)
    {
        val += input[idx];
    }


    // Warp 内规约
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    __shared__ float warp_results[32];
    if (lane == 0) warp_results[wid] = val;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    if (tid == 0) output[blockIdx.x] = val;
}

extern "C" void solve(float* input, float* output, int N) {
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    int grid_size  = num_sms * 4;   // 432 for A100
    int block_size = 256;

    cudaMemset(output, 0, grid_size * sizeof(float));

    reduce_v7<<<grid_size, block_size>>>(input, output, N);
    cudaDeviceSynchronize();
}
