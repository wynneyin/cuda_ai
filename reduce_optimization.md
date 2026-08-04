# CUDA Reduce 优化之旅：从 Naive 到接近带宽极限

## 概述

Reduce（归约）是 GPU 计算中最基础的操作之一：把一个数组的所有元素累加成一个标量。本文通过 8 个递进版本（v0～v7），逐步展示每一步优化针对的瓶颈是什么，以及如何用代码解决。

每一个版本都只暴露同一个接口：

```c
extern "C" void solve(float* input, float* output, int N);
```

---

## v0：最朴素的版本——Warp Divergence

### 代码

```cuda
__global__ void reduce_v0(float* input, float* output, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (int step = 1; step < blockDim.x; step *= 2) {
        if (tid % (2 * step) == 0) {          // ← 问题在这里
            if (tid + step < blockDim.x) {
                smem[tid] += smem[tid + step];
            }
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, smem[0]);
}
```

### 思路

每一轮迭代，步长 `step` 翻倍，选出 `tid % (2*step) == 0` 的线程做加法：

```
Round 1 (step=1):  tid=0,2,4,6,...  做加法
Round 2 (step=2):  tid=0,4,8,12,... 做加法
Round 3 (step=4):  tid=0,8,16,...   做加法
...
```

### 问题：Warp Divergence

GPU 以 **warp（32 个线程）** 为单位执行指令。当同一 warp 内的线程走不同分支时，GPU 必须串行执行两个分支，效率减半（或更差）。

第一轮 `step=1`：warp 0 包含 tid=0..31，其中偶数线程走 if、奇数不走 → **同一 warp 内分支分歧**。

```
warp 0: tid[0,2,4,...30] 执行加法，tid[1,3,5,...31] 空转
        → 串行执行两个路径，吞吐减半
```

---

## v1：消除 Warp Divergence——活跃线程连续排列

### 代码

```cuda
for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    int index = tid * 2 * s;           // ← 改变寻址方式
    if (index < blockDim.x) {
        smem[index] += smem[index + s];
    }
    __syncthreads();
}
```

### 思路

把活跃线程改为从低 tid 连续排列：

```
Round 1 (s=1):  tid=0..127 活跃，tid=128..255 不参与
Round 2 (s=2):  tid=0..63  活跃
Round 3 (s=4):  tid=0..31  活跃
...
```

每一轮，活跃线程全在低位，整个 warp 要么全活跃要么全不活跃，消除了 warp divergence。

### 残留问题：Shared Memory Bank Conflict

CUDA shared memory 有 32 个 bank，每个 bank 宽度 4 字节。如果一个 warp 中的多个线程访问**同一 bank 的不同地址**，就会产生 bank conflict，导致串行化。

第一轮 `s=1`：
- tid=0 访问 `smem[0]` 和 `smem[1]`，步长为 2
- tid=1 访问 `smem[2]` 和 `smem[3]`，步长为 2

`smem[0]` 在 bank 0，`smem[2]` 在 bank 2，`smem[4]` 在 bank 4……看起来没冲突。但实际上 `index = tid * 2 * s`，当 s 较大时，间距扩大，多个线程会映射到同一 bank。

---

## v2：消除 Bank Conflict——反转步长

### 代码

```cuda
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {   // ← s 从大到小
    if (tid < s) {
        smem[tid] += smem[tid + s];
    }
    __syncthreads();
}
```

### 思路

步长从大到小（`blockDim/2 → 1`），活跃线程始终是 `tid < s` 的连续线程，每次访问的两个地址相差 `s`：

```
Round 1 (s=128): tid=0..127，smem[tid] += smem[tid+128]
  tid=0: smem[0] vs smem[128]  → bank 0 vs bank 0（但 128 个线程全不冲突）
  tid=1: smem[1] vs smem[129]  → bank 1 vs bank 1
  ...
```

相邻线程访问相邻地址（步长1），32 个线程映射到 32 个不同 bank，**无冲突**。

### 残留问题：线程利用率低

第一轮只有一半线程（128/256）有效工作，其余线程仅做了数据加载就闲置了。

---

## v3：提升线程利用率——加载时做一次加法

### 代码

```cuda
__global__ void reduce_v3(float* input, float* output, int n) {
    int gid = (blockDim.x * 2) * blockIdx.x + tid;   // ← 每个 block 处理两倍数据

    float val = 0.0f;
    if (gid < n)             val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];  // ← 加载时顺带加法

    smem[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, smem[0]);
}
```

### 思路

原来每个 block 处理 `blockDim.x` 个元素，现在让每个 block 处理 `blockDim.x * 2` 个元素。每个线程在加载数据到 shared memory 之前，先把两个元素加在一起：

```
线程 tid=0：
  v2 时：加载 input[gid] 到 smem[0]（一半线程 idle）
  v3 时：加载 input[gid] + input[gid+256] 到 smem[0]（所有线程都在工作）
```

这相当于把 v2 第一轮的计算提前到数据加载阶段，**所有线程从一开始就参与计算**，减少了一半的 block 数量，同时提高了内存访问效率。

---

## v4：消除 __syncthreads() 开销——Warp Unrolling

### 代码

```cuda
__device__ void warpReduce(volatile float* smem, int tid) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
}

__global__ void reduce_v4(float* input, float* output, int n) {
    // ...加载同 v3...

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {   // ← 只到 s>32
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid < 32) warpReduce(smem, tid);    // ← 最后一个 warp 手动展开

    if (tid == 0) atomicAdd(output, smem[0]);
}
```

### 思路

当 `s <= 32` 时，只有 1 个 warp（32 个线程）在工作。**同一 warp 内的线程天然同步**（lockstep 执行），不需要 `__syncthreads()`。

用 `volatile` 关键字防止编译器把 shared memory 读写缓存到寄存器，确保每次都从 shared memory 读取最新值。

```
s=32: 只有 tid 0..31 活跃 → 1 个 warp，无需 __syncthreads()
s=16: 同上
...
s=1:  同上
```

把最后 5 次 `__syncthreads()` 全部省掉，减少了同步开销。

---

## v5：完全循环展开——编译期优化

### 代码

```cuda
template <unsigned int BLOCK_SIZE>
__global__ void reduce_v5(float* input, float* output, int n) {
    // ...加载同 v3...

    // 编译期展开：不满足的分支被编译器直接删除
    if (BLOCK_SIZE >= 512) { if (tid < 256) smem[tid] += smem[tid + 256]; __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (tid <  64) smem[tid] += smem[tid +  64]; __syncthreads(); }

    if (tid < 32) {
        volatile float* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        // ...
    }
}
```

### 思路

用 C++ 模板将 `BLOCK_SIZE` 作为编译期常量。编译器在生成代码时：
- `BLOCK_SIZE=256` 时，`if (BLOCK_SIZE >= 512)` 的分支被**静态删除**，不产生任何代码
- 不满足的分支不仅不执行，甚至不存在于二进制中

相比 v4 中的运行时循环判断，v5 生成的是完全展开的直线代码，**没有循环控制开销，没有条件跳转**。

```cpp
// 调用时选择对应特化版本
switch (blockSize) {
    case 512: reduce_v5<512><<<...>>>(...); break;
    case 256: reduce_v5<256><<<...>>>(...); break;
    case 128: reduce_v5<128><<<...>>>(...); break;
}
```

---

## v6：用 Warp Shuffle 替代 Shared Memory——两级规约

### 代码

```cuda
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void reduce_v6(float* input, float* output, int n) {
    int lane = tid % 32;
    int wid  = tid / 32;

    // 每个线程累加两个元素（同 v3/v4）
    float val = 0.0f;
    if (gid < n)               val += input[gid];
    if (gid + blockDim.x < n)  val += input[gid + blockDim.x];

    // Level 1：warp 内规约，完全在寄存器中完成
    val = warpReduceSum(val);

    // 每个 warp 的 lane 0 把结果写到 shared memory
    __shared__ float smem[32];
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    // Level 2：warp 0 对 8 个 warp 结果再做一次 warp 规约
    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warpReduceSum(val);
    }

    if (tid == 0) atomicAdd(output, val);
}
```

### 思路

`__shfl_down_sync` 是 **warp 内线程间直接交换寄存器值**的指令，不经过 shared memory：

```
offset=16: val[0]  += val[16], val[1]  += val[17], ...
offset=8:  val[0]  += val[8],  val[1]  += val[9],  ...
offset=4:  val[0]  += val[4],  ...
offset=2:  val[0]  += val[2],  ...
offset=1:  val[0]  += val[1]
→ val[0] 包含 32 个线程的总和
```

**两级规约架构**（256 线程 = 8 个 warp）：

```
Level 1: 每个 warp 内 32 个线程 → 1 个值（8 次 shuffle，纯寄存器）
Level 2: 8 个 warp 的结果 → warp 0 再做一次 warp 规约

Shared memory 仅用于 8 个值的中转（极小的 smem 压力）
```

**收益**：shared memory 的访问延迟约 20-30 个 cycle，而 warp shuffle 只需 ~4 个 cycle。

---

## v7：向量化读取 + Grid Stride Loop——突破内存带宽

### 代码

```cuda
__global__ void reduce_v7(float* input, float* output, int n) {
    float4* input4 = reinterpret_cast<float4*>(input);
    int n4 = n / 4;

    float val = 0.0f;

    // Grid Stride Loop：每个线程处理多个 float4
    for (int idx = blockIdx.x * blockDim.x + tid;
         idx < n4;
         idx += gridDim.x * blockDim.x)
    {
        float4 data = input4[idx];
        val += data.x + data.y + data.z + data.w;
    }

    // 处理尾部（n 不是 4 的倍数）
    int tail_start = n4 * 4;
    for (int idx = tail_start + blockIdx.x * blockDim.x + tid;
         idx < n;
         idx += gridDim.x * blockDim.x)
    {
        val += input[idx];
    }

    // Warp 规约（同 v6）
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    __shared__ float warp_results[32];
    if (lane == 0) warp_results[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (tid == 0) output[blockIdx.x] = val;  // ← 注意：写到 output[blockIdx.x]，不用 atomicAdd
}

extern "C" void solve(float* input, float* output, int N) {
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    int grid_size  = num_sms * 4;   // A100: 108 * 4 = 432
    int block_size = 256;

    cudaMemset(output, 0, grid_size * sizeof(float));
    reduce_v7<<<grid_size, block_size>>>(input, output, N);
    cudaDeviceSynchronize();
    // 注：此处 output[0..grid_size-1] 存各 block 结果，调用方可做第二级归约
}
```

### 两个关键优化

#### 1. float4 向量化读取

```cuda
float4* input4 = reinterpret_cast<float4*>(input);
float4 data = input4[idx];
val += data.x + data.y + data.z + data.w;
```

一条 128-bit 的宽加载指令 = 4 个 float，相比 4 次 32-bit 加载：
- **减少内存事务数量**（1 次 vs 4 次）
- **更好地利用内存总线带宽**
- 编译器能更好地做指令级并行

**前提**：input 必须 16 字节对齐（通常 cudaMalloc 保证对齐）。

#### 2. Grid Stride Loop

```
传统方式：grid 中的 block 数 = 数据量 / 每 block 处理量
          N=10^8 时需要数万个 block

Grid Stride Loop：固定 grid_size = num_sms * 4（A100 上约 432）
                  每个 block 以步长 grid_size * block_size 迭代整个数组
```

**为什么固定小 grid 更好？**

- **隐藏内存延迟**：每个 SM 有多个 block 驻留，可以在等待内存时切换执行其他 block
- **减少 kernel 启动开销**：block 数越多，调度开销越大
- **更好的 L2 Cache 复用**：局部性更强

```
A100: 108 个 SM，grid_size = 432
每个 block 迭代：N / (432 * 256) ≈ 9000 次（N=10^8 时）
每次迭代加载 float4 = 16 bytes → 每个线程总加载 ≈ 144KB
```

---

## 优化总结

| 版本 | 优化点 | 解决的问题 |
|------|--------|------------|
| v0 | 基线：交错寻址 | — |
| v1 | 活跃线程连续排列 | Warp Divergence |
| v2 | 步长从大到小 | Shared Memory Bank Conflict |
| v3 | 加载时做第一次加法 | 线程利用率低（50% idle） |
| v4 | 手动展开最后一个 warp | `__syncthreads()` 开销 |
| v5 | 模板完全展开循环 | 运行时循环/分支开销 |
| v6 | Warp Shuffle 两级规约 | Shared Memory 延迟 |
| v7 | float4 + Grid Stride Loop | 内存带宽利用率 |

### 性能提升的层次

```
v0 → v1：消除 warp divergence，理论提升约 2x（最差情况）
v1 → v2：消除 bank conflict，提升约 2x
v2 → v3：线程利用率从 50% → 100%，减少 block 数，提升约 2x
v3 → v4：省去 5 次 __syncthreads()，小幅提升
v4 → v5：编译期展开，减少指令数，小幅提升
v5 → v6：warp shuffle 替代 shared memory，延迟从 ~25 cycle → ~4 cycle
v6 → v7：向量化读取提升带宽利用率，grid stride loop 优化 SM 占用
```

---

## 关键概念速查

**Warp Divergence**：同一 warp 内线程走不同分支，GPU 必须串行执行两条路径。

**Shared Memory Bank Conflict**：32 个 bank，访问同一 bank 不同地址时串行化。避免方法：让相邻线程访问相邻地址（步长为 1 或 32 的倍数）。

**`__syncthreads()`**：block 内所有线程的同步屏障，代价约 10-20 cycle。尽量减少调用次数。

**`__shfl_down_sync`**：warp 内线程直接交换寄存器，无需经过 shared memory，延迟约 4 cycle。

**`float4`**：128-bit 向量加载，减少内存事务数，充分利用内存总线宽度。

**Grid Stride Loop**：用固定小 grid 配合步长迭代，让每个 SM 保持足够的工作量以隐藏内存延迟。

**`volatile`**：告诉编译器每次都从 shared memory 读取，不缓存到寄存器，在 warp unrolling 时确保正确性（v5 后 shuffle 取代了这种用法）。