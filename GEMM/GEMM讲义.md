# CUDA GEMM 优化讲义

矩阵乘法 C = A × B，其中 A 是 M×K，B 是 K×N，C 是 M×N。

---

## 前置知识：GPU 内存层级

```
速度  慢 ──────────────────────────────── 快
容量  大 ──────────────────────────────── 小

Global Memory → L2 Cache → Shared Memory → Register
  几十GB          几十MB       ~100KB         ~256个/线程
  ~600 cycles               ~30 cycles       1 cycle
  所有线程可读              同一block内共享    线程私有
```

关键规律：**越靠右越快，但越小。优化的核心就是让数据尽量住在右边。**

矩阵在内存里是一维数组，访问 `A[row][col]` 用公式 `A[row * 列数 + col]`。

---

## 第一版：Naive（`baseline.cu`）

### 代码

```cuda
__global__ void sgemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### 思路

用二维线程网格，每个线程对应 C 的一个元素。线程知道自己的 `row` 和 `col`，就沿 K 方向把 A 的一行和 B 的一列点乘起来。

```
线程(row=1, col=2) 负责计算 C[1][2]：

A 矩阵          B 矩阵
[. . . .]       [. . . .]
[a b c d]  ×   [. . * .]   →  C[1][2] = a*B[0][2] + b*B[1][2] + ...
[. . . .]       [. . * .]
                [. . * .]
```

### 问题：每次都去 Global Memory 读

循环 K 次，每次从 Global Memory 读 `A[row][k]` 和 `B[k][col]`，共 2K 次 Global Memory 访问，只做 1 次 FMA。

**算术强度 = 1 FMA / 2次全局内存读 = 极低**

Global Memory 延迟 600 cycles，线程大部分时间在等数据，GPU 的计算单元被闲置。

---

## 第二版：Shared Memory Tiling（`Tiling.cu`）

### 核心思路

A 和 B 都很大，没法全放进 Shared Memory。但可以**分块**：

把 C 切成 BM×BN 的小块，每个 CUDA block 只负责算 C 的一个小块。  
沿 K 方向每次搬 BK 列/行进 Shared Memory，算完这一段，再搬下一段。

```
C 矩阵（M×N）
┌────┬────┬────┐
│blk │blk │blk │  ← 每格由一个 CUDA block 负责，所有 block 同时跑
│(0,0)│(0,1)│(0,2)│
├────┼────┼────┤
│blk │blk │blk │
│(1,0)│(1,1)│(1,2)│
└────┴────┴────┘
```

block(1,0) 要算的部分：

```
C[BM..2BM][0..BN] = A[BM..2BM][:] × B[:][0..BN]

K 太大，沿 K 方向切片：
  第0轮：搬 A[BM..2BM][0..BK]   和 B[0..BK][0..BN]   进 Shared Memory，算部分和
  第1轮：搬 A[BM..2BM][BK..2BK] 和 B[BK..2BK][0..BN]，继续累加
  ...
  结果累加在寄存器 c_frag 里
```

### 关键：Shared Memory 是覆盖写，不需要"卸载"

```
循环第1轮：写入 As、Bs → 全体线程读取计算 → 结果进寄存器 → __syncthreads()
循环第2轮：直接覆盖 As、Bs（旧数据消失）→ 继续累加到同一寄存器
...
最后：把寄存器 c_frag 写回 Global Memory
```

Shared Memory 只是**临时中转站**，寄存器才是**跨循环持有累加结果**的地方。

### 代码结构

```cuda
template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void sgemm_tiled(...) {
    __shared__ float As[BM][BK];   // Shared Memory，BK 列的 A 切片
    __shared__ float Bs[BK][BN];   // Shared Memory，BK 行的 B 切片

    // 本 block 在全局矩阵中的起始坐标
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tid = threadIdx.x;   // 一维线程编号

    for (int k = 0; k < K; k += BK) {
        // ① 256个线程协作，把 A、B 的一个切片搬进 Shared Memory
        load_As(...);
        load_Bs(...);
        __syncthreads();   // 确保所有线程都搬完了再开始算

        // ② 用 Shared Memory 里的数据计算，结果累加进寄存器
        for (int p = 0; p < BK; p++)
            c_frag[i][j] += As[row][p] * Bs[p][col];

        __syncthreads();   // 确保所有线程都算完了再覆盖 Shared Memory
    }

    // ③ 把寄存器结果写回 Global Memory
}
```

### 为什么用一维线程编号？

kernel 启动时用 `dim3 block(256)`，线程是一维的，只有 `threadIdx.x`。

好处是：加载 As 和 Bs 时可以用**不同的形状**来切分同一批线程，分别对应各自 tile 的形状，减少 bank conflict。

```cpp
// 加载 As 时：把256个线程脑补成 32行×8列
int a_load_row = tid / BK;    // BK=8，tid 0~7 都是第0行
int a_load_col = tid % BK;

// 加载 Bs 时：把256个线程脑补成 8行×32列
int b_load_row = tid / 32;
int b_load_col = tid % 32;
```

---

## 第三版：Register Tiling / 外积优化（`outer_poduct.cu`）

### 问题：第二版的 Shared Memory 复用不够

第二版里每个线程只算 C 的一个元素：

```
线程(0,0) 负责 C[0][0]：读 As[0][k]，用一次，扔掉
线程(0,1) 负责 C[0][1]：读 As[0][k]，用一次，扔掉  ← 同一个值被两个线程分别读了
```

`As[0][k]` 被重复从 Shared Memory 读了多次。Shared Memory 虽然比 Global Memory 快，但比寄存器还是慢 30 倍。

### 解决：让每个线程负责 TM×TN 个 C 元素

```
线程(0,0) 同时负责 C[0][0], C[0][1], ..., C[TM-1][TN-1]
```

这样在一轮 k 迭代里：

```
从 Shared Memory 读 As 的一列（TM个元素）→ 存进寄存器 a_frag[TM]
从 Shared Memory 读 Bs 的一行（TN个元素）→ 存进寄存器 b_frag[TN]

然后纯寄存器运算：
for i in TM:
    for j in TN:
        c_frag[i][j] += a_frag[i] * b_frag[j]   ← TM×TN 次 FMA
```

`a_frag[0]` 从 Shared Memory 读一次，参与了 TN=8 次 FMA，复用了 8 次。

### 算术强度对比

| 版本 | Shared Memory 读次数 | FMA 次数 | 比值 |
|------|---------------------|---------|------|
| 内积（每线程1个元素） | 2 | 1 | 0.5 |
| 外积 TM=TN=8 | TM+TN=16 | TM×TN=64 | **4** |

外积写法的 Shared Memory 访问效率提升 8 倍。

### 为什么叫"外积"？

内积是向量点乘，结果是标量：`[a b c] · [d e f] = ad+be+cf`

外积是列向量 × 行向量，结果是矩阵：

```
[a]               [ad ae af]
[b]  × [d e f] = [bd be bf]
[c]               [cd ce cf]
```

每轮 k 迭代，`a_frag[TM]` 是 As 的一列，`b_frag[TN]` 是 Bs 的一行，做的正是外积，结果累加进 `c_frag[TM][TN]`。

### 线程如何分工

BLOCK_SIZE=256 个线程排成 16×16 的逻辑网格，每个线程负责 8×8 的 C 子块：

```
C 的 128×128 区域（由本 block 负责）

        0    16   32  ... 112  128
      ┌────┬────┬────┬───┬────┐
  0   │ t0 │ t1 │ t2 │...│t15 │
      ├────┼────┼────┼───┼────┤
  16  │t16 │t17 │    │   │    │
      ├────┼────┼────┼───┼────┤
  ... │    │    │    │   │    │
      ├────┼────┼────┼───┼────┤
 112  │    │    │    │   │t255│
      └────┴────┴────┴───┴────┘

每格 = 8×8 个 C 元素，由一个线程负责
```

### 完整代码

```cuda
template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void sgemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tid = threadIdx.x;

    // 加载线程布局
    constexpr int A_LOAD_ROWS = BLOCK_SIZE / BK;
    const int a_load_row = tid / BK;
    const int a_load_col = tid % BK;

    constexpr int B_LOAD_COLS = 32;
    const int b_load_row = tid / B_LOAD_COLS;
    const int b_load_col = tid % B_LOAD_COLS;

    // 计算线程布局
    constexpr int C_THREAD_COLS = 16;
    constexpr int C_THREAD_ROWS = BLOCK_SIZE / C_THREAD_COLS;
    const int thread_row = tid / C_THREAD_COLS;
    const int thread_col = tid % C_THREAD_COLS;

    constexpr int TM = BM / C_THREAD_ROWS;  // 8
    constexpr int TN = BN / C_THREAD_COLS;  // 8
    float c_frag[TM][TN] = {};

    for (int k = 0; k < K; k += BK) {
        // 协作加载 As
        for (int i = 0; i < BM; i += A_LOAD_ROWS) {
            int row = block_row + a_load_row + i;
            int col = k + a_load_col;
            As[a_load_row + i][a_load_col] = (row < M && col < K) ? A[row * K + col] : 0.0f;
        }
        // 协作加载 Bs
        for (int j = 0; j < BN; j += B_LOAD_COLS) {
            int row = k + b_load_row;
            int col = block_col + b_load_col + j;
            Bs[b_load_row][b_load_col + j] = (row < K && col < N) ? B[row * N + col] : 0.0f;
        }
        __syncthreads();

        // 外积累加
        for (int p = 0; p < BK; p++) {
            float a_frag[TM], b_frag[TN];
            for (int i = 0; i < TM; i++)
                a_frag[i] = As[thread_row + i * C_THREAD_ROWS][p];
            for (int j = 0; j < TN; j++)
                b_frag[j] = Bs[p][thread_col + j * C_THREAD_COLS];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    c_frag[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }

    // 写回
    for (int i = 0; i < TM; i++) {
        int row = block_row + thread_row + i * C_THREAD_ROWS;
        for (int j = 0; j < TN; j++) {
            int col = block_col + thread_col + j * C_THREAD_COLS;
            if (row < M && col < N)
                C[row * N + col] = c_frag[i][j];
        }
    }
}
```

---

## 三版对比总结

```
版本          瓶颈              优化手段                  数据住在哪
──────────────────────────────────────────────────────────────────
Naive         反复读Global Mem  无                        Global Memory
Tiling        Shared Mem读多   分块搬到 Shared Memory     Shared Memory
外积           Shared Mem复用低 每线程算TM×TN个元素，      寄存器
                               数据在寄存器里复用
```

优化路径一句话：

> **Global Memory → Shared Memory → Register，数据住的越靠右，程序越快。**

### TM=TN=8 为什么不能更大？

每个线程寄存器用量 = `a_frag[TM] + b_frag[TN] + c_frag[TM][TN]` = TM + TN + TM×TN。

TM=TN=8 时：8 + 8 + 64 = **80个寄存器**，加上索引变量约 100 个，接近单线程上限 255 的一半，可以保证 SM 同时驻留足够多的 block（occupancy）。

TM=TN=16 时：16 + 16 + 256 = **288个寄存器**，超过上限，触发 register spill（溢出到 Local Memory，实际是 Global Memory），反而变慢。
