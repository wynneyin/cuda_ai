#include <cuda_runtime.h>

template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void sgemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 本 block 在全局矩阵中的起始坐标
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tid = threadIdx.x;

    // 加载 As 的线程布局：A_LOAD_ROWS × BK
    constexpr int A_LOAD_ROWS = BLOCK_SIZE / BK;
    const int a_load_row = tid / BK;
    const int a_load_col = tid % BK;

    // 加载 Bs 的线程布局：BK × B_LOAD_COLS
    constexpr int B_LOAD_COLS = 32;
    constexpr int B_LOAD_ROWS = BLOCK_SIZE / B_LOAD_COLS;  // = BK
    const int b_load_row = tid / B_LOAD_COLS;
    const int b_load_col = tid % B_LOAD_COLS;

    // 计算 C 时的线程布局：C_THREAD_ROWS × C_THREAD_COLS
    constexpr int C_THREAD_COLS = 16;
    constexpr int C_THREAD_ROWS = BLOCK_SIZE / C_THREAD_COLS;  // 16
    const int thread_row = tid / C_THREAD_COLS;
    const int thread_col = tid % C_THREAD_COLS;

    // 每个线程负责 TM×TN 个 C 元素
    constexpr int TM = BM / C_THREAD_ROWS;  // 8
    constexpr int TN = BN / C_THREAD_COLS;  // 8
    float c_frag[TM][TN] = {};

    for (int k = 0; k < K; k += BK) {
        // 协作加载 As（BM×BK），每个线程加载 BM/A_LOAD_ROWS 行
        #pragma unroll
        for (int i = 0; i < BM; i += A_LOAD_ROWS) {
            int row = block_row + a_load_row + i;
            int col = k + a_load_col;
            As[a_load_row + i][a_load_col] = (row < M && col < K) ? A[row * K + col] : 0.0f;
        }

        // 协作加载 Bs（BK×BN），每个线程加载 BN/B_LOAD_COLS 列
        #pragma unroll
        for (int j = 0; j < BN; j += B_LOAD_COLS) {
            int row = k + b_load_row;
            int col = block_col + b_load_col + j;
            Bs[b_load_row][b_load_col + j] = (row < K && col < N) ? B[row * N + col] : 0.0f;
        }

        __syncthreads();

        // 外积累加：先把一列 As 和一行 Bs 读进寄存器，再做 TM×TN 次 FMA
        #pragma unroll
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

    // 写回全局内存
    for (int i = 0; i < TM; i++) {
        int row = block_row + thread_row + i * C_THREAD_ROWS;
        for (int j = 0; j < TN; j++) {
            int col = block_col + thread_col + j * C_THREAD_COLS;
            if (row < M && col < N)
                C[row * N + col] = c_frag[i][j];
        }
    }
}

void launch_sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 8, BLOCK_SIZE = 256;

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_tiled<BM, BN, BK, BLOCK_SIZE><<<grid, block>>>(A, B, C, M, N, K);
}
