#include <cuda_runtime.h>

template <int BM, int BN, int BK,int BLOCK_SIZE>
__global__ void sgemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int r0= blockIdx.y * BM;
    int c0= blockIdx.x * BN;
    int tid = threadIdx.x;

    // 加载 tile A 时的线程重排
    constexpr int A_BLOCK_X =BK; // =8 
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X; //  32
    int a_row = tid / A_BLOCK_X;
    int a_col = tid % A_BLOCK_X;

    // 加载 tile B 时的线程重排
    constexpr int B_BLOCK_X = 32; // =8
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X; //  32
    int b_row = tid / B_BLOCK_X;
    int b_col = tid % B_BLOCK_X;

    // 加载 tile c 的线程排布 16*16
    constexpr int C_BLOCK_X = 16; // =16
    constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X; // =16
    int c_row = tid / C_BLOCK_X;
    int c_col = tid % C_BLOCK_X;

    //每个线程负责 Tm x Tn 的计算
    const int Tm= BM / C_BLOCK_Y; // 8
    const int Tn= BN / C_BLOCK_X; // 8
    float c[Tm][Tn] = {0.0f};

    // loop
    for(int k=0;k<K;k+=BK){
        // load tile A
        #pragma unroll
        for(int i=a_row;i<BM;i++){
            int r = r0 + i, c = k + a_row;
            As[i][a_row] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }

        // load tile B
        #pragma unroll
        for(int i=b_col;i<BN;i++){
            int r = r0 + i, c = k + b_col;
            Bs[i][b_col] = (r < M && c < K) ? B[r * K + c] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for(int p=0;p<BK;p++){
            for(int i=0;i<Tm;i++){
                int row=c_row+i*(BOLCK_SIZE/C_BLOCK_Y);
                for(int j=0;j<Tn;j++){
                    int col=c_col+j*(BLOCK_SIZE/C_BLOCK_X);
                    Ct[i][j] += As[row][p] * Bs[p][col];
                }
            }
        }

        __syncthreads();
    }

     // 写回结果
    for (int i = 0; i < Tm; i++) {
        int r = r0 + c_thread_y + i * C_BLOCK_Y;
        for (int j = 0; j < Tn; j++) {
            int c = c0 + c_thread_x + j * C_BLOCK_X;
            if (r < M && c < N) C[r * N + c] = Ct[i][j];
        }
    }
