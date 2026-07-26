#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include "reduce.h"   // ← 只依赖头文件

// ============================================================
// CPU 参考实现
// ============================================================
static double cpu_sum(const std::vector<float>& arr) {
    double s = 0.0;
    for (float v : arr) s += (double)v;
    return s;
}

// ============================================================
// 测试结构体
// ============================================================
struct TestCase {
    std::string name;
    std::vector<float> input;
    float expected;
    float tol;
};

// ============================================================
// 运行单个测试
// ============================================================
static bool run_test(const TestCase& tc) {
    int N = (int)tc.input.size();

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input,  N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, tc.input.data(), N * sizeof(float),
               cudaMemcpyHostToDevice);

    solve(d_input, d_output, N);

    float result = 0.0f;
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    float diff = fabsf(result - tc.expected);
    bool  pass = diff <= tc.tol;

    printf("  [%s] %-30s N=%-10d expected=%-12.3f got=%-12.3f diff=%.3f\n",
           pass ? "PASS" : "FAIL",
           tc.name.c_str(), N,
           tc.expected, result, diff);
    return pass;
}

// ============================================================
// 性能测试
// ============================================================
static void benchmark(int N, int warmup = 3, int repeat = 20) {
    std::vector<float> h(N, 1.0f);
    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input,  N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < warmup; i++) solve(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) solve(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= repeat;

    double bw = (double)N * sizeof(float) / (ms / 1000.0) / 1e9;
    printf("  N=%-12d  %.3f ms  %.1f GB/s\n", N, ms, bw);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
}

// ============================================================
// 构造所有测试用例
// ============================================================
static std::vector<TestCase> make_tests() {
    std::vector<TestCase> T;
    srand(42);

    auto add = [&](std::string name, std::vector<float> v, float tol = 1.0f) {
        float exp = (float)cpu_sum(v);
        T.push_back({name, v, exp, tol});
    };

    // --- 边界 ---
    add("single_element",          {42.0f});
    add("two_elements",            {100.0f, -100.0f});
    add("all_zeros_1024",          std::vector<float>(1024, 0.0f));

    // --- 2的幂边界 ---
    add("all_ones_N=256",          std::vector<float>(256,  1.0f));
    add("all_ones_N=1023",         std::vector<float>(1023, 1.0f));
    add("all_ones_N=1024",         std::vector<float>(1024, 1.0f));
    add("all_ones_N=1025",         std::vector<float>(1025, 1.0f));
    add("all_ones_N=2048",         std::vector<float>(2048, 1.0f));

    // --- 题目样例 ---
    add("example_1",               {1,2,3,4,5,6,7,8});
    add("example_2",               {-2.5f,1.5f,-1.0f,2.0f});

    // --- 数值特征 ---
    add("all_negative",            std::vector<float>(512, -3.14f));
    add("alternating_pos_neg",     [](){
        std::vector<float> v(10000);
        for (int i = 0; i < 10000; i++) v[i] = (i%2==0) ? 1.0f : -1.0f;
        return v;
    }());
    add("extreme_values",          {1000.0f,-1000.0f,999.9f,-999.9f});

    // --- 等差序列 ---
    add("seq_1_to_100",            [](){
        std::vector<float> v(100);
        for(int i=0;i<100;i++) v[i]=(float)(i+1);
        return v;
    }());

    // --- 大数组 ---
    add("large_1M_all_ones",       std::vector<float>(1<<20, 1.0f), 100.0f);
    add("large_4M_all_ones",       std::vector<float>(1<<22, 1.0f), 100.0f);

    // --- 随机 ---
    auto rand_vec = [&](int n) {
        std::vector<float> v(n);
        for (int i = 0; i < n; i++)
            v[i] = ((float)rand()/RAND_MAX) * 2000.0f - 1000.0f;
        return v;
    };
    add("random_N=10000",   rand_vec(10000),   10.0f);
    add("random_N=100000",  rand_vec(100000),  50.0f);
    add("random_N=4194304", rand_vec(4194304), 500.0f);

    // --- 最大规模 ---
    add("max_N=100M_ones", std::vector<float>(100000000, 1.0f), 1000.0f);

    return T;
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s  |  SM x%d  |  %.0f GB/s 理论带宽\n\n",
           prop.name, prop.multiProcessorCount,
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth/8) / 1e6);

    // 正确性
    printf("========== 正确性测试 ==========\n");
    auto tests  = make_tests();
    int pass    = 0;
    for (auto& tc : tests)
        if (run_test(tc)) pass++;
    printf("\n%d / %d 通过\n\n", pass, (int)tests.size());

    // 性能（可用 --bench 跳过）
    bool do_bench = true;
    for (int i = 1; i < argc; i++)
        if (std::string(argv[i]) == "--no-bench") do_bench = false;

    if (do_bench) {
        printf("========== 性能测试 ==========\n");
        benchmark(1 << 20);
        benchmark(1 << 22);
        benchmark(1 << 24);
        benchmark(1 << 26);
        benchmark(100000000);
    }

    return (pass == (int)tests.size()) ? 0 : 1;  // 失败时返回非0
}
