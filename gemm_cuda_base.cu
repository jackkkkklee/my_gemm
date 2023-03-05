/*
 * C = alpha*A dot B + beta*C
 *  A[M][K];
 *  B[K][N];
 *  C[M][N];
 *  GPU 中的一个线程去计算 矩阵C的一个位置的值
 */
__global__ void gemmKernel(const float * A,const float * B, float * C,
                           float alpha, float beta,
                           unsigned M, unsigned N,unsigned K) {
    //blockDim is the size of block; blockIdx is the index of block
    //行
    unsigned int m = threadIdx.x + blockDim.x * blockIdx.x;
    //列
    unsigned int n = threadIdx.y + blockDim.y * blockIdx.y;
    if (m >= M || n >= N)
        return;
    float c = 0;
    for (unsigned k = 0; k < K; ++k) {
        //用一维数组去模拟
        c += A[m * K + k] // 行遍历
                *
                B[k * N + n];//列遍历
    }
    c = c * alpha;
    float result = c;
    if (beta != 0) {
        result = result + C[m * N + n] * beta; // m * N +n 是C矩阵本次计算的位置
    }
    C[m * N + n] = result;
}
/*
 * 分配 grid 和 block 并调用kernel
 */
void gemmBasic(const float *A, const float *B, float *C,
               float alpha, float beta, unsigned M,
               unsigned N, unsigned K) {
    //每个block 256个线程
    dim3 block(16, 16);
    dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);

    gemmKernel<<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
}

int main() {
    int gpu_rank = 0;
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, gpu_rank);
    cudaSetDevice(gpu_rank);
    printf("GPU %s status: ", deviceProp.name);
    double boostFrequency = deviceProp.clockRate / 1e6; //kHZ to GHZ

    //get device  fp32 core num
    int fp32CoresNum;
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&fp32CoresNum, cudaDevAttrMultiprocessorCount, dev);
    fp32CoresNum *= _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);

    //估计浮点运算性能极限 吞吐量 = 每个核心的时钟频率 × 每个核心的计算能力(e.g. 2) × 并行处理能力*核心数
    double peakPerformance = boostFrequency * fp32CoresNum * 2;
    printf(
            "clock rate %.3f GHz, FP32 cores num %d, FP32 peak throughput %.3f "
            "GFLOPS\n",
            boostFrequency, fp32CoresNum, peakPerformance);

    //init matrix
    omp_set_num_threads(omp_get_num_procs());
    unsigned M = 1024, N = 1024, K = 1024;
    float alpha = 1., beta = 0.;
    float *deviceAPrt, *deviceBPtr, *deviceCPtr;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A{M, K},
            B{K, N}, C{M, N};
    A.setRandom();
    B.setRandom();
    C.setRandom();

    //分配GPU内存 并 复制
    cudaMalloc(&deviceAPrt, M * K * sizeof(float));
    cudaMemcpy(deviceAPrt, A.data(), M * K * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMalloc(&deviceBPtr, K * N * sizeof(float));
    cudaMemcpy(deviceBPtr, B.data(), K * N * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMalloc(&deviceCPtr, M * N * sizeof(float));
    cudaMemcpy(deviceCPtr, C.data(), M * N * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);

    gemmBasic(deviceAPrt, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    printf("GPU use: %.3f(ms)\n", milliseconds);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            hostResult{M, N}, deviceResult{M, N};
    clock_t begin, end;
    begin = clock();
    hostResult = alpha * (A * B) + beta * C;
    end = clock();
    printf("CPU use: %.3f(ms)\n", double(end - begin) / CLOCKS_PER_SEC * 1e3);

    cudaMemcpy(deviceResult.data(), deviceCPtr, M * N * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    //计算误差
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> diffArray =
            (hostResult - deviceResult).array().abs();
    printf("Max Error: %f\n", diffArray.maxCoeff());

    //my gemm 性能估计方式为 运行总浮点操作数 / 运行时间 / 10^9
    double GFLOPS = 2 * 1e-9 * M * N * K / (milliseconds * 1e-3);
    printf("GPU Throughput: %.3f GFLOPS\n", GFLOPS);
}


