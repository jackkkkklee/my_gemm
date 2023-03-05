//
// Created by 76919 on 2023/3/3.
//

#include "gemm_cpu.h"
void gemm_cpu(unsigned M ,unsigned N, unsigned K,float **A, float **B, float **C){
//    float A[M][K];
//    float B[K][N];
//    float C[M][N];
    float alpha, beta;

    for (unsigned m = 0; m < M; ++m) {
        for (unsigned n = 0; n < N; ++n) {
            float c = 0;
            for (unsigned k = 0; k < K; ++k) {
                c += A[m][k] * B[k][n];
            }
            C[m][n] = alpha * c + beta * C[m][n];
        }
    }
}