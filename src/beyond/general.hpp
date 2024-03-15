#include <cmath>

#pragma once

namespace SNICIT_BEY {

__global__ void dense_input(
    const float* Y0,
    const float* weight,
    const float* bias,
    const int M, const int N, const int K,
    float* Y1
) {
    extern __shared__ float shRow[];
    int shRowSize = K;
    
    if (threadIdx.y == 0) {
        shRow[threadIdx.x] = bias[threadIdx.x];
    }
    __syncthreads();

    for (int i = threadIdx.y; i < N; i += blockDim.y) {
        float valY = Y0[blockIdx.x * N + i];
        if (valY == 0) {
            continue;
        }
        float valW = weight[i * K + threadIdx.x];
        atomicAdd(&shRow[threadIdx.x], valY * valW);
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        Y1[blockIdx.x * K + threadIdx.x] = min(float(1.0), max(float(0), shRow[threadIdx.x]));
    }
}

__global__ void sparse_hidden(
    const float* Y0,
    const int* roffW,
    const int* colsW,
    const float* valsW,
    const float* bias,
    const int M, const int N, const int K,
    float* Y1
) {
    extern __shared__ float shRow[];
    int shRowSize = K;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    if (tid < K) {
        shRow[tid] = bias[tid];
    }
    __syncthreads();

    for (int i = threadIdx.y; i < N; i += blockDim.y) {
        float valY = Y0[blockIdx.x * N + i];
        if (valY == 0) {
            continue;
        }

        int begOffW = roffW[i] + threadIdx.x;
        int endOffW = roffW[i + 1];
        for (int k = begOffW; k < endOffW; k += blockDim.x) {
            int colW = colsW[k];
            float valW = valsW[k];
            atomicAdd(&shRow[colW], valY * valW);
        }
    }
    __syncthreads();

    if (tid < K) {
        Y1[blockIdx.x * K + tid] = min(float(1.0), max(float(0), shRow[tid]));
    }
}

__global__ void dense_output(
    const float* Y0,
    const float* weight,
    const float* bias,
    const int M, const int N, const int K,
    float* Y1
) {
    extern __shared__ float shRow[];
    int shRowSize = K;

    if (threadIdx.y == 0) {
        shRow[threadIdx.x] = bias[threadIdx.x];
    }
    __syncthreads();

    for (int i = threadIdx.y; i < N; i += blockDim.y) {
        float valY = Y0[blockIdx.x * N + i];
        if (valY == 0) {
            continue;
        }
        float valW = weight[i * K + threadIdx.x];
        atomicAdd(&shRow[threadIdx.x], valY * valW);
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        Y1[blockIdx.x * K + threadIdx.x] = shRow[threadIdx.x];
    }
}

__global__ void check_acc(
    float* Y, int num_classes, int num_input, int* label, int* cnt
) {
    extern __shared__ int shcnt[];
    int shcntSize = 1;

    if (threadIdx.x == 0)
        shcnt[0] = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < num_input; i += blockDim.x) {
        int argmax = 0;
        float tmpmax = -10000.0;
        for (int j = 0; j < num_classes; j++) {
            if (Y[i * num_classes + j] > tmpmax) {
                argmax = j;
                tmpmax = Y[i * num_classes + j];
            }
        }

        if (argmax == label[i])
            atomicAdd(&shcnt[0], 1);
        __syncthreads();
    }

    cnt[0] = shcnt[0];
}

}
