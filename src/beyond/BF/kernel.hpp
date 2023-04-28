#pragma once

namespace SNICIT_BEY{
__global__ void bf_inference(
    
  const float* Y0,
  const size_t nerowsY,
  const int* rowsY0,
  int* rlenY0,
  const int* roffW,
  const int* colsW,
  const float* valsW,
  const float* bias,
  const int M, 
  const int N, 
  const int K,
  float* Y1,
  int* rlenY1
) {
    if(blockIdx.x >= nerowsY) {
        return;
    }
    // (8, 128)
    extern  __shared__ float shRow[];
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int rid = rowsY0[blockIdx.x];
    if (tid < K) {
        shRow[tid] = bias[tid]; 
    }
    __syncthreads();
    if(tid == 0) {
        rlenY0[rid] = 0;
        rlenY1[rid] = 0;
    }

    for (int i = threadIdx.y; i < N; i += blockDim.y) {
        float valY = Y0[blockIdx.x * N + i];
        if(valY == 0) {
            continue;
        }

        int begOffW = roffW[i] + threadIdx.x;
        int endOffW = roffW[i + 1];
        for(int k = begOffW; k < endOffW; k += blockDim.x) { // += blockDim.x
            int colW = colsW[k];
            float valW = valsW[k];
            atomicAdd(&shRow[colW], valY * valW);
        }
    }
    __syncthreads();
    float v = tid < K ? shRow[tid] : -1;
    if (tid < K) {
        Y1[blockIdx.x * K+tid] = min(float(1.0), max(float(0), shRow[tid]));
    }
    int count = 0;
    count += __syncthreads_count(v > 0);
    if(tid == 0) {
      rlenY1[rid] += count;
    }
}

}