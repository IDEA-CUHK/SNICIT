#pragma once

namespace SNICIT_SDGC{

__global__ 
void bf_inference(
  const float* Y0,
  const size_t nerowsY,
  const int* rowsY0,
  int* rlenY0,
  const size_t COL_BLK,
  const size_t N_SLAB,
  const size_t num_neurons_per_layer,
  const int* roffW,
  const int* colsW,
  const float* valsW,
  const float bias,
  float* Y1,
  int* rlenY1
);

//-----------------------------------------------------------------------------
//Definition of task function
//-----------------------------------------------------------------------------

__global__ 
void bf_inference(
  const float* Y0,
  const size_t nerowsY,
  const int* rowsY0,
  int* rlenY0,
  const size_t COL_BLK,
  const size_t N_SLAB,
  const size_t num_neurons_per_layer,
  const int* roffW,
  const int* colsW,
  const float* valsW,
  const float bias,
  float* Y1,
  int* rlenY1
) {

  if(blockIdx.x >= nerowsY) {
    return;
  }

  extern  __shared__ float shRow[];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int rid = rowsY0[blockIdx.x];
  __syncthreads();
  if(tid == 0) {
    rlenY0[rid] = 0;
    rlenY1[rid] = 0;
  }

  for(size_t i = 0; i < N_SLAB; i++) {
    __syncthreads();
    for(size_t j = threadIdx.x; j < COL_BLK; j++) {
      shRow[j] = 0;  
    }
    __syncthreads();
    for(size_t j = threadIdx.y; j < num_neurons_per_layer; j += blockDim.y) {
      float valY = Y0[rid * num_neurons_per_layer + j];
      if(valY == 0) {
        continue;
      }
      int begOffW = roffW[i * num_neurons_per_layer + j] + threadIdx.x;
      int endOffW = roffW[i * num_neurons_per_layer + j + 1];
      for(int k = begOffW; k < endOffW; k += blockDim.x) {
        int colW = colsW[k];
        float valW = valsW[k];
        atomicAdd(&shRow[colW - i * COL_BLK], valY * valW);
      }
    }
    __syncthreads();
    int count = 0;
    for(size_t j = 0; j < COL_BLK; j += blockDim.x * blockDim.y) {
      float v = j + tid < COL_BLK ? shRow[j + tid] + bias : -1;
      count += __syncthreads_count(v > 0);
      if(j + tid < COL_BLK) {
        Y1[rid * num_neurons_per_layer + i * COL_BLK + j + tid] = min(float(32), max(float(0), v));
      }
    }
    if(tid == 0) {
      rlenY1[rid] += count;
    }
  }
}

}// end of namespace snig ----------------------------------------------
