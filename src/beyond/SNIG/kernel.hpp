#pragma once

namespace SNICIT_BEY{


__global__ 
void snig_inference(
  const float* Y_0,
  const bool* is_nonzero_row_0,
  const int num_neurons,
  const int* col_w,
  const int* row_w,
  const float* val_w,
  const float* bias,
  bool* is_nonzero_row_1,
  float* Y_1
) {
    // (8, 128)
  int tid = threadIdx.y * blockDim.x + threadIdx.x;

  int num_threads = blockDim.x * blockDim.y;

  //num_secs is small enough to compute by each single thread
  bool is_all_zero = true;
  is_all_zero &= !is_nonzero_row_0[blockIdx.x];


  if(is_all_zero) {
    //incremental memory resetting
    //avoid calling cudaMemset
    if(is_nonzero_row_1[blockIdx.x]) {
      for(size_t j = tid; j < num_neurons; j += num_threads) {
        Y_1[blockIdx.x * num_neurons + j] = 0;
      }
      __syncthreads();
      if(tid == 0) {
        is_nonzero_row_1[blockIdx.x] = false;
      } 
    }
    return;
  }

  //forward feeding
  extern __shared__ float results[];

  //set results to bias directly
  if (tid < num_neurons) {
    results[tid] = bias[tid]; 
  }

  //use bool array size of 2 (is_nonzero) in share memory to avoid synchronization
  //is_nonzero[1] represents whether this row is nonzero
  //if is_nonzero[1] is true, this row is nonzero
  __shared__ bool is_nonzero[2];
  if(tid == 0) {
    is_nonzero[1] = false;
  }
  __syncthreads();

  for(size_t j = threadIdx.y; j < num_neurons; j += blockDim.y) {
    float valY = Y_0[blockIdx.x * num_neurons + j];
    if(valY == 0) {
      continue;
    }
    int beg_w = col_w[j] + threadIdx.x;
    int end_w = col_w[j + 1];
    for(int k = beg_w; k < end_w; k += blockDim.x) {
      int roww = row_w[k];
      float valw = val_w[k];
      atomicAdd(&results[roww], valY * valw);
    }
  }

  __syncthreads();
  if (tid < num_neurons) {
    float v = min(float(1.0), max(results[tid], float(0.0)));
    Y_1[blockIdx.x * num_neurons + tid] = v;
    is_nonzero[v != 0] = true;
  }

  //if one thread sets is_nonzero[1] to true
  //meaning this row is nonzero
  //toggle is_nonzero_row_1[this row] to true
  __syncthreads();
  if(tid == 0) {
    is_nonzero_row_1[blockIdx.x] = is_nonzero[1];
  }
}

}// end of namespace snig ----------------------------------------------
