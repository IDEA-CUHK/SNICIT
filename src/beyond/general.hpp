#include <cmath>

#pragma once

namespace SNICIT_BEY {

__global__ void y_star_gen(
    const float* Y0,
    int *y_star_row,
    const int num_input,
    const int neurons,
    const int seed_size
) {
    int row_idx = threadIdx.y * num_input / blockDim.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int shRowSize = neurons + seed_size + 1024;
    extern __shared__ float shRow[];
    float* shRowPtr = shRow;
    if (threadIdx.x == 0) {
        shRowPtr[neurons+seed_size+threadIdx.y] = (float)row_idx;
    }
    __syncthreads();
    for (int i = 0; i < seed_size; i++) {
        if (shRowPtr[neurons+seed_size+i]!=-1.0) {
            if (tid < neurons) {
                shRowPtr[tid] = Y0[neurons*(int)shRowPtr[neurons+seed_size+i]+tid];
            }
            if (tid < seed_size) {
                shRowPtr[neurons+tid] = 0;
            }
            __syncthreads();
            if (shRowPtr[neurons+seed_size+threadIdx.y] != -1.0) {
                for (int j = threadIdx.x; j < neurons; j += blockDim.x) {
                    if (abs(Y0[neurons*row_idx+j] - shRowPtr[j]) > 0.03) {
                        atomicAdd(&shRowPtr[neurons+threadIdx.y], 1);
                    }
                }
            }
            __syncthreads();
            if (threadIdx.y!=i && shRowPtr[neurons+threadIdx.y] < neurons*0.03) {
                shRowPtr[neurons+seed_size+threadIdx.y] = -1.0;
            }
            __syncthreads();
        }
    }
    if (tid < seed_size) {
        y_star_row[tid] = (int)shRowPtr[neurons+seed_size+tid];
    }
    __syncthreads();
}

__global__ void coarse_cluster(
    float* Y0,
    const int *y_star_row,
    bool *ne_record,
    const int y_star_cnt,
    int *centroid_LUT,
    const int neurons
) {
    if (centroid_LUT[blockIdx.x] == -1) {
        ne_record[blockIdx.x] = true;
        return;
    }
    int thisRowSize = neurons + 60;
    extern __shared__ float thisRow[];
    float* thisRowPtr = thisRow;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid < neurons) {
        thisRowPtr[tid] = Y0[blockIdx.x*neurons+tid];
    }
    if (tid < y_star_cnt) {
        thisRowPtr[neurons+tid] = 0;
    }
    __syncthreads();
    for(int i = threadIdx.x; i < neurons; i += blockDim.x) {
        if (abs(Y0[neurons*y_star_row[threadIdx.y]+i] - thisRowPtr[i]) > 0.04) {
            atomicAdd(&thisRowPtr[neurons+threadIdx.y], 1);
        }
    }
    __syncthreads();
    int argmin=-10;
    float min_num = neurons+1;
    if (tid == 0) {
        for (int i = 0; i < y_star_cnt; i++) {
            if (min_num > thisRowPtr[neurons+i]) {
                min_num = thisRowPtr[neurons+i];
                argmin = y_star_row[i];
            }
        }
        centroid_LUT[blockIdx.x] = argmin;
    }
    __syncthreads();
    argmin = centroid_LUT[blockIdx.x];
    float v = ((tid < neurons) && (abs(thisRowPtr[tid]-Y0[neurons*argmin+tid])>0.04)) ?
        thisRowPtr[tid]-Y0[neurons*argmin+tid] : 0;
    if (tid < neurons) {
        Y0[blockIdx.x*neurons+tid] = v;
    }
    int count = __syncthreads_count(v > 0);
    if (tid == 0) {
        if (count == 0) ne_record[blockIdx.x] = false;
        else ne_record[blockIdx.x] = true;
    }
}

__global__ void sparse_hidden(
    const float* Y0,
    const int* dev_cur_row_offset,
    const int* dev_cur_delta_index,
    const float* dev_cur_nonzero_values,
    const float* dev_cur_bias,
    const int* dev_cur_slope,
    const int K,
    float* Y1
) {
    extern __shared__ float shRow[];
    int shRowSize = K;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int rid = blockIdx.x;

    if (tid < K) {
        shRow[tid] = dev_cur_bias[tid];
    }
    __syncthreads();

    int row_nnz = dev_cur_row_offset[rid + 1] - dev_cur_row_offset[rid];
    for (int i = threadIdx.y; i < row_nnz; i += blockDim.y) {
        int Di = dev_cur_delta_index[dev_cur_row_offset[rid] + i];
        float valY = Y0[rid * K + Di];
        if (valY == 0) {
            continue;
        }

        int colW = Di + i * dev_cur_slope[rid];
        float valW = dev_cur_nonzero_values[dev_cur_row_offset[rid] + i];
        atomicAdd(&shRow[colW], valY * valW);
    }

    __syncthreads();
    if (tid < K) {
        Y1[rid * K + tid] = shRow[tid];
    }
}

__global__ void update_post(
    const int *rowsY,
    const int *centroid_LUT,
    const float* Y0,
    const float* bias,
    const int neurons,
    bool* ne_record,
    float* Y1
) {
    int tid = threadIdx.x;
    int rid = rowsY[blockIdx.x];
    float b = bias[threadIdx.x];
    if (centroid_LUT[rid] == -1) {
        Y1[rid * neurons+tid] = min(float(1.0), max(float(0), Y0[rid * neurons+tid]+b));
        ne_record[rid] = true;
        return;
    }
    float wy_centroid = Y0[neurons * centroid_LUT[rid] + tid];
    float wdelta_y = Y0[neurons * rid + tid];
    float true_diff = min(float(1.0), max(float(0), wy_centroid+b+wdelta_y))-min(float(1.0), max(float(0), wy_centroid+b));
    float val = (abs(true_diff)>0.05)?true_diff:0;
    int count = __syncthreads_count(val != 0);
    Y1[rid * neurons+tid] = val;
    if (tid == 0) {
        if (count == 0) ne_record[rid] = false;
        else ne_record[rid] = true;
    }
}

__global__ void recover(
    float* Y0,
    const int *centroid_LUT,
    const int neurons
) {
    extern __shared__ float shRow[];
    int shRowSize = neurons;
    if (centroid_LUT[blockIdx.x] == -1) {
        return;
    }
    int tid = threadIdx.x;
    shRow[tid] = Y0[blockIdx.x*neurons+tid] + Y0[centroid_LUT[blockIdx.x]*neurons+tid];
    __syncthreads();
    Y0[blockIdx.x*neurons+tid] = shRow[tid];
}

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
    const int *rowsY,
    const float* Y0,
    const int *dev_cur_delta_index,
    const float *dev_cur_nonzero_values,
    const int *dev_cur_minimum,
    const int *dev_cur_row_offset,
    const int *dev_cur_avg_nnz,
    const int *dev_cur_slope,
    const int M, const int K,
    float* Y1
) {
    extern __shared__ float shRow[];
    int shRowSize = K;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int rid = rowsY[blockIdx.x];
    if (tid < K) {
        shRow[tid] = 0;
    }
    __syncthreads();

    int row_nnz = dev_cur_avg_nnz[rid] + dev_cur_row_offset[rid];
    for (int i = threadIdx.y; i < row_nnz; i += blockDim.y) {
        int Di = dev_cur_delta_index[rid * row_nnz + i] + dev_cur_minimum[rid];
        float valY = Y0[rid * dev_cur_minimum[rid] + Di];
        if (valY == 0) {
            continue;
        }

        int colW = Di + i * dev_cur_slope[rid];
        float valW = dev_cur_nonzero_values[rid * row_nnz + i];
        atomicAdd(&shRow[colW], valY * valW);
    }

    __syncthreads();
    if (tid < K) {
        Y1[rid * K + tid] = shRow[tid];
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
