#define MINIBATCH 8
#define UNROLL 8
#define YSTARTOP 20
#include <XY/kernel.hpp>
namespace SNICIT_SDGC {


__global__ void coarse_cluster(
    float* Y0,
    float* Y1,
    const int *y_star_row,
    bool *ne_record,
    const int y_star_cnt,
    int *centroid_map,
    const int neurons,
    const int batch
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (centroid_map[tid] == -1) {
        ne_record[tid] = true;
    }
    extern __shared__ float sh_y_star[];
    int dist = neurons+1;
    int clustered_idx;
    int this_dist;
    int final_len = 0;
    for (int y_star_idx = 0; y_star_idx < y_star_cnt; y_star_idx++) {
        this_dist = 0;
        // int ptr = 2;  // idx0 is reserved for non zero element length, idx1 is reserved for the corresponding centroid
        for (int r = 0; r < neurons / 1024; r++) {
            sh_y_star[threadIdx.x] = Y0[y_star_row[y_star_idx]+batch*(r*1024+threadIdx.x)]; 
            __syncthreads();
            if (tid < batch) {
                for (int k = 0; k < 1024; k++) {
                    float reg = Y0[batch*(r*1024+k)+tid];
                    if (sh_y_star[k] != reg) {
                        float diff = reg-sh_y_star[k];
                        if (diff != 0)
                            this_dist += 1;
                    }
                }
            }
            __syncthreads();
        }
        if (this_dist < dist) {
            dist = this_dist;
            clustered_idx = y_star_row[y_star_idx];
        }
    }
    for (int r = 0; r < neurons; r++) {
        if (tid < batch ) {
            float reg = Y0[batch*r+tid];
            if (centroid_map[tid] != -1)
                Y1[batch*r+tid] = reg - Y0[y_star_row[clustered_idx]+batch*r];
            else
                Y1[batch*r+tid] = reg;
        }
    }
    if (tid < batch) {
        if (dist == 0 && centroid_map[tid] != -1) {
            ne_record[tid] = false;
        }
        else {
            ne_record[tid] = true;
        }
        if (centroid_map[tid] != -1)
            centroid_map[tid] = clustered_idx;
    }
}

__global__ void post_spMM(
    float *A, 
    float *C, 
    int *rowsY,
    float * __restrict__ B, 
    int* __restrict__ index, 
    int batch,
    int neuron
) {
    extern __shared__ float shared[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = rowsY[blockIdx.y];
    int begin_idx = blockIdx.x * OUT_CHANNEL / 16 *  32;
    
    float result = 0;
    int idx = begin_idx;
    for(int r = 0; r < 32/OUT_CHANNEL; r++) {
        int row_idx = index[idx + r*OUT_CHANNEL+threadIdx.x];  // check every?
        shared[r*OUT_CHANNEL+threadIdx.x] = A[rid*neuron+row_idx]; // Y_star[row_idx];
        // result += 32 * B[(tid * 32) + r];
    }
    __syncthreads();
    for(int r = 0; r < 32; ++r) {
        float val = shared[r];
        if (val != 0)
            result += val * B[(blockIdx.x * OUT_CHANNEL * 32) + r*OUT_CHANNEL+threadIdx.x];
        // if (blockIdx.y == 1 && blockIdx.x * blockDim.x + threadIdx.x == 0)
        //     printf("\n");
    }
    C[rid*neuron+(tid)] = result;
}

__global__ void post_minus(
    float *  A, 
    float *  C, 
    bool *ne_record,
    int *centroid_map,
    int *rowsY,
    int neuron, 
    float bias,
    int batch
) {
    int rid = rowsY[blockIdx.x];
    if (centroid_map[rid] == -1) {
        for (int i = threadIdx.x; i < neuron; i += blockDim.x) {
            C[rid*neuron + i] = __ReLU(A[rid*neuron + i] + bias); //Y0[rid * neurons+tid];
        }
        ne_record[rid] = true;
        return;
    }
    int count = 0;
    for (int i = threadIdx.x; i < neuron; i += blockDim.x) {
        float wy_centroid = A[centroid_map[rid]*neuron + i];
        float wdelta_y = A[rid*neuron + i];
        float val = __ReLU(wy_centroid+bias+wdelta_y)-__ReLU(wy_centroid+bias);
        int cnt = __syncthreads_count(val != 0);
        count += cnt;
        C[rid * neuron+ i] = val;
    }
    
    if (threadIdx.x == 0) {
        if (count == 0) ne_record[rid] = false;
        else ne_record[rid] = true;
    }
};

__global__ void reduction(int *y_star_idx, float *A,
    float *reduced_Y, int neuron) {
    extern __shared__ float sum[];  // each block is in charge of one sample, we define the sum-bin here
    sum[threadIdx.y] = 0;  // initialize
    int chosen_star = blockIdx.x;  // chosen rows to sample
    if (threadIdx.x + threadIdx.y == 0) {
        y_star_idx[blockIdx.x] = chosen_star; // record the chosen rows
    }
    __syncthreads();
    for (int r = threadIdx.x; r < neuron / blockDim.y; r += blockDim.x) { // histogram sum calculation
        float val = A[chosen_star*neuron+threadIdx.y*blockDim.x+r];
        atomicAdd(sum+threadIdx.y, val);
    }
    __syncthreads();
    reduced_Y[threadIdx.y+(blockIdx.x)*(blockDim.y)] = sum[threadIdx.y]; // write back to global mem
}


__global__ void centroid_sel(
    int *y_star_idx, float *reduced_Y, int reduced_size, int sample_size
) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    extern __shared__ float sh_arr[]; // create a shared mem
    __shared__ float diff_arr[32]; // sample size
    __shared__ int tmp_star_row[32]; // sample size
    if (threadIdx.x == 0) {
        tmp_star_row[threadIdx.y] = y_star_idx[threadIdx.y];
    }

    __syncthreads(); // copy the information in ystar_list to shared mem
    for (int cmp_idx = 0; cmp_idx < sample_size; cmp_idx++) {
        if (tmp_star_row[cmp_idx] != -1) {
            if (tid < reduced_size) {
                sh_arr[tid] = reduced_Y[reduced_size*tmp_star_row[cmp_idx]+tid]; // to be compared
            }
            if (tid < sample_size) {
                diff_arr[tid] = 0;
            }
            __syncthreads();
            if (tmp_star_row[threadIdx.y] != -1) {
                if (abs(reduced_Y[reduced_size*threadIdx.y+threadIdx.x] - sh_arr[threadIdx.x]) > 0.03) { // epsilon1 = 0.03
                    atomicAdd(&diff_arr[threadIdx.y], 1); // L0 norm aggregation
                }
            }
            __syncthreads();
            if (threadIdx.x == 0 && threadIdx.y != cmp_idx 
                && diff_arr[threadIdx.y] < (float)reduced_size*0.03
            ) { // epsilon2 = 0.03
                tmp_star_row[threadIdx.y] = -1;
            }
            __syncthreads();
        }
    }
    if (tid < sample_size) {
        y_star_idx[tid] = tmp_star_row[tid]; // write back to global mem
    }
}


__global__ void recover(
    float *Y, int *active_d, int* centroid_map, int neuron, int batch
) {
    int rid = blockIdx.x;
    int cnt = 0;
    if (centroid_map[rid] == -1) {
        for (int i = threadIdx.x; i < neuron; i += blockDim.x) {
            float v = Y[rid * neuron + i];
            cnt += __syncthreads_count(v > 0);
        }
    }
    else {
        for (int i = threadIdx.x; i < neuron; i += blockDim.x) {
            float v = Y[rid * neuron + i]+Y[centroid_map[rid] * neuron + i];
            cnt += __syncthreads_count(v > 0);
        }
    }

    if (threadIdx.x == 0) {
        active_d[rid] = cnt > 0? 1: 0;
    }
}


}
