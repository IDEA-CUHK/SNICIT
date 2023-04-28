#pragma once

namespace SNICIT_SDGC{

__device__ inline float __ReLU(float x){
   return x<0.0?0.0:x>32.0?32.0:x;
}

#define MINIBATCH 8
#define UNROLL 8

__global__ void n16384l1_kernel(
	float * __restrict__ A, 
	float * __restrict__ B, 
	float * __restrict__ C, 
	int* __restrict__ index, 
    int* categories,
    int* active,
	int batch, 
    int neuron, 
	float bias)     
{
	extern __shared__ float shared[];
	int start_idx = index[blockIdx.y];
	int col_gropu = threadIdx.x / 16;
	int last_load = ((neuron / 16) % 7) * 16 + 16;
	int load_num = (blockIdx.y + 1) == gridDim.y ? last_load : 128;
	for(int n = threadIdx.x; n < load_num; n += blockDim.x){
		for(int f = 0; f < MINIBATCH; ++f) {
			shared[f * 128 + n] = A[(blockIdx.x * MINIBATCH + f) * neuron + (start_idx + n) % neuron];
		}
	}
	__syncthreads();
	int last_thread = (neuron % 112);
	if(col_gropu == 7 || ((blockIdx.y + 1) == gridDim.y && threadIdx.x >= last_thread)) return;
    float res[MINIBATCH] = {0.0};
    for(int r = 0; r < 32; ++r) {
        float val = B[(blockIdx.y * 128 * 32) + r * 128 + threadIdx.x];
        int idx = col_gropu * 16 + r;
        for(int f = 0; f < MINIBATCH / UNROLL; ++f) {
            res[0 + f * UNROLL] += shared[(f * UNROLL + 0) * 128 + idx] * val;
            res[1 + f * UNROLL] += shared[(f * UNROLL + 1) * 128 + idx] * val;
            res[2 + f * UNROLL] += shared[(f * UNROLL + 2) * 128 + idx] * val;
            res[3 + f * UNROLL] += shared[(f * UNROLL + 3) * 128 + idx] * val;
            res[4 + f * UNROLL] += shared[(f * UNROLL + 4) * 128 + idx] * val;
            res[5 + f * UNROLL] += shared[(f * UNROLL + 5) * 128 + idx] * val;
            res[6 + f * UNROLL] += shared[(f * UNROLL + 6) * 128 + idx] * val;
            res[7 + f * UNROLL] += shared[(f * UNROLL + 7) * 128 + idx] * val;
        }
    }
    __syncthreads();
    for(int f = 0; f < MINIBATCH; ++f) {
        if(C[(blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * 112 + threadIdx.x] = __ReLU(res[f] + bias)) {
            active[blockIdx.x * MINIBATCH + f] = 1;
        }
    }
}

__global__ void n16384_l2_l11_kernel(
	float * __restrict__ A, 
	float * __restrict__ B, 
	float * __restrict__ C, 
    int* __restrict__ categories,
    int* __restrict__ active,
	int stride,
	int batch, 
    int neuron, 
	float bias) {

	extern __shared__ float shared[];
	int start_idx1 = (blockDim.x / 16) * (blockIdx.y) * 16;
	int start_idx2 = (blockDim.x / 16) * (blockIdx.y) * 16 + stride;
    int load_num = stride > blockDim.x ? 32 * (blockDim.x / 16) : stride + 16 * (blockDim.x / 16);
	int shared_size = ((load_num + 31) / 32) * 32;
	int col_gropu = threadIdx.x / 16;
	
	

	for(int n = threadIdx.x; n < load_num * MINIBATCH; n += blockDim.x){
		int f = n / load_num;
		int k = n % load_num;
        int a_k = ((stride > blockDim.x) && (k >= blockDim.x)) ? (k - blockDim.x) + start_idx2 : k + start_idx1;
		shared[f * shared_size + k] = A[categories[(blockIdx.x * MINIBATCH + f)] * neuron + (a_k) % neuron];
	}

	__syncthreads();
    int gap = stride >= blockDim.x ? blockDim.x : stride;
	float res[MINIBATCH] = {0.0};
	for(int r = 0; r < 32; ++r) {
        float val = B[(blockIdx.y * blockDim.x * 32) + r * blockDim.x + threadIdx.x];
        int idx = col_gropu * 16 + (r >= 16? r + gap - 16 : r);
        for(int f = 0; f < MINIBATCH / UNROLL; ++f) {
            res[0 + f * UNROLL] += shared[(f * UNROLL + 0) * shared_size + idx] * val;
            res[1 + f * UNROLL] += shared[(f * UNROLL + 1) * shared_size + idx] * val;
            res[2 + f * UNROLL] += shared[(f * UNROLL + 2) * shared_size + idx] * val;
            res[3 + f * UNROLL] += shared[(f * UNROLL + 3) * shared_size + idx] * val;
            res[4 + f * UNROLL] += shared[(f * UNROLL + 4) * shared_size + idx] * val;
            res[5 + f * UNROLL] += shared[(f * UNROLL + 5) * shared_size + idx] * val;
            res[6 + f * UNROLL] += shared[(f * UNROLL + 6) * shared_size + idx] * val;
            res[7 + f * UNROLL] += shared[(f * UNROLL + 7) * shared_size + idx] * val;
        }
    }
    for(int f = 0; f < MINIBATCH ; ++f) {
        if(C[(blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * blockDim.x + threadIdx.x] = __ReLU(res[f] + bias)) {
            active[blockIdx.x * MINIBATCH + f] = 1;
        }
	}
}

#define OUT_CHANNEL 16
__global__ void n16384_l11_kernel(
    float * __restrict__ A, 
    float * __restrict__ B, 
    float * __restrict__ C, 
    int* __restrict__ index, 
    int* __restrict__ active,
    int batch, 
    int neuron, 
    float bias
) {
    
    extern __shared__ float shared[];


    for(int n = threadIdx.x; n < OUT_CHANNEL * 32; n += blockDim.x){
        shared[n] = B[(blockIdx.y * OUT_CHANNEL * 32) + n];
    }
    __syncthreads();

    if((blockIdx.x * blockDim.x + threadIdx.x) >= batch) return;
    int begin_idx = blockIdx.y * OUT_CHANNEL / 16 * 32;
    for(int o_r = 0; o_r < OUT_CHANNEL / 16; ++o_r) {
        float reduce[16] = {0.0};
        int idx = begin_idx + o_r * 32;
        for(int r = 0; r < 32; ++r) {
            int row_idx = index[idx + r];  // check every?
            float val = A[row_idx * batch + blockIdx.x * blockDim.x + threadIdx.x];
            for(int c = 0; c < 16; ++c) {
                reduce[c] += val * shared[o_r * 32 * 16 + r * 16 + c];
            }
        }
        for(int c = 0; c < 16; ++c) {
            if(C[(blockIdx.y * OUT_CHANNEL  + o_r * 16 + c) * batch + blockIdx.x * blockDim.x + threadIdx.x]
                = __ReLU(reduce[c] + bias)) {
                active[blockIdx.x * blockDim.x + threadIdx.x] = 1;
            }
        }
    }
}

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void matrix_transpose(float * __restrict__ odata, float * __restrict__ idata, int neuron, int batch) {

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM && (y + j) < batch && x < neuron; j += BLOCK_ROWS) {
        tile[(threadIdx.y + j)][threadIdx.x] = idata[(y + j) * neuron + x];
    }

    __syncthreads();


    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM && x < batch && y + j < neuron; j += BLOCK_ROWS) {
        odata[(y+j) * batch + x] = tile[threadIdx.x][threadIdx.y + j];
    } 
};

__global__ void matrix_re_transpose_and_delete(
    float * __restrict__ odata, 
    float * __restrict__ idata,
    int * __restrict__ old_to_new_map,
    int neuron, int batch) {

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM && x < batch; j += BLOCK_ROWS) {
        tile[(threadIdx.y + j)][threadIdx.x] = idata[(y + j) * batch + x];
    }

    __syncthreads();


    x = blockIdx.y * TILE_DIM + threadIdx.x;  // old row
    y = blockIdx.x * TILE_DIM + threadIdx.y;  // old batch
    

    for (int j = 0; j < TILE_DIM && (y+j) < batch; j += BLOCK_ROWS) {
        if(old_to_new_map[y + j] == -1) continue;
        int tmp = old_to_new_map[y + j]; // new batch
        odata[tmp * neuron + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

}