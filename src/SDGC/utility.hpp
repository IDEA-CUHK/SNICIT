#pragma once
#include <functional>
#include <algorithm>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <fstream>
#include <numeric>
#include <vector>
#include <string>
#include <sstream>
#include <thrust/scan.h>
#include <filesystem>
#include <thrust/execution_policy.h>

namespace SNICIT_SDGC {

size_t get_sec_size(const size_t num_neurons) {

  //only for the same GPUs
  //
  //get tuned shared memory size
  //num_neurons must be divisible by shared memory (a.k.a. sec_size)
  //only for double float
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  size_t sec_size{0};

  size_t max_num_per_block = props.sharedMemPerBlock / sizeof(float);
  if(num_neurons <= max_num_per_block) {
    sec_size = num_neurons;
  }
  else{
    int max_divisor = 2;
    while((num_neurons % max_divisor != 0) || 
          (max_num_per_block < (num_neurons / max_divisor))) {
      ++max_divisor;
    }
    sec_size = num_neurons / max_divisor;
  }
  return sec_size;
}


inline
std::string read_file_to_string(const std::string & path) {
  
  using namespace std::literals::string_literals;

  std::ifstream f{ path };

  if(!f) {
    throw std::runtime_error("cannot open the file" + path);
  }

  std::stringstream sstream;
  sstream << f.rdbuf();
  return sstream.str();
}

inline
size_t count_nnz(const std::string& s) {
  return std::count(s.begin(), s.end(), '\n');
}

inline
size_t find_max_nnz(
  const std::string & weight_dir,
  const size_t num_layers,
  const size_t num_neurons_per_layer
) {

  size_t max_nnz{0};
  for(size_t i = 0; i < num_layers; ++i) {
    std::string p = weight_dir;
    p += "/n" + std::to_string(num_neurons_per_layer) + "-l"
      + std::to_string(i + 1) + ".tsv";
    auto data_str = read_file_to_string(p);
    max_nnz = std::max(max_nnz, count_nnz(data_str));
  }

  return max_nnz;
}


void tsv_string_to_CSR_packed_array(
  const std::string& s,
  const size_t rows,
  const size_t cols,
  const size_t nnz,
  const size_t COL_BLK,
  const size_t N_SLAB,
  int* arr
) {

  typedef Eigen::Triplet<float> E;
  std::string line;
  std::vector<E> triplet_list;
  triplet_list.reserve(nnz);
  std::istringstream read_s(s);
  std::vector<std::string> tokens;

  while(std::getline(read_s, line)) {
    std::istringstream lineStream(line);
    std::string token;
    tokens.clear();
    while(std::getline(lineStream, token, '\t')) {
      tokens.push_back(std::move(token));
    }
    triplet_list.emplace_back(
      std::stoi(tokens[0]) - 1 + 
      rows * ((std::stoi(tokens[1]) - 1) / COL_BLK),
      std::stoi(tokens[1]) - 1,
      std::stof(tokens[2])
    );
  }

  Eigen::SparseMatrix<float, Eigen::RowMajor> eigen_mat(rows * N_SLAB, cols);
  eigen_mat.reserve(triplet_list.size());
  eigen_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());

  std::copy(eigen_mat.outerIndexPtr(), eigen_mat.outerIndexPtr() + rows * N_SLAB + 1, arr);
  std::copy(eigen_mat.innerIndexPtr(), eigen_mat.innerIndexPtr() + nnz, arr + rows * N_SLAB + 1);

  float* tmp = reinterpret_cast<float*>(arr + rows * N_SLAB + 1 + nnz);
  std::copy(eigen_mat.valuePtr(), eigen_mat.valuePtr() + nnz, tmp);
}


void read_weight(
  const std::string& weight_dir,
  const size_t num_neurons_per_layer,
  const size_t max_nnz_per_layer,
  const size_t num_layers,
  const size_t COL_BLK,
  const size_t N_SLAB,
  const size_t pad,
  int* arr
) {
  for(size_t i = 0; i < num_layers; ++i) {
    std::string p = weight_dir;
    p += "/n" + std::to_string(num_neurons_per_layer) + "-l"
      + std::to_string(i + 1) + ".tsv";
    auto data_str = read_file_to_string(p);

    tsv_string_to_CSR_packed_array(
      data_str,
      num_neurons_per_layer, 
      num_neurons_per_layer,
      max_nnz_per_layer,
      COL_BLK,  
      N_SLAB,
      arr + i * (num_neurons_per_layer * N_SLAB + 1
        + max_nnz_per_layer + pad + (sizeof(float) / sizeof(int)) * max_nnz_per_layer)
    );

  }
}


void tsv_string_to_1D_array(
  const std::string& s,
  const size_t cols,
  float* arr
) {
  std::string line;
  std::istringstream read_s(s);

  std::vector<std::string> tokens;

  while(std::getline(read_s, line)) {
    std::istringstream lineStream(line);
    std::string token;
    tokens.clear();
    while(std::getline(lineStream, token, '\t')) {
      tokens.push_back(std::move(token));
    }
    arr[(std::stoi(tokens[0]) - 1) * cols + std::stoi(tokens[1]) - 1] = std::stof(tokens[2]);
  }

}


void read_input(
  const std::string& input_path,
  const size_t num_features,
  float* arr
) {

  auto input_str = read_file_to_string(input_path);
  tsv_string_to_1D_array(input_str, num_features, arr);
}


__global__
void identify(
  float* target_arr,
  const size_t batch_size,
  const size_t num_neurons_per_layer,
  int* result_arr
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < batch_size; i += gridDim.x * blockDim.x) {
    float sum = thrust::reduce(
      thrust::device,
      target_arr + i * num_neurons_per_layer,
      target_arr + (i + 1) * num_neurons_per_layer,
      0,
      thrust::plus<float>()
    );
    result_arr[i] = sum > 0 ? 1 : 0;
  }
}

inline
Eigen::Matrix<int, Eigen::Dynamic, 1> arr_to_Eigen_int(
  const int* arr,
  const size_t arr_len
) {
  Eigen::Matrix<int, Eigen::Dynamic, 1> result(arr_len, 1);
  for(size_t i = 0; i < arr_len; ++i) {
    result(i, 1) = arr[i];
  }
  return result;
};


inline
std::stringstream read_file_to_sstream(const std::string& path) {
  
  using namespace std::literals::string_literals;

  std::ifstream f{ path };

  if(!f) {
    throw std::runtime_error("cannot open the file"s + path);
  }

  std::stringstream sstream;
  sstream << f.rdbuf();
  return sstream;
}

inline
Eigen::Matrix<int, Eigen::Dynamic, 1> read_golden(
  const std::string& golden_path,
  const size_t num_inputs
) {
  std::string line;
  std::stringstream read_s = read_file_to_sstream(golden_path);
  Eigen::Matrix<int, Eigen::Dynamic, 1> golden = Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(num_inputs, 1);

  while(std::getline(read_s, line)) {
    golden(std::stoi(line) - 1, 0) = 1;
  }   
  return golden;
}

inline
bool is_passed(
  const Eigen::Matrix<int, Eigen::Dynamic, 1>& output,
  const Eigen::Matrix<int, Eigen::Dynamic, 1>& golden
) {
  int check = output.rows() - output.cwiseEqual(golden).count();
  std::cout << "\nNumber of different categories: " << check << std::endl;
  return (check == 0);
}

void read_input_xy(
  const std::string input_file_name, 
  std::vector<std::vector<float>> &input, 
  const int neuron, 
  const int batch, 
  const int offset) 
{
  std::ifstream input_file(input_file_name);
  if(!input_file){
      std::cout << "FILE:" << input_file_name << " does not exists.\n";
      exit(-1);
  }
  int b, n;
  float val;
  long read_num = 0;
  while(input_file >> b >> n >> val) {
      if(b <= batch+offset && b > offset) {
          read_num++;
          input[b-offset - 1][n - 1] = val;
          if(val != 1.00) {
              printf("read input %d, %f\n", b, val);
          }
      }
  }
  std::cout << "Read Input success! read_number = " << read_num << std::endl;
}


}