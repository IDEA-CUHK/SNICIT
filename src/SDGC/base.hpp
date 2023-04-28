#pragma once

#include <utility.hpp>
#include <chrono>
#include <iostream>
#include <cuda_error.hpp>
#include <fstream>


namespace SNICIT_SDGC {

class Base {

  protected:

    //model configuration
    float _bias;
    size_t _num_neurons;
    size_t _num_layers;
    size_t _num_inputs;
    
    //Both SNIG and BF use maximum external shared memory
    size_t _num_secs;
    size_t _sec_size;

    //weights
    int* _host_pinned_weight;
    size_t _max_nnz;
    size_t _pad {0};
    size_t _p_w_index_len;
    size_t _pp_w_index_len;
    size_t _pp_wlen;
    size_t _pp_wsize;


    Base(
      const std::string& weight_path,
      const float bias,
      const size_t num_neurons,
      const size_t num_layers
    );

    virtual ~Base();

  private:

    void _load_weight(const std::string& weight_path); 

    size_t num_neurons() const;

    size_t num_layers() const;

    virtual void _preprocess(const std::string& input_path) = 0;
    
    virtual void _weight_alloc() = 0;

    virtual void _input_alloc() = 0;

    virtual void _result_alloc() = 0;

    virtual void _infer() = 0;

};

// ----------------------------------------------------------------------------
// Definition of Base
// ----------------------------------------------------------------------------

Base::Base(
  const std::string& weight_path,
  const float bias,
  const size_t num_neurons,
  const size_t num_layers
) : 
  _bias{bias},
  _num_neurons{num_neurons},
  _num_layers{num_layers}
{
  _sec_size = get_sec_size(Base::_num_neurons);
  _num_secs = (Base::_num_neurons) / _sec_size;
  _load_weight(weight_path);
}

Base::~Base() {
  checkCuda(cudaFreeHost(_host_pinned_weight));
}


void Base::_load_weight(const std::string& weight_path) {
  std::cout<<("Loading the weight......")<<std::endl;
  auto _tic = std::chrono::steady_clock::now();
  _max_nnz = find_max_nnz(
               weight_path,
               _num_layers,
               _num_neurons
             );

  // total length of row and col index
  // value index should consider sizeof(T)
  _p_w_index_len  = _num_neurons * _num_secs + _max_nnz + 1;

  //handle aligned
  if((sizeof(int) * _p_w_index_len) % sizeof(float) != 0) {
    ++_pad;
  }

  _pp_w_index_len = _p_w_index_len + _pad;
  

  //pad packed weight length
  //max_nnz should be even, otherwis it needs to be padded
  _pp_wlen = _pp_w_index_len + (sizeof(float) / sizeof(int)) * _max_nnz;

  //pad packed weight size
  _pp_wsize = sizeof(int) * (_pp_w_index_len) + sizeof(float) * _max_nnz;
  
  checkCuda(cudaMallocHost(
    (void**)&_host_pinned_weight,
    _pp_wsize * _num_layers
  ));

  std::memset(
    _host_pinned_weight,
    0,
    _pp_wsize * _num_layers
  );

  read_weight(
    weight_path,
    _num_neurons,
    _max_nnz,
    _num_layers,
    _sec_size,
    _num_secs,
    _pad,
    _host_pinned_weight
  );
  auto _toc = std::chrono::steady_clock::now();
  auto _duration = std::chrono::duration_cast<std::chrono::microseconds>(_toc - _tic).count();
  std::cout<<"Finish reading DNN layers with "<< _duration / 1000.0<<" ms"<<std::endl;
}

size_t Base::num_neurons() const {
   return _num_neurons; 
}

size_t Base::num_layers() const { 
  return _num_layers; 
}



}