#pragma once
#include <Eigen/Core>
#include <utility.hpp>
#include <base.hpp>
#include <BF/kernel.hpp>


namespace SNICIT_SDGC{

class BF : public Base {
  //Since we don't have NVlink,
  //this implementation doesn't do load balancing at each iteration.
  //It actually let GPUs work in their own partitioned input data
  
  private:

    //Both BF and SNIG use the maximum externel shared memory for inference
    // COL_BLK == Base<T>::_sec_size
    // N_SLAB  == Base<T>::_num_secs

    std::vector<int*> _rowsY{2, nullptr};
    std::vector<int*> _rlenY{2, nullptr};
    std::vector<float*> _Y{2, nullptr};
    
    //use dev_Y, dev_rowsY,  dev_rlenY, and dev_nerowsY to record each GPUs' own data
    //Since each GPU owns parts of inputs
    //each rowsY in _dev_rowsY is indexed individually by each GPU, rather than indexed by whole inputs
    std::vector<int*> _dev_W;
    std::vector<int*> _dev_rowsY;
    std::vector<int*> _dev_rlenY;
    std::vector<float*> _dev_Y;
    size_t _dev_nerowsY;
    size_t _dev_num_inputs;

    int* _results;

    void _infer();

    void _non_empty_rows(const size_t buff);

    void _set_parameters(
      const size_t num_inputs
    );
    
    void _preprocess(const std::string& input_path);
    
    void _weight_alloc();

    void _input_alloc();

    void _result_alloc();

  public:

    BF(
      const std::string& weight_path,
      const float bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120
    );

    ~BF();
    
    void infer(
      const std::string& input_path,
      const std::string& golden_path,
      const size_t num_inputs
    );

};

// ----------------------------------------------------------------------------
// Definition of BF
// ----------------------------------------------------------------------------
BF::BF(
  const std::string& weight_path,
  const float bias,
  const size_t num_neurons,
  const size_t num_layers
):
  Base(weight_path, bias, num_neurons, num_layers)
{
  std::cout<<("Constructing BF method......")<<std::endl;
}

BF:: ~BF() {
  for(auto& each_Y : _Y) {
    checkCuda(cudaFree(each_Y));
  }
  for(auto& each_rowsY : _rowsY) {
    checkCuda(cudaFree(each_rowsY));
  }
  for(auto& each_rlenY : _rlenY) {
    checkCuda(cudaFree(each_rlenY));
  }
  for(auto& w : _dev_W) {
    checkCuda(cudaFree(w));
  }
  checkCuda(cudaFree(_results));
}

void BF::infer(
  const std::string& input_path,
  const std::string& golden_path,
  const size_t num_inputs
) {
  _set_parameters(num_inputs);

  _preprocess(input_path);

  _infer();

  auto results = arr_to_Eigen_int(_results, Base::_num_inputs);

  auto golden = read_golden(golden_path, Base::_num_inputs);
  if(is_passed(results, golden)) {
    std::cout << "CHALLENGE PASSED\n";
  }
  else{
    std::cout << "CHALLENGE FAILED\n";
  }


}


void BF::_set_parameters(
  const size_t num_inputs
) {
  std::cout<<"Total input size : " << num_inputs<<std::endl;

  Base::_num_inputs = num_inputs; 
}

void BF::_preprocess(const std::string& input_path) {
  std::cout<<"Preprocessing...... "<<std::endl;
  auto _tic = std::chrono::steady_clock::now();

  //weight allocation
  _weight_alloc();

  //input allocation
  _input_alloc();

  //final results allocation
  _result_alloc();
  
  //read input
  read_input(input_path, Base::_num_neurons,_Y[0]);

  auto _toc = std::chrono::steady_clock::now();
  auto _duration = std::chrono::duration_cast<std::chrono::microseconds>(_toc - _tic).count();
  std::cout<<("Finish preprocessing with ", _duration / 1000.0, " ms", "\n")<<std::endl;
}


void BF::_infer() {
  std::cout<<("Start inference...... ")<<std::endl;
  auto _tic = std::chrono::steady_clock::now();
  //store results
  int* dev_results;
  dev_results = _results;

  std::vector<cudaStream_t> dev_stream(2);

  checkCuda(cudaStreamCreate(&dev_stream[0]));
  checkCuda(cudaStreamCreate(&dev_stream[1]));
  for(size_t cur_layer = 0; cur_layer < Base::_num_layers; ++cur_layer) {
    if(cur_layer != Base::_num_layers - 1) {
      checkCuda(cudaMemcpyAsync(
        _dev_W[(cur_layer + 1) % 2],
        Base::_host_pinned_weight + (cur_layer + 1) * (Base::_pp_wlen),
        Base::_pp_wsize,
        cudaMemcpyHostToDevice,
        dev_stream[0]
      ));
    }

    int* roffw = _dev_W[cur_layer % 2];
    int* colsw = _dev_W[cur_layer % 2] + Base::_num_neurons * Base::_num_secs + 1;
    float* valsw = (float*)(_dev_W[cur_layer % 2] + Base::_p_w_index_len);

    bf_inference<<<_dev_nerowsY, dim3(2, 512, 1), sizeof(float) * Base::_sec_size, dev_stream[1]>>>(
      _dev_Y[cur_layer % 2],
      _dev_nerowsY,
      _dev_rowsY[cur_layer % 2],
      _dev_rlenY[cur_layer % 2],
      Base::_sec_size,
      Base::_num_secs,
      Base::_num_neurons,
      roffw,
      colsw,
      valsw,
      Base::_bias,
      _dev_Y[(cur_layer + 1) % 2],
      _dev_rlenY[(cur_layer + 1) % 2]
    );
    checkCuda(cudaStreamSynchronize(dev_stream[1]));

    _non_empty_rows((cur_layer + 1) % 2);

    //Rolling swap requires resetting memory for next iteration
    checkCuda(cudaMemset(
      _dev_Y[cur_layer % 2],
      0,
      _dev_num_inputs * Base::_num_neurons * sizeof(float)
    ));

    checkCuda(cudaStreamSynchronize(dev_stream[0]));
  }
  identify<<<16, 512>>>(_dev_Y[0], _dev_num_inputs, Base::_num_neurons, dev_results);
  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaStreamDestroy(dev_stream[0]));
  checkCuda(cudaStreamDestroy(dev_stream[1]));
  

  auto _toc = std::chrono::steady_clock::now();
  auto _duration = std::chrono::duration_cast<std::chrono::microseconds>(_toc - _tic).count();
  std::cout<<"BF info: runtime "<<_duration / 1000.0<<" ms"<<std::endl;
}

void BF::_non_empty_rows(const size_t buff) {
  _dev_nerowsY = 0;
  for(size_t i = 0; i < _dev_num_inputs; ++i) {
    if((_dev_rlenY[buff])[i] > 0) {
      (_dev_rowsY[buff])[_dev_nerowsY++] = i;
    }
  }
}

void BF::_weight_alloc() {
  std::vector<int*> W(2, nullptr);

  checkCuda(cudaMallocManaged(
    &W[0],
    Base::_pp_wsize
  ));
  checkCuda(cudaMallocManaged(
    &W[1],
    Base::_pp_wsize
  ));
  checkCuda(cudaMemcpy(
    W[0],
    Base::_host_pinned_weight,
    Base::_pp_wsize,
    cudaMemcpyHostToDevice
  ));

  _dev_W = W;
}

void BF::_input_alloc() {
  size_t ylen = Base::_num_inputs * Base::_num_neurons;
  size_t ysize = ylen * sizeof(float);
  size_t ry_size = Base::_num_inputs * sizeof(int);

  for(int buff = 0; buff < 2; ++buff) {
    checkCuda(cudaMallocManaged(&_rowsY[buff], ry_size));
    checkCuda(cudaMallocManaged(&_rlenY[buff], ry_size));
    checkCuda(cudaMallocManaged(&_Y[buff], ysize));
    checkCuda(cudaMemset(_rowsY[buff], 0, ry_size));
  }
  checkCuda(cudaMemset(_rlenY[0], 1, ry_size));
  checkCuda(cudaMemset(_rlenY[1], 0, ry_size));

  //partition
  size_t each_partition = Base::_num_inputs;

  //use dev_Y, dev_rowsY,  dev_rlenY, and dev_nerowsY to record each GPUs' own data
  std::vector<int*> each_GPU_rowsY(2, nullptr);
  std::vector<int*> each_GPU_rlenY(2, nullptr);
  std::vector<float*> each_GPU_Y(2, nullptr);

  for(int buff = 0; buff < 2; ++buff) {
    each_GPU_rowsY[buff] = _rowsY[buff]; 
    each_GPU_rlenY[buff] = _rlenY[buff]; 
    each_GPU_Y[buff] = _Y[buff];
  }
  _dev_rowsY = each_GPU_rowsY;
  _dev_rlenY = each_GPU_rlenY;
  _dev_Y = each_GPU_Y;
  _dev_nerowsY = each_partition;
  _dev_num_inputs = each_partition;

  
  //find non-empty rows at the beginning
  //reindex rowsY
  _non_empty_rows(0);
  
  //Advise
  for(int buff = 0; buff < 2; ++buff) {
    checkCuda(cudaMemAdvise(
      _dev_rowsY[buff],
      _dev_num_inputs * sizeof(int),
      cudaMemAdviseSetPreferredLocation,
      0 
    ));
    checkCuda(cudaMemAdvise(
      _dev_rlenY[buff],
      _dev_num_inputs * sizeof(int),
      cudaMemAdviseSetPreferredLocation,
      0 
    ));
    checkCuda(cudaMemAdvise(
      _dev_Y[buff],
      _dev_num_inputs * Base::_num_neurons * sizeof(float),
      cudaMemAdviseSetPreferredLocation,
      0 
    ));
  }

}

void BF::_result_alloc() {
  //final results allocation
  checkCuda(cudaMallocManaged(&_results, sizeof(int) * Base::_num_inputs));
  checkCuda(cudaMemset(_results, 0, sizeof(int) * Base::_num_inputs));
}

}
