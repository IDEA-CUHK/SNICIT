#pragma once

#include <Eigen/Core>
#include <SNIG/kernel.hpp>
#include <base.hpp>
#include <utility.hpp>
#include <taskflow/taskflow.hpp>
#include <taskflow/cudaflow.hpp>


namespace SNICIT_SDGC {


class SNIG : public Base {

  
  private:
    
    size_t _batch_size;
    size_t _num_weight_buffers;
    float* _source_Y;
    bool* _source_is_nonzero_row;
    std::vector<float*>  _dev_Y;
    std::vector<bool*>  _dev_is_nonzero_row;
    std::vector<int*> _dev_W;

    size_t _batch_ylen;
    size_t _batch_ysize;
    int* _results;
    // size_t _num_duplicates; // duplicate inputs for experiments on updating methods
    //                            // number of inputs = _num_duplicates x _num_inputs

    void _set_parameters(
      const size_t num_inputs,
      const size_t batch_size,
      const size_t num_weight_buffers
    );

    void _preprocess(const std::string& input_path);
  
    void  _infer();

    void _input_alloc();

    void _weight_alloc();

    void _result_alloc();

  public:

    SNIG(
      const std::string& weight_path,
      const float bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120
    );

    ~SNIG();

    void infer(
      const std::string& input_path,
      const std::string& golden_path,
      const size_t num_inputs,
      const size_t batch_size,
      const size_t num_buff
    );

};

// ----------------------------------------------------------------------------
// Definition of SNIG
// ----------------------------------------------------------------------------

SNIG::SNIG(
  const std::string& weight_path,
  const float bias,
  const size_t num_neurons_per_layer,
  const size_t num_layers
):
  Base(weight_path, bias, num_neurons_per_layer, num_layers)
{
  std::cout<<("Constructing SNIG method......")<<std::endl;
}


SNIG::~SNIG() {

  checkCuda(cudaFree(_source_Y));
  checkCuda(cudaFree(_source_is_nonzero_row));

  for(auto& each_W : _dev_W) {
    checkCuda(cudaFree(each_W));
  }

  checkCuda(cudaFree(_dev_Y[1])); 
  checkCuda(cudaFree(_dev_is_nonzero_row[1]));

  checkCuda(cudaFree(_results));
}

void SNIG::infer(
  const std::string& input_path,
  const std::string& golden_path,
  const size_t num_inputs,
  const size_t batch_size,
  const size_t num_weight_buffers
) {
  
  std::cout<<"Total input size : "<<num_inputs<<std::endl;
  std::cout<<"Input batch size : "<<batch_size<<std::endl;
  std::cout<<"Number of weight buffers : "<<num_weight_buffers<<std::endl;

  _set_parameters(
    num_inputs,
    batch_size,
    num_weight_buffers
  );

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

void SNIG::_set_parameters(
  const size_t num_inputs,
  const size_t batch_size,
  const size_t num_weight_buffers
) {
  Base::_num_inputs = num_inputs;
  _num_weight_buffers = num_weight_buffers;

  _batch_size = batch_size;
  _batch_ylen = _batch_size * Base::_num_neurons;
  _batch_ysize = _batch_ylen * sizeof(float);

}

void SNIG::_preprocess(const std::string& input_path) {
  std::cout <<"Preprocessing...... "<<std::endl;
  auto _tic = std::chrono::steady_clock::now();

  //weight allocation
  _weight_alloc();
  //input allocation
  _input_alloc();
  //final results allocation
  _result_alloc();
  
  //read input
  read_input(input_path, Base::_num_neurons, _source_Y);
  auto _toc = std::chrono::steady_clock::now();
  auto _duration = std::chrono::duration_cast<std::chrono::microseconds>(_toc - _tic).count();
  std::cout<<("Finish preprocessing with ", _duration / 1000.0, " ms", "\n")<<std::endl;
}


void SNIG::_infer() {
  std::cout <<"Start inference...... "<<std::endl;
  auto _tic = std::chrono::steady_clock::now();
  size_t accumulated_duplicates{0}; 

  //Use taskflow and cudaGraph to implement task graph
  tf::Taskflow taskflow("SNIG");
  tf::Executor executor;
  tf::Task stop_inners;
  tf::Task first_fetchs;
  tf::Task cudaflows;
  tf::Task fetchs;
//   first_fetchs.reserve(Base<T>::_num_gpus);
//   cudaflows.reserve(Base<T>::_num_gpus);
//   fetchs.reserve(Base<T>::_num_gpus);

  std::atomic<size_t> finished_inputs{0};
  int* dev_results = nullptr;

  dim3 grid_dim(_batch_size, Base::_num_secs, 1);

  tf::Task start = taskflow.emplace([](){}).name("start");
  tf::Task start_inner = taskflow.emplace([](){}).name("start_inner");

  
    first_fetchs = (taskflow.emplace([&](){
        int is_end = 1;
        size_t beg_inputs = finished_inputs.fetch_add(_batch_size);
        if(beg_inputs < Base::_num_inputs) {
        _dev_Y[0] = _source_Y + beg_inputs * Base::_num_neurons;
        _dev_is_nonzero_row[0] = _source_is_nonzero_row + beg_inputs * Base::_num_secs;
        dev_results = _results + beg_inputs;
        checkCuda(cudaMemPrefetchAsync(_dev_Y[0], _batch_ysize, 0, NULL));
        checkCuda(cudaMemPrefetchAsync(_dev_is_nonzero_row[0], sizeof(bool) * _batch_size * Base::_num_secs, 0, NULL));
        checkCuda(cudaMemPrefetchAsync(dev_results, sizeof(int) * _batch_size, 0, NULL));
        is_end = 0;
        }
        return is_end;
    }).name("first_fetch"));

    cudaflows = (taskflow.emplace_on([&](tf::cudaFlow& cf){
        std::vector<tf::cudaTask> weight_copies;
        std::vector<tf::cudaTask> infers;
        weight_copies.reserve(Base::_num_layers);
        infers.reserve(Base::_num_layers);

    for(size_t cur_layer = 0; cur_layer < Base::_num_layers; cur_layer += _num_weight_buffers) {
      for(size_t k = 0; k < _num_weight_buffers; ++k) {
          //tasks of cudaflow
          weight_copies.emplace_back(cf.copy(
          _dev_W[k],
          Base::_host_pinned_weight + (cur_layer + k) * Base::_pp_wlen,
          Base::_pp_wlen
          ).name("weight_copy"));

          // transformed CSC weight matrix equals to CSR with exchanged row and col
          int* col_w = _dev_W[k];
          int* row_w = _dev_W[k] + Base::_num_neurons * Base::_num_secs + 1;
          float* val_w = (float*)(_dev_W[k] + Base::_p_w_index_len);
          infers.emplace_back(cf.kernel(
          grid_dim,
          dim3(2, 512, 1),
          sizeof(float) * Base::_sec_size,
          snig_inference,
          _dev_Y[k % 2],
          _dev_is_nonzero_row[k % 2],
          Base::_sec_size,
          Base::_num_secs,
          Base::_num_neurons,
          col_w,
          row_w,
          val_w,
          Base::_bias,
          _dev_is_nonzero_row[(k + 1) % 2],
          _dev_Y[(k + 1) % 2]
          ).name("Inference"));
      }
      }

      // TODO: consider parameterizing the thread numbers
      tf::cudaTask ident = cf.kernel(16, 512, 0, identify, _dev_Y[0], _batch_size, Base::_num_neurons, dev_results);

      //dependencies of cudaflow
      for(size_t cur_layer = 0; cur_layer < Base::_num_layers; ++cur_layer) {
      weight_copies[cur_layer].precede(infers[cur_layer]);

      if(cur_layer + _num_weight_buffers < Base::_num_layers) {
          infers[cur_layer].precede(weight_copies[cur_layer + _num_weight_buffers]);
      }
      if(cur_layer + 1 < Base::_num_layers) {
          infers[cur_layer].precede(infers[cur_layer + 1]);
      }
    }
    infers[Base::_num_layers - 1].precede(ident);
  }, 0).name("GPU"));

  fetchs = taskflow.emplace([&](){
      int is_end = 1;
      size_t beg_inputs = finished_inputs.fetch_add(_batch_size);
      if(beg_inputs < Base::_num_inputs) {
      _dev_Y[0] = _source_Y + beg_inputs * Base::_num_neurons;
      _dev_is_nonzero_row[0] = _source_is_nonzero_row + beg_inputs * Base::_num_secs;
      dev_results = _results + beg_inputs;
      checkCuda(cudaMemPrefetchAsync(_dev_Y[0], _batch_ysize, 0, NULL));
      checkCuda(cudaMemPrefetchAsync(_dev_is_nonzero_row[0], sizeof(bool) * _batch_size * Base::_num_secs, 0, NULL));
      checkCuda(cudaMemPrefetchAsync(dev_results, sizeof(int) * _batch_size, 0, NULL));
      is_end = 0;
      }
      return is_end;
  }).name("fetch");

  stop_inners = taskflow.emplace([](){}).name("stop_inner");
  
  
  tf::Task stop = taskflow.emplace([](){}).name("stop");


  tf::Task duplicate = taskflow.emplace([&](){
    finished_inputs = 0;
    ++accumulated_duplicates;
    return (accumulated_duplicates < 1) ? 0 : 1;
  }).name("duplicate");


    //dependencies of taskflow
    start.precede(start_inner);
    start_inner.precede(first_fetchs);
    first_fetchs.precede(cudaflows, duplicate);
    cudaflows.precede(fetchs);
    fetchs.precede(cudaflows, stop_inners);
    stop_inners.precede(duplicate);
    duplicate.precede(start_inner, stop);
  
  executor.run(taskflow).wait();

  auto _toc = std::chrono::steady_clock::now();
  auto _duration = std::chrono::duration_cast<std::chrono::microseconds>(_toc - _tic).count();
  std::cout<<"SNIG info: runtime "<<_duration / 1000.0<< " ms"<<std::endl;
}

void SNIG::_weight_alloc() {
  std::vector<int*> W(_num_weight_buffers, nullptr);

    for(auto& each_W : W) {
        checkCuda(cudaMalloc(
        &each_W,
        Base::_pp_wsize
        ));
    }
    _dev_W = W;

}

void SNIG::_input_alloc() {
  size_t ylen = Base::_num_inputs *  Base::_num_neurons;
  size_t ysize = ylen * sizeof(float);

  checkCuda(cudaMallocManaged(&_source_Y, ysize));
  checkCuda(cudaMallocManaged(&_source_is_nonzero_row, sizeof(bool) * Base::_num_inputs * Base::_num_secs));
  checkCuda(cudaMemset(_source_is_nonzero_row, 1, sizeof(bool) * Base::_num_inputs * Base::_num_secs));

  std::vector<float*> Y{2, nullptr};
  std::vector<bool*> is_nonzero_row{2, nullptr};
  {
    checkCuda(cudaMalloc(&Y[1], _batch_ysize));
    checkCuda(cudaMalloc(&is_nonzero_row[1], sizeof(bool) * _batch_size * Base::_num_secs));
    checkCuda(cudaMemset(Y[1], 0, _batch_ysize));
    checkCuda(cudaMemset(is_nonzero_row[1], 0, sizeof(bool) * _batch_size * Base::_num_secs));
    _dev_Y = Y;
    _dev_is_nonzero_row = is_nonzero_row;
  }
}

void SNIG::_result_alloc() {
  checkCuda(cudaMallocManaged(&_results, sizeof(int) * Base::_num_inputs));
  checkCuda(cudaMemset(_results, 0, sizeof(int) * Base::_num_inputs));
}

}
