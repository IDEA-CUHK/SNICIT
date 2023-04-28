#include <BF/kernel.hpp>
#include <cuda_error.hpp>
#include <chrono>
#include <fstream>
#include <general.hpp>
#include <string>

namespace SNICIT_BEY{

class BF{

  private:
    std::vector<float*> _dev_Y_hidden; // 2 buffers
    std::vector<int*> _dev_hidden_roffw; // hidden layer W row offset
    std::vector<int*> _dev_hidden_colsw; // hidden layer W cols
    std::vector<float*> _dev_hidden_valsw; // hidden layer W vals
    std::vector<float*> _dev_hidden_bias; // hidden layer W vals

    std::vector<int*>  _dev_rowsY;
    std::vector<int*>  _dev_rlenY;

    float *Y_input; // on cpu
    float *_dev_Y_output_whole; // on gpu

    float* _dev_Y_input;
    float* _dev_Y_output;
    float* _dev_input_weight;
    float* _dev_input_bias;
    float* _dev_output_weight;
    float* _dev_output_bias;
    int* _dev_result_label;

    size_t _dev_nerowsY;

    std::string weight_path, bias_path;
    int num_hidden_neurons, num_layers;
    int input_size, num_classes, batch_size, num_input;
    int nnz, threshold;
    float density;
    bool is_cifar;

    void _infer();
    
    void _preprocess(const std::string& input_path);

    void _non_empty_rows(const size_t buff);
        
    void _weight_bias_alloc_read();

    void _input_alloc_read(const std::string& input_path);

    void _result_alloc_read(const std::string& input_path);

  public:
    BF(
      const std::string& _weight_path,
      const std::string& _bias_path,
      const int _num_hidden_neurons,
      const int _num_layers,
      const int _threshold,
      const float _density,
      const int _batch_size,
      const int _num_input,
      const bool _is_cifar
    );

    ~BF();
    
    void infer(const std::string& input_path);

};

BF::BF(
    const std::string& _weight_path,
    const std::string& _bias_path,
    const int _num_hidden_neurons,
    const int _num_layers,
    const int _threshold,
    const float _density,
    const int _batch_size,
    const int _num_input,
    const bool _is_cifar
) : weight_path(_weight_path), bias_path(_bias_path), 
    num_hidden_neurons(_num_hidden_neurons), num_layers(_num_layers), 
    num_classes(10), density(_density), threshold(_threshold),
    nnz(std::round(_num_hidden_neurons*_num_hidden_neurons*_density)), 
    batch_size(_batch_size), num_input(_num_input), is_cifar(_is_cifar)
 {
  std::cout<<"Constructing BF method......\n";
  input_size = is_cifar ? _num_hidden_neurons : 784;
}

BF::~BF() {
  for(auto& each_Y : _dev_Y_hidden) {
    checkCuda(cudaFree(each_Y));
  }
  for(auto& each_dev_hidden_roffw : _dev_hidden_roffw) {
    checkCuda(cudaFree(each_dev_hidden_roffw));
  }
  for(auto& each_dev_hidden_colsw : _dev_hidden_colsw) {
    checkCuda(cudaFree(each_dev_hidden_colsw));
  }
  for(auto& each_dev_hidden_valsw : _dev_hidden_valsw) {
    checkCuda(cudaFree(each_dev_hidden_valsw));
  }
  for(auto& each_dev_hidden_bias : _dev_hidden_bias) {
    checkCuda(cudaFree(each_dev_hidden_bias));
  }
  for(auto& each_rowsY : _dev_rowsY) {
    checkCuda(cudaFree(each_rowsY));
  }
  for(auto& each_rlenY : _dev_rlenY) {
    checkCuda(cudaFree(each_rlenY));
  }

  checkCuda(cudaFree(_dev_output_weight));
  checkCuda(cudaFree(_dev_output_bias));
  if (!is_cifar) {
    checkCuda(cudaFree(_dev_Y_input));
    checkCuda(cudaFree(_dev_input_weight));
    checkCuda(cudaFree(_dev_input_bias));
  }
  checkCuda(cudaFree(_dev_Y_output));
  checkCuda(cudaFree(_dev_result_label));
  checkCuda(cudaFree(_dev_Y_output_whole));

  delete [] Y_input;

}

void BF::infer(const std::string& input_path) {
  _preprocess(input_path);

  _infer();

}

void BF::_preprocess(const std::string& input_path) {
  std::cout<<"preprocessing......\n";
  auto _tic = std::chrono::steady_clock::now();

  _weight_bias_alloc_read();
  
  _input_alloc_read(input_path);
  // std::cout<<"!!!!!!!!!!!\n";
  _result_alloc_read(input_path);


  auto _toc = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(_toc - _tic).count();
  std::cout<<"finished preprocessing in "<<duration<< "ms"<<std::endl;
}


void BF::_weight_bias_alloc_read() {
  std::string line;
  std::ifstream MyReadFile;
  int ptr = 0;
  int file_offset;
  if (!is_cifar) {
    file_offset = 2;
    // allocate input layer's weight and bias
    checkCuda(cudaMalloc(
      &_dev_input_weight,
      input_size * num_hidden_neurons * sizeof(float)
    ));
    checkCuda(cudaMalloc(
      &_dev_input_bias,
      num_hidden_neurons * sizeof(float)
    ));
    // read input layer's weight and bias
    float *input_weight;
    float *input_bias;
    input_weight = new float[input_size * num_hidden_neurons];
    input_bias = new float[num_hidden_neurons];

    MyReadFile = std::ifstream(weight_path+"l1-dense.tsv");
    if (MyReadFile.is_open()) {
      while(std::getline(MyReadFile, line)){
        input_weight[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: open weight file " << weight_path+"l1-dense.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();
    MyReadFile = std::ifstream(bias_path+"l1-dense.tsv");
    if (MyReadFile.is_open()) {
      ptr = 0;
      while(std::getline(MyReadFile, line)){
        input_bias[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: open bias file " << weight_path+"l1-dense.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();
    // copy input layer's weight and bias
    checkCuda(cudaMemcpy(_dev_input_weight, input_weight, 
      input_size * num_hidden_neurons * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(_dev_input_bias, input_bias, 
      num_hidden_neurons * sizeof(float), cudaMemcpyHostToDevice));
    delete [] input_weight;
    delete [] input_bias;
  }
  else {file_offset = 1;}
  for(int hidden_layer = 0; hidden_layer < num_layers; hidden_layer++) {
    int *hidden_roffw;
    int *hidden_colsw;
    float *hidden_valsw;
    float *hidden_bias;

    hidden_roffw = new int[num_hidden_neurons+1];
    hidden_colsw = new int[nnz];
    hidden_valsw = new float[nnz];
    hidden_bias = new float[num_hidden_neurons];
    memset(hidden_roffw, 0, (num_hidden_neurons+1)*sizeof(int));
    int* dev_cur_roffw;
    int* dev_cur_colsw;
    float* dev_cur_valsw;
    float* dev_cur_bias;

    // allocate hidden layer's weight and bias
    checkCuda(cudaMallocManaged(
      &dev_cur_roffw,
      (num_hidden_neurons+1) * sizeof(int)
    ));
    checkCuda(cudaMallocManaged(
      &dev_cur_colsw,
      nnz * sizeof(int)
    ));
    checkCuda(cudaMallocManaged(
      &dev_cur_valsw,
      nnz * sizeof(float)
    ));

    // read hidden layer
    
    MyReadFile = std::ifstream(weight_path+"l"+std::to_string(hidden_layer+file_offset)+"-sparse.tsv");
    if (MyReadFile.is_open()) {
      ptr = 0;
      std::vector<std::string> tokens;
      while(std::getline(MyReadFile, line)){

        std::stringstream lineStream(line);
        std::string token;
        tokens.clear();
        while(std::getline(lineStream, token, '\t')) {
          tokens.push_back(std::move(token));
        }
        hidden_roffw[std::stoi(tokens[0])+1]++;
        hidden_colsw[ptr] = std::stoi(tokens[1]);
        hidden_valsw[ptr] = std::stof(tokens[2]);
        ptr++;
      }
      for (int i = 0; i < num_hidden_neurons; i++)
      {
        hidden_roffw[i + 1] += hidden_roffw[i];
      }
    }
    else {
      std::cout << "ERROR: open weight file " << weight_path+"l"+
        std::to_string(hidden_layer+file_offset)+"-sparse.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();
    // copy hidden layer
    checkCuda(cudaMemcpy(dev_cur_roffw, hidden_roffw, 
      (num_hidden_neurons+1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_cur_colsw, hidden_colsw, 
      nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_cur_valsw, hidden_valsw, 
      nnz * sizeof(float), cudaMemcpyHostToDevice));
    _dev_hidden_roffw.emplace_back(dev_cur_roffw);
    _dev_hidden_colsw.emplace_back(dev_cur_colsw);
    _dev_hidden_valsw.emplace_back(dev_cur_valsw);

    checkCuda(cudaMallocManaged(
      &dev_cur_bias,
      num_hidden_neurons * sizeof(float)
    ));


    MyReadFile = std::ifstream(bias_path+"l"+std::to_string(hidden_layer+file_offset)+"-sparse.tsv");
    if (MyReadFile.is_open()) {
      ptr = 0;
      while(std::getline(MyReadFile, line)){
        hidden_bias[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: open bias file " << bias_path+"l"+
        std::to_string(hidden_layer+file_offset)+"-sparse.tsv"<<std::endl;
      exit(1);
    }
    MyReadFile.close();

    checkCuda(cudaMemcpy(dev_cur_bias, hidden_bias, 
      num_hidden_neurons * sizeof(float), cudaMemcpyHostToDevice));

    _dev_hidden_bias.emplace_back(dev_cur_bias);

    delete [] hidden_roffw;
    delete [] hidden_colsw;
    delete [] hidden_valsw;
    delete [] hidden_bias;
  }

  checkCuda(cudaMalloc(
    &_dev_output_weight,
    num_hidden_neurons * num_classes * sizeof(float)
  ));
  checkCuda(cudaMalloc(
    &_dev_output_bias,
    num_classes * sizeof(float)
  ));
  float *output_weight;
  float *output_bias;
  output_weight = new float[num_hidden_neurons * num_classes];
  output_bias = new float[num_classes];
  MyReadFile = std::ifstream(weight_path+"l"+std::to_string(num_layers+file_offset)+"-dense.tsv");
  if (MyReadFile.is_open()) {
    ptr = 0;
    while(std::getline(MyReadFile, line)){
      output_weight[ptr++] = std::stof(line);
    }
  }
  else {
    std::cout << "ERROR: open weight file " << weight_path+"l"+
      std::to_string(num_layers+file_offset)+"-dense.tsv"<<std::endl;
    exit(1);
  }
  MyReadFile.close();
  MyReadFile = std::ifstream(bias_path+"l"+std::to_string(num_layers+file_offset)+"-dense.tsv");
  if (MyReadFile.is_open()) {
    ptr = 0;
    while(std::getline(MyReadFile, line)){
      output_bias[ptr++] = std::stof(line);
    }
  }
  else {
    std::cout << "ERROR: open weight file " << bias_path+"l"+
      std::to_string(num_layers+file_offset)+"-dense.tsv"<<std::endl;
    exit(1);
  }
  MyReadFile.close();

  checkCuda(cudaMemcpy(_dev_output_weight, output_weight, 
    num_hidden_neurons * num_classes * sizeof(float), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(_dev_output_bias, output_bias, 
    num_classes * sizeof(float), cudaMemcpyHostToDevice));

  delete [] output_weight;
  delete [] output_bias;
}


void BF::_non_empty_rows(const size_t buff) {
  _dev_nerowsY = 0;
  for(size_t i = 0; i < batch_size; ++i) {
    if((_dev_rlenY[buff])[i] > 0) {
      (_dev_rowsY[buff])[_dev_nerowsY++] = i;
    }
  }
}

void BF::_input_alloc_read(const std::string& input_path) {
  Y_input = new float[num_input*input_size];
  if (!is_cifar) {
    std::ifstream file(input_path+"t10k-images-idx3-ubyte", std::ios::binary);
    if (file.is_open())
    {
      int magic_number=0;
      int number_of_images=0;
      int n_rows=0;
      int n_cols=0;
      file.read((char*)&magic_number,sizeof(magic_number));
      file.read((char*)&number_of_images,sizeof(number_of_images));
      file.read((char*)&n_rows,sizeof(n_rows));
      file.read((char*)&n_cols,sizeof(n_cols));
      for(int i=0;i<10000;++i)
      {
        for(int r=0;r<input_size;++r)
        {
          unsigned char temp=0;
          file.read((char*)&temp,sizeof(temp));
          Y_input[i*input_size+r]= ((float)temp)/255.0;
        }
      }
      checkCuda(cudaMallocManaged(
        &_dev_Y_input,
        batch_size*input_size * sizeof(float)
      ));
    }
    else {
      std::cout << "ERROR: MNIST input file open failed" << std::endl;
      exit(1);
    }
    file.close();
  }
  else {
    std::ifstream file(input_path+"cifar-input.txt");
    if (file.is_open()) {
      std::string line;
      int ptr = 0;
      while(std::getline(file, line)){
        Y_input[ptr++] = std::stof(line);
      }
    }
    else {
      std::cout << "ERROR: CIFAR-10 input file open failed" << std::endl;
      exit(1);
    }
    file.close();
  }


  for(int buff=0; buff<2; buff++) {
    float *_dev_buff_Y;
    checkCuda(cudaMallocManaged(
      &_dev_buff_Y,
      batch_size*num_hidden_neurons * sizeof(float)
    ));
    _dev_Y_hidden.emplace_back(_dev_buff_Y);
  }
  checkCuda(cudaMallocManaged(
    &_dev_Y_output_whole,
    num_input*num_classes * sizeof(float)
  ));
  checkCuda(cudaMallocManaged(
    &_dev_Y_output,
    batch_size*num_classes * sizeof(float)
  ));
  int* _dev_rowsY0;
  int* _dev_rowsY1;
  _dev_rowsY.emplace_back(_dev_rowsY0);
  _dev_rowsY.emplace_back(_dev_rowsY1);

  int* _dev_rlenY0;
  int* _dev_rlenY1;
  _dev_rlenY.emplace_back(_dev_rlenY0);
  _dev_rlenY.emplace_back(_dev_rlenY1);

  size_t ry_size = num_input * sizeof(int);
  for(int buff = 0; buff < 2; ++buff) {
    checkCuda(cudaMallocManaged(&_dev_rowsY[buff], ry_size));
    checkCuda(cudaMallocManaged(&_dev_rlenY[buff], ry_size));
    checkCuda(cudaMemset(_dev_rowsY[buff], 0, ry_size));
  }
  checkCuda(cudaMemset(_dev_rlenY[0], 1, ry_size));
  checkCuda(cudaMemset(_dev_rlenY[1], 0, ry_size));

  _non_empty_rows(0);
}

void BF::_result_alloc_read(const std::string& input_path) {
  int *label;
  label = new int[num_input];
  if (!is_cifar) {
    std::ifstream file(input_path+"t10k-labels-idx1-ubyte", std::ios::binary);
    if (file.is_open())
    {
      int magic_number=0;
      int number_of_images=0;
      file.read((char*)&magic_number,sizeof(magic_number));
      file.read((char*)&number_of_images,sizeof(number_of_images));
      for(int i = 0; i < 10000; ++i)
      {
        unsigned char temp=0;
        file.read((char*)&temp, sizeof(temp));
        label[i] = (int)temp;
      }
    }
    else {
      std::cout << "ERROR: MNIST result file open failed" << std::endl;
      exit(1);
    }
    file.close();
  }
  else {
    std::ifstream file;
    file.open(input_path+"cifar-label.bin", std::ios::in | std::ios::binary | std::ios::ate);
    if (!file) {
        std::cout << "ERROR: CIFAR-10 result file open failed" << std::endl;
        exit(1);
    }

    auto file_size = file.tellg();
    std::unique_ptr<char[]> buffer(new char[file_size]);

    //Read the entire file at once
    file.seekg(0, std::ios::beg);
    file.read(buffer.get(), file_size);
    file.close();

    for(std::size_t i = 0; i < num_input; ++i){
      int l = buffer[i * 3073];
      label[i] = l;
    }
  }

  checkCuda(cudaMallocManaged(
    &_dev_result_label,
    num_input * sizeof(int)
  ));

  checkCuda(cudaMemcpy(_dev_result_label, label, 
    num_input * sizeof(int), cudaMemcpyHostToDevice));

  delete [] label;
}

void BF::_infer() {
  std::cout<<"inferring......\n";
  auto _tic = std::chrono::steady_clock::now();
  double sparse_duration = 0.0;
  double post_duration = 0.0;
  cudaStream_t dev_stream;
  checkCuda(cudaStreamCreate(&dev_stream));
  for (int round = 0; round < num_input / batch_size; round++) {
    std::cout<<"[round "<<round<<"] begins: "<<std::endl;
    if (!is_cifar) {
      checkCuda(cudaMemcpy(_dev_Y_input, Y_input+round * batch_size * input_size, 
        batch_size*input_size * sizeof(float), cudaMemcpyHostToDevice));

      dense_input<<<batch_size, dim3(num_hidden_neurons, (int)(1024/num_hidden_neurons), 1), 
        sizeof(float)*num_hidden_neurons, dev_stream>>>(_dev_Y_input, 
        _dev_input_weight, _dev_input_bias, batch_size, input_size, num_hidden_neurons, _dev_Y_hidden[0]);
    }
    else {
      checkCuda(cudaMemcpy(_dev_Y_hidden[0], Y_input+round * batch_size * input_size, 
        batch_size*input_size * sizeof(float), cudaMemcpyHostToDevice));
    }
    checkCuda(cudaStreamSynchronize(dev_stream));
    auto sparse_tic = std::chrono::steady_clock::now();

    for(int cur_layer = 0; cur_layer < num_layers; cur_layer++) {
      auto layer_tic = std::chrono::steady_clock::now();
      // BF
      bf_inference<<<_dev_nerowsY, dim3((int)(1024/num_hidden_neurons), num_hidden_neurons, 1), 
        sizeof(float)*num_hidden_neurons, dev_stream>>>(
        _dev_Y_hidden[cur_layer%2], 
        _dev_nerowsY,
        _dev_rowsY[cur_layer % 2],
        _dev_rlenY[cur_layer % 2],
        _dev_hidden_roffw[cur_layer], 
        _dev_hidden_colsw[cur_layer], 
        _dev_hidden_valsw[cur_layer], 
        _dev_hidden_bias[cur_layer], 
        batch_size, 
        num_hidden_neurons, 
        num_hidden_neurons, 
        _dev_Y_hidden[(cur_layer+1) % 2],
        _dev_rlenY[(cur_layer+1) % 2]
      );
      checkCuda(cudaStreamSynchronize(dev_stream));
      _non_empty_rows((cur_layer + 1) % 2);
      // checkCuda(cudaStreamSynchronize(dev_stream));
      checkCuda(cudaMemset(
        _dev_Y_hidden[cur_layer % 2],
        0,
        batch_size*num_hidden_neurons*sizeof(float)
      ));
      
      auto layer_toc = std::chrono::steady_clock::now();
      auto layer_duration = std::chrono::duration_cast<std::chrono::microseconds>(layer_toc - layer_tic).count();
      if (cur_layer >= threshold) {post_duration += std::chrono::duration_cast<std::chrono::microseconds>(layer_toc - layer_tic).count();}
      std::cout<<"finished layer "<< cur_layer <<" in "<< layer_duration/1000.0<< "ms"<<std::endl;
    }

    auto sparse_toc = std::chrono::steady_clock::now();
    sparse_duration += std::chrono::duration_cast<std::chrono::microseconds>(sparse_toc - sparse_tic).count();
    
    dense_output<<<batch_size, dim3(num_classes, (int)(1024/num_classes), 1), 
        sizeof(float)*num_classes, dev_stream>>>(_dev_Y_hidden[0], 
        _dev_output_weight, _dev_output_bias, batch_size, num_hidden_neurons, num_classes, _dev_Y_output);
    checkCuda(cudaStreamSynchronize(dev_stream));
    checkCuda(cudaMemcpy(_dev_Y_output_whole+round * batch_size * num_classes, _dev_Y_output,
      batch_size*num_classes * sizeof(float), cudaMemcpyDeviceToDevice));
  }
  int* cnt;
  checkCuda(cudaMallocManaged(
    &cnt,
    sizeof(int)
  ));
  check_acc<<<1, 1024, sizeof(int), dev_stream>>>(_dev_Y_output_whole, num_classes, num_input, _dev_result_label, cnt);
  checkCuda(cudaStreamSynchronize(dev_stream));
  checkCuda(cudaDeviceSynchronize());
  
  std::cout<<"BF info: accuracy "<<100*((float)cnt[0]/(float)num_input)<<"%"<<" runtime "<< sparse_duration/1000.0<< "ms"<<
  " avgpost "<<post_duration/(1000.0*(num_layers-threshold))<< "ms"<<std::endl;

  auto _toc = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(_toc - _tic).count();
  std::cout<<"[Total] finished inferring in "<<duration/1000.0<< "ms"<<std::endl;
  
}

}