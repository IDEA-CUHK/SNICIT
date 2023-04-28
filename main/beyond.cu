#pragma once

#include <CLI11/CLI11.hpp>
#include <BF/BF.hpp>
#include <SNIG/SNIG.hpp>
#include <SNICIT/SNICIT.hpp>

#include <iostream>

int main(int argc, char* argv[]) {
  CLI::App app{"SNICIT-beyond"};
  std::string mode = "SNICIT";
  std::string benchmark = "A";
  app.add_option(
    "-m, --mode", 
    mode, 
    "select mode (SNICIT, BF, SNIG), default is SNICIT"
  );

  app.add_option(
    "-k, --benchmark", 
    benchmark, 
    "A, B, C, D"
  );
  
  std::string weight_path;
  std::string bias_path;
  std::string input_path;
  int num_hidden_neurons;
  int num_layers;
  float density;
  int layer_threshold = -1;
  bool is_cifar;


  int num_input = 10000;
  int batch_size = 10000;
  int seed_size = 128;
  
  app.add_option(
    "-n, --num_input", 
    num_input, 
    "default is 10000"
  );
  app.add_option(
    "-b, --batch_size", 
    batch_size, 
    "default is 10000"
  );
  app.add_option(
    "-t, --threshold", 
    layer_threshold, 
    "default is l/2"
  );
  // get path done!!!! big time
  CLI11_PARSE(app, argc, argv);
  if (benchmark == "A") {
    weight_path = "../dataset/beyond/networks/tsv_weights/n128-l18-acc94.94/";
    bias_path = "../dataset/beyond/networks/tsv_biases/n128-l18-acc94.94/";
    input_path = "../dataset/beyond/MNIST/";
    num_hidden_neurons = 128;
    num_layers = 18;
    density = 0.6;
    is_cifar = false;
    if (layer_threshold == -1) layer_threshold = 8;
  }
  else if (benchmark == "B") {
    weight_path = "../dataset/beyond/networks/tsv_weights/n256-l18-acc96.88/";
    bias_path = "../dataset/beyond/networks/tsv_biases/n256-l18-acc96.88/";
    input_path = "../dataset/beyond/MNIST/";
    num_hidden_neurons = 256;
    num_layers = 18;
    density = 0.6;
    is_cifar = false;
    if (layer_threshold == -1) layer_threshold = 8;
  }
  else if (benchmark == "C") {
    weight_path = "../dataset/beyond/networks/tsv_weights/n256-l12-acc95.61/";
    bias_path = "../dataset/beyond/networks/tsv_biases/n256-l12-acc95.61/";
    input_path = "../dataset/beyond/MNIST/";
    num_hidden_neurons = 256;
    num_layers = 12;
    density = 0.5;
    is_cifar = false;
    if (layer_threshold == -1) layer_threshold = 6;
  }
  else if (benchmark == "D") {
    weight_path = "../dataset/beyond/networks/tsv_weights/n256-l12-acc75.86/";
    bias_path = "../dataset/beyond/networks/tsv_biases/n256-l12-acc75.86/";
    input_path = "../dataset/beyond/CIFAR-10/";
    num_hidden_neurons = 256;
    num_layers = 12;
    density = 0.5;
    is_cifar = true;
    if (layer_threshold == -1) layer_threshold = 6;
  }
  else {
    using namespace std::literals::string_literals;
    throw std::runtime_error("Error benchmark. Please correct your benchmark name");
  }

  std::cout << "Current mode: " << mode << std::endl;
  std::cout << "Benchmark: " << benchmark << std::endl;
  if(mode == "BF") {
    SNICIT_BEY::BF BF_obj(
      weight_path, 
      bias_path,
      num_hidden_neurons, 
      num_layers,
      layer_threshold,
      density,
      batch_size,
      num_input,
      is_cifar
    );
    BF_obj.infer(input_path);
  }
  else if (mode == "SNIG") {
    SNICIT_BEY::SNIG SNIG_obj(
      weight_path, 
      bias_path,
      num_hidden_neurons, 
      num_layers,
      layer_threshold,
      density,
      batch_size,
      num_input,
      is_cifar
    );
    SNIG_obj.infer(input_path);
  }
  else if (mode == "SNICIT") {
    SNICIT_BEY::SNICIT SNICIT_obj(
      weight_path, 
      bias_path,
      num_hidden_neurons, 
      num_layers,
      density,
      seed_size,
      layer_threshold,
      batch_size,
      num_input,
      is_cifar
    );
    SNICIT_obj.infer(input_path);
  }
  else {
    using namespace std::literals::string_literals;
    throw std::runtime_error("Error mode. Please correct your mode name");
  }

}