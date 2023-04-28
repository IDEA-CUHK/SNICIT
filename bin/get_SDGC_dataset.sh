#usage: 
#       $1 num_neurons,  --all , or -h

bold=$(tput bold)
normal=$(tput sgr0)

get_command() {
  # help message
  if [[ "$1" == "-h" ]]; then
    echo "usage : ./get_dataset.sh (num_neurons, --all, or -h)"
    echo ""
    echo "\"./get_dataset.sh 1024\" would download and extract benchmarks with 1024 neurons"
    echo "\"./get_dataset.sh --all\" would download and extract all benchmarks"
    echo "All files would be stored in ../dataset"
    exit
  fi

  # make target directories
  if [ ! -d "../dataset/SDGC/MNIST" ]; then
    mkdir ../dataset/SDGC/MNIST
  fi
  if [ ! -d "../dataset/SDGC/tsv_weights" ]; then
    mkdir ../dataset/SDGC/tsv_weights
  fi

  # dowload and extract
  if [[ "$1" == "--all" ]]; then
    echo "${bold}Downloading all files...${normal}"
    num_neurons=(1024 4096 16384 65536)
    for (( k = 0; k < 4; ++k ))
    do
      download ${num_neurons[k]}
      extract ${num_neurons[k]}
    done
  elif [[ $1 == 1024 || $1 == 4096 || $1 == 16384 || $1 == 65536 ]]; then
    download $1
    extract $1
  else
    echo "wrong format!"
    echo "usage : ./get_dataset (num_neurons, --all, or -h)"
    echo ""
    echo "\"./get_dataset 1024\" would download and extract benchmarks with 1024 neurons"
    echo "\"./get_dataset --all\" would download and extract all benchmarks"
  fi

}

download() {

  num_layers=(120 480 1920)

  for (( i = 0; i < 3; ++i ))
  do
    wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron$1-l${num_layers[i]}-categories.tsv -P ../dataset/SDGC/MNIST
  done
  wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron$1.tar.gz -P ../dataset/SDGC/tsv_weights
  wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/mnist/sparse-images-$1.tsv.gz -P ../dataset/SDGC/MNIST
  
}

extract() {
  echo "${bold}Extracting files...${normal}"
  tar -xzf ../dataset/SDGC/tsv_weights/neuron$1.tar.gz -C ../dataset/SDGC/tsv_weights
  gunzip ../dataset/SDGC/MNIST/sparse-images-$1.tsv.gz

  echo "${bold}Removing compressed files...${normal}"
  rm ../dataset/SDGC/tsv_weights/neuron$1.tar.gz
}

get_command $1
