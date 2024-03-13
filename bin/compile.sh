nvcc -std=c++17 --extended-lambda -lstdc++fs ../main/beyond.cu -I ../3rd-party/ -I ../src/beyond -O3 -o beyond
