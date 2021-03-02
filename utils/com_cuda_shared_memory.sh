g++ --std=c++11 -fPIC  -shared -W `pkg-config --cflags cuda-11.1` -o libcuda_shared.so  cuda_shared.cc  `pkg-config --libs cuda-11.1` -lrt 
