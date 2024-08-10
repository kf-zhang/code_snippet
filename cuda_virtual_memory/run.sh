cd $(dirname $0)
nvcc -o main main.cu -arch=sm_80 -lcuda
./main