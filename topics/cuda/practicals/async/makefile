NVCC_FLAGS=-arch=sm_35 -std=c++11 -O3

LINK=-lcublas -lcuda

CUDA_BENCHMARKS=memcopy1.cuda memcopy2.cuda memcopy3.cuda

all : $(CUDA_BENCHMARKS)

#----------------------------------------
# CUDA benchmarks
#----------------------------------------
memcopy1.cuda : memcopy1.cu util.h
	nvcc $(NVCC_FLAGS) memcopy1.cu -o memcopy1.cuda $(LINK)

memcopy2.cuda : memcopy2.cu util.h
	nvcc $(NVCC_FLAGS) memcopy2.cu -o memcopy2.cuda $(LINK)

memcopy3.cuda : memcopy3.cu util.h
	nvcc $(NVCC_FLAGS) memcopy3.cu -o memcopy3.cuda $(LINK)

clean :
	rm -f *.cuda
