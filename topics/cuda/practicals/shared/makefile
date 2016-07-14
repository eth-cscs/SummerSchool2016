NVCC_FLAGS=-arch=sm_35 -std=c++11 -O3
GCC_FLAGS=-std=c++11 -O3 -fopenmp

LINK=-lcublas -lcuda

CUDA_BENCHMARKS= blur.cuda blur_twice.cuda dot.cuda string_reverse.cuda
OPENMP_BENCHMARKS= blur_twice.omp

all : $(CUDA_BENCHMARKS) $(OPENMP_BENCHMARKS)

#----------------------------------------
# CUDA benchmarks
#----------------------------------------
blur.cuda : blur.cu util.h
	nvcc $(NVCC_FLAGS) blur.cu -o blur.cuda $(LINK)

blur_twice.cuda : blur_twice.cu util.h
	nvcc $(NVCC_FLAGS) blur_twice.cu -o blur_twice.cuda $(LINK)

dot.cuda : dot.cu util.h
	nvcc $(NVCC_FLAGS) dot.cu -o dot.cuda $(LINK)

string_reverse.cuda : string_reverse.cu util.h
	nvcc $(NVCC_FLAGS) string_reverse.cu -o string_reverse.cuda $(LINK)

#----------------------------------------
# openmp benchmarks
#----------------------------------------
blur_twice.omp : blur_twice.cpp util.h
	g++ $(GCC_FLAGS) blur_twice.cpp -o blur_twice.omp

clean :
	rm -f *.cuda
	rm -f *.omp
