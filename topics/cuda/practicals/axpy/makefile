NVCC_FLAGS=-arch=sm_35 -std=c++11 -O3
GCC_FLAGS=-std=c++11 -O3 -fopenmp

LINK=-lcublas -lcuda

CUDA_BENCHMARKS= axpy_cublas.cuda axpy_kernel.cuda newton.cuda
OPENMP_BENCHMARKS= axpy.omp

BENCHMARKS=$(CUDA_BENCHMARKS) $(OPENMP_BENCHMARKS)

all : $(BENCHMARKS)

axpy_cublas.cuda : axpy_cublas.cu util.h
	nvcc $(NVCC_FLAGS) axpy_cublas.cu -o axpy_cublas.cuda $(LINK)

axpy_kernel.cuda : axpy_kernel.cu util.h
	nvcc $(NVCC_FLAGS) axpy_kernel.cu -o axpy_kernel.cuda $(LINK)

newton.cuda : newton.cu util.h
	nvcc $(NVCC_FLAGS) newton.cu -o newton.cuda $(LINK)

axpy.omp : axpy_omp.cpp
	g++ $(GCC_FLAGS) axpy_omp.cpp -o axpy.omp

clean :
	rm -f *.cuda
	rm -f *.omp
