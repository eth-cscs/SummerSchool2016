flags=-std=c++11
ifeq ($(PE_ENV),GNU)
	flags+=-fopenmp
endif
ifeq ($(PE_ENV),INTEL)
	flags+=-openmp
endif

all : test.omp test.mpi

test.omp: test_omp.cpp
	CC $(flags) test_omp.cpp -o test.omp

test.mpi: test_mpi.cpp
	CC $(flags) test_mpi.cpp -o test.mpi

clean:
	rm -rf test.omp
	rm -rf test.mpi


