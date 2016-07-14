FLAGS=-arch=sm_35
FLAGS+=-std=c++11
FLAGS+=-O3

LINK=-lcublas -lcuda

BENCHMARKS= diffusion2d.cuda diffusion2d_mpi.cuda

all : $(BENCHMARKS)

#----------------------------------------
# diffusion2D benchmarks
#----------------------------------------
diffusion2d.cuda : diffusion2d.cu util.h
	nvcc $(FLAGS) diffusion2d.cu $(LINK) -o diffusion2d.cuda

# an aditional linking stage required for MPI
diffusion2d_mpi.cuda : diffusion2d_mpi.cu util.h
	nvcc -c -std=c++11 -O3 diffusion2d_mpi.cu -I$(CRAY_MPICH2_DIR)/include
	CC diffusion2d_mpi.o -o diffusion2d_mpi.cuda -lcuda -lcudart

clean :
	rm -f *.cuda *.o
	rm -f output.*
