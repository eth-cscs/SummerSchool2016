.PHONY: all clean

CRAY ?= 1
PGI  ?= 0
GCC  ?= 0
VERBOSE ?= 0

CXX     = CC
MPI_CXX = CC
FC      = ftn
MPI_FC  = ftn

ifeq ($(CRAY), 1)
CXX = CC
FC  = ftn
CXXFLAGS = -hstd=c++11 -O3 -hacc -homp -dynamic
FCFLAGS  = -O3 -homp -hacc -dynamic
ifeq ($(VERBOSE), 1)
CXXFLAGS += -hmsgs
FCFLAGS  += -hmsgs
endif
endif

ifeq ($(PGI), 1)
CXX = pgc++
FC  = pgf90
CXXFLAGS = -DPGI -std=c++11 -O3 -acc -Minfo=accel
FCFLAGS  = -O3 -acc -Minfo=accel
LDFLAGS += -L$(CRAY_CUDATOOLKIT_DIR)/lib64 -lcudart
endif

ifeq ($(GCC), 1)
# not tested
CXX = g++
FC  = gfortran
CXXFLAGS = -std=c++11 -O3 -fopenmp -fopenacc -I$(CRAY_CUDATOOLKIT_DIR)/include
FCFLAGS  = -O3 -fopenacc -fopenmp
LDFLAGS += -L$(CRAY_CUDATOOLKIT_DIR)/lib64 -lcudart
endif

CXX_BENCHMARKS = diffusion2d.openacc diffusion2d.openacc.cuda diffusion2d.omp \
	diffusion2d.openacc.mpi diffusion2d.openacc.cuda.mpi
FTN_BENCHMARKS = diffusion2d.openacc.fort diffusion2d.openacc.fort.mpi

BENCHMARKS=$(CXX_BENCHMARKS) $(FTN_BENCHMARKS)

all: $(BENCHMARKS)

util.o: util.f03
	$(FC) -c $(FCFLAGS) $<

diffusion2d.o: diffusion2d.f03
	$(FC) -c $(FCFLAGS) $<

diffusion2d.openacc: diffusion2d_openacc.cpp diffusion2d.hpp
	$(CXX) $(CXXFLAGS) -DOPENACC_DATA $< -o $@ $(LDFLAGS)

diffusion2d.openacc.cuda: diffusion2d_openacc.cpp diffusion2d.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

diffusion2d.openacc.mpi: diffusion2d_openacc_mpi.cpp diffusion2d.hpp
	$(MPI_CXX) $(CXXFLAGS) -DOPENACC_DATA $< -o $@ $(LDFLAGS)

diffusion2d.openacc.cuda.mpi: diffusion2d_openacc_mpi.cpp diffusion2d.hpp
	$(MPI_CXX) $(CXXFLAGS) $< -o $@

diffusion2d.omp: diffusion2d_omp.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

diffusion2d.openacc.fort: diffusion2d_openacc.f90 util.o diffusion2d.o
	$(FC) $(FCFLAGS) $< util.o diffusion2d.o -o $@

diffusion2d.openacc.fort.mpi: diffusion2d_openacc_mpi.f90 util.o diffusion2d.o
	$(MPI_FC) $(FCFLAGS) $< util.o diffusion2d.o -o $@

clean:
	rm -f $(BENCHMARKS) *.cub *.ptx *.o *.mod
