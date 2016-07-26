.PHONY: all clean

CRAY ?= 1
PGI  ?= 0
GCC  ?= 0
VERBOSE ?= 0

ifeq ($(CRAY), 1)
CXX = CC
FC  = ftn
CXXFLAGS = -hstd=c++11 -O3 -hacc -dynamic
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
CXX = g++
FC  = gfortran
CXXFLAGS = -std=c++11 -O3 -fopenmp -fopenacc -I$(CRAY_CUDATOOLKIT_DIR)/include
FCFLAGS  = -O3 -fopenacc -fopenmp
LDFLAGS += -L$(CRAY_CUDATOOLKIT_DIR)/lib64 -lcudart
endif

OPENACC_BENCHMARKS= blur.openacc blur.openacc.fort dot.openacc dot.openacc.fort
BENCHMARKS=$(OPENACC_BENCHMARKS)

all: $(BENCHMARKS)

util.o: util.f03
	$(FC) -c $(FCFLAGS) util.f03

blur.openacc: blur_openacc.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

blur.openacc.fort: blur_openacc.f90 util.o
	$(FC) $(FCFLAGS) $< util.o -o $@

dot.openacc: dot_openacc.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

dot.openacc.fort: dot_openacc.f90 util.o
	$(FC) $(FCFLAGS) $< util.o -o $@

clean:
	rm -f $(BENCHMARKS) *.cub *.ptx *.o *.mod
