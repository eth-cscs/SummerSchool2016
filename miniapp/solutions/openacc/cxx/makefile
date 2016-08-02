.PHONY: all clean
.DEFAULT: all

# by default select gcc
CXX      = g++
CXXFLAGS = -g -O3 -fopenmp
MKDIR_P  = mkdir -p
RM       = /bin/rm

# if on a cray system, set flags according to the compilation enviroment
ifeq ($(PE_ENV), GNU)
	CXX=CC
	CXXFLAGS=-g -O3 -fopenmp
endif
ifeq ($(PE_ENV), CRAY)
	CXX = CC
	CXXFLAGS = -O3 -hstd=c++11 -hacc -homp -dynamic
	ifeq ($(VERBOSE), 1)
		CXXFLAGS += -hmsgs
	endif
endif
ifeq ($(PE_ENV), INTEL)
	CXX = CC
	CXXFLAGS = -g -O3 -openmp
endif
ifeq ($(PE_ENV), PGI)
	CXX = CC
	CXXFLAGS = -g -std=c++11 -O3 -mp -acc
	ifeq ($(VERBOSE), 1)
		CXXFLAGS += -Minfo=accel
	endif
endif

PROGRAM = main
SOURCES = stats.cpp data.cpp operators.cpp linalg.cpp main.cpp
OBJ     = $(SOURCES:%.cpp=%.o)

all: $(PROGRAM)

DEPDIR = .deps
df = $(DEPDIR)/$(*D)/$(*F)
%.o: %.cpp
	@$(MKDIR_P) $(DEPDIR)/$(*D)
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c -MMD -o $@ $<
	@cp $*.d $(df).P; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $*.d >> $(df).P; \
	$(RM) -f $*.d

-include $(SOURCES:%.cpp=$(DEPDIR)/%.P)

main: $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS)

clean:
	$(RM) -rf $(DEPDIR)
	$(RM) -f $(OBJ) $(PROGRAM) *.i *.lst output.*
