# SBD Build System
# Prerequisites: source ../env_setup.sh before building
# Usage: make [GPU=1] [DEBUG=1] [VERBOSE=1]

# User options
GPU ?= 1
TRADMODE ?= 1
DEBUG ?= 0
VERBOSE ?= 1

# Paths
SBD_PATH = ../../..
SBD_INCLUDE_DIR = $(SBD_PATH)/include

# make sure your mpicxx picks up the desired clang compiler
# export OMPI_CXX=/path/to/bin/clang++
  export MPICH_CXX=/path/to/bin/clang++
  CXX = mpicxx

# On Cray systems : use Cray C++ compiler wrapper CC
# CXX = CC 


# Base flags : BLAS and Lapack are required; one option is OpenBLAS
CXXFLAGS = -std=c++17 -fopenmp -I$(SBD_INCLUDE_DIR) 
LDFLAGS = -fopenmp /path/to/OpenBLAS-0.3.30/lib/libopenblas.a 

# Build mode
ifeq ($(DEBUG),1)
    CXXFLAGS += -O0 -g -DDEBUG
else
    CXXFLAGS += -O3 -march=native -funroll-loops
endif

# Traditional mode
ifeq ($(TRADMODE),1)
    CXXFLAGS += -DSBD_TRADMODE 
endif

#-----------------------------------------------------------------------------
# for AMD GPUs: -fopenmp-targets=amdgcn-amd-amdhsa --offload-arch=gfx90a
# use rocminfo | grep gfx to get the offload architecture
#-----------------------------------------------------------------------------
# for NVIDIA GPUS: -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_100 
# NVIDIA A100 sm_80,  H100 sm_90,  GB200 sm_100, etc.
#-----------------------------------------------------------------------------
 
ifeq ($(GPU),1)

    CXXFLAGS += -DUSE_GPU -DUSE_DET_CACHE_OMP -DUSE_HIJ_OMP_OFFLOAD \
		-fopenmp-targets=amdgcn-amd-amdhsa --offload-arch=gfx90a \
                -fopenmp-offload-mandatory -foffload-lto
    LDFLAGS += -foffload-lto
else
    CXXFLAGS += -DUSE_DET_CACHE_OMP

endif

# Verbosity
ifeq ($(VERBOSE),1)
    $(info === Build Configuration ===)
    $(info Compiler: $(CXX))
    $(info GPU: $(GPU))
    $(info DEBUG: $(DEBUG))
    $(info ==========================)
else
    Q = @
endif

# Targets
TARGET = diag
OBJECTS = main.o

.DEFAULT_GOAL := $(TARGET)

$(TARGET): clean $(OBJECTS)
	@echo "[LINK] $@"
	$(Q)$(CXX) $(CXXFLAGS) main.o -o diag $(LDFLAGS)

%.o: %.cc
	@echo "[CXX]  $<"
	$(Q)$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	$(Q)rm -f $(OBJECTS) ./diag

.PHONY: clean
