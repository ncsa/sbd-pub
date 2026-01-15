# SBD Build System
# Usage: make [GPU=1] [DEBUG=1] [VERBOSE=1] [ROCM_PATH=/path/to/rocm]

# User options
GPU ?= 1
TRADMODE ?= 1
DEBUG ?= 0
VERBOSE ?= 0

# Paths
SBD_PATH = ../../..
SBD_INCLUDE_DIR = $(SBD_PATH)/include

# Compiler: Override Make's built-in default (CXX = g++)
# Use := to set default but still allow command-line override: make CXX=other-compiler
ifeq ($(origin CXX), default)
    CXX := CC
endif

# Base flags
CXXFLAGS = -std=c++17 -fopenmp -I$(SBD_INCLUDE_DIR)
LDFLAGS ?= -fopenmp -llapack -lblas
# For Frontier, use "make LDFLAGS='-fopenmp -L/opt/cray/pe/libsci/25.09.0/CRAYCLANG/20.0/x86_64/lib -lsci_cray_mp'

# Build mode
ifeq ($(DEBUG),1)
    CXXFLAGS += -O0 -g -DDEBUG
else
    CXXFLAGS += -O3 -march=native
endif

# Traditional mode
ifeq ($(TRADMODE),1)
    CXXFLAGS += -DSBD_TRADMODE
endif

# GPU configuration
ifeq ($(GPU),1)
    # ROCm installation path - must be set via environment or command line
    # Common locations: /opt/rocm, /usr, or from module environment ($ROCM_PATH)
    ROCM_PATH ?= /opt/rocm
    ROCM_INCLUDE := $(ROCM_PATH)/include
    ROCM_LIB := $(ROCM_PATH)/lib

    CXXFLAGS += -DUSE_GPU -D__HIP_PLATFORM_AMD__ \
                -DUSE_DET_CACHE_OMP -DUSE_HIJ_OMP_OFFLOAD \
                -I$(ROCM_INCLUDE) \
                -I$(ROCM_INCLUDE)/hip \
                -fopenmp-targets=amdgcn-amd-amdhsa \
                -Xopenmp-target=amdgcn-amd-amdhsa \
                -march=gfx90a

    LDFLAGS += -L$(ROCM_LIB) -lrocblas -lamdhip64
endif

# Verbosity
ifeq ($(VERBOSE),1)
    $(info === Build Configuration ===)
    $(info Compiler: $(CXX))
    $(info GPU: $(GPU))
    $(info DEBUG: $(DEBUG))
    $(info ROCM_PATH: $(ROCM_PATH))
    $(info ROCM_INCLUDE: $(ROCM_INCLUDE))
    $(info ROCM_LIB: $(ROCM_LIB))
    $(info ==========================)
else
    Q = @
endif

# Targets
TARGET = diag
OBJECTS = main.o

.DEFAULT_GOAL := $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "[LINK] $@"
	$(Q)$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	@echo "[CXX]  $<"
	$(Q)$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	$(Q)rm -f $(TARGET) $(OBJECTS)

.PHONY: clean
