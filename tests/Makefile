# Top-level Makefile for SBD Testing
# Coordinates functionality tests
# Usage: make [GPU=1] [CXX=CC] [VERBOSE=1]

# User options (use ?= to allow override from command line or environment)
GPU ?= 1
VERBOSE ?= 0

# Compiler: Override Make's built-in default (CXX = g++)
# Use := to set default but still allow command-line override: make CXX=other-compiler
ifeq ($(origin CXX), default)
    CXX := CC
endif

# Export variables to subdirectories
export GPU
export CXX
export VERBOSE

SUBDIRS = functionality

.PHONY: all test clean help $(SUBDIRS)

# Default target shows help
.DEFAULT_GOAL := help

all:
	@for dir in $(SUBDIRS); do \
		echo "Building $$dir..."; \
		$(MAKE) -C $$dir all; \
	done

test:
	@echo "Running functionality tests..."
	@$(MAKE) -C functionality test

clean:
	@for dir in $(SUBDIRS); do \
		echo "Cleaning $$dir..."; \
		$(MAKE) -C $$dir clean; \
	done

help:
	@echo "SBD Test Suite"
	@echo ""
	@echo "Main targets:"
	@echo "  make all       - Build all test executables"
	@echo "  make test      - Run all correctness tests"
	@echo "  make clean     - Clean all build artifacts"
	@echo ""
	@echo "GPU tests in functionality/:"
	@echo "  Build with GPU=1 to enable GPU tests:"
	@echo "    cd functionality && make GPU=1"
	@echo "    cd functionality && make GPU=1 test"
	@echo "  GPU tests include:"
	@echo "    test_gpu_correctness      - Integration tests for GPU pipeline"
	@echo "    test_gpu_fe4s4_minimal    - Real-data regression"
	@echo ""
	@echo "Workflow for GPU development:"
	@echo "  1. Implement GPU version of function"
	@echo "  2. Build and run tests with GPU=1 to verify correctness"
	@echo "  3. Debug with individual test executables in functionality/ directory"
