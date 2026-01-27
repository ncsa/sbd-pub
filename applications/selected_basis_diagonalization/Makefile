# Selected Basis Diagonalization Application

# User options
GPU ?= 1

.PHONY: all clean diag

# Default target builds the application
all: diag

# Build target (creates src/diag executable)
diag:
	@echo "=== Building SBD Application ==="
	@$(MAKE) -C src GPU=$(GPU)

# Clean build artifacts
clean:
	@$(MAKE) -C src clean
