# Selected Basis Diagonalization Application

.PHONY: all clean diag

# Default target builds the application
all: diag

# Build target (creates src/diag executable)
diag:
	@echo "=== Building SBD Application ==="
	@$(MAKE) -C src GPU=1

# Clean build artifacts
clean:
	@$(MAKE) -C src clean
