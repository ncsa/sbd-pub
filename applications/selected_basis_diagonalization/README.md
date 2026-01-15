# Selected Basis Diagonalization (SBD)

High-performance quantum chemistry ground state solver using sampled basis diagonalization with MPI parallelization and GPU acceleration support.

## Overview

This application computes the ground state of quantum chemistry Hamiltonians based on sampled bitstrings representing alpha-spin orbitals. For detailed theory background, input file formats (FCIDUMP and AlphaDets.txt), and algorithm details, please see `../../samples/selected_basis_diagonalization/README.md`.

### Key Features

- Parallel Davidson algorithm implementation
- MPI-based distributed computing
- GPU acceleration support (AMD ROCm)
- Support for large-scale quantum chemistry problems

## Installation & Build

### Build Configuration

Edit the `src/Makefile` to set the following variables according to your system:

- **CC**: MPI C++ compiler wrapper (e.g., `mpicxx`, `CC` for Cray systems)
- **ROCM_PATH**: Path to ROCm installation (required for GPU support)
- **BLAS/LAPACK**: Linker flags for BLAS and LAPACK libraries

Optional build variables:
- **VERBOSE=1**: Show full compiler commands during build
- **GPU=1**: Enable GPU acceleration (default)
- **DEBUG=1**: Build with debugging symbols

### Building

```bash
# Clean build
make clean
make

# Verbose build (shows compiler commands)
make VERBOSE=1
```

The executable will be created at `src/diag`.

## Usage

### Running with mpirun

```bash
# Basic run with auto-detected MPI decomposition
mpirun -n 8 ./src/diag --fcidump <fcidump_file> --adetfile <alpha_dets_file>

# Custom MPI decomposition (2 alpha × 4 beta × 1 task = 8 ranks)
mpirun -n 8 ./src/diag --fcidump <fcidump_file> --adetfile <alpha_dets_file> \
    --adet_comm_size 2 --bdet_comm_size 4 --task_comm_size 1
```

### Running with srun (SLURM)

```bash
# Single node, 8 GPUs
srun -N 1 -n 8 --gpus-per-task=1 ./src/diag \
    --fcidump <fcidump_file> --adetfile <alpha_dets_file> \
    --adet_comm_size 2 --bdet_comm_size 4

# Multi-node
srun -N 2 -n 16 --gpus-per-task=1 ./src/diag \
    --fcidump <fcidump_file> --adetfile <alpha_dets_file> \
    --adet_comm_size 4 --bdet_comm_size 4
```

### Command-Line Options

| Option              | Description                             | Default  |
| ------------------- | --------------------------------------- | -------- |
| `--fcidump`         | FCIDUMP file with Hamiltonian integrals | Required |
| `--adetfile`        | Alpha-spin orbital bitstrings file      | Required |
| `--iteration`       | Number of Davidson restarts             | 2        |
| `--block`           | Maximum Davidson subspace size          | 10       |
| `--tolerance`       | Convergence threshold                   | 1.0e-4   |
| `--method`          | 0=on-the-fly, 1=stored Hamiltonian      | 0        |
| `--adet_comm_size`  | Alpha determinant MPI partitions        | Auto     |
| `--bdet_comm_size`  | Beta determinant MPI partitions         | Auto     |
| `--task_comm_size`  | Task communicator size                  | 1        |
| `--carryover_ratio` | Weight retention ratio                  | 0.5      |
| `--shuffle`         | Shuffle input determinants (0/1)        | 0        |
| `--rdm`             | Compute reduced density matrices (0/1)  | 0        |

