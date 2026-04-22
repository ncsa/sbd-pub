# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules for All Work in This Project

**These rules apply to everything you do in this repository:**

1. **ALWAYS use relative error when comparing floating-point numbers.** Never use absolute error comparisons for numerical validation or convergence tests in quantum chemistry calculations.

2. **NEVER attempt to elevate privileges.** Do not use `sudo`, `su`, or any other privilege escalation methods. All operations must be performed with standard user permissions.

3. **Do NOT add redundant "Generated with Claude Code" messages in commit messages.** Simply include `Co-Authored-By: Claude <noreply@anthropic.com>` - mentioning Claude as co-author is sufficient.

4. **After every code modification that changes functionality, ensure tests pass, then commit.** Do not accumulate multiple changes before committing. Each logical change should be its own commit.

5. **After each optimization step, generate a new performance profile.**
   - For CPU optimizations: Use `gprof`
   - For GPU optimizations: Use `rocprof --stats`
   - Always record total runtime from test execution
   - Compare against previous baseline

6. **Keep a timestamped log of performance results** in PERFORMANCE_LOG.md with commit hash, date/time, total runtime, and key metrics (newest results at bottom). Format:
   ```
   YYYY-MM-DD HH:MM:SS | commit_hash | step_description | total_time=X.XXs | metric1 | metric2
   ```

7. **Keep a log of FAILED optimization attempts** in FAILED_OPTIMIZATIONS.md with timestamp, what was tried, why it failed, and lessons learned. Successful optimizations will be documented in commit messages.

## Project Overview

SBD (Selected Basis Diagonalization) is a header-only C++ library for diagonalizing quantum chemistry Hamiltonians using a selected basis approach with MPI+OpenMP parallelization. The library is specifically designed to handle wavefunction vectors too large to fit in single-node memory by distributing them across multiple MPI ranks.

**Key characteristics:**

- Header-only library (no installation required, just include `sbd/sbd.h`)
- Targets quantum chemistry Configuration Interaction (CI) calculations
- Uses Tensor Product Basis (TPB) decomposition: factorizes Hilbert space as α-spin ⊗ β-spin sectors
- Implements Davidson iterative eigensolver for ground state computation
- Three-level MPI parallelization: task communicator, α-det communicator, β-det communicator

## Building and Running

### Sample Code Compilation

The main sample is in `samples/selected_basis_diagonalization/`:

1. Edit `Configuration` file to set:
   - `CCCOM`: MPI C++ compiler (e.g., `mpicxx` or path to your MPI wrapper)
   - `CCFLAGS`: Compiler flags (must include `-std=c++17 -fopenmp -O3`)
   - `SYSLIB`: Linker flags for BLAS/LAPACK (e.g., `-L/usr/lib/x86_64-linux-gnu -llapack -lblas -fopenmp`)
   - `SBD_PATH`: Path to SBD library root (default `../../`)

2. Build:

   ```bash
   cd samples/selected_basis_diagonalization
   make
   ```

3. Run (see `run.sh` for examples):
   ```bash
   ./diag --fcidump fcidump.txt --adetfile alphadets.txt [options]
   ```

**Important:** The sample expects input files named `fcidump.txt` and `alphadets.txt`. You may need to create symlinks to the actual data files (e.g., `ln -sf fcidump_Fe4S4.txt fcidump.txt`).

### Key Command-Line Options

- `--method 0`: Matrix-free mode (compute H\*v on-the-fly, lower memory)
- `--method 1`: Store Hamiltonian matrix (faster but higher memory)
- `--task_comm_size N`: MPI ranks for task parallelization
- `--adet_comm_size M`: Partitions for α-determinants
- `--bdet_comm_size K`: Partitions for β-determinants
- `--iteration N`: Number of Davidson restart iterations
- `--block B`: Davidson subspace size
- `--tolerance T`: Convergence threshold (e.g., `1e-8`)
- `--rdm 1`: Compute 1-particle and 2-particle reduced density matrices

Total MPI ranks required: `task_comm_size × adet_comm_size × bdet_comm_size`

## Code Architecture

### Directory Structure

```
include/sbd/
├── framework/          # Core utilities (MPI, bit manipulation, FCIDUMP parser, dm_vector)
├── basic/              # General-purpose diagonalization routines
└── chemistry/          # Quantum chemistry specific code
    ├── basic/          # Determinant operations, integrals, Hamiltonian
    └── tpb/            # Tensor Product Basis (TPB) specialization
```

### Core Components

**1. Bitstring Representation (`framework/bit_manipulation.h`)**

- Slater determinants encoded as `std::vector<size_t>` with configurable `bit_length`
- Little-endian convention within each `size_t` element
- Functions: `makestring()`, `from_string()` for I/O conversion

**2. Distributed Vectors (`framework/dm_vector.h`)**

- Template class for MPI-distributed dense vectors
- Each rank owns a contiguous chunk
- Supports BLAS operations and MPI reductions

**3. Quantum Chemistry Hamiltonian (`chemistry/basic/`)**

- `determinants.h`: Critical functions:
  - `DetFromAlphaBeta()`: Interleaves α and β bitstrings into full determinant (32.8% of runtime in profiling)
  - `Hij()`: Computes Hamiltonian matrix element between determinants using Slater-Condon rules (51% of runtime)
  - `parity()`: Computes fermion anticommutation sign (8.5% of runtime)
- `integrals.h`: FCIDUMP integral storage and lookup
  - `twoInt::Value()`: 2-electron integral lookup (4.9% of runtime)
  - Compact storage using 8-fold symmetry

**4. TPB Implementation (`chemistry/tpb/`)**

- `mult.h`: Hamiltonian-vector multiplication (H\*v) for TPB factorization
  - Iterates over (α_bra, β_bra) pairs
  - For each bra, generates connected kets via single/double excitations
  - Calls `DetFromAlphaBeta()` and `Hij()` repeatedly (major bottleneck)
- `davidson.h`: Davidson eigensolver
  - Builds Krylov subspace from trial vectors {C[0], C[1], ...} and {H*C[0], H*C[1], ...}
  - Projects H into small subspace, diagonalizes with LAPACK
  - Computes preconditioned residual: `C[new] = R/(E[0] - diagonal)`
  - Gram-Schmidt orthogonalizes new vectors against existing subspace
- `helper.h`: MPI task distribution metadata

### Parallelization Strategy

Three-level MPI decomposition:

1. **Task communicator** (`t_comm`): Distributes column blocks of Hamiltonian during mult()
2. **α-det communicator** (`a_comm`): Partitions α-determinant set
3. **β-det communicator** (`b_comm`): Partitions β-determinant set

Each MPI rank computes matrix elements for a specific (task, α-range, β-range) tuple. Results are reduced via `MPI_Allreduce` on appropriate sub-communicators.

## Performance Bottlenecks (from profiling)

Based on `ibms_small_profile.130905.0.txt`:

1. **Hij (51%)**: Slater-Condon rule evaluation
2. **DetFromAlphaBeta (32.8%)**: Bit interleaving
3. **parity (8.5%)**: Sign computation
4. **twoInt::Value (4.9%)**: Integral lookup

All are called within `mult()` inner loops. **GPU acceleration target:** Port these functions to AMD Radeon Instinct GPUs using HIP/ROCm.

## Data Formats

**FCIDUMP:** Standard quantum chemistry integral file

- Two-electron integrals: `v[i,j,k,l]`
- One-electron integrals: `h[i,j]`
- See Knowles & Handy, Comput. Phys. Commun. 54(1), 75 (1989)

**Alpha-determinant file:** Plain text, one bitstring per line

- Bitstring ordered right-to-left (rightmost bit = orbital 1)
- Example line: `1101` means orbitals 1,3,4 occupied

## Development Notes

- **No Fortran compiler required** (Fortran MPI bindings disabled in sample builds)
- **Davidson algorithm**: NOT a pure Krylov method (uses preconditioning, not powers of H)
- **Initial state**: Hartree-Fock determinant (first bitstring in sorted α-det list)
- **License:** Apache License 2.0
- **Status:** Research code, paper unpublished (do not cite yet)

## GPU Porting Strategy (AMD Radeon Instinct)

**Precomputation approach for DetFromAlphaBeta:**

- Each MPI rank works on `(nadets/adet_comm_size) × (nbdets/bdet_comm_size)` determinants
- For 30K α-dets × 30K β-dets with 64 ranks (8×8): ~14M dets/rank → ~1.1 GB cache
- Precompute all full determinants once, eliminating 32.8% bottleneck

**Fused HIP kernel for Hij:**

- Batch (bra, ket) pairs to single kernel launch
- Coalesced memory access for precomputed determinants
- Use `__popcll()`, `__ffsll()` for bit operations (AMD GPU intrinsics)

**Target functions to port:**

- `DetFromAlphaBeta()` → precompute or GPU kernel with lookup tables
- `Hij()` → `Hij_GPU_fused()` kernel
- `mult()` outer loops → orchestrate GPU kernels
