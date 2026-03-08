# A sample program for parallelized selected basis diagonalization for creation/annilation operator Hamiltonain

Sample code for computing the ground state of a Hamiltonian written by the creation/annihilation operator on a selected bitstrings representing the particle occupations.

## Build

- Open the `Configuration` file and edit the environment variables according to your system: the compiler command (CCCOM), compiler options (CCFLAGS), and linker options (SYSLIB), and the path to the sbd library (SBD_PATH).

- After editing the configuration, run the make command to generate the executable:
    ```
    make
    ```

## Command-Line Arguments for Executable

For usage examples, please refer to the `run.sh` script included in this directory. It shows how to run the executable with various command-line options.

Below is an explanation of each command-line option.

- `--hamfile` (str):  
  Specifies the path to the file describing the Hamiltonian. For the file format, refer to `hamiltonian.txt` in the same folder (for the 1D hardcore boson model).
- `--basisfiles` (strs):  
  Specifiles one or more files containing the bitstrings used as basis states. Multiple files can be provided by separating them with commans (e.g., `--basisfiles basis0.txt,basis1.txt,basis2.txt`). The number of files does not need to match the value of `b_comm_size` described below.
- `--loadname` (str):  
  Specifies the filename of the binary data used as the initial wavefunction.
- `--savename` (str):  
  Specifies the filename to save the resulting ground-state wavefunction as binary data.
- `--t_comm_size` (int):  
  Size of the communicator used for column-wise parallelization during Hamiltonian application.
- `--b_comm_size` (int):  
  Number of nodes used to distribute and store the wavefunction data.
- `--h_comm_size` (int):  
  Size of the communicator used for row-wise parallelization during Hamiltonian application.
- `--method` (int):  
  Selects diagonalization method and how the Hamiltonian is handled
    - `0`: (Davidson) Do not store Hamiltonian data: perform Hamiltonian application on-the-fly.
    - `1`: (Davidson) Store Hamiltonian data to accelerate Hamiltonian application.
    - `2`: (Lanczos) Do not store Hamiltonian data: perform Hamiltonian application on-the-fly.
    - `3`: (Lanczos) Store Hamiltonian data to accelerate Hamiltonian application.
- `--iteration` (int):  
  Number of restart cycles in the Davidson algorithm.
- `--block` (int):  
  Maximum dimension of the reduced space when constructing the projected Hamiltonian in the Davidson algorithm.
- `--numivec` (int):  
  Number of initial vectors (initial subspace dimension) for the Davidson method. This option controls how many basis vectors are used to form the initial Krylov subspace. When restarting, the lowest-energy eigenstates obtained from the previous reduced subspace diagonalization are reused as initial vectors, up to the number specified by this option.
- `--tolerance` (float):  
  Convergence threshold for the Davidson algorithm. The iteration terminates once the norm of the residual vector falls below this value.
- `--system_size` (int):  
  System size, i.e., the length of each bitstring.
- `--bit_length` (int):  
  Number of bits stored in each `size_t` when representing a bitstring as a `std::vector<size_t>`.
- `--fermionsign` (int):  
  Set to `1` to include fermionic sign factors; set to `0` to ignore them.
- `--init` (int):  
  Specifies how to initial state is generated. Currently, only `0` (a random initial vector) is supported.
- `--do_sort_basis` (int):  
  Set to `1` to sort the bitstrings loaded from `basisfiles` across all nodes. This is manly used when the input files contain overlapping bitstrings. Set to `0` if sorting is unnecessary.
- `--do_redist_basis` (int):  
  Set to `1` to redistribute the bitstrings from `basisfiles` uniformly across the nodes specified by `b_comm_size`. Set to `0` if redistribution is unnecessary. Note: If `--sort_basis` is set to `1`, uniform redistribution is performed automatically, and this option is ignored.

---

## `gen_ham.py` — Hamiltonian Generator for Extended Hard-Core Boson Models

This script generates Hamiltonian data for testing **selected-basis diagonalization codes** written in
creation/annihilation-operator format.  
The output strictly follows the operator notation used in the reference example (`hamiltonian.txt`).

The generated Hamiltonian corresponds to an **extended hard-core boson model**, including nearest-neighbor
hopping and repulsive interaction terms.

### Output Format

The script produces plain text with the following structure:

1. **Line 1**:
  ```
  +1
  ```
  This indicates a **bosonic model** (fermionic models would use `-1`, but this script always outputs `+1`).
2. **Subsequent lines**: Individual Hamiltonian terms
- Hopping terms
  ```
  t bdag i b j
  t bdag j b i
  ```
- Interaction terms
  ```
  V bdag i bdag j b j b i
  ```
Here `i` and `j` are site indices, and the format follows:
```
(coefficient) (operator type) (site index) (operator type) (site index) ...
```

### Features

- Supports **1D open-boudary chains** via `--nsites`
- Supports **arbitrary graphs** via repeated `--edge i j` specifications.
- Allows mixing both: `1D chain edges and user-specified edge`.
- Adjustable physical parameters (`t`, `V`)
- Output via standard output or saved to a file

### Command-Line Options
| Option | Argument | Description |
|--------|----------|-------------|
| `--nsites N` | int | Generates a 1D open chain with sites `0..N-1`. Edge list includes `(i,i+1)`. |
| `--edge i j` | int int | Adds an undirected edge `(i, j)`. Can be used multiple times for arbitrary graphs. |
| `--t VAL` | float | Hopping amplitude \( t \). Default: `1.0`. |
| `--V VAL` | float | Interaction strength \( V \). Default: `2.0`. |
| `--output FILE`, `-o FILE` | str | Output file path. Use `-` for stdout (default). |

At least **one of** `--nsites` or `--edge` must be provided.

### Usage Examples

#### 1. Generate an 8-site 1D open chain (matching the sample file)
```
python gen_ham.py --nsites 8 > hamiltonian.txt
```
#### 2. Change hopping and interation parameters
```
python gen_ham.py --nsites 10 --t 0.5 --V 3.0 -o ham_N10_custom.txt
```
#### 3. Generate a custom graph
```
python gen_ham.py \
       --edge 0 1 \
       --edge 0 2 \
       --edge 2 3 \
       -o ham_graph.txt
```
#### 4. Combine a 1D chain with addditional long-range edges
```
python gen_ham.py \
    --nsites 6 \
    --edge 0 5 \
    -o ham_chain_plus_longlink.txt0
```

### Requirements
- Pytyhon 3.7+
- No external libraries required (standard library only).

### Notes
- Self-edges (`i==j`) are ignored.
- Edges are treated as undirected and stored as sorted pairs (`min(i,j), max(i,j)`)
- Duplicate edges are automatically removed.
- Output ordering is deterministic: edges are sorted lexicographyically.

---
## `gen_bits.py`: Bitstring Generator (U(1)-Symmetric)
This script is a simple and flexible tool for generating bitstrings with fixed Hamming weight, corresponding to a **U(1)-symmetric sector** (fixed particle number).
It is desgined for preparing test data for selected-basis diagonalization, CI/FCI sampling, and various quantum many-body simulations.

The generator currently supports:
- Random bitstring generation with fixed number of 1s
- Optional uniqueness filtering
- Output to one or multiple files
- Even (round-robin) distribution of bitstrings across multiple output files
- Random seed control for reproducibility
Additional generation methods can be added via the `--method` option.

### Basic Usage
```
python gen_bits.py --bitlength 16 --numones 4 --num 100
```
This generates:
- Bitstrings of length 16
- With exactly 4 bits set to 1
- A total of 100 samples
- Output sent to stdout

### command-Line Options
This section describes all available options.
#### Required Options
| Option | Argument | Description |
|--------|----------|-------------|
| `--bitlength L` | `int` | Length of each bitstring (system size). |
| `--numones N` | `int` | Number of bits set to 1 (particle number in the U(1) sector). |

#### Primary Options
| Option | Argument | Description |
|--------|----------|-------------|
| `--num M` | `int` | Number of bitstrings to generate (default: 10). |
| `--method m` | `int` | `0`: Random generation with fixed Hamming weight. |
| `--unique` | -- | If specified, only unique bitstrings are output. |
| `--seed R` | `int` | Set random seed to make the output reproducible. |
| `--outfile files1 [files2 ...]` | `str` | Specify one or more output files. |

### Usage Examples
- **Generate 100 bitstrings to stdout**. 
  ```
  python gen_bits.py --bitlength 12 --numones 6 --num 100
  ```
- **Reproducible random generation**. 
  ```
  python gen_bits.py --bitlength 20 --numones 8 --num 50 --seed 123
  ```
- **Unique bitstrings only**.  
  ```
  python gen_bits.py --bitlength 20 --numones 10 --num 500 --unique
  ```
- **Split output across 2 files**. 
  ```
  python gen_bits.py --bitlength 18 --numones 9 --num 200 \
          --outfile part1,txt part2.txt
  ```
### Implementation Notes
- Random samples are generated by shuffling a base bit pattern with fixed Hamming weight.
- The `--unique` option operates by collecting results in a Python `set`.
- Output distribution uses a simple round-robin assignment, suitable for HPC batch jobs.
- The code structure is modular and ready for adding new generation methods.
