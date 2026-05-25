# SBD Testing Suite

This directory contains a comprehensive test suite for the `sbd-amd` library. Its primary purpose is to verify the **correctness** of library functions and provide a structured workflow for developing and optimizing high-performance CPU and GPU functions.

The suite contains:

- `functionality/`: A set of tests to verify the correctness of library functions.

---

## Prerequisites

Before you begin, ensure you have the following installed and configured on your system:

- A C++17 compliant compiler (e.g., `g++`).
- An MPI implementation (e.g., OpenMPI, MPICH). The `mpicxx` compiler wrapper and `mpirun` command must be in your `PATH`.
- BLAS and LAPACK libraries.

---

## Quick Start

1.  **Build All Tests**: Compile all test executables.

    ```bash
    make all
    ```

2.  **Run Functionality Tests**: Verify that all implementations are working correctly.

    ```bash
    make test
    ```

3.  **Clean Up**: Remove all compiled artifacts.
    ```bash
    make clean
    ```

---

## Running the Suite in Detail

While the `make` targets are convenient, it's important to understand how to run the executables directly with `mpirun` for proper parallel testing.

### Functionality Tests (Correctness)

These tests ensure your library functions produce the correct results. They will exit with an error if any `TEST_ASSERT` fails.

- **Simple Execution (Single MPI Process)**:
  ```bash
  make test
  ```
- **Parallel Execution with `mpirun` (Recommended)**:
  To properly test MPI-aware code, you should use `mpirun` to launch the tests with multiple processes.

  ```bash
  # Example: Run the inner product test with 4 MPI processes
  mpirun -np 4 ./functionality/test_inner_product
  ```

---

## The GPU Development Workflow ✨

This test suite is designed for an incremental, test-driven approach to porting functions to a GPU. Follow these steps to ensure a correct and performant port.

### Step 1: Implement the GPU Function

First, implement the GPU version of your target function within the main `sbd` library. For example, create a `InnerProduct_GPU` function. Ensure this code is conditionally compiled using a preprocessor flag (e.g., `#ifdef HAS_GPU_INNER_PRODUCT`).

### Step 2: Enable the Functionality Test for the GPU

Now, verify that your new GPU implementation produces the exact same results as the trusted CPU version.

1.  **Edit `functionality/Makefile`** and uncomment the appropriate GPU flag:

    ```makefile
    # In functionality/Makefile
    ...
    # GPU flags (uncomment as implementations become available)
    # CXXFLAGS += -DHAS_GPU_DETERMINANTS
    CXXFLAGS += -DHAS_GPU_INNER_PRODUCT  # <--- UNCOMMENT THIS
    ```

2.  **Modify the C++ test file** (e.g., `functionality/test_inner_product.cc`) to call your GPU function and compare its result against the CPU result within the `TEST_ASSERT` macros.

### Step 3: Verify Correctness

Run the functionality tests. This is the most critical step. If this fails, there is a bug in your GPU kernel.

```bash
make clean && make test
```

**Do not proceed until all tests pass!** Fix any bugs in your GPU implementation and re-run the tests.

### Step 4: Optimize and Repeat

Now, you can iterate. Optimize your GPU kernel, and then repeat step 3 to verify that your changes didn't break correctness. This iterative cycle of **Verify -> Optimize** is key to developing correct high-performance code.
