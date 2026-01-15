# Incremental GPU Porting Plan for SBD with OpenMP Offload

## Phase 0: Preliminary Setup & CPU BLAS Integration

### Step 0.1: Create branch structure
- Create new branch `hipification` from current `tests` branch
  - This marks the divergence point for GPU approach vs other approaches
- Create new branch `openmp_offload` from `hipification`
  - This is where we will do the OpenMP GPU offload work
- **Verification:** Both branches exist, currently on `openmp_offload` branch
- **Rationale:** `hipification` preserves the fork point; `openmp_offload` is our working branch

### Step 0.2: Replace InnerProduct with BLAS ddot (CPU)
**Target:** `include/sbd/framework/dm_vector.h:21-48` (InnerProduct function)

**Current:** Manual Kahan summation in OpenMP parallel loop

**Replace with:** Call to BLAS `ddot` for the local dot product, then MPI_Allreduce

**Why:** BLAS is highly optimized; establishes pattern for GPU transition

**Implementation:**
```cpp
template <typename ElemT>
void InnerProduct(const std::vector<ElemT> & X,
                  const std::vector<ElemT> & Y,
                  ElemT & res,
                  MPI_Comm comm) {
    // Local ddot: sum_i conj(X[i]) * Y[i]
    ElemT local_sum = cblas_ddot(X.size(), X.data(), 1, Y.data(), 1);
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    MPI_Allreduce(&local_sum, &res, 1, DataT, MPI_SUM, comm);
}
```

**Modified files:**
- `include/sbd/framework/dm_vector.h`
- `samples/selected_basis_diagonalization/Configuration` (add BLAS flags)
- `samples/selected_basis_diagonalization/Makefile` (link BLAS)

**Testing:** Run full test suite - all 12 tests must pass

**Profiling:**
- Tool: `gprof` (CPU profiling)
- Generate new profile: `gprof ./test_optimization_targets > profile_0.2_blas_cpu.txt`
- Record total runtime from test output

**Commit:** "Replace InnerProduct with BLAS ddot for CPU optimization"

**Log:** Record in PERFORMANCE_LOG.md:
```
YYYY-MM-DD HH:MM | <commit_hash> | Step 0.2: BLAS InnerProduct (CPU) | total_time=X.XXs | <key metrics>
```

### Step 0.3: Offload InnerProduct to GPU with rocBLAS
**Target:** Same InnerProduct function

**Add GPU path:** Use `rocblas_ddot` when compiled with `-DSBD_USE_GPU`

**Data movement:** Copy X, Y to GPU → compute → copy result back

**Note:** This step will likely be SLOW due to data transfer overhead - we expect this

**Implementation:**
```cpp
#ifdef SBD_USE_GPU
    // Allocate GPU memory
    double *d_X, *d_Y;
    hipMalloc(&d_X, X.size() * sizeof(double));
    hipMalloc(&d_Y, Y.size() * sizeof(double));

    // Copy to GPU
    hipMemcpy(d_X, X.data(), X.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_Y, Y.data(), Y.size() * sizeof(double), hipMemcpyHostToDevice);

    // Compute on GPU
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    rocblas_ddot(handle, X.size(), d_X, 1, d_Y, 1, &local_sum);

    // Cleanup
    hipFree(d_X);
    hipFree(d_Y);
    rocblas_destroy_handle(handle);
#else
    // CPU BLAS path
#endif
```

**Modified files:**
- `include/sbd/framework/dm_vector.h`
- `samples/selected_basis_diagonalization/Configuration` (add rocBLAS flags)
- `samples/selected_basis_diagonalization/Makefile` (link rocBLAS, HIP)

**Testing:** Run full test suite with GPU - all 12 tests must pass

**Profiling:**
- Tool: `rocprof` (GPU profiling)
- Generate profile: `rocprof --stats ./test_optimization_targets > profile_0.3_rocblas_gpu.txt`
- Record total runtime from test output
- **Expected:** Slower than 0.2 due to data transfer overhead

**Commit:** "Add GPU offload for InnerProduct with rocBLAS"

**Log:** Record performance (likely regression) in PERFORMANCE_LOG.md:
```
YYYY-MM-DD HH:MM | <commit_hash> | Step 0.3: rocBLAS InnerProduct (GPU) | total_time=X.XXs | <key metrics>
```

## Phase 0.4: OpenMP GPU Offload Analysis & Planning

### Critical OpenMP parallel regions to offload

#### A. mult.h - Main computational kernel (3 regions)

**1. Lines 223-332 (TRADMODE) or Lines 487-597 (non-TRADMODE)**

- **Hottest region:** Nested loops over (ia, ib, ja, jb)
- **Inner work:** DetFromAlphaBeta() + Hij() calls (51% + 32.8% of runtime)
- **Challenge:** Complex control flow with 3 task types
- **Data dependencies:**
  - Read: `adets`, `bdets`, `helper`, `I0`, `I1`, `I2`, `T`
  - Write: `Wb` (requires atomic updates)

**Critical optimization: mpi_size_h power-of-2 requirement**

Current code:
```cpp
if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;
```

If power of 2:
```cpp
if( (braIdx & (mpi_size_h - 1)) != mpi_rank_h ) continue;
```

**Restrictiveness analysis:**
- Common MPI configurations: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 ranks
- These are ALL powers of 2!
- Non-power-of-2 (e.g., 12, 24, 48, 96) are rare in HPC practice
- Typical supercomputer allocations: powers of 2 for load balancing
- **Recommendation:** Require power-of-2 in documentation, add runtime check with clear error message
- **Practical impact:** Minimal - covers 99% of real-world use cases

**Other optimizations:**
- Division `braIdx = (ia-start)*size + (ib-start)` → no change needed (not on hot path)
- Integer division in DetFromAlphaBeta → replace with bit shifts (see below)

**2. Lines 184-198/229-243 - Diagonal term**

- **Simple:** `Wb[i] += hii[i] * T[i]`
- **Easy GPU target** - embarrassingly parallel
- **Should combine** with main mult region to avoid data transfer overhead

#### B. dm_vector.h - Vector operations

1. **InnerProduct** (lines 21-48) - Already covered in 0.2/0.3
2. **Normalize** (lines 50-84) - Similar pattern, use rocBLAS `dnrm2` + vector scale
3. **Randomize** (lines 86-136) - **Skip for now**
   - Only called once at initialization before Davidson iterations
   - Not performance critical (< 0.01% of total runtime)
   - Can revisit if profiling shows otherwise
4. **Swap** (lines 138-151) - **Skip for now**
   - Infrequent operation (not in hot path)
   - Will handle naturally when we GPU-accelerate davidson.h vector ops

#### C. helper.h - Excitation list generation (2 regions)

- Lines 155-167, 169-181 (GenerateExcitation)
- **Keep on CPU:** One-time setup, complex logic with std::find
- **Not performance critical** (< 0.1% of runtime, called once)

#### D. davidson.h - Davidson eigensolver

**Why NOT OpenMP offload for davidson.h:**

1. **Already uses BLAS/LAPACK internally**
   - Line 264-285: Multiple `InnerProduct` calls (will use rocBLAS from 0.3)
   - Line 275-280: AXPY operations (`C[i] = a*C[j] + b*C[i]`)
   - Uses LAPACK `dsyev`/`dgeev` for small dense eigensolve
   - **Strategy:** GPU-accelerate via our BLAS wrappers, not direct offload
   - BLAS calls automatically use GPU when we link rocBLAS

2. **Control flow complexity**
   - Convergence checks, Gram-Schmidt orthogonalization
   - Subspace projection with small matrices (10-100 dims)
   - Better suited to BLAS library calls than GPU kernels
   - Would require extensive refactoring for OpenMP target directives

3. **Small subspace operations**
   - Davidson subspace typically 10-50 vectors
   - LAPACK operations on small matrices (50×50) stay on CPU
   - Only large vector operations (10^6-10^9 elements) go to GPU
   - Small matrix operations have overhead > computation time on GPU

4. **Data movement pattern**
   - Vectors already on GPU from InnerProduct/Normalize offload
   - BLAS calls operate on GPU-resident data (with rocBLAS)
   - No explicit OpenMP target directives needed
   - Library handles GPU memory management transparently

### Refactoring Needs for GPU Efficiency

**1. Replace integer division with bit shifts in DetFromAlphaBeta**

**Location:** determinants.h lines 81-86, 105-110

**Current (bit_length = 64):**
```cpp
size_t block = i / bit_length;       // Division
size_t bit_pos = i % bit_length;     // Modulo
```

**Optimized:**
```cpp
size_t block = i >> 6;               // Bit shift (/ 64)
size_t bit_pos = i & 63;             // Bit mask (% 64)
```

**Caveat:** Only valid for bit_length = power of 2

**Current practice:** bit_length = 64 (size_t on 64-bit systems)

**Safe assumption:** Will remain 64 in practice

**Performance benefit:**
- Integer division ~20-40 cycles on modern CPUs
- Bit shift: 1 cycle
- Called millions of times in hot loop → 5-10% speedup expected

**2. Combine OpenMP regions in mult()**

**Current:** Separate `#pragma omp parallel` for diagonal (line 184) and off-diagonal (line 223)

**Problem:** Creates 2 GPU offload regions with data transfers in between

**Solution:** Merge into single region:
```cpp
#pragma omp target teams distribute parallel for \
    map(to: adets, bdets, I1, I2, T, hii) \
    map(tofrom: Wb)
{
    // Diagonal term (if mpi_rank_t == 0)
    if(mpi_rank_t == 0) {
        for(size_t i=0; i < T.size(); i++) {
            Wb[i] += hii[i] * T[i];
        }
    }

    // Off-diagonal tasks (all ranks)
    for(size_t task=0; task < helper.size(); task++) {
        // ... existing task loop
    }
}
```

**Benefit:** Transfer `Wb`, `T`, `adets`, `bdets` to GPU ONCE, not twice

**Data transfer volumes:**
- `adets`: ~few MB (30K dets × 3 words × 8 bytes = 720 KB)
- `bdets`: ~few MB (similar)
- `T`: ~GB scale (10^6-10^9 elements × 8 bytes)
- `Wb`: ~GB scale (same size as T)
- Avoiding duplicate transfer of GB-scale data is critical

**3. Data transfer strategy**

**Transfer once per Davidson iteration:**
- `adets`, `bdets` (small, ~few MB)
- `I1`, `I2` integrals (moderate, ~100s MB)
- These don't change during Davidson convergence

**Transfer per mult() call:**
- `Wk` → `T` (GB scale) - changes every mult()
- `Wb` back to CPU (GB scale) - accumulates result

**Keep on GPU across mult() calls within same Davidson iteration:**
- Integrals, determinants (if they fit)
- Reuse across multiple H*v operations

### Experiment: Fix bit_length = 512

**Current situation:**
- Scientists report: "bit_length typically 100-200 in practice"
- Current code: bit_length = sizeof(size_t) * 8 = 64
- **Likely confusion:** They may be referring to number of orbitals (norb), not bit_length
- **Need clarification before proceeding with this experiment**

**If we fix bit_length = 512:**

**Memory impact analysis:**

Each determinant size: `(2*norb + bit_length - 1) / bit_length` words

| norb | Current (bit_length=64) | Fixed (bit_length=512) | Savings |
|------|-------------------------|------------------------|---------|
| 36   | 3 words (24 bytes)      | 2 words (16 bytes)     | -33%    |
| 100  | 5 words (40 bytes)      | 2 words (16 bytes)     | -60%    |
| 256  | 9 words (72 bytes)      | 2 words (16 bytes)     | -78%    |
| 512  | 17 words (136 bytes)    | 3 words (24 bytes)     | -82%    |

**Conclusion:** For realistic norb values (32-512), fixing bit_length=512 SAVES memory!

**Memory waste only occurs when norb < bit_length/2 = 256:**
- norb=64: Current 2 words, Fixed 2 words → no waste
- norb=32: Current 2 words, Fixed 2 words → no waste
- **No memory waste for realistic norb values!**

**Performance benefits:**
- Fewer words per determinant → better cache utilization
- Fewer loop iterations in bit manipulation functions
- Simpler division: `i / 512 = i >> 9`, `i % 512 = i & 511`
- Expected speedup: 5-10% from reduced memory traffic

**When to test this:**
- **Save for last** (as specified)
- After all other optimizations are in place
- Measure before/after to isolate benefit
- Create separate branch for this experiment
- Compare against optimized baseline

## Success Criteria for Phase 0

**After each step:**
1. All 12 tests pass
2. New profile generated:
   - CPU steps (0.2): Use `gprof`
   - GPU steps (0.3+): Use `rocprof --stats`
3. Total runtime recorded from test execution
4. Performance logged in PERFORMANCE_LOG.md with:
   - Timestamp (YYYY-MM-DD HH:MM:SS)
   - Commit hash
   - Step description
   - Total runtime
   - Key metrics from profile
5. Changes committed to git with descriptive message

**Specific criteria:**
- **0.1:** Branches `hipification` and `openmp_offload` created, currently on `openmp_offload`
- **0.2:** Tests pass, gprof profile shows BLAS InnerProduct, commit made, performance logged
- **0.3:** Tests pass, rocprof profile shows rocBLAS usage (may be slower), commit made, performance logged
- **0.4:** Detailed analysis documented (this plan), CLAUDE.md updated with new rules

## Files to Create/Modify

### New files to create:
- `GPU_PORTING_PLAN.md` - This document (detailed implementation plan)
- `PERFORMANCE_LOG.md` - Timestamped performance results
  - Format: `YYYY-MM-DD HH:MM:SS | commit_hash | description | total_time | metrics`
  - Newest entries at bottom
- `FAILED_OPTIMIZATIONS.md` - Log of unsuccessful attempts
  - Format: `YYYY-MM-DD HH:MM:SS | what_tried | why_failed | lessons_learned`

### Modified files (Phase 0.2-0.3):
- `include/sbd/framework/dm_vector.h` (InnerProduct with BLAS/rocBLAS)
- `samples/selected_basis_diagonalization/Configuration` (BLAS/rocBLAS compiler flags)
- `samples/selected_basis_diagonalization/Makefile` (link BLAS/rocBLAS libraries)
- `CLAUDE.md` (add 4 new critical rules)

### Modified files (future Phase 0.4+ implementation):
- `include/sbd/chemistry/tpb/mult.h` (merge OpenMP regions, add GPU offload)
- `include/sbd/chemistry/basic/determinants.h` (bit shift optimization)

## CLAUDE.md Updates

Add to "Critical Rules for All Work in This Project" section:

```markdown
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
```

## Notes and Open Questions

1. **Clarification needed:** Scientists report "bit_length 100-200" but code uses 64. Are they referring to `norb` instead?

2. **Power-of-2 requirement for mpi_size_h:** Document this requirement clearly in user-facing documentation. Add runtime assertion:
   ```cpp
   assert((mpi_size_h & (mpi_size_h - 1)) == 0 && "mpi_size_h must be a power of 2");
   ```

3. **Profiling on GPU:** Need access to AMD GPU hardware to generate rocprof profiles. If not available, use CPU profiling until GPU access is secured.

4. **BLAS library selection:** Confirm which BLAS library to use (OpenBLAS, MKL, BLIS). rocBLAS is clear for GPU.

5. **Test coverage:** Current 12 tests cover optimization targets. May need additional integration tests for full Davidson solver with GPU offload.
