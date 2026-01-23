# run_with_timing.sh - Working Test Examples

These are the 4 tested and working examples for running benchmarks with the timing wrapper.
All examples use Fe4S4 dataset for fast execution (~20 seconds).

**KEY PRINCIPLE:** One script to rule them all - `run_with_timing.sh` sources environment automatically, no bash -c wrappers needed!

## Prerequisites

Before running any example, ensure you're in the correct directory:
```bash
cd /home/jujaykka/projects/sbd-amd/applications/selected_basis_diagonalization
```

---

## Case 1: Make Target (Local or Compute Node)

**Description:** Simplest method - just run make target

**Command:**
```bash
make my_new_benchmark
```

**Works on:** Login nodes (with Cray assertion error) or compute nodes (clean execution)

**Notes:**
- Makefile automatically does `cd benchmarks` and uses relative paths
- Environment sourced automatically inside run_with_timing.sh
- On login nodes: Will fail with Cray GPU assertion but wrapper captures error correctly
- On compute nodes: Runs successfully with GPU

**To test on compute node via srun:**
```bash
srun --nodes=1 --ntasks=1 --gpus-per-task=1 --partition=MI250X_A1_COS_OK make my_new_benchmark
```

---

## Case 2: Interactive salloc Session

**Description:** Allocate resources first, then run inside interactive session

**Step 1 - Allocate resources:**
```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 \
       --partition=MI250X_A1_COS_OK --time=00:30:00
```

**Step 2 - Inside the interactive session:**
```bash
cd /home/jujaykka/projects/sbd-amd/applications/selected_basis_diagonalization/benchmarks

srun ./run_with_timing.sh ../src/diag \
  --fcidump ../data/fe4s4/fcidump_Fe4S4.txt \
  --adetfile ../data/fe4s4/AlphaDets.txt \
  --method 0 --block 10 --iteration 1 --tolerance 1.0e-4 \
  --adet_comm_size 1 --bdet_comm_size 1 --task_comm_size 1 \
  --init 0 --shuffle 0 --carryover_ratio 0.5 --rdm 0
```

**Step 3 - Exit when done:**
```bash
exit
```

**Key simplifications:**
- No bash -c wrapper needed!
- Environment sourced automatically inside run_with_timing.sh
- Just call `srun ./run_with_timing.sh` directly
- CRITICAL: `--gpus-per-task=1` required in salloc for GPU access

---

## Case 3: sbatch Direct

**Description:** Submit batch job to SLURM queue (no separate script file needed)

**Command:**
```bash
cd /home/jujaykka/projects/sbd-amd/applications/selected_basis_diagonalization/benchmarks

sbatch --job-name=fe4s4_test --nodes=1 --ntasks=1 --cpus-per-task=8 \
       --gpus-per-task=1 --partition=MI250X_A1_COS_OK --time=00:10:00 \
       ./run_with_timing.sh ../src/diag \
       --fcidump ../data/fe4s4/fcidump_Fe4S4.txt \
       --adetfile ../data/fe4s4/AlphaDets.txt \
       --method 0 --block 10 --iteration 1 --tolerance 1.0e-4 \
       --adet_comm_size 1 --bdet_comm_size 1 --task_comm_size 1 \
       --init 0 --shuffle 0 --carryover_ratio 0.5 --rdm 0
```

**Monitor:**
```bash
squeue -u $USER
```

**Check results:**
```bash
# SLURM output
cat slurm-<JOBID>.out

# Wrapper timing output
ls -lht run_*.log | head -1
cat run_*.log | tail -20
```

**Notes:**
- No separate script file needed - pass arguments directly to sbatch
- Environment sourced automatically inside run_with_timing.sh
- Output saved to slurm-<JOBID>.out
- CRITICAL: `--gpus-per-task=1` required for GPU access

---

## Case 4: Direct srun

**Description:** One-command execution (no separate job script file)

**Command:**
```bash
cd /home/jujaykka/projects/sbd-amd/applications/selected_basis_diagonalization/benchmarks

srun --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 \
     --partition=MI250X_A1_COS_OK --time=00:10:00 \
     ./run_with_timing.sh ../src/diag \
       --fcidump ../data/fe4s4/fcidump_Fe4S4.txt \
       --adetfile ../data/fe4s4/AlphaDets.txt \
       --method 0 --block 10 --iteration 1 --tolerance 1.0e-4 \
       --adet_comm_size 1 --bdet_comm_size 1 --task_comm_size 1 \
       --init 0 --shuffle 0 --carryover_ratio 0.5 --rdm 0
```

**Notes:**
- Same syntax as Case 2 but without prior salloc
- srun waits in queue, allocates resources, runs, then releases
- Good for one-off runs
- No bash -c wrapper - just call run_with_timing.sh directly
- CRITICAL: `--gpus-per-task=1` required for GPU access
- On some systems, direct srun may be disabled (use Case 2 or 3 instead)

---

## Verifying Success

After any run completes successfully, check:

1. **Exit code in output:**
   ```bash
   grep "Exit code:" run_*.log | tail -1
   ```
   Should show: `Exit code:         0`

2. **Wall time:**
   ```bash
   grep "Wall time elapsed:" run_*.log | tail -1
   ```
   Should show: `Wall time elapsed: <N> seconds` (typically 15-25 seconds for Fe4S4)

3. **Energy convergence:**
   ```bash
   grep "Energy =" run_*.log | tail -1
   ```
   Should show final energy value

4. **Metadata captured:**
   ```bash
   head -40 run_*.log
   ```
   Should show full metadata header with modules, environment, etc.

---

## Expected Output Filename Pattern

- **Local/Case 1:** `run_YYYYMMDD_HHMMSS_local_PID.log`
- **SLURM/Case 2-4:** `run_YYYYMMDD_HHMMSS_JOBID_PID.log`

Example: `run_20251125_071500_38568_123456.log`

---

## Troubleshooting

### Problem: "librocblas.so.4: cannot open shared object file"
**Solution:** This should not happen anymore - run_with_timing.sh sources env_setup.sh automatically.
If it still occurs, check that env_setup.sh exists in the parent directory.

### Problem: "Failed to open FCIDUMP file"
**Solution:** Wrong working directory or relative paths incorrect. Use relative paths:
```bash
--fcidump ../data/fe4s4/fcidump_Fe4S4.txt  # NOT data/fe4s4/...
```

### Problem: Cray assertion failure on login node
**Solution:** This is expected. Run on compute node via srun:
```bash
srun --nodes=1 --ntasks=1 --gpus-per-task=1 --partition=MI250X_A1_COS_OK <command>
```

### Problem: "no ROCm-capable device is detected"
**Solution:** Missing GPU allocation. Add `--gpus-per-task=1` to srun/salloc command.

---

## Quick Comparison of All 4 Methods

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **Case 1 (make)** | Development | Simplest, auto-navigates | Requires Makefile target |
| **Case 2 (salloc)** | Interactive debugging | Multiple tests in one allocation | Manual resource allocation |
| **Case 3 (sbatch)** | Production runs | Queued execution, automated | Requires script file |
| **Case 4 (srun)** | One-off runs | No script file needed | May be disabled on some systems |

**ALL CASES:** Just call `run_with_timing.sh` directly - no bash -c wrappers needed!

---

## Performance Note

Fe4S4 with iteration=1 takes approximately **15-25 seconds** on a compute node with 8 OpenMP threads.

For production benchmarks, use H2O dataset with iteration=2 (takes several minutes).
