#!/bin/bash
# === Define Paths ===
BASE_DIR=$(cd "$SLURM_SUBMIT_DIR/.." && pwd)
SRC_DIR="$BASE_DIR/src"
DATA_DIR="$BASE_DIR/data"
BENCHMARK_DIR="$BASE_DIR/benchmarks"
EXE_PATH="$SRC_DIR/diag"

# Allow user to override decomposition via environment variables
if [ -n "$ADET_COMM" ] && [ -n "$BDET_COMM" ] && [ -n "$TASK_COMM" ]
then
    echo "Using user-specified MPI decomposition: ADET=$ADET_COMM, BDET=$BDET_COMM, TASK=$TASK_COMM"
else
    # Use defaults
    export ADET_COMM=8
    export BDET_COMM=8
    export TASK_COMM=4
    echo "Defult MPI decomposition: ADET=$ADET_COMM, BDET=$BDET_COMM, TASK=$TASK_COMM"
fi

# Validate the decomposition matches total tasks
PRODUCT=$((ADET_COMM * BDET_COMM * TASK_COMM))
if [ $PRODUCT -ne ${SLURM_NTASKS} ]; then
  echo "========================================="
  echo "ERROR: MPI decomposition invalid!"
  echo "========================================="
  echo "  TOTAL_TASKS = $SLURM_NTASKS"
  echo "  ADET_COMM × BDET_COMM × TASK_COMM = $ADET_COMM × $BDET_COMM × $TASK_COMM = $PRODUCT"
  echo ""
  echo "The product must equal the total number of tasks."
  echo ""
  echo "To manually specify decomposition, use:"
  echo "  sbatch --export=ADET_COMM=X,BDET_COMM=Y,TASK_COMM=Z,... sbatch_run.sh"
  echo "========================================="
  exit 1
fi

# === Job Info ===
echo "========================================="
echo "SBD ${MODE^^} Job - MI250X System"
echo "========================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Mode:            $MODE"
echo "Nodes:           $SLURM_NNODES ($SLURM_NODELIST)"
echo "Tasks/Node:      $SLURM_NTASKS_PER_NODE"
echo "CPUs/Task:       $SLURM_CPUS_PER_TASK"
echo "Total Tasks:     $SLURM_NTASKS"
echo "MPI Decomp:      ADET=$ADET_COMM × BDET=$BDET_COMM × TASK=$TASK_COMM = $PRODUCT"
echo "Dataset:         $DATASET"
echo "========================================="

# === Load Environment ===
echo module restore $1
module restore $1

# Ensure LD_LIBRARY_PATH is exported to srun tasks
export LD_LIBRARY_PATH

# === Create Results Directory ===
# Allow override via environment variable (for sweep mode)
if [ -n "$RESULTS_DIR" ]; then
  # Use provided RESULTS_DIR (e.g., for decomposition sweeps)
  RESULTS_DIR="$RESULTS_DIR"
elif [ "$MODE" = "benchmark" ]; then
  # Add decomposition to directory name if specified
  if [ -n "$ADET_COMM" ] && [ -n "$BDET_COMM" ]; then
    RESULTS_DIR="$BENCHMARK_DIR/results/slurm_${SLURM_JOB_ID}_${DATASET}_${SLURM_NNODES}nodes_${ADET_COMM}x${BDET_COMM}"
  else
    RESULTS_DIR="$BENCHMARK_DIR/results/slurm_${SLURM_JOB_ID}_${DATASET}_${SLURM_NNODES}nodes"
  fi
else
  RESULTS_DIR="$BENCHMARK_DIR/profile_results/slurm_${SLURM_JOB_ID}_${DATASET}"
  ITERATIONS=1 # Force single iteration for profiling
fi
mkdir -p "$RESULTS_DIR"
cp "$0" "$RESULTS_DIR/job_script.sh"
cp "$ENV_SETUP" "$RESULTS_DIR/env_setup.sh" 2>/dev/null
env >"$RESULTS_DIR/job_environment.txt"

# === Dataset Path ===
# ADETFILE can be overridden via environment variable (from Makefile)
DATASET=h2o
FCIDUMP="$DATA_DIR/h2o/fcidump.txt"
ADETFILE="${DATA_DIR}/h2o/h2o-1em8-alpha.txt"

# === Run Job ===
echo ""
echo "Starting $MODE at $(date --iso=seconds)"
echo ""
LOGFILE="$RESULTS_DIR/benchmark_rep${i}.log"

then=$(date +%s) && \
    srun --export=ALL --label \
	 ${EXE_PATH} --fcidump ../data/h2o/fcidump.txt \
	 --adetfile ../data/h2o/h2o-1em8-alpha.txt \
	 --method 0 \
	 --block 10 \
	 --iteration 1 \
	 --tolerance 1.0e-4 \
	 --adet_comm_size 8 \
	 --bdet_comm_size 8 \
	 --task_comm_size 4 \
	 --init 0 \
	 --shuffle 1 \
	 --carryover_ratio 0.5 \
	 --rdm 0
echo "EXIT_CODE: $? $(hostname)"
now=$(date +%s)
echo "Wall time elapsed: $((${now} - ${then}))"

echo "========================================="
echo "Results saved in: $RESULTS_DIR"
echo "========================================="
