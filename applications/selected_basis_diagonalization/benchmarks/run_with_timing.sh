#!/usr/bin/env bash
#
# run_with_timing.sh - Universal wrapper for running diag with timing and metadata capture
#
# This script works in all execution contexts:
#   1. make target:       make my_new_benchmark
#   2. Interactive salloc: srun ./run_with_timing.sh ../src/diag --args
#   3. sbatch script:     srun ./run_with_timing.sh ../src/diag --args
#   4. Direct srun:       srun --nodes=N ... ./run_with_timing.sh ../src/diag --args
#
# Features:
#   - Monotonic timing via /proc/uptime (seconds since boot, immune to clock adjustments)
#   - Auto-capture to unique timestamped file
#   - Rich metadata header (modules, MPI layout, environment, git commit)
#   - Timing happens AFTER srun scheduling (excludes queue wait time)
#
# Usage:
#   ./run_with_timing.sh <command> [args...]
#
# Example:
#   ./run_with_timing.sh ../src/diag --fcidump data/n2/fcidump.txt --adetfile data/n2/alphadets.txt --method 0
#

set -euo pipefail

# ============================================================================
# Source environment setup (ROCm libraries, etc.)
# ============================================================================
# Try multiple strategies to find env_setup.sh (handles all execution contexts)
if [ -f "../env_setup.sh" ]; then
  # Strategy 1: Relative to current directory (works for srun, make when pwd=benchmarks)
  ENV_SETUP="../env_setup.sh"
elif [ -n "$SLURM_SUBMIT_DIR" ] && [ -f "${SLURM_SUBMIT_DIR}/../env_setup.sh" ]; then
  # Strategy 2: Relative to SLURM submit dir (works for sbatch from benchmarks/)
  ENV_SETUP="${SLURM_SUBMIT_DIR}/../env_setup.sh"
elif [ -n "$SLURM_SUBMIT_DIR" ] && [ -f "${SLURM_SUBMIT_DIR}/env_setup.sh" ]; then
  # Strategy 3: In SLURM submit dir (works for make via srun from parent)
  ENV_SETUP="${SLURM_SUBMIT_DIR}/env_setup.sh"
else
  # Strategy 4: Relative to script location (fallback)
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ENV_SETUP="${SCRIPT_DIR}/../env_setup.sh"
fi

if [ -f "$ENV_SETUP" ]; then
  source "$ENV_SETUP"
else
  echo "WARNING: env_setup.sh not found at $ENV_SETUP"
  echo "Continuing without environment setup..."
fi

# ============================================================================
# Generate unique output filename
# ============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-local}
PID=$$
OUTPUT_FILE="run_${TIMESTAMP}_${JOB_ID}_${PID}.log"

# ============================================================================
# Helper function: Get git commit hash
# ============================================================================
get_git_commit() {
  if git rev-parse --git-dir > /dev/null 2>&1; then
    git rev-parse --short HEAD 2>/dev/null || echo "unknown"
  else
    echo "not-a-repo"
  fi
}

# ============================================================================
# Helper function: Get MPI rank info (if running under MPI)
# ============================================================================
get_mpi_info() {
  local rank="${PMI_RANK:-${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-unknown}}}"
  local size="${PMI_SIZE:-${SLURM_NTASKS:-${OMPI_COMM_WORLD_SIZE:-unknown}}}"
  echo "MPI Rank $rank / $size"
}

# ============================================================================
# Start monotonic timing (seconds since boot from /proc/uptime)
# ============================================================================
START_TIME=$(awk '{print $1}' /proc/uptime)

# ============================================================================
# Write metadata header to output file
# ============================================================================
{
  echo "================================================================================"
  echo "                           RUN METADATA"
  echo "================================================================================"
  echo ""
  echo "Execution Details:"
  echo "  Date:              $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "  Hostname:          $(hostname)"
  echo "  Working Directory: $(pwd)"
  echo "  Command:           $*"
  echo "  Git Commit:        $(get_git_commit)"
  echo ""

  echo "SLURM Environment:"
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "  Job ID:            $SLURM_JOB_ID"
    echo "  Job Name:          ${SLURM_JOB_NAME:-N/A}"
    echo "  Partition:         ${SLURM_JOB_PARTITION:-N/A}"
    echo "  Nodes:             ${SLURM_JOB_NUM_NODES:-N/A}"
    echo "  Node List:         ${SLURM_JOB_NODELIST:-N/A}"
    echo "  Total Tasks:       ${SLURM_NTASKS:-N/A}"
    echo "  Tasks per Node:    ${SLURM_TASKS_PER_NODE:-N/A}"
    echo "  CPUs per Task:     ${SLURM_CPUS_PER_TASK:-N/A}"
    echo "  GPUs per Task:     ${SLURM_GPUS_PER_TASK:-N/A}"
    echo "  Account:           ${SLURM_ACCOUNT:-N/A}"
  else
    echo "  (Not running under SLURM)"
  fi
  echo ""

  echo "MPI Configuration:"
  echo "  $(get_mpi_info)"
  echo ""

  echo "OpenMP Configuration:"
  echo "  OMP_NUM_THREADS:   ${OMP_NUM_THREADS:-unset}"
  echo "  OMP_PROC_BIND:     ${OMP_PROC_BIND:-unset}"
  echo "  OMP_PLACES:        ${OMP_PLACES:-unset}"
  echo ""

  echo "Loaded Modules:"
  if command -v module > /dev/null 2>&1; then
    module list 2>&1 | sed 's/^/  /'
  else
    echo "  (module command not available)"
  fi
  echo ""

  echo "Key Environment Variables:"
  echo "  PATH:              ${PATH}"
  echo "  LD_LIBRARY_PATH:   ${LD_LIBRARY_PATH:-unset}"
  echo "  ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-unset}"
  echo "  HIP_VISIBLE_DEVICES:  ${HIP_VISIBLE_DEVICES:-unset}"
  echo ""

  echo "================================================================================"
  echo "                           PROGRAM OUTPUT"
  echo "================================================================================"
  echo ""
} | tee "$OUTPUT_FILE"

# ============================================================================
# Execute the command and capture output
# ============================================================================
# Use PIPESTATUS to capture the exit code of the command, not tee
"$@" 2>&1 | tee -a "$OUTPUT_FILE"
EXIT_CODE=${PIPESTATUS[0]}

# ============================================================================
# End monotonic timing
# ============================================================================
END_TIME=$(awk '{print $1}' /proc/uptime)

# Calculate elapsed time (integer seconds, sufficient for ~10 min runs)
ELAPSED=$(awk "BEGIN {printf \"%.0f\", $END_TIME - $START_TIME}")

# ============================================================================
# Write timing footer to output file
# ============================================================================
{
  echo ""
  echo "================================================================================"
  echo "                           RUN SUMMARY"
  echo "================================================================================"
  echo ""
  echo "Wall time elapsed: $ELAPSED seconds"
  echo "Exit code:         $EXIT_CODE"
  echo "Output saved to:   $OUTPUT_FILE"
  echo ""
  echo "================================================================================"
} | tee -a "$OUTPUT_FILE"

# Also write a machine-readable timing line for easy parsing
echo "TIMING_RESULT: elapsed=${ELAPSED}s exit_code=${EXIT_CODE} output_file=${OUTPUT_FILE}" | tee -a "$OUTPUT_FILE"

exit $EXIT_CODE
