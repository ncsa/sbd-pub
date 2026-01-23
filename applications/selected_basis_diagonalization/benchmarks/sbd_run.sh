#!/usr/bin/env bash
#SBATCH --job-name=sbd_run
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --gpus-per-task=1

# NOTE: User MUST specify --nodes, --ntasks-per-node, --cpus-per-task on command line

set -euo pipefail

# ============================================================================
# Help text
# ============================================================================
show_help() {
    cat <<EOF
Usage: sbd_run.sh [OPTIONS] EXECUTABLE [EXECUTABLE_ARGS...]

Universal wrapper for running SBD executables with timing, metadata capture,
and optional orchestration features. Preserves full flexibility to run any
executable with any arguments.

OPTIONS:
  --replicates N        Run N benchmark replicates with statistical analysis
  --profile             Enable profiling mode (wraps execution with rocprof)
  --output-dir DIR      Set output directory (default: auto-timestamped)
  --env MODE            Environment setup mode:
                          inherit         - Use inherited environment (no sourcing)
                          NAME            - Restore saved module collection NAME
                          /path/to/file   - Source specified file
                          ./path/to/file  - Source relative path file
                        (default: source ../env_setup.sh)
  --help                Show this help message

EXAMPLES:
  # Simple run with timing and metadata:
  sbatch --nodes=1 --ntasks-per-node=8 --cpus-per-task=7 --gpus-per-task=1 \\
         ./sbd_run.sh ./diag --fcidump data/n2/fcidump.txt --method 0

  # Benchmark mode (5 replicates):
  sbatch --nodes=1 --ntasks-per-node=8 --cpus-per-task=7 --gpus-per-task=1 \\
         ./sbd_run.sh --replicates 5 ../src/diag --fcidump data/fe4s4/fcidump.txt

  # Profile mode:
  sbatch --nodes=1 --ntasks-per-node=8 --cpus-per-task=7 --gpus-per-task=1 \\
         ./sbd_run.sh --profile ./diag --fcidump data/h2o/fcidump.txt

  # With salloc then srun (srun starts SLURM_NTASKS copies simultaneously):
  salloc --nodes=1 --ntasks=8 --cpus-per-task=7 --gpus-per-task=1
  srun ./sbd_run.sh ./diag --fcidump data/fe4s4/fcidump.txt

  # Environment modes:
  srun ./sbd_run.sh --env inherit ./diag --fcidump data/n2/fcidump.txt
  srun ./sbd_run.sh --env my_saved_modules ./diag --fcidump data/n2/fcidump.txt
  srun ./sbd_run.sh --env /custom/env.sh ./diag --fcidump data/n2/fcidump.txt
EOF
}

# ============================================================================
# Parse script options
# ============================================================================
REPLICATES=1
PROFILE_MODE=0
OUTPUT_DIR=""
ENV_MODE=""  # Empty = default (../env_setup.sh)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            show_help
            exit 0
            ;;
        --replicates)
            REPLICATES="$2"
            shift 2
            ;;
        --profile)
            PROFILE_MODE=1
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --env)
            ENV_MODE="$2"
            shift 2
            ;;
        *)
            # First non-option argument is the executable
            break
            ;;
    esac
done

# Remaining arguments: executable and its arguments
if [ $# -eq 0 ]; then
    echo "ERROR: No executable specified"
    show_help
    exit 1
fi

EXECUTABLE="$1"
shift
EXECUTABLE_ARGS=("$@")

# Validate executable exists and is executable
if [ ! -f "$EXECUTABLE" ]; then
    echo "ERROR: Executable not found: $EXECUTABLE"
    exit 1
fi
if [ ! -x "$EXECUTABLE" ]; then
    echo "ERROR: File is not executable: $EXECUTABLE"
    exit 1
fi

# ============================================================================
# Detect execution context
# ============================================================================
RANK=${SLURM_PROCID:-0}
IS_RANK0=$([ $RANK -eq 0 ] && echo 1 || echo 0)

echo_rank0() {
    [ $IS_RANK0 -eq 1 ] && echo "$@" || echo "$@" >/dev/null
}

# ============================================================================
# Path setup and environment
# ============================================================================
# Determine base directory robustly (handle symlinks correctly)
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ANALYZER_PY="$SCRIPT_DIR/summarize.py"
ANALYSIS_SCRIPT="$SCRIPT_DIR/show-profile-tree.py"

# Environment setup via --env flag:
# --env inherit            - Use inherited environment (no sourcing)
# --env NAME               - Restore saved module collection NAME
# --env /path/to/file      - Source absolute path file
# --env ./path/to/file     - Source relative path file
# --env ../path/to/file    - Source relative path file
# (default if --env not specified: source ../env_setup.sh)

if [ -z "$ENV_MODE" ]; then
    # Default: source ../env_setup.sh
    ENV_FILE="$BASE_DIR/env_setup.sh"
    if [ -f "$ENV_FILE" ]; then
        echo_rank0 "Sourcing environment from $ENV_FILE (default)"
        source "$ENV_FILE"
    else
        echo_rank0 "WARNING: Default env file not found: $ENV_FILE"
        echo_rank0 "Continuing with inherited environment..."
    fi
elif [ "$ENV_MODE" = "inherit" ]; then
    echo_rank0 "Using inherited environment (--env inherit)"
elif [[ "$ENV_MODE" == /* ]] || [[ "$ENV_MODE" == ./* ]] || [[ "$ENV_MODE" == ../* ]]; then
    # Path (starts with /, ./, or ../)
    if [ -f "$ENV_MODE" ]; then
        echo_rank0 "Sourcing environment from $ENV_MODE"
        source "$ENV_MODE"
    else
        echo_rank0 "ERROR: Environment file not found: $ENV_MODE"
        exit 1
    fi
else
    # Module collection name
    echo_rank0 "Restoring module collection: $ENV_MODE"
    module restore "$ENV_MODE"
fi

# ============================================================================
# Git commit and output directory
# ============================================================================
# Capture git commit for provenance
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date +%Y%m%dT%H%M%S)  # Almost-ISO format (: is problematic in shell)
JOB_ID=${SLURM_JOB_ID:-local}
PID=$$

# Determine output directory or file
if [ -n "$OUTPUT_DIR" ]; then
    # User specified output directory
    RESULTS_DIR="$OUTPUT_DIR"
else
    # Auto-generate timestamped directory/file
    EXECUTABLE_BASENAME=$(basename "$EXECUTABLE")
    if [ $PROFILE_MODE -eq 1 ]; then
        RESULTS_DIR="profile_${EXECUTABLE_BASENAME}_${GIT_COMMIT}_${TIMESTAMP}_${JOB_ID}"
    elif [ $REPLICATES -gt 1 ]; then
        RESULTS_DIR="benchmark_${EXECUTABLE_BASENAME}_${GIT_COMMIT}_${TIMESTAMP}_${JOB_ID}"
    else
        # Single run - output to specific log file, not directory
        # Invent new name if file already exists (avoid appending to old logs)
        OUTPUT_FILE="run_${EXECUTABLE_BASENAME}_${GIT_COMMIT}_${TIMESTAMP}_${JOB_ID}_${PID}.log"
        COUNTER=1
        while [ -f "$OUTPUT_FILE" ]; do
            OUTPUT_FILE="run_${EXECUTABLE_BASENAME}_${GIT_COMMIT}_${TIMESTAMP}_${JOB_ID}_${PID}_${COUNTER}.log"
            COUNTER=$((COUNTER + 1))
        done
    fi
fi

# Create results directory if needed (rank-0 only to avoid race)
if [ $IS_RANK0 -eq 1 ] && [ -n "$RESULTS_DIR" ]; then
    mkdir -p "$RESULTS_DIR"
fi

# ============================================================================
# DO NOT Set OMP_NUM_THREADS
# ============================================================================
# Let SLURM cgroups limit CPUs; OpenMP will auto-detect
# DO NOT: export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# OpenMP uses omp_get_num_procs() which respects cgroup limits
#
# If user wants to override, they can set it in environment before calling script

# ============================================================================
# Metadata Header (Rank-0 Only)
# ============================================================================
# Determine output file for metadata (single run vs replicate/profile)
if [ -n "$RESULTS_DIR" ]; then
    METADATA_LOG="$RESULTS_DIR/run_metadata.log"
    # For replicate/profile modes, check if metadata log exists and invent new name if needed
    if [ $IS_RANK0 -eq 1 ] && [ -f "$METADATA_LOG" ]; then
        COUNTER=1
        while [ -f "${RESULTS_DIR}/run_metadata_${COUNTER}.log" ]; do
            COUNTER=$((COUNTER + 1))
        done
        METADATA_LOG="${RESULTS_DIR}/run_metadata_${COUNTER}.log"
    fi
else
    METADATA_LOG="$OUTPUT_FILE"
fi

# Note: Do NOT touch file before tee - let tee create it
# We've already ensured unique filename above, so no risk of appending to old logs

{
    echo_rank0 "================================================================================"
    echo_rank0 "                           RUN METADATA"
    echo_rank0 "================================================================================"
    echo_rank0 ""
    echo_rank0 "Execution Details:"
    echo_rank0 "  Date:              $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo_rank0 "  Hostname:          $(hostname)"
    echo_rank0 "  Working Directory: $(pwd)"
    echo_rank0 "  Executable:        $EXECUTABLE"
    echo_rank0 "  Arguments:         ${EXECUTABLE_ARGS[*]}"
    echo_rank0 "  Git Commit:        $GIT_COMMIT"
    echo_rank0 "  Mode:              $([ $PROFILE_MODE -eq 1 ] && echo "PROFILE" || ([ $REPLICATES -gt 1 ] && echo "BENCHMARK ($REPLICATES replicates)" || echo "SINGLE RUN"))"
    echo_rank0 ""

    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo_rank0 "SLURM Environment:"
        echo_rank0 "  Job ID:            $SLURM_JOB_ID"
        echo_rank0 "  Job Name:          ${SLURM_JOB_NAME:-N/A}"
        echo_rank0 "  Partition:         ${SLURM_JOB_PARTITION:-N/A}"
        echo_rank0 "  Nodes:             ${SLURM_JOB_NUM_NODES:-N/A}"
        echo_rank0 "  Node List:         ${SLURM_JOB_NODELIST:-N/A}"
        echo_rank0 "  Total Tasks:       ${SLURM_NTASKS:-N/A}"
        echo_rank0 "  Tasks per Node:    ${SLURM_TASKS_PER_NODE:-N/A}"
        echo_rank0 "  CPUs per Task:     ${SLURM_CPUS_PER_TASK:-N/A}"
        echo_rank0 "  GPUs per Task:     ${SLURM_GPUS_PER_TASK:-N/A}"
        echo_rank0 ""
    else
        echo_rank0 "SLURM Environment: (Not running under SLURM)"
        echo_rank0 ""
    fi

    echo_rank0 "OpenMP Configuration:"
    echo_rank0 "  OMP_NUM_THREADS:   ${OMP_NUM_THREADS:-unset (auto-detect via cgroups)}"
    echo_rank0 "  OMP_PROC_BIND:     ${OMP_PROC_BIND:-unset}"
    echo_rank0 "  OMP_PLACES:        ${OMP_PLACES:-unset}"
    echo_rank0 ""

    echo_rank0 "Loaded Modules:"
    if command -v module > /dev/null 2>&1; then
        module list 2>&1 | sed 's/^/  /'
    else
        echo_rank0 "  (module command not available)"
    fi
    echo_rank0 ""

    echo_rank0 "Key Environment Variables:"
    echo_rank0 "  PATH:              ${PATH}"
    echo_rank0 "  LD_LIBRARY_PATH:   ${LD_LIBRARY_PATH:-unset}"
    echo_rank0 "  ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-unset}"
    echo_rank0 "  HIP_VISIBLE_DEVICES:  ${HIP_VISIBLE_DEVICES:-unset}"
    echo_rank0 ""
    echo_rank0 "================================================================================"
    echo_rank0 "                           PROGRAM OUTPUT"
    echo_rank0 "================================================================================"
    echo_rank0 ""
} | tee -a "$METADATA_LOG"

# ============================================================================
# Detect execution context and determine if we need to call srun
# ============================================================================
# SLURM_STEP_ID is only set by srun, not by sbatch
# - Under srun: SLURM_STEP_ID is set (script is one of SLURM_NTASKS copies)
# - Under sbatch only: SLURM_STEP_ID is NOT set (script runs once, must call srun)

if [ -n "${SLURM_STEP_ID:-}" ]; then
    # Running under srun - script is one of SLURM_NTASKS copies
    UNDER_SRUN=1
else
    # Not under srun - script runs once
    UNDER_SRUN=0
fi

# Helper function to execute command (with or without srun)
execute_command() {
    if [ $UNDER_SRUN -eq 0 ]; then
        # Script runs once, need to launch tasks
        if [ -n "${SLURM_JOB_ID:-}" ]; then
            # Have SLURM allocation - use srun to launch tasks
            # Inherit task configuration from sbatch allocation
            # Always request GPU allocation (required for SBD application)
            srun --ntasks="${SLURM_NTASKS}" --gpus-per-task=1 "$@"
        else
            # No SLURM - run directly (serial mode for testing)
            "$@"
        fi
    else
        # Already under srun - execute directly (no nesting!)
        "$@"
    fi
}

# ============================================================================
# Execution logic based on mode
# ============================================================================

if [ $REPLICATES -gt 1 ]; then
    # === BENCHMARK MODE (multiple replicates) ===
    SUMMARY="$RESULTS_DIR/summary.txt"

    for i in $(seq 1 $REPLICATES); do
        echo_rank0 "=== Replicate $i/$REPLICATES ===" | tee -a "$SUMMARY"

        LOGFILE="$RESULTS_DIR/benchmark_rep${i}_${TIMESTAMP}_${GIT_COMMIT}.log"

        # Monotonic wall time measurement (external to application)
        START_TIME=$(awk '{print $1}' /proc/uptime)

        # Execute the command (srun added automatically if needed)
        execute_command "$EXECUTABLE" "${EXECUTABLE_ARGS[@]}" 2>&1 | tee -a "$LOGFILE"

        EXIT_CODE=${PIPESTATUS[0]}
        END_TIME=$(awk '{print $1}' /proc/uptime)
        WALL_TIME=$(awk "BEGIN {printf \"%.2f\", $END_TIME - $START_TIME}")

        # Extract internal timings if available (application-specific, optional)
        DAVIDSON_TIME=$(grep "Elapsed time for davidson" "$LOGFILE" 2>/dev/null | tail -1 | awk '{print $5}' || echo "N/A")
        ENERGY=$(grep " Energy = " "$LOGFILE" 2>/dev/null | tail -1 | awk '{print $5}' || echo "N/A")

        echo_rank0 "Replicate $i: Wall=${WALL_TIME}s Davidson=${DAVIDSON_TIME}s Energy=${ENERGY} Exit=${EXIT_CODE}" | tee -a "$SUMMARY"

        # Machine-readable output
        echo_rank0 "BENCHMARK_RESULT: replicate=$i wall_time=${WALL_TIME}s davidson_time=${DAVIDSON_TIME}s energy=${ENERGY} exit_code=${EXIT_CODE} git_commit=${GIT_COMMIT} executable=${EXECUTABLE} logfile=${LOGFILE}" | tee -a "$SUMMARY"

        [ $i -lt $REPLICATES ] && sleep 5
    done

    # Statistical analysis (rank-0 only)
    if [ $IS_RANK0 -eq 1 ]; then
        echo_rank0 ""
        echo_rank0 "=== Benchmark Summary ==="
        echo_rank0 ""
        echo_rank0 "Benchmark completed at $(date --iso=seconds)"

        if [ -f "$ANALYZER_PY" ]; then
            python3 "$ANALYZER_PY" "$RESULTS_DIR"/benchmark_rep*.log > "$RESULTS_DIR/final_summary.txt" 2>&1
            cat "$RESULTS_DIR/final_summary.txt"
        fi
    fi

elif [ $PROFILE_MODE -eq 1 ]; then
    # === PROFILE MODE ===
    PROFILE_LOG="$RESULTS_DIR/profile_${GIT_COMMIT}.log"

    START_TIME=$(awk '{print $1}' /proc/uptime)

    # Per-rank rocprof to avoid SQLite conflicts
    # Note: rocprof must wrap the actual executable, so we need special handling
    RANK_ID=${SLURM_PROCID:-0}

    if [ $UNDER_SRUN -eq 0 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
        # Not under srun but have allocation - wrap srun call with rocprof
        srun rocprof --stats --timestamp on \
            -o "$RESULTS_DIR/rocprof_rank\${SLURM_PROCID}.csv" \
            "$EXECUTABLE" "${EXECUTABLE_ARGS[@]}" \
            2>&1 | tee -a "$PROFILE_LOG"
    else
        # Already under srun or no SLURM - wrap executable directly
        rocprof --stats --timestamp on \
            -o "$RESULTS_DIR/rocprof_rank${RANK_ID}.csv" \
            "$EXECUTABLE" "${EXECUTABLE_ARGS[@]}" \
            2>&1 | tee -a "$PROFILE_LOG"
    fi

    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(awk '{print $1}' /proc/uptime)
    WALL_TIME=$(awk "BEGIN {printf \"%.2f\", $END_TIME - $START_TIME}")

    echo_rank0 "PROFILE_RESULT: wall_time=${WALL_TIME}s exit_code=${EXIT_CODE} git_commit=${GIT_COMMIT} executable=${EXECUTABLE}" | tee -a "$PROFILE_LOG"

    # Analysis (rank 0 only)
    if [ $IS_RANK0 -eq 1 ] && [ -f "$ANALYSIS_SCRIPT" ] && [ -f "$RESULTS_DIR/rocprof_rank0.csv" ]; then
        echo_rank0 ""
        echo_rank0 "Running profile analysis..."
        python3 "$ANALYSIS_SCRIPT" "$RESULTS_DIR/rocprof_rank0.csv" > "$RESULTS_DIR/analysis.txt" 2>&1
        echo_rank0 "=== Top Functions (Rank 0) ==="
        head -30 "$RESULTS_DIR/analysis.txt"
        echo_rank0 ""
        echo_rank0 "Full analysis saved to: $RESULTS_DIR/analysis.txt"
    fi

else
    # === SINGLE RUN MODE (default, like run_with_timings_real.sh) ===

    # Monotonic wall time measurement
    START_TIME=$(awk '{print $1}' /proc/uptime)

    # Execute the command (srun added automatically if needed)
    execute_command "$EXECUTABLE" "${EXECUTABLE_ARGS[@]}" 2>&1 | tee -a "$OUTPUT_FILE"

    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(awk '{print $1}' /proc/uptime)
    ELAPSED=$(awk "BEGIN {printf \"%.0f\", $END_TIME - $START_TIME}")

    # Write timing footer
    {
        echo_rank0 ""
        echo_rank0 "================================================================================"
        echo_rank0 "                           RUN SUMMARY"
        echo_rank0 "================================================================================"
        echo_rank0 ""
        echo_rank0 "Wall time elapsed: $ELAPSED seconds"
        echo_rank0 "Exit code:         $EXIT_CODE"
        echo_rank0 "Output saved to:   $OUTPUT_FILE"
        echo_rank0 ""
        echo_rank0 "================================================================================"
    } | tee -a "$OUTPUT_FILE"

    # Machine-readable timing line
    echo_rank0 "TIMING_RESULT: elapsed=${ELAPSED}s exit_code=${EXIT_CODE} output_file=${OUTPUT_FILE} git_commit=${GIT_COMMIT} executable=${EXECUTABLE}" | tee -a "$OUTPUT_FILE"
fi

exit $EXIT_CODE
