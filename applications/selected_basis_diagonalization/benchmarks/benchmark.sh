#!/usr/bin/env bash
#
# benchmark.sh - SBD Application Benchmarking Suite
#
# Usage:
#     ./benchmark.sh [OPTIONS]
#
# Options:
#     --dataset NAME       Dataset (fe4s4, n2, skqd, or custom)
#     --tasks N            Number of MPI tasks (default: 1)
#     --replicates N       Number of benchmark runs (default: 5)
#     --iterations N       Davidson iterations (default: 2)
#     --threads N          OpenMP threads (default: 64)
#     --fcidump FILE       Custom FCIDUMP file path
#     --adetfile FILE      Custom alpha determinants file path
#     --set-baseline       Save current performance as baseline
#     --compare-baseline   Compare against saved baseline
#     --output-dir DIR     Output directory (default: ./results)
#     --tag TAG            Tag for output files (default: timestamp)
#     --help               Show this help message
#
# Examples:
#     ./benchmark.sh --dataset fe4s4                   # Quick test (1 task)
#     ./benchmark.sh --dataset n2 --tasks 8 --reps 10  # 8-task benchmark
#     ./benchmark.sh --dataset n2 --set-baseline       # Before GPU port
#     ./benchmark.sh --dataset n2 --compare-baseline   # After GPU port

set -e

# --- Define Absolute Base Paths ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BASE_DIR=$(dirname "$SCRIPT_DIR")

SRC_DIR="$BASE_DIR/src"
DATA_DIR="$BASE_DIR/data"
BENCHMARK_DIR="$BASE_DIR/benchmarks"
EXE_PATH="$SRC_DIR/diag"
ANALYZER_PY="$BENCHMARK_DIR/summarize.py"
DEFAULT_OUTPUT_DIR="$BENCHMARK_DIR/results"
DEFAULT_BASELINE_DIR="$BENCHMARK_DIR/baseline"
# ------------------------------------

# Default configuration
DATASET="fe4s4"
TASKS=1
REPLICATES=5
ITERATIONS=2
THREADS=64
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
BASELINE_DIR="$DEFAULT_BASELINE_DIR"
TAG=$(date +%Y%m%d_%H%M%S)
SET_BASELINE=0
COMPARE_BASELINE=0
FCIDUMP=""
ADETFILE=""

# Dataset presets (now with absolute paths)
declare -A DATASETS_FCIDUMP=(
  ["fe4s4"]="$DATA_DIR/fe4s4/fcidump_Fe4S4.txt"
  ["n2"]="$DATA_DIR/n2/fcidump.txt"
  ["skqd"]="$DATA_DIR/skqd_paper/fcidump.txt"
)

declare -A DATASETS_ADET=(
  ["fe4s4"]="$DATA_DIR/fe4s4/AlphaDets.txt"
  ["n2"]="$DATA_DIR/n2/AlphaDets.txt"
  ["skqd"]="$DATA_DIR/skqd_paper/AlphaDets.txt"
)

# Expected runtimes for user guidance
declare -A EXPECTED_TIMES=(
  ["fe4s4"]="~10-15 seconds"
  ["n2"]="~85-120 seconds"
  ["skqd"]="~30-45 seconds"
)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
  --dataset)
    DATASET="$2"
    shift 2
    ;;
  --tasks)
    TASKS="$2"
    shift 2
    ;;
  --replicates)
    REPLICATES="$2"
    shift 2
    ;;
  --iterations)
    ITERATIONS="$2"
    shift 2
    ;;
  --threads)
    THREADS="$2"
    shift 2
    ;;
  --fcidump)
    FCIDUMP="$2" # User can provide relative or absolute
    DATASET="custom"
    shift 2
    ;;
  --adetfile)
    ADETFILE="$2" # User can provide relative or absolute
    shift 2
    ;;
  --output-dir)
    OUTPUT_DIR="$2"
    shift 2
    ;;
  --tag)
    TAG="$2"
    shift 2
    ;;
  --set-baseline)
    SET_BASELINE=1
    shift
    ;;
  --compare-baseline)
    COMPARE_BASELINE=1
    shift
    ;;
  --help)
    # Updated head count to include --tasks
    head -n 25 "$0" | tail -n +2 | sed 's/^# //'
    exit 0
    ;;
  *)
    echo "Unknown option: $1"
    echo "Use --help for usage information"
    exit 1
    ;;
  esac
done

# Set dataset files if not custom
if [[ "$DATASET" != "custom" ]]; then
  FCIDUMP="${DATASETS_FCIDUMP[$DATASET]}"
  ADETFILE="${DATASETS_ADET[$DATASET]}"
fi

# --- Validate files and build (no cd) ---
if [[ ! -f "$FCIDUMP" ]]; then
  echo "Error: FCIDUMP file not found: $FCIDUMP"
  echo "(Note: Custom paths are relative to where you ran the script)"
  exit 1
fi

if [[ ! -f "$ADETFILE" ]]; then
  echo "Error: Alpha determinants file not found: $ADETFILE"
  echo "(Note: Custom paths are relative to where you ran the script)"
  exit 1
fi

if [[ ! -f "$EXE_PATH" ]]; then
  echo "Error: $EXE_PATH binary not found. Building..."
  make -C "$SRC_DIR"
fi
# -----------------------------------------

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$BASELINE_DIR"

# Get system info
HOSTNAME=$(hostname)
CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
TOTAL_CORES=$(nproc)

# Get git info (use -C to run git from the project root)
COMMIT=$(git -C "$BASE_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
COMMIT_MSG=$(git -C "$BASE_DIR" log -1 --pretty=%B 2>/dev/null | head -1 || echo "unknown")

# Summary file (now an absolute path, includes task count)
SUMMARY_FILE="$OUTPUT_DIR/benchmark_${DATASET}_${TASKS}tasks_${TAG}.summary"

# Print configuration
echo "=== SBD Application Benchmark ===" | tee "$SUMMARY_FILE"
echo "Host: $HOSTNAME" | tee -a "$SUMMARY_FILE"
echo "CPU: $CPU_MODEL ($TOTAL_CORES cores)" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Configuration:" | tee -a "$SUMMARY_FILE"
echo "  Dataset: $DATASET ${EXPECTED_TIMES[$DATASET]:-}" | tee -a "$SUMMARY_FILE"
echo "  FCIDUMP: $FCIDUMP" | tee -a "$SUMMARY_FILE"
echo "  AlphaDETS: $ADETFILE" | tee -a "$SUMMARY_FILE"
echo "  MPI Tasks: $TASKS" | tee -a "$SUMMARY_FILE"
echo "  OMP Threads per Task: $THREADS" | tee -a "$SUMMARY_FILE"
echo "  Total Thread Parallelism: $((TASKS * THREADS))" | tee -a "$SUMMARY_FILE"
echo "  Replicates: $REPLICATES" | tee -a "$SUMMARY_FILE"
echo "  Iterations: $ITERATIONS" | tee -a "$SUMMARY_FILE"
echo "  Git commit: $COMMIT - $COMMIT_MSG" | tee -a "$SUMMARY_FILE"
echo "  Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Arrays to collect timing data
declare -a WALL_TIMES
declare -a DAVIDSON_TIMES
declare -a ENERGIES

# Run benchmark replicates
echo "=== Running Benchmarks ===" | tee -a "$SUMMARY_FILE"
# (Estimate logic is unchanged)
EXPECTED_STR_SPACED=${EXPECTED_TIMES[$DATASET]//[^0-9]/ }
EXPECTED_NUMS=($EXPECTED_STR_SPACED)
EXPECTED_TIME_PER_RUN=${EXPECTED_NUMS[0]}
TOTAL_EXPECTED=$((REPLICATES * EXPECTED_TIME_PER_RUN)) 2>/dev/null || TOTAL_EXPECTED=60

# --- No cd ../src here ---
for i in $(seq 1 $REPLICATES); do
  # LOGFILE path is now absolute (includes task count)
  LOGFILE="$OUTPUT_DIR/benchmark_${DATASET}_${TASKS}tasks_${TAG}_rep${i}.log"

  # tee paths are now correct
  echo "[$(date '+%H:%M:%S')] Running replicate $i/$REPLICATES..." | tee -a "$SUMMARY_FILE"

  # Run with time command for detailed metrics
  # Use absolute $EXE_PATH and variable $TASKS
  /usr/bin/time -v mpirun -np $TASKS --bind-to none -x OMP_NUM_THREADS=$THREADS "$EXE_PATH" \
    --fcidump "$FCIDUMP" \
    --adetfile "$ADETFILE" \
    --method 0 \
    --block 10 \
    --iteration $ITERATIONS \
    --tolerance 1.0e-4 \
    --adet_comm_size 1 \
    --bdet_comm_size 1 \
    --task_comm_size 1 \
    --init 0 \
    --shuffle 0 \
    --carryover_ratio 0.5 \
    --rdm 0 \
    2>&1 | tee "$LOGFILE"

  # Extract metrics from log
  WALL_TIME=$(grep "Elapsed (wall clock)" "$LOGFILE" | awk '{print $8}' | awk -F: '{if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2}')
  DAVIDSON_TIME=$(grep "Elapsed time for davidson" "$LOGFILE" | tail -1 | awk '{print $5}')
  ENERGY=$(grep " Energy = " "$LOGFILE" | tail -1 | awk '{print $5}')

  # Store results
  if [[ -n "$WALL_TIME" ]]; then
    WALL_TIMES+=("$WALL_TIME")
    echo "  Wall time: ${WALL_TIME}s" | tee -a "$SUMMARY_FILE"
  fi

  if [[ -n "$DAVIDSON_TIME" ]]; then
    DAVIDSON_TIMES+=("$DAVIDSON_TIME")
    echo "  Davidson time: ${DAVIDSON_TIME}s" | tee -a "$SUMMARY_FILE"
  fi

  if [[ -n "$ENERGY" ]]; then
    ENERGIES+=("$ENERGY")
    echo "  Energy: $ENERGY Ha" | tee -a "$SUMMARY_FILE"
  fi

  # Brief pause between replicates
  if [[ $i -lt $REPLICATES ]]; then
    sleep 2
  fi
done
# --- No cd ../benchmarks here ---

echo "" | tee -a "$SUMMARY_FILE"

# Calculate statistics
if [[ ${#WALL_TIMES[@]} -gt 0 ]]; then
  echo "=== Performance Statistics ===" | tee -a "$SUMMARY_FILE"

  # Find min, max, mean
  MIN_TIME=$(printf '%s\n' "${WALL_TIMES[@]}" | sort -n | head -1)
  MAX_TIME=$(printf '%s\n' "${WALL_TIMES[@]}" | sort -n | tail -1)

  # Calculate mean and stddev
  SUM=0
  for t in "${WALL_TIMES[@]}"; do
    SUM=$(echo "$SUM + $t" | bc)
  done
  MEAN=$(echo "scale=2; $SUM / ${#WALL_TIMES[@]}" | bc)

  # Calculate standard deviation
  SUM_SQ=0
  for t in "${WALL_TIMES[@]}"; do
    DIFF=$(echo "$t - $MEAN" | bc)
    SUM_SQ=$(echo "$SUM_SQ + $DIFF * $DIFF" | bc)
  done
  VARIANCE=$(echo "scale=4; $SUM_SQ / ${#WALL_TIMES[@]}" | bc)
  STDDEV=$(echo "scale=2; sqrt($VARIANCE)" | bc)
  # Avoid division by zero if MEAN is 0
  if (($(echo "$MEAN == 0" | bc -l))); then
    CV="N/A"
  else
    CV=$(echo "scale=2; 100 * $STDDEV / $MEAN" | bc)
  fi

  echo "Wall Time:" | tee -a "$SUMMARY_FILE"
  echo "  Fastest: ${MIN_TIME}s" | tee -a "$SUMMARY_FILE"
  echo "  Slowest: ${MAX_TIME}s" | tee -a "$SUMMARY_FILE"
  echo "  Mean: ${MEAN}s ± ${STDDEV}s (CV: ${CV}%)" | tee -a "$SUMMARY_FILE"

  # Check energy consistency
  if [[ ${#ENERGIES[@]} -gt 0 ]]; then
    UNIQUE_ENERGIES=$(printf '%s\n' "${ENERGIES[@]}" | sort -u | wc -l)
    if [[ $UNIQUE_ENERGIES -eq 1 ]]; then
      echo "Energy: ${ENERGIES[0]} Ha ✓ (consistent)" | tee -a "$SUMMARY_FILE"
    else
      echo "⚠ WARNING: Energy values differ across runs!" | tee -a "$SUMMARY_FILE"
      printf '%s\n' "${ENERGIES[@]}" | sort -u | tee -a "$SUMMARY_FILE"
    fi
  fi

  # Handle baseline operations
  BASELINE_FILE="$BASELINE_DIR/${DATASET}_${TASKS}tasks_baseline.txt"

  if [[ $SET_BASELINE -eq 1 ]]; then
    echo "" | tee -a "$SUMMARY_FILE"
    echo "=== Setting Baseline ===" | tee -a "$SUMMARY_FILE"
    cat >"$BASELINE_FILE" <<EOF
DATASET=$DATASET
TASKS=$TASKS
MIN_TIME=$MIN_TIME
MEAN_TIME=$MEAN
ENERGY=${ENERGIES[0]}
THREADS=$THREADS
COMMIT=$COMMIT
DATE=$(date '+%Y-%m-%d %H:%M:%S')
EOF
    echo "Baseline saved: $BASELINE_FILE" | tee -a "$SUMMARY_FILE"
  fi

  if [[ $COMPARE_BASELINE -eq 1 ]] && [[ -f "$BASELINE_FILE" ]]; then
    echo "" | tee -a "$SUMMARY_FILE"
    echo "=== Baseline Comparison ===" | tee -a "$SUMMARY_FILE"

    # Load baseline
    source "$BASELINE_FILE"
    BASELINE_MIN=$MIN_TIME
    BASELINE_MEAN=$MEAN_TIME

    # Calculate speedup
    SPEEDUP=$(echo "scale=2; $BASELINE_MIN / $MIN_TIME" | bc)
    MEAN_SPEEDUP=$(echo "scale=2; $BASELINE_MEAN / $MEAN" | bc)

    echo "Baseline ($TASKS tasks): ${BASELINE_MIN}s (best), ${BASELINE_MEAN}s (mean)" | tee -a "$SUMMARY_FILE"
    echo "Current ($TASKS tasks):  ${MIN_TIME}s (best), ${MEAN}s (mean)" | tee -a "$SUMMARY_FILE"
    echo "Speedup:  ${SPEEDUP}x (best), ${MEAN_SPEEDUP}x (mean)" | tee -a "$SUMMARY_FILE"
  fi
fi

echo "" | tee -a "$SUMMARY_FILE"
echo "=== Benchmark Complete ===" | tee -a "$SUMMARY_FILE"
echo "Results: $OUTPUT_DIR/" | tee -a "$SUMMARY_FILE"
echo "Summary: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"

# Run Python analyzer if available
if [[ -f "$ANALYZER_PY" ]]; then
  echo ""
  # Use the updated log file pattern
  "$ANALYZER_PY" $OUTPUT_DIR/benchmark_${DATASET}_${TASKS}tasks_${TAG}_rep*.log >>"$SUMMARY_FILE"
fi
