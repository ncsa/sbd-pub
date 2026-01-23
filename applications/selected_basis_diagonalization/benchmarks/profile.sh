#!/usr/bin/env bash
#
# profile.sh - SBD Application Profiling Script
#
# Usage:
#     ./profile.sh [OPTIONS]
#
# Options:
#     --dataset NAME       Dataset to profile (fe4s4, n2, skqd)
#     --tasks N            Number of MPI tasks (default: 1)
#     --iterations N       Davidson iterations (default: 1 for profiling)
#     --threads N          OpenMP threads (default: 64)
#     --tool TOOL          Profiling tool (rocprof, perf, time) (default: rocprof)
#     --output-dir DIR     Output directory (default: ./profile_results)
#     --analyze            Run analysis after profiling (if available)
#     --help               Show this help message

set -e

# --- Define Absolute Base Paths ---
# Get the directory this script lives in
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# Get the project root directory (one level up)
BASE_DIR=$(dirname "$SCRIPT_DIR")

SRC_DIR="$BASE_DIR/src"
DATA_DIR="$BASE_DIR/data"
BENCHMARK_DIR="$BASE_DIR/benchmarks"
EXE_PATH="$SRC_DIR/diag"
ANALYSIS_SCRIPT="$BENCHMARK_DIR/show-profile-tree.py"
# ------------------------------------

# Default configuration
DATASET="fe4s4"
ITERATIONS=1 # Single iteration for faster profiling
TASKS=1      # Default to 1 MPI task
THREADS=64
TOOL="rocprof"
# Default output dir is now absolute
OUTPUT_DIR="$BENCHMARK_DIR/profile_results"
ANALYZE=0

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

# Parse arguments
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
  --iterations)
    ITERATIONS="$2"
    shift 2
    ;;
  --threads)
    THREADS="$2"
    shift 2
    ;;
  --tool)
    TOOL="$2"
    shift 2
    ;;
  --output-dir)
    OUTPUT_DIR="$2"
    shift 2
    ;;
  --analyze)
    ANALYZE=1
    shift
    ;;
  --help)
    # Updated head count to show new --tasks option
    head -n 15 "$0" | tail -n +2 | sed 's/^# //'
    exit 0
    ;;
  *)
    echo "Unknown option: $1"
    exit 1
    ;;
  esac
done

# Validate dataset
if [[ -z "${DATASETS_FCIDUMP[$DATASET]}" ]]; then
  echo "Error: Unknown dataset '$DATASET'. Available: fe4s4, n2, skqd"
  exit 1
fi

FCIDUMP="${DATASETS_FCIDUMP[$DATASET]}"
ADETFILE="${DATASETS_ADET[$DATASET]}"

# --- Validate files and build (no cd) ---
if [[ ! -f "$FCIDUMP" ]]; then
  echo "Error: FCIDUMP not found: $FCIDUMP"
  exit 1
fi

if [[ ! -f "$ADETFILE" ]]; then
  echo "Error: AlphaDets not found: $ADETFILE"
  exit 1
fi

if [[ ! -f "$EXE_PATH" ]]; then
  echo "Building application..."
  # Use make -C to build in the src directory without cd'ing
  make -C "$SRC_DIR"
fi
# -----------------------------------------

# Create output directory
PROFILE_DIR="$OUTPUT_DIR/${DATASET}_${TASKS}tasks_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PROFILE_DIR"

# --- Split commands for flexible profiler wrapping ---
MPI_CMD="mpirun -np $TASKS --bind-to none -x OMP_NUM_THREADS=$THREADS"
APP_CMD="$EXE_PATH \
    --fcidump $FCIDUMP \
    --adetfile $ADETFILE \
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
    --rdm 0"

echo "=== SBD Application Profiling ==="
echo "Dataset: $DATASET"
# Note: Tool will be updated below if 'rocprof' is selected
echo "Tool: $TOOL"
echo "MPI Tasks: $TASKS"
echo "OMP Threads: $THREADS"
echo "Iterations: $ITERATIONS (profiling mode)"
echo "Output: $PROFILE_DIR"
echo ""

# --- Profiler Detection Logic ---
# If tool is rocprof, find the specific binary to use
if [[ "$TOOL" == "rocprof" ]]; then
  PROFILER_CMD=""
  # List of profilers to try, in your specified order of preference
  PROFILERS_TO_TRY=("rocsys" "rocprof-sys-run" "rocprofv3" "rocprofv2" "rocprof")

  for cmd in "${PROFILERS_TO_TRY[@]}"; do
    if command -v "$cmd" &>/dev/null; then
      PROFILER_CMD=$(command -v "$cmd")
      break
    fi
  done

  # If no profiler was found, error out with a helpful message
  if [[ -z "$PROFILER_CMD" ]]; then
    echo "Error: Could not find a ROCm profiler for --tool=rocprof."
    echo "Tried to find: ${PROFILERS_TO_TRY[*]}"
    echo "Please ensure a rocm module is loaded and a profiler is in your PATH."
    exit 1
  fi

  # Store the name (rocsys, rocprof, etc.) for the analysis step
  PROFILER_NAME=$(basename "$PROFILER_CMD")

  # If using rocsys, we also need to find the collector (rocprofv3/v2)
  if [[ "$PROFILER_NAME" == "rocsys" ]]; then
    if command -v "rocprofv3" &>/dev/null; then
      ROCSYS_COLLECTOR="rocprofv3"
    elif command -v "rocprofv2" &>/dev/null; then
      ROCSYS_COLLECTOR="rocprofv2"
    else
      echo "Warning: 'rocsys' found, but 'rocprofv3' or 'rocprofv2' not found. Profiling may fail."
      # Fallback to just rocprofv2 as a string, as in the user's help output
      ROCSYS_COLLECTOR="rocprofv2"
    fi
  fi

  echo "Tool: rocprof (resolved to $PROFILER_NAME)"
fi
# ---------------------------------

# --- Run profiling ---
# This is the *only* cd. We cd into the profile dir to dump logs.
cd "$PROFILE_DIR"

# Add debug print as requested
echo "--- [DEBUG] Running command: ---"
set -x

case $TOOL in
rocprof)
  echo "Running with $PROFILER_NAME..."

  # Execute the correct command based on which profiler was found
  # Every case is now wrapped in /usr/bin/time -v to capture stats
  case "$PROFILER_NAME" in
  rocsys)
    # rocsys launches mpirun, which launches the collector, which launches the app
    echo "Using 'rocsys' to launch '$ROCSYS_COLLECTOR'..."
    SESSION_NAME=$(basename "$PROFILE_DIR")

    # Define collector arguments based on version
    if [[ "$ROCSYS_COLLECTOR" == "rocprofv3" ]]; then
      # v3 needs a trace flag + stats. -r is a good default.
      ROCSYS_COLLECTOR_ARGS="-r --stats --output-format csv -o profile.csv"
    else
      # v2/v1 can use --stats --timestamp
      ROCSYS_COLLECTOR_ARGS="--stats --timestamp on -o profile.csv"
    fi

    /usr/bin/time -v rocsys --session "$SESSION_NAME" launch \
      $MPI_CMD $ROCSYS_COLLECTOR $ROCSYS_COLLECTOR_ARGS -- $APP_CMD 2>&1 | tee profile.log
    ;;

  rocprof-sys-run)
    # This tool wraps the mpirun command
    echo "Using legacy 'rocprof-sys-run'."
    /usr/bin/time -v rocprof-sys-run \
      --use-sampling --frequency 100 \
      --trace-thread-state \
      --use-mpip \
      --output $(pwd) \
      --label "${DATASET}_profile" \
      -- $MPI_CMD $APP_CMD 2>&1 | tee profile.log
    ;;

  rocprofv3)
    # v3 needs a trace flag + stats. -r is a good default.
    echo "Using standard '$PROFILER_NAME' for summary stats."
    /usr/bin/time -v $MPI_CMD \
      $PROFILER_NAME -r --stats --output-format csv -o "profile.csv" -- \
      $APP_CMD 2>&1 | tee profile.log
    ;;

  rocprofv2 | rocprof)
    # v2 and v1 are assumed to support --timestamp and default stats
    echo "Using standard '$PROFILER_NAME' for summary stats."
    /usr/bin/time -v $MPI_CMD \
      $PROFILER_NAME --stats --timestamp on -o "profile.csv" -- \
      $APP_CMD 2>&1 | tee profile.log
    ;;
  esac
  ;;

perf)
  # perf wraps mpirun
  echo "Running with perf..."
  /usr/bin/time -v perf record -F 99 -a -g -o perf.data -- \
    $MPI_CMD $APP_CMD 2>&1 | tee profile.log
  perf report --stdio >perf_report.txt
  ;;

time | *)
  echo "Running with /usr/bin/time..."
  /usr/bin/time -v $MPI_CMD $APP_CMD 2>&1 | tee profile.log
  ;;
esac
# -----------------------

# Stop debug printing
set +x
echo "--- [DEBUG] Command finished ---"

# Extract basic metrics
echo ""
echo "=== Profile Summary ==="

WALL_TIME=$(grep "Elapsed (wall clock)" profile.log 2>/dev/null | awk '{print $8}' || echo "N/A")
if [[ "$WALL_TIME" != "N/A" ]]; then
  echo "Wall time: $WALL_TIME"
fi

DAVIDSON_TIME=$(grep "Elapsed time for davidson" profile.log 2>/dev/null | tail -1 | awk '{print $5}' || echo "N/A")
if [[ "$DAVIDSON_TIME" != "N/A" ]]; then
  echo "Davidson time: ${DAVIDSON_TIME}s"
fi

MAX_RSS=$(grep "Maximum resident set size" profile.log 2>/dev/null | awk '{print $6}' || echo "N/A")
if [[ "$MAX_RSS" != "N/A" ]]; then
  echo "Peak memory: $((MAX_RSS / 1024)) MB"
fi

# Run analysis if requested
if [[ $ANALYZE -eq 1 ]]; then
  echo ""
  echo "=== Running Analysis ==="

  # Determine the input file for the analysis script
  ANALYSIS_INPUT_FILE=""
  case $TOOL in
  rocprof)
    case "$PROFILER_NAME" in
    rocsys | rocprofv3 | rocprofv2 | rocprof)
      # Modern tools produce CSV files. Prefer kernel stats.
      if [[ -f "profile.csv_kernel_stats.csv" ]]; then
        ANALYSIS_INPUT_FILE="profile.csv_kernel_stats.csv"
      elif [[ -f "profile.csv" ]]; then
        # Fallback if only the base CSV exists (older rocprof?)
        ANALYSIS_INPUT_FILE="profile.csv"
      fi
      ;;
    rocprof-sys-run)
      # Old tool produces JSON
      if [[ -f "sampling_wall_clock-0.json" ]]; then
        ANALYSIS_INPUT_FILE="sampling_wall_clock-0.json"
      fi
      ;;
    esac
    ;;
  perf)
    # Perf analysis is handled directly below, not via python script
    ;;
  time | *)
    # No analysis for 'time' tool
    ;;
  esac

  # Run the Python analysis script if an input file was found
  if [[ -f "$ANALYSIS_SCRIPT" ]] && [[ -n "$ANALYSIS_INPUT_FILE" ]] && [[ -f "$ANALYSIS_INPUT_FILE" ]]; then
    echo "Analyzing $ANALYSIS_INPUT_FILE..."
    # Always use --flat --times for CSV, allow user choice for JSON
    ANALYSIS_FLAGS="--times"
    if [[ "$ANALYSIS_INPUT_FILE" == *.csv ]]; then
      ANALYSIS_FLAGS="--flat --times"
    fi

    python3 "$ANALYSIS_SCRIPT" "$ANALYSIS_INPUT_FILE" $ANALYSIS_FLAGS >analysis.txt

    echo "Analysis saved to: $PROFILE_DIR/analysis.txt"
    echo ""
    echo "Top functions:"
    # Use head -n based on format (flat list is usually longer)
    if [[ "$ANALYSIS_INPUT_FILE" == *.csv ]]; then
      head -30 analysis.txt
    else
      head -20 analysis.txt # Keep original head for JSON tree
    fi
  elif [[ "$TOOL" == "rocprof" ]]; then
    echo "Analysis script or suitable profile output file not found."
  fi

  # Handle perf analysis separately
  if [[ "$TOOL" == "perf" ]] && [[ -f "perf_report.txt" ]]; then
    echo "Top functions (from perf):"
    head -30 perf_report.txt | grep -A 20 "Overhead"
  fi

fi # End if [[ $ANALYZE -eq 1 ]]

echo ""
echo "Profile complete. Results in: $PROFILE_DIR"
