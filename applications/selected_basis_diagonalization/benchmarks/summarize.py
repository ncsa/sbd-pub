#!/usr/bin/env python3
"""
summarize_benchmark.py - Analyze and summarize benchmark results

Usage:
    ./summarize_benchmark.py [OPTIONS] LOGFILES...

Options:
    -h, --help              Show this help message
    --by-commit             Group results by commit hash
    --csv                   Output in CSV format
    --baseline COMMIT       Specify baseline commit for speedup analysis
    --filter-old HOURS      Filter out log files older than N hours (default: no filter)
    --output FILE           Write output to file instead of stdout
    --pattern PATTERN       Filename pattern for commit extraction (default: benchmark_([a-f0-9]{7})_)

Examples:
    # Analyze all recent logs
    ./summarize_benchmark.py run_replicate*.log

    # Group by commit with speedup analysis
    ./summarize_benchmark.py --by-commit --baseline 6535271 benchmark_*.log

    # Filter files from last 24 hours only
    ./summarize_benchmark.py --filter-old 24 run_replicate*.log

    # CSV output for automated processing
    ./summarize_benchmark.py --csv --by-commit benchmark_*.log > results.csv
"""

import argparse
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


def parse_time(time_str):
    """Parse time string in format 'h:mm:ss', 'm:ss', or 'ss.ss' to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:  # h:mm:ss
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:  # m:ss
        return float(parts[0]) * 60 + float(parts[1])
    return float(time_str)


def extract_data(logfile, commit_pattern):
    """Extract timing and energy data from benchmark log file."""
    data = {"logfile": str(logfile)}
    try:
        with open(logfile, "r") as f:
            content = f.read()

            # Extract wall clock time
            wall_match = re.search(
                r"Elapsed \(wall clock\) time.*?:\s+([0-9:.]+)", content
            )
            if wall_match:
                data["wall_time"] = parse_time(wall_match.group(1))

            # Extract energy
            energy_match = re.search(r"Energy = (-?[0-9.]+)", content)
            if energy_match:
                data["energy"] = float(energy_match.group(1))

            # Extract exit status
            exit_match = re.search(r"Exit status:\s+(\d+)", content)
            if exit_match:
                data["exit_status"] = int(exit_match.group(1))

            # Extract commit hash from filename using provided pattern
            filename = Path(logfile).name
            try:
                commit_match = re.search(commit_pattern, filename)
                if commit_match and commit_match.groups():
                    data["commit"] = commit_match.group(1)
                else:
                    # No commit in filename - use 'current' as placeholder
                    data["commit"] = "current"
            except (IndexError, AttributeError):
                data["commit"] = "current"

    except Exception as e:
        print(f"Error reading {logfile}: {e}", file=sys.stderr)
        return None

    return data if "wall_time" in data else None


def filter_by_age(logfiles, max_age_hours):
    """Filter log files by modification time."""
    if max_age_hours is None:
        return logfiles

    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    filtered = []
    for logfile in logfiles:
        path = Path(logfile)
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if mtime >= cutoff:
                filtered.append(logfile)
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and summarize benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("logfiles", nargs="+", help="Log files to analyze")
    parser.add_argument(
        "--by-commit", action="store_true", help="Group results by commit hash"
    )
    parser.add_argument("--csv", action="store_true", help="Output in CSV format")
    parser.add_argument(
        "--baseline", metavar="COMMIT", help="Baseline commit for speedup analysis"
    )
    parser.add_argument(
        "--filter-old",
        type=float,
        metavar="HOURS",
        help="Filter out files older than N hours",
    )
    parser.add_argument(
        "--output", metavar="FILE", help="Write output to file instead of stdout"
    )
    parser.add_argument(
        "--pattern",
        default=r"benchmark_([a-f0-9]{7})_",
        help="Regex pattern for extracting commit from filename",
    )

    args = parser.parse_args()

    # Filter files by age if requested
    logfiles = filter_by_age(args.logfiles, args.filter_old)

    if not logfiles:
        print("Error: No log files found (after filtering)", file=sys.stderr)
        sys.exit(1)

    # Parse all log files
    results = []
    for logfile in logfiles:
        data = extract_data(logfile, args.pattern)
        if data:
            results.append(data)

    if not results:
        print("Error: No valid benchmark data found", file=sys.stderr)
        sys.exit(1)

    # Set up output destination
    output = open(args.output, "w") if args.output else sys.stdout

    try:
        # Determine if we need commit-based analysis
        by_commit = args.by_commit or args.csv or args.baseline

        if by_commit:
            # Group by commit
            commits = defaultdict(list)
            for r in results:
                commits[r["commit"]].append(r)

            if args.csv:
                # CSV output
                print(
                    "commit,fastest_time_s,mean_time_s,std_dev_s,cv_percent,n_runs,energy,all_passed",
                    file=output,
                )
                for commit in sorted(commits.keys()):
                    runs = commits[commit]
                    times = [r["wall_time"] for r in runs]
                    energies = [r["energy"] for r in runs if "energy" in r]
                    exit_codes = [r.get("exit_status", 0) for r in runs]

                    fastest = min(times)
                    mean = statistics.mean(times)
                    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
                    cv = (stdev / mean) * 100
                    energy = energies[0] if energies else ""
                    all_passed = (
                        "yes" if all(code == 0 for code in exit_codes) else "no"
                    )

                    print(
                        f"{commit},{fastest:.2f},{mean:.2f},{stdev:.2f},{cv:.2f},{len(times)},{energy},{all_passed}",
                        file=output,
                    )
            else:
                # Human-readable output by commit
                print("=== Benchmark Summary (by commit) ===\n", file=output)
                for commit in sorted(commits.keys()):
                    runs = commits[commit]
                    times = [r["wall_time"] for r in runs]
                    energies = [r["energy"] for r in runs if "energy" in r]
                    exit_codes = [r.get("exit_status", 0) for r in runs]

                    fastest = min(times)
                    mean = statistics.mean(times)
                    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
                    cv = (stdev / mean) * 100

                    print(f"Commit {commit}:", file=output)
                    print(f"  Runs: {len(times)}", file=output)
                    print(
                        f"  Times: {', '.join(f'{t:.2f}s' for t in sorted(times))}",
                        file=output,
                    )
                    print(f"  Fastest: {fastest:.2f}s", file=output)
                    print(f"  Mean: {mean:.2f}s ± {stdev:.2f}s", file=output)
                    print(f"  Std dev: {stdev:.2f}s ({cv:.2f}% CV)", file=output)

                    if not all(code == 0 for code in exit_codes):
                        failed = sum(1 for code in exit_codes if code != 0)
                        print(
                            f"  EXIT STATUS: {failed}/{len(exit_codes)} runs FAILED",
                            file=output,
                        )

                    if energies:
                        energy_consistent = all(
                            abs(e - energies[0]) < 1e-10 for e in energies
                        )
                        if energy_consistent:
                            print(
                                f"  Energy: {energies[0]:.16f} Ha (consistent)",
                                file=output,
                            )
                        else:
                            print(
                                f"  Energy: {energies[0]:.16f} Ha (WARNING: inconsistent!)",
                                file=output,
                            )
                    print(file=output)

                # Speedup analysis
                if len(commits) > 1:
                    print("=== Speedup Analysis ===", file=output)

                    # Determine baseline
                    if args.baseline:
                        if args.baseline not in commits:
                            print(
                                f"Warning: Baseline commit {args.baseline} not found in results",
                                file=sys.stderr,
                            )
                            baseline_commit = sorted(commits.keys())[0]
                        else:
                            baseline_commit = args.baseline
                    else:
                        baseline_commit = sorted(commits.keys())[0]

                    baseline_time = min(
                        [r["wall_time"] for r in commits[baseline_commit]]
                    )

                    print(
                        f"Baseline ({baseline_commit}): {baseline_time:.2f}s",
                        file=output,
                    )
                    for commit in sorted(commits.keys()):
                        if commit == baseline_commit:
                            continue
                        commit_time = min([r["wall_time"] for r in commits[commit]])
                        speedup = ((baseline_time - commit_time) / baseline_time) * 100
                        print(
                            f"Commit {commit}: {commit_time:.2f}s ({speedup:+.1f}%)",
                            file=output,
                        )
        else:
            # Overall summary (no grouping)
            times = [r["wall_time"] for r in results]
            energies = [r["energy"] for r in results if "energy" in r]
            exit_codes = [r.get("exit_status", 0) for r in results]

            print("=== Benchmark Summary ===", file=output)
            print(f"Total runs: {len(times)}", file=output)
            print(
                f"Times: {', '.join(f'{t:.2f}s' for t in sorted(times))}", file=output
            )
            print(f"Fastest time: {min(times):.2f}s", file=output)
            print(
                f"Mean time: {statistics.mean(times):.2f}s ± {statistics.stdev(times) if len(times) > 1 else 0:.2f}s",
                file=output,
            )

            if not all(code == 0 for code in exit_codes):
                failed = sum(1 for code in exit_codes if code != 0)
                print(
                    f"EXIT STATUS: {failed}/{len(exit_codes)} runs FAILED", file=output
                )

            if energies:
                energy_consistent = all(abs(e - energies[0]) < 1e-10 for e in energies)
                if energy_consistent:
                    print(f"Energy: {energies[0]:.16f} Ha (consistent)", file=output)
                else:
                    print(f"Energy: varied across runs (WARNING!)", file=output)

    finally:
        if args.output:
            output.close()


if __name__ == "__main__":
    main()
