#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
import re
import sys


def calculate_max_width(vertex, indent="", collapse_patterns=None):
    """
    Recursively calculate the maximum width of the JSON tree structure.
    (Only used for JSON input)
    """
    node = vertex["node"]
    function_name = node["prefix"]
    truncated_name = function_name[:20]  # Keep truncation consistent

    if indent == "":
        current_width = len(truncated_name)
    else:
        current_width = len(f"{indent}|_{truncated_name}")

    max_width = current_width

    # Check if this node should be collapsed (children hidden)
    should_collapse = False
    if collapse_patterns:
        for pattern in collapse_patterns:
            if re.search(pattern, function_name):
                should_collapse = True
                break

    children = vertex["children"]
    if children and not should_collapse:
        child_indent = indent + "  "
        for child in children:
            child_width = calculate_max_width(child, child_indent, collapse_patterns)
            max_width = max(max_width, child_width)
    return max_width


def print_tree(vertex, indent="", show_times=False, max_width=0, collapse_patterns=None):
    """
    Recursively print the JSON profile tree structure.
    (Only used for JSON input)
    """
    node = vertex["node"]
    function_name = node["prefix"]
    truncated_name = function_name[:20]  # Keep truncation consistent

    if show_times:
        inclusive_time = node["inclusive"]["entry"]["value"]
        exclusive_time = node["exclusive"]["entry"]["value"]
        if indent == "":
            tree_part = truncated_name
        else:
            tree_part = f"{indent}|_{truncated_name}"
        print(
            f"{tree_part:<{max_width}}  {inclusive_time:10.4f}  {exclusive_time:10.4f}"
        )
    else:
        if indent == "":
            print(function_name)
        else:
            print(f"{indent}|_{function_name}")

    # Check if this node should be collapsed (children hidden)
    should_collapse = False
    if collapse_patterns:
        for pattern in collapse_patterns:
            if re.search(pattern, function_name):
                should_collapse = True
                break

    children = vertex["children"]
    if children and not should_collapse:
        child_indent = indent + "  "
        for child in children:
            print_tree(child, child_indent, show_times, max_width, collapse_patterns)


def collect_all_function_calls_json(vertex, calls=None, collapse_patterns=None):
    """
    Recursively collect function calls from JSON data.
    (Only used for JSON input)
    """
    if calls is None:
        calls = []
    node = vertex["node"]
    full_function_name = node["prefix"]
    tid = tuple(node["tid"])
    inclusive_time = node["inclusive"]["entry"]["value"]
    exclusive_time = node["exclusive"]["entry"]["value"]
    inc_stddev = node["inclusive"]["stats"].get("stddev", 0.0)  # Use .get for safety
    exc_stddev = node["exclusive"]["stats"].get("stddev", 0.0)  # Use .get for safety

    calls.append(
        (
            full_function_name,
            tid,
            inclusive_time,
            exclusive_time,
            inc_stddev,
            exc_stddev,
        )
    )

    # Check if this node should be collapsed (children excluded from collection)
    should_collapse = False
    if collapse_patterns:
        for pattern in collapse_patterns:
            if re.search(pattern, full_function_name):
                should_collapse = True
                break

    # Only recurse into children if not collapsed
    if not should_collapse:
        for child in vertex["children"]:
            collect_all_function_calls_json(child, calls, collapse_patterns)
    return calls


def parse_rocprof_csv(filename):
    """
    Parse the rocprof CSV file (e.g., _kernel_stats.csv) to extract flat function calls.
    Returns data suitable for the merge_function_calls_flat function.
    """
    calls = []
    try:
        with open(filename, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            # Try to determine column names flexibly
            name_col = next(
                (k for k in reader.fieldnames if "Name" in k), None
            )  # KernelName or FunctionName
            dur_col = next(
                (
                    k
                    for k in reader.fieldnames
                    if "DurationNs" in k or "TotalDuration" in k
                ),
                None,
            )  # Duration
            calls_col = next(
                (k for k in reader.fieldnames if "Calls" in k or "Count" in k), None
            )  # Number of calls

            if not name_col or not dur_col:
                print(
                    f"Error: Could not find required columns ('Name', 'Duration') in CSV: {filename}",
                    file=sys.stderr,
                )
                return []

            for row in reader:
                try:
                    full_name = row[name_col]
                    # Duration is usually in ns, convert to seconds
                    # CSV provides total duration for all calls, treat as exclusive time here
                    duration_sec = float(row[dur_col]) / 1e9
                    # We don't have inclusive time or stddev from basic kernel stats CSV
                    inclusive_time = duration_sec  # Best guess: inclusive = exclusive for kernel-only stats
                    exclusive_time = duration_sec
                    inc_stddev = 0.0
                    exc_stddev = 0.0
                    tid = (0,)  # CSV doesn't provide thread ID, use placeholder

                    # If Calls column exists, duration is per-call avg, multiply back
                    # This might need adjustment based on exact CSV content
                    if calls_col and row.get(calls_col):
                        num_calls = int(row[calls_col])
                        if num_calls > 1:
                            # Assuming Duration is average, recalculate total
                            total_duration_sec = duration_sec * num_calls
                            inclusive_time = total_duration_sec
                            exclusive_time = total_duration_sec
                            # Store the average time in stddev field for now? Or just 0.
                            # exc_stddev = duration_sec # Not really stddev, but avg duration

                    calls.append(
                        (
                            full_name,
                            tid,
                            inclusive_time,
                            exclusive_time,
                            inc_stddev,
                            exc_stddev,
                        )
                    )
                except (ValueError, KeyError, TypeError) as e:
                    print(
                        f"Warning: Skipping row due to parsing error: {row}. Error: {e}",
                        file=sys.stderr,
                    )
                    continue
    except FileNotFoundError:
        print(f"Error: CSV file not found: {filename}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error reading CSV file {filename}: {e}", file=sys.stderr)
        return []

    return calls


def merge_function_calls_flat(calls):
    """
    Merge function calls by (full_name, thread) and return sorted by exclusive time.
    (Works for both JSON and CSV derived call lists)
    """
    merged = {}
    for full_name, tid, inclusive, exclusive, inc_stddev, exc_stddev in calls:
        # For CSV, tid is always (0,), so effectively group by name
        key = (
            full_name,
            tid if len(tid) > 1 else full_name,
        )  # Group by name if single 'thread'
        if key not in merged:
            merged[key] = {
                "inclusive": 0.0,
                "exclusive": 0.0,
                "inc_stddev": 0.0,
                "exc_stddev": 0.0,
                "num_threads": 0,
                "tids": set(),
            }
        merged[key]["inclusive"] += inclusive
        merged[key]["exclusive"] += exclusive
        merged[key]["inc_stddev"] = max(merged[key]["inc_stddev"], inc_stddev)
        merged[key]["exc_stddev"] = max(merged[key]["exc_stddev"], exc_stddev)
        merged[key]["tids"].add(tid)  # Track unique TIDs merged

    result = []
    for (full_name, _), times in merged.items():
        # Correctly calculate num_threads based on unique thread IDs
        # TID can be a tuple like (0,) for single thread or (3,4,5,6,7,8,9) for multiple threads
        # We need to flatten all TID tuples and count unique thread IDs
        all_thread_ids = set()
        for tid_tuple in times["tids"]:
            all_thread_ids.update(tid_tuple)

        # If all TIDs are (0,), it's likely a single-threaded placeholder
        num_threads = len(all_thread_ids) if all_thread_ids != {0} else 1

        result.append(
            (
                full_name,
                times["tids"],  # Store the set of TIDs
                times["inclusive"],
                times["exclusive"],
                times["inc_stddev"],
                times["exc_stddev"],
                num_threads,
            )
        )

    result.sort(key=lambda x: x[3], reverse=True)  # Sort by total exclusive time
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse and display rocprof profile data (JSON or CSV)"
    )
    parser.add_argument("filename", help="Profile JSON or CSV file to parse")
    parser.add_argument(
        "--times",
        action="store_true",
        help="Display timing information (tree for JSON, flat for CSV)",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Display flat list of functions sorted by exclusive time (required for CSV)",
    )
    parser.add_argument(
        "--collapse",
        action="append",
        metavar="REGEX",
        help="Regex pattern for functions to collapse (hide children). Can be specified multiple times.",
    )
    # --compiler-functions and pruning are removed as they don't apply well to CSV
    args = parser.parse_args()

    # Compile collapse patterns
    collapse_patterns = []
    if args.collapse:
        for pattern in args.collapse:
            try:
                collapse_patterns.append(pattern)
            except re.error as e:
                print(f"Error: Invalid regex pattern '{pattern}': {e}", file=sys.stderr)
                sys.exit(1)

    file_path = Path(args.filename)
    file_type = file_path.suffix.lower()

    all_calls = []

    if file_type == ".json":
        try:
            with open(args.filename, "r") as f:
                data = json.load(f)
            # Adjust navigation based on potential JSON structures
            # timemory will have one child, but different profiles use different key names
            graph = []
            if "timemory" in data:
                timemory = data["timemory"]
                # Find the first child key that has a "graph" field
                for key in timemory:
                    if isinstance(timemory[key], dict) and "graph" in timemory[key]:
                        graph = timemory[key].get("graph", [])
                        break
            # Fallback: check for graph at top level
            if not graph and isinstance(data.get("graph"), list):
                graph = data["graph"]

            if graph and isinstance(graph, list) and len(graph) > 0:
                root_vertices = graph[0]  # Assume first process/thread group

                if args.flat:
                    # Collect calls for flat view (respecting collapse patterns)
                    for vertex in root_vertices:
                        all_calls.extend(collect_all_function_calls_json(vertex, collapse_patterns=collapse_patterns))
                elif args.times:
                    # Calculate max width for tree view
                    max_width = 0
                    for vertex in root_vertices:
                        width = calculate_max_width(vertex, collapse_patterns=collapse_patterns)
                        max_width = max(max_width, width)
                    max_width += 1
                    # Print tree header
                    print(
                        f"{'Function Name':<{max_width}}  {'Inclusive':>10}  {'Exclusive':>10}"
                    )
                    print(f"{'-'*max_width}  {'-'*10}  {'-'*10}")
                    # Print tree
                    for vertex in root_vertices:
                        print_tree(vertex, show_times=True, max_width=max_width, collapse_patterns=collapse_patterns)
                else:
                    # Default: Print basic tree (names only)
                    for vertex in root_vertices:
                        print_tree(vertex, collapse_patterns=collapse_patterns)
            else:
                print(
                    f"Error: Could not find expected graph structure in JSON file: {args.filename}",
                    file=sys.stderr,
                )
                sys.exit(1)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file {args.filename}: {e}", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError:
            print(f"Error: JSON file not found: {args.filename}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error processing JSON file {args.filename}: {e}", file=sys.stderr)
            sys.exit(1)

    elif file_type == ".csv":
        if not args.flat and args.times:
            print(
                "Warning: Tree view (--times without --flat) is not supported for CSV input. Showing flat list.",
                file=sys.stderr,
            )
            args.flat = True  # Force flat view for CSV
        elif not args.times:
            # If neither --times nor --flat is given, default to flat for CSV
            args.flat = True

        if args.flat:
            all_calls = parse_rocprof_csv(args.filename)
        else:
            print(
                "Error: CSV input requires --flat flag to display data.",
                file=sys.stderr,
            )
            sys.exit(1)

    else:
        print(
            f"Error: Unrecognized file type: {args.filename}. Expecting .json or .csv",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Print Flat List (if requested and calls were collected) ---
    if args.flat and all_calls:
        merged_calls = merge_function_calls_flat(all_calls)

        calls_with_per_thread = []
        for (
            full_name,
            tids,
            inclusive,
            exclusive,
            inc_stddev,
            exc_stddev,
            num_threads,
        ) in merged_calls:
            inclusive_per_thread = (
                inclusive / num_threads if num_threads > 0 else inclusive
            )
            exclusive_per_thread = (
                exclusive / num_threads if num_threads > 0 else exclusive
            )
            calls_with_per_thread.append(
                (
                    full_name,
                    tids,
                    inclusive,
                    exclusive,
                    inc_stddev,
                    exc_stddev,
                    num_threads,
                    inclusive_per_thread,
                    exclusive_per_thread,
                )
            )

        # Sort by exclusive per thread (descending)
        calls_with_per_thread.sort(key=lambda x: x[8], reverse=True)

        max_name_width = 80  # Max width for function names in flat view

        # Print flat header (Exclusive first)
        print(
            f"{'Function Name':<{max_name_width}}  {'#Thr':>4}  {'Excl/Thr':>10}  {'±StdDev':>10}  {'Incl/Thr':>10}  {'±StdDev':>10}"
        )
        print(f"{'-'*max_name_width}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        # Print flat list
        for (
            full_name,
            tids,
            inclusive,
            exclusive,
            inc_stddev,
            exc_stddev,
            num_threads,
            inclusive_per_thread,
            exclusive_per_thread,
        ) in calls_with_per_thread:
            display_name = full_name[:max_name_width]
            print(
                f"{display_name:<{max_name_width}}  {num_threads:4d}  {exclusive_per_thread:10.4f}  {exc_stddev:10.4f}  {inclusive_per_thread:10.4f}  {inc_stddev:10.4f}"
            )

    elif args.flat and not all_calls:
        print("No function calls found to display in flat list.", file=sys.stderr)
