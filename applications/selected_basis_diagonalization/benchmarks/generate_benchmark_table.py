#!/usr/bin/env python3
"""
Generate markdown table from benchmark log files.

Usage: ./generate_benchmark_table.py [--no-commands] <directory>

Parses all .out files in the specified directory and generates a markdown
table with benchmark results.

Options:
  --no-commands    Omit full run command values from table (column remains empty)
"""

import sys
import os
import re
import argparse
from pathlib import Path
from datetime import datetime


def parse_log_file(log_path):
    """Parse a single log file and extract benchmark information."""
    with open(log_path, 'r') as f:
        content = f.read()

    data = {}

    # Extract date (first line: ISO 8601 format)
    date_match = re.search(r'^(\d{4}-\d{2}-\d{2})T', content, re.MULTILINE)
    if date_match:
        data['date'] = date_match.group(1)
    else:
        data['date'] = ''

    # Extract srun parameters
    srun_match = re.search(
        r'srun --nodes=(\d+) --tasks-per-node=(\d+) --cpus-per-task=(\d+) --gpus-per-task=(\d+)',
        content
    )
    if srun_match:
        data['nodes'] = int(srun_match.group(1))
        data['ranks_per_node'] = int(srun_match.group(2))
        data['threads_per_rank'] = int(srun_match.group(3))
        data['gpu_per_rank'] = int(srun_match.group(4))
    else:
        data['nodes'] = ''
        data['ranks_per_node'] = ''
        data['threads_per_rank'] = ''
        data['gpu_per_rank'] = ''

    # Extract communicator sizes
    adet_match = re.search(r'--adet_comm_size (\d+)', content)
    bdet_match = re.search(r'--bdet_comm_size (\d+)', content)
    task_match = re.search(r'--task_comm_size (\d+)', content)

    if adet_match and bdet_match and task_match:
        A = int(adet_match.group(1))
        B = int(bdet_match.group(1))
        T = int(task_match.group(1))

        # Calculate R (h_comm_size)
        total_ranks = data['nodes'] * data['ranks_per_node'] if data['nodes'] and data['ranks_per_node'] else 0
        if total_ranks > 0:
            R = total_ranks // (A * B * T)
            # Verify it's an integer division
            if total_ranks != R * A * B * T:
                print(f"WARNING: {log_path.name}: total_ranks={total_ranks} is not divisible by A*B*T={A*B*T}", file=sys.stderr)
                R = '?'
        else:
            R = ''

        data['comm_shape'] = f"a={A}, b={B}, t={T}, r={R}"
        data['A'] = A
        data['B'] = B
        data['T'] = T
        data['R'] = R
    else:
        data['comm_shape'] = ''


    # Extract final energy
    final_energy_match = re.search(r'  Energy = ([0-9.+-]+)', content)
    if final_energy_match:
        data['final_energy'] = float(final_energy_match.group(1))
    
    # Extract Davidson time
    davidson_match = re.search(r'Elapsed time for davidson ([\d.]+) \(sec\)', content)
    if davidson_match:
        data['davidson_time'] = float(davidson_match.group(1))
    else:
        data['davidson_time'] = ''

    # Extract wall time (at end of log)
    wall_match = re.search(r'Wall time elapsed: ([\d.]+)', content)
    if wall_match:
        data['wall_time'] = float(wall_match.group(1))
    else:
        data['wall_time'] = ''

    # Extract full run command (line after "Running this:")
    cmd_match = re.search(r'Running this:\n(.+)', content)
    if cmd_match:
        data['command'] = cmd_match.group(1).strip()
    else:
        data['command'] = ''

    # Extract dataset name from --adetfile argument
    adetfile_match = re.search(r'--adetfile\s+(\S+)', content)
    if adetfile_match:
        adetfile_path = adetfile_match.group(1)
        # Get basename and remove .txt extension
        dataset_name = os.path.basename(adetfile_path)
        if dataset_name.endswith('.txt'):
            dataset_name = dataset_name[:-4]
        data['dataset'] = dataset_name
    else:
        data['dataset'] = ''

    return data


def parse_directory(directory):
    """Parse all log files in a directory and return list of results."""
    # Find all .out files
    log_dir = Path(directory)
    log_files = sorted(log_dir.glob('*.out'))

    if not log_files:
        print(f"No .out files found in {directory}", file=sys.stderr)
        return []

    # Parse all log files
    results = []
    for log_file in log_files:
        try:
            data = parse_log_file(log_file)
            # Dataset is now extracted from --adetfile in parse_log_file
            results.append(data)
        except Exception as e:
            print(f"Error parsing {log_file.name}: {e}", file=sys.stderr)
            continue

    return results


def print_table_rows(results, include_commands=True):
    """Print markdown table rows for given results."""
    developer_name = "Juha Jäykkä"  # With ä!

    for data in results:
        date = data.get('date', '')
        dataset = data.get('dataset', '')
        nodes = str(data.get('nodes', ''))
        ranks_per_node = str(data.get('ranks_per_node', ''))
        threads_per_rank = str(data.get('threads_per_rank', ''))
        gpu_per_rank = str(data.get('gpu_per_rank', ''))
        comm_shape = data.get('comm_shape', '')
        wall_time = f"{data['wall_time']:.1f}" if data.get('wall_time') else ''
        final_energy = f"{data['final_energy']:.16f}" if data.get('final_energy') else "unavail"
        davidson_time = f"{data['davidson_time']:.2f}" if data.get('davidson_time') else ''
        git_hash = ''  # To be filled in manually
        command = data.get('command', '') if include_commands else ''
        comments = ''

        # Pad fields for alignment
        date_pad = date.ljust(10)
        dataset_pad = dataset.ljust(18)
        nodes_pad = nodes.rjust(5)
        ranks_pad = ranks_per_node.rjust(10)
        threads_pad = threads_per_rank.rjust(12)
        gpu_pad = gpu_per_rank.rjust(8)
        comm_pad = comm_shape.rjust(20)
        wall_pad = wall_time.rjust(13)
        final_energy_pad = final_energy.rjust(20)
        davidson_pad = davidson_time.rjust(17)
        git_pad = git_hash.ljust(8)
        dev_pad = developer_name.ljust(14)

        print(f"| {date_pad} | {dataset_pad} | {nodes_pad} | {ranks_pad} | {threads_pad} | {gpu_pad} | {comm_pad} | {wall_pad} | {davidson_pad} | {git_pad} | {dev_pad} | {final_energy} | {command} | {comments} |")


def main():
    parser = argparse.ArgumentParser(
        description='Generate markdown table from benchmark log files.',
        epilog='Example: ./generate_benchmark_table.py scan_logs/h2o/h2o-1em8-alpha.txt'
    )
    parser.add_argument('directory', nargs='+', help='Directory(ies) containing .out log files')
    parser.add_argument('--no-commands', action='store_true',
                        help='Omit full run command values from table (column remains empty)')

    args = parser.parse_args()

    # Validate all directories exist
    for directory in args.directory:
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a directory", file=sys.stderr)
            sys.exit(1)

    # Print header once
    print("| Date       | Dataset            | Nodes | Ranks/Node | Threads/Rank | GPU/Rank | Comm shape           | Wall Time (s) | Davidson Time (s) | git hash | Developer name | Final Energy       | Full run command | Comments |")
    print("|------------|--------------------| ------|-----------:|-------------:|---------:|---------------------:|--------------:|------------------:|----------|----------------|--------------------|------------------|----------|")

    # Process directories in command-line order
    found_any = False
    for directory in args.directory:
        results = parse_directory(directory)

        if not results:
            continue

        found_any = True

        # Sort results within this directory by wall time only
        def sort_key(x):
            wall_time = x.get('wall_time', '')
            if wall_time == '' or wall_time is None:
                wall_time = float('inf')
            return wall_time

        results.sort(key=sort_key)

        # Print rows for this directory
        print_table_rows(results, include_commands=not args.no_commands)

    if not found_any:
        print("No valid log files parsed from any directory", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
