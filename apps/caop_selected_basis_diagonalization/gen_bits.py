#!/usr/bin/env python3
import argparse
import random
import sys

def method_0_random(bit_length: int, num_ones: int, num_bitstrings: int, rng: random.Random):
    """
    Method 0:
    Random generation of bitstrings with fixed Hamming weight (U(1) symmetry).
    """
    base_bits = [1] * num_ones + [0] * (bit_length - num_ones)
    for _ in range(num_bitstrings):
        rng.shuffle(base_bits)
        yield "".join(str(b) for b in base_bits)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate bitstrings under U(1) symmetry with optional uniqueness."
    )

    parser.add_argument(
        "--bitlength", type=int, required=True,
        help="Total bitstring length (system size)."
    )

    parser.add_argument(
        "--numones", type=int, required=True,
        help="Number of 1-bits (particle number)."
    )

    parser.add_argument(
        "--num", type=int, default=10,
        help="How many bitstrings to generate (default: 10)."
    )

    parser.add_argument(
        "--method", type=int, default=0,
        help="Generation method. Currently only 0 (random)."
    )

    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (optional)."
    )

    parser.add_argument(
        "--unique", action="store_true",
        help="If specified, output only unique bitstrings."
    )

    parser.add_argument(
        "-o", "--outfile", nargs="+",
        help=(
            "Output file name(s). "
            "If multiple files are given, lines are distributed roughly evenly. "
            "If omitted, write to stdout."
        )
    )

    return parser.parse_args()

def write_outputs(bitstrings, outfiles):
    """
    Write bitstrings either to stdout (if outfiles is None)
    or distribute them across multiple files in round-robin fashion.
    """
    if not outfiles:
        # No output file specified: write to stdout
        for bs in bitstrings:
            print(bs)
        return

    n_files = len(outfiles)
    files = [open(name, "w") for name in outfiles]
    try:
        for idx, bs in enumerate(bitstrings):
            f = files[idx % n_files]
            f.write(bs + "\n")
    finally:
        for f in files:
            f.close()

def main():
    args = parse_args()

    if args.bitlength <= 0:
        print("Error: --bitlength must be positive.", file=sys.stderr)
        sys.exit(1)

    if not (0 <= args.numones <= args.bitlength):
        print("Error: --numones must satisfy 0 <= numones <= bitlength.", file=sys.stderr)
        sys.exit(1)

    if args.num <= 0:
        print("Error: --num must be positive.", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)

    # Select method
    if args.method == 0:
        generator = method_0_random(args.bitlength, args.numones, args.num, rng)
    else:
        print(f"Error: Unknown method {args.method}.", file=sys.stderr)
        sys.exit(1)

    if args.unique:
        # Collect unique bitstrings
        seen = set()
        for bs in generator:
            seen.add(bs)
        # Sort for reproducibility (optional; remove sorted() if不要なら)
        outputs = sorted(seen)
    else:
        outputs = generator

    write_outputs(outputs, args.outfile)

if __name__ == "__main__":
    main()
