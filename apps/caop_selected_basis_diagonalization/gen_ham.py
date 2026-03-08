#!/usr/bin/env python3
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate extended hard-core boson Hamiltonian in creation/annihilation-operator format.\n\n"
            "The output format is:\n"
            "  +1                       (first line: bosonic model)\n"
            "  t bdag i b j             (hopping term)\n"
            "  t bdag j b i             (hopping term, hermitian conjugate)\n"
            "  V bdag i bdag j b j b i  (density-density / repulsion term)\n\n"
            "You can specify either a 1D open chain (--nsites) or an arbitrary graph via --edge."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--nsites",
        type=int,
        default=None,
        help="Number of sites for a 1D open chain. "
             "Edges (i, i+1) for i = 0..nsites-2 will be added.",
    )

    parser.add_argument(
        "--edge",
        type=int,
        nargs=2,
        action="append",
        metavar=("I", "J"),
        help="Add an undirected edge (I, J). "
             "Can be specified multiple times for an arbitrary graph.",
    )

    parser.add_argument(
        "--t",
        type=float,
        default=1.0,
        help="Hopping amplitude t (default: 1.0).",
    )

    parser.add_argument(
        "--V",
        type=float,
        default=2.0,
        help="Nearest-neighbor repulsion strength V (default: 2.0).",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="-",
        help="Output file path. Use '-' (default) for stdout.",
    )

    args = parser.parse_args()

    # Sanity check: at least one source of edges must be provided
    if args.nsites is None and args.edge is None:
        parser.error("You must specify either --nsites or at least one --edge.")

    return args


def build_edge_list(nsites, edge_specs):
    """
    Build a sorted list of undirected edges from:
      - 1D chain edges (0,1), (1,2), ... if nsites is not None
      - manual edges from --edge options

    Returns a sorted list of (i, j) with i < j and duplicates removed.
    """
    edges = set()

    # 1D open chain
    if nsites is not None:
        if nsites < 2:
            raise ValueError("--nsites must be >= 2 for a nontrivial chain.")
        for i in range(nsites - 1):
            edges.add((i, i + 1))

    # Arbitrary edges from CLI
    if edge_specs is not None:
        for i, j in edge_specs:
            if i == j:
                # self-edgeは無視（必要ならここを変える）
                continue
            a, b = sorted((i, j))
            edges.add((a, b))

    if not edges:
        raise ValueError("No edges generated. Check --nsites and --edge options.")

    # 整理して決まった順序で出力
    return sorted(edges)


def open_output(path):
    if path == "-" or path is None:
        return sys.stdout
    return open(path, "w")


def main():
    args = parse_args()

    try:
        edges = build_edge_list(args.nsites, args.edge)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    out = open_output(args.output)

    try:
        # 1行目：ボゾンを意味する +1 （今回は常にボゾン）
        print("+1", file=out)

        t = args.t
        V = args.V

        # まずホッピング項
        for i, j in edges:
            # t bdag i b j
            print(f"{t} bdag {i} b {j}", file=out)
            # t bdag j b i （エルミート共役）
            print(f"{t} bdag {j} b {i}", file=out)

        # 次に斥力（拡張ハードコアボゾン）
        for i, j in edges:
            # V bdag i bdag j b j b i
            print(f"{V} bdag {i} bdag {j} b {j} b {i}", file=out)

    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
