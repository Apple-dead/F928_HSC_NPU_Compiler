#!/usr/bin/env python3
"""Compare two text files and print a Linux-diff-like unified diff."""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path


def read_lines(path: Path) -> list[str]:
    """Read text while keeping line endings for accurate diff output."""
    try:
        return path.read_text(encoding="utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk").splitlines(keepends=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two text files and print a unified diff.",
        usage="python diff.py FILE1 FILE2",
    )
    parser.add_argument("file1", type=Path, help="old/original file")
    parser.add_argument("file2", type=Path, help="new/modified file")
    parser.add_argument(
        "-n",
        "--context-lines",
        type=int,
        default=3,
        help="number of unchanged context lines to show, default: 3",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    for path in (args.file1, args.file2):
        if not path.is_file():
            print(f"diff.py: {path}: No such file", file=sys.stderr)
            return 2

    old_lines = read_lines(args.file1)
    new_lines = read_lines(args.file2)

    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=str(args.file1),
            tofile=str(args.file2),
            lineterm="",
            n=args.context_lines,
        )
    )

    if not diff_lines:
        return 0

    for line in diff_lines:
        print(line, end="" if line.endswith("\n") else "\n")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
