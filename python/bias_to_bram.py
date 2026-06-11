#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bias_to_bram.py

Read a bias text file, divide each bias by a configurable divisor, expand every processed bias
into an L x L square matrix, pad the number of bias values to a multiple of 4,
and write the result as a 32-bit word-granularity Xilinx COE file.

Usage:
    python bias_to_bram.py -length <side_length> [-move <divisor>] <bias_file> <output_coe_file>

Example:
    python bias_to_bram.py -length 256 data/layer1_0_bias.txt coe/layer1_bias.coe

Default behavior:
    - Bias numbers can be separated by spaces, commas, brackets, or newlines.
    - Each bias is divided by the -move value, defaulting to 128.
    - The divided value is floored by default, matching common quantized feature-map flow.
    - The result is first represented as signed int8 two's-complement bytes.
    - Every 4 consecutive bytes are packed into one 32-bit COE item.
    - Low-address byte is placed on the right / low 8 bits.
    - Example: byte sequence 20,22,1F,00 -> COE word 001F2220.
    - If the original bias count is not a multiple of 4, zero bias values are appended.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, List


_NUMBER_RE = re.compile(
    r"""
    (?P<hex>[+-]?0[xX][0-9a-fA-F]+)                 # hex integer, e.g. -0x10
    |(?P<float>[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)  # int/float/scientific
    """,
    re.VERBOSE,
)


def parse_numbers(path: Path) -> List[float]:
    """Parse numeric values from a text file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    nums: List[float] = []

    for m in _NUMBER_RE.finditer(text):
        token = m.group(0)
        if m.group("hex") is not None:
            nums.append(float(int(token, 16)))
        else:
            nums.append(float(token))

    if not nums:
        raise ValueError(f"No numeric bias value was found in: {path}")

    return nums


def quantize_after_divisor(value: float, divisor: float, mode: str) -> int:
    """Divide by divisor and convert to an integer according to the selected mode."""
    scaled = value / divisor

    if mode == "floor":
        return math.floor(scaled)
    if mode == "trunc":
        return int(scaled)  # truncate toward zero
    if mode == "round":
        return int(round(scaled))

    raise ValueError(f"Unsupported round mode: {mode}")


def int_to_byte(value: int, *, clamp: bool = False) -> int:
    """Convert a signed int8 value to unsigned two's-complement byte."""
    if clamp:
        value = max(-128, min(127, value))
    elif value < -128 or value > 127:
        raise ValueError(
            f"Processed bias value {value} is out of signed int8 range [-128, 127]. "
            "Use --clamp if you intentionally want saturation."
        )

    return value & 0xFF


def pad_to_multiple_of_4(values: List[float]) -> List[float]:
    """Append zeros until len(values) is a multiple of 4."""
    pad_num = (-len(values)) % 4
    if pad_num:
        values = values + [0.0] * pad_num
    return values


def expand_biases_to_bytes(
    bias_values: Iterable[float],
    length: int,
    divisor: float,
    round_mode: str,
    clamp: bool,
) -> List[int]:
    """
    For each bias value, generate an L x L matrix filled with that processed bias.
    Output order: bias index -> row -> column.
    """
    items: List[int] = []
    matrix_items = length * length

    for bias in bias_values:
        q = quantize_after_divisor(bias, divisor, round_mode)
        byte_val = int_to_byte(q, clamp=clamp)
        items.extend([byte_val] * matrix_items)

    return items


def bytes_to_words(data: List[int], word_bytes: int, endian: str) -> List[int]:
    if word_bytes <= 0:
        raise ValueError("word_bytes must be a positive integer.")
    if len(data) % word_bytes != 0:
        raise ValueError(f"Byte count {len(data)} is not divisible by word_bytes={word_bytes}.")
    raw = bytes(data)
    return [
        int.from_bytes(raw[i : i + word_bytes], byteorder=endian, signed=False)
        for i in range(0, len(raw), word_bytes)
    ]


def format_word(value: int, radix: int, word_bytes: int) -> str:
    if radix == 16:
        return f"{value:0{word_bytes * 2}X}"
    if radix == 10:
        return str(value)
    raise ValueError("Only radix=16 or radix=10 is supported.")


def write_coe(words: List[int], out_path: Path, radix: int, word_bytes: int, values_per_line: int) -> None:
    """Write word-granularity COE file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    values = [format_word(w, radix=radix, word_bytes=word_bytes) for w in words]

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"memory_initialization_radix={radix};\n")
        f.write("memory_initialization_vector=\n")

        for i in range(0, len(values), values_per_line):
            chunk = values[i : i + values_per_line]
            is_last = (i + values_per_line) >= len(values)
            f.write(", ".join(chunk) + (";\n" if is_last else ",\n"))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert bias text data to 32-bit word-granularity BRAM COE. "
            "Each bias is divided by -move, expanded to an LxL matrix, "
            "and the bias count is padded to a multiple of 4."
        )
    )
    parser.add_argument(
        "-length",
        type=int,
        required=True,
        help="Side length L of the square matrix generated for every bias.",
    )
    parser.add_argument(
        "-move",
        type=float,
        default=128.0,
        help="Divisor applied to every bias value before integer conversion. Default: 128.",
    )
    parser.add_argument("bias_file", type=Path, help="Input bias text file.")
    parser.add_argument("output_coe_file", type=Path, help="Output .coe file path.")
    parser.add_argument(
        "--round-mode",
        choices=("floor", "trunc", "round"),
        default="floor",
        help="How to convert bias/-move to integer before writing int8 hex. Default: floor.",
    )
    parser.add_argument(
        "--clamp",
        action="store_true",
        help="Clamp processed values to signed int8 range instead of reporting overflow.",
    )
    parser.add_argument("--radix", type=int, choices=(10, 16), default=16, help="COE radix. Default: 16.")
    parser.add_argument(
        "--word-bytes",
        type=int,
        default=4,
        help="Bytes per BRAM word. Default: 4, i.e. 32-bit COE items.",
    )
    parser.add_argument(
        "--word-endian",
        choices=("little", "big"),
        default="little",
        help="Packing endian. Default little: low-address byte is on the right / low 8 bits.",
    )
    parser.add_argument("--values-per-line", type=int, default=1, help="COE words per line. Default: 1.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if args.length <= 0:
        raise ValueError("-length must be a positive integer.")
    if args.move == 0:
        raise ValueError("-move must not be zero.")
    if args.values_per_line <= 0:
        raise ValueError("--values-per-line must be a positive integer.")
    if not args.bias_file.is_file():
        raise FileNotFoundError(f"Bias file does not exist: {args.bias_file}")

    raw_biases = parse_numbers(args.bias_file)
    padded_biases = pad_to_multiple_of_4(raw_biases)
    byte_items = expand_biases_to_bytes(
        padded_biases,
        length=args.length,
        divisor=args.move,
        round_mode=args.round_mode,
        clamp=args.clamp,
    )
    words = bytes_to_words(byte_items, word_bytes=args.word_bytes, endian=args.word_endian)
    write_coe(words, args.output_coe_file, radix=args.radix, word_bytes=args.word_bytes, values_per_line=args.values_per_line)

    print("bias_to_bram done")
    print(f"  input_bias_count   = {len(raw_biases)}")
    print(f"  padded_bias_count  = {len(padded_biases)}")
    print(f"  matrix_length      = {args.length}")
    print(f"  divisor            = {args.move}")
    print(f"  byte_items         = {len(byte_items)}")
    print(f"  bram_words         = {len(words)} words, {args.word_bytes} byte(s)/word")
    print("  packing_rule       = low-address byte on the right / low 8 bits")
    print(f"  output             = {args.output_coe_file}")


if __name__ == "__main__":
    main()
