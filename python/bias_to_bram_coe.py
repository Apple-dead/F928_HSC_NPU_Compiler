#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, List, Tuple


NUMBER_RE = re.compile(
    r"""
    (?P<hex>[+-]?0[xX][0-9a-fA-F]+)
    |(?P<float>[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
    """,
    re.VERBOSE,
)
LAYER_BIAS_RE = re.compile(r"^layer(?P<layer>\d+)_bias$")


def parse_numbers(path: Path) -> List[float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    nums: List[float] = []
    for match in NUMBER_RE.finditer(text):
        token = match.group(0)
        nums.append(float(int(token, 16)) if match.group("hex") is not None else float(token))
    if not nums:
        raise ValueError(f"No numeric bias value was found in: {path}")
    return nums


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_bias_moves(path: Path) -> dict[int, float]:
    data = load_json(path)
    raw_moves = data.get("BIAS_MOVE_BY_LAYER")
    if not isinstance(raw_moves, dict):
        raise ValueError(f"{path} must contain object field BIAS_MOVE_BY_LAYER")

    moves: dict[int, float] = {}
    for key, value in raw_moves.items():
        layer = int(key)
        move = float(value)
        if move == 0:
            raise ValueError(f"BIAS_MOVE_BY_LAYER[{layer}] must not be zero")
        moves[layer] = move
    return moves


def align_up(value: int, alignment: int = 4) -> int:
    if value <= 0:
        raise ValueError("channel count must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def quantize_after_divisor(value: float, divisor: float) -> int:
    return round(value / divisor)


def int_to_byte(value: int) -> int:
    if value < -128 or value > 127:
        raise ValueError(f"Processed bias value {value} is out of signed int8 range [-128, 127].")
    return value & 0xFF


def pad_to_multiple_of_4(values: List[float]) -> List[float]:
    pad_num = (-len(values)) % 4
    return values + [0.0] * pad_num if pad_num else values


def npu_channel_groups(out_ch: int, aligned_out_ch: int, max_channels: int = 8) -> List[Tuple[int, int, bool]]:
    groups: List[Tuple[int, int, bool]] = []
    start = 0
    while start < aligned_out_ch:
        remaining_valid = max(0, out_ch - start)
        group_size = max_channels if remaining_valid >= max_channels else align_up(max(remaining_valid, 1), 4)
        group_size = min(group_size, aligned_out_ch - start)
        valid_count = max(0, min(out_ch - start, group_size))
        groups.append((start, group_size, valid_count < group_size))
        start += group_size
    return groups


def expand_biases_to_bytes(bias_values: Iterable[float], length: int, divisor: float, out_ch: int) -> List[int]:
    biases = list(bias_values)
    if len(biases) != out_ch:
        raise ValueError(f"bias count mismatch: file has {len(biases)} values, expected out_ch={out_ch}")

    bias_bytes = [int_to_byte(quantize_after_divisor(bias, divisor)) for bias in pad_to_multiple_of_4(biases)]
    matrix_items = length * length
    items: List[int] = []
    for i in range(0, len(bias_bytes), 4):
        items.extend(bias_bytes[i : i + 4] * matrix_items)
    return items


def bytes_to_words(data: List[int], word_bytes: int = 4, endian: str = "little") -> List[int]:
    if len(data) % word_bytes != 0:
        raise ValueError(f"Byte count {len(data)} is not divisible by word_bytes={word_bytes}.")
    raw = bytes(data)
    return [
        int.from_bytes(raw[i : i + word_bytes], byteorder=endian, signed=False)
        for i in range(0, len(raw), word_bytes)
    ]


def write_coe(words: List[int], out_path: Path, values_per_line: int = 1) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        values = [f"{word:08X}" for word in words]
        for i in range(0, len(values), values_per_line):
            chunk = values[i : i + values_per_line]
            is_last = i + values_per_line >= len(values)
            f.write(", ".join(chunk) + (";\n" if is_last else ",\n"))


def iter_bias_tensors(memory_plan: dict) -> list[tuple[int, str, dict]]:
    tensors = memory_plan.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError("memory plan must contain tensors object")

    items: list[tuple[int, str, dict]] = []
    for name, tensor in tensors.items():
        match = LAYER_BIAS_RE.match(name)
        if match:
            items.append((int(match.group("layer")), name, tensor))
    return sorted(items)


def bias_shape(tensor_name: str, tensor: dict) -> tuple[int, int, int]:
    shape = tensor.get("shape_nchw")
    if not shape or len(shape) != 4:
        raise ValueError(f"{tensor_name} does not contain shape_nchw")
    _n, out_ch, height, width = [int(value) for value in shape]
    if height != width:
        raise ValueError(f"{tensor_name} is not square: {height}x{width}")
    return out_ch, height, width


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate all parsed layer bias COE files from memory_plan.json and bias_move.json."
    )
    parser.add_argument("--memory-plan", type=Path, required=True, help="Input data/memory_plan.json.")
    parser.add_argument("--model-params", type=Path, required=True, help="Directory containing layerN_0_bias.txt files.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for generated layerN_bias.coe files.")
    parser.add_argument("--move", type=Path, required=True, help="JSON file containing BIAS_MOVE_BY_LAYER.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    memory_plan = load_json(args.memory_plan)
    moves = load_bias_moves(args.move)
    bias_tensors = iter_bias_tensors(memory_plan)
    if not bias_tensors:
        raise ValueError(f"No layerN_bias tensors found in {args.memory_plan}")

    for layer, tensor_name, tensor in bias_tensors:
        if layer not in moves:
            raise KeyError(f"Missing BIAS_MOVE_BY_LAYER entry for layer {layer}")

        out_ch, height, _width = bias_shape(tensor_name, tensor)
        bias_path = args.model_params / f"layer{layer}_0_bias.txt"
        output_path = args.out_dir / f"layer{layer}_bias.coe"
        raw_biases = parse_numbers(bias_path)
        aligned_out_ch = align_up(out_ch, 4)
        byte_items = expand_biases_to_bytes(raw_biases, length=height, divisor=moves[layer], out_ch=out_ch)
        words = bytes_to_words(byte_items)
        write_coe(words, output_path)
        groups = npu_channel_groups(out_ch, aligned_out_ch)

        print(f"[OK] bias COE generated: {output_path}")
        print(f"     layer={layer}, tensor={tensor_name}, move={moves[layer]:g}")
        print(f"     channels={out_ch}, aligned_channels={aligned_out_ch}, matrix={height}x{height}")
        print(f"     bytes={len(byte_items)}, words={len(words)}")
        for idx, (start_channel, group_channels, has_padding) in enumerate(groups):
            offset = start_channel * height * height
            print(
                f"     group{idx}: start_channel={start_channel}, channels={group_channels}, "
                f"offset=0x{offset:08X}, has_padding={has_padding}"
            )


if __name__ == "__main__":
    main()
