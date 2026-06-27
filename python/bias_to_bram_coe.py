#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List


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


def align_up(value: int, alignment: int = 4) -> int:
    if value <= 0:
        raise ValueError("channel count must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def pad_to_multiple_of_4(values: List[float]) -> List[float]:
    pad_num = (-len(values)) % 4
    return values + [0.0] * pad_num if pad_num else values


def bias_values_to_words(bias_values: List[float], out_ch: int) -> List[int]:
    biases = list(bias_values)
    if len(biases) != out_ch:
        raise ValueError(f"bias count mismatch: file has {len(biases)} values, expected out_ch={out_ch}")

    words: List[int] = []
    for index, value in enumerate(pad_to_multiple_of_4(biases)):
        rounded = round(value)
        if abs(value - rounded) > 1e-6:
            raise ValueError(f"bias[{index}]={value!r} is not an integer value and cannot be emitted as raw int32")
        if not -(1 << 31) <= rounded <= (1 << 31) - 1:
            raise ValueError(f"bias[{index}]={rounded} is out of signed int32 range")
        words.append(int(rounded) & 0xFFFFFFFF)
    return words


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


def bias_channels(tensor_name: str, tensor: dict) -> tuple[int, int]:
    out_ch = int(tensor.get("channels", 0))
    aligned_out_ch = int(tensor.get("aligned_channels", 0))
    if out_ch <= 0 or aligned_out_ch <= 0:
        shape = tensor.get("shape")
        storage_shape = tensor.get("storage_shape")
        if not shape or not storage_shape:
            raise ValueError(f"{tensor_name} does not contain bias channel metadata")
        out_ch = int(shape[0])
        aligned_out_ch = int(storage_shape[0])
    return out_ch, aligned_out_ch


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate all parsed layer bias COE files from memory_plan.json."
    )
    parser.add_argument("--memory-plan", type=Path, required=True, help="Input data/memory_plan.json.")
    parser.add_argument("--model-params", type=Path, required=True, help="Directory containing layerN_0_bias.txt files.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for generated layerN_bias.coe files.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    memory_plan = load_json(args.memory_plan)
    bias_tensors = iter_bias_tensors(memory_plan)
    if not bias_tensors:
        raise ValueError(f"No layerN_bias tensors found in {args.memory_plan}")

    for layer, tensor_name, tensor in bias_tensors:
        out_ch, aligned_out_ch = bias_channels(tensor_name, tensor)
        bias_path = args.model_params / f"layer{layer}_0_bias.txt"
        output_path = args.out_dir / f"layer{layer}_bias.coe"
        raw_biases = parse_numbers(bias_path)
        expected_aligned = align_up(out_ch, 4)
        if aligned_out_ch != expected_aligned:
            raise ValueError(
                f"{tensor_name} aligned_channels={aligned_out_ch}, expected {expected_aligned} for out_ch={out_ch}"
            )
        words = bias_values_to_words(raw_biases, out_ch=out_ch)
        expected_size = int(tensor.get("size_bytes", len(words) * 4))
        if len(words) * 4 != expected_size:
            raise ValueError(
                f"{tensor_name} size mismatch: memory_plan={expected_size} bytes, generated={len(words) * 4} bytes"
            )
        write_coe(words, output_path)

        print(f"[OK] bias COE generated: {output_path}")
        print(f"     layer={layer}, tensor={tensor_name}")
        print(f"     channels={out_ch}, aligned_channels={aligned_out_ch}")
        print(f"     bytes={len(words) * 4}, words={len(words)}")


if __name__ == "__main__":
    main()
