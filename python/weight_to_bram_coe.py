#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np


NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
LAYER_WEIGHT_RE = re.compile(r"^layer(?P<layer>\d+)_weight$")


def read_weight_txt(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    nums = [float(x) for x in NUMBER_RE.findall(text)]
    if not nums:
        raise ValueError(f"No numeric weight value was found in: {path}")

    arr_float = np.asarray(nums, dtype=np.float64)
    arr_int = np.rint(arr_float).astype(np.int64)
    if not np.allclose(arr_float, arr_int, atol=1e-6):
        bad_idx = np.where(~np.isclose(arr_float, arr_int, atol=1e-6))[0][:10]
        bad_vals = arr_float[bad_idx].tolist()
        raise ValueError(f"Non-integer int8 weights detected: {list(zip(bad_idx.tolist(), bad_vals))}")
    return arr_int


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def align_up(value: int, alignment: int = 4) -> int:
    if value <= 0:
        raise ValueError("channel count must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def check_int8_range(values: np.ndarray) -> None:
    if values.size and (values.min() < -128 or values.max() > 127):
        raise ValueError(f"weights exceed signed int8 range: min={values.min()}, max={values.max()}")


def signed_to_unsigned_twos_complement(value: int) -> int:
    return int(value) & 0xFF


def bytes_to_words(units: List[int], word_bytes: int = 4, endian: str = "little") -> List[int]:
    if len(units) % word_bytes != 0:
        raise ValueError(f"Weight byte count {len(units)} is not divisible by word_bytes={word_bytes}.")
    data = bytes(units)
    return [
        int.from_bytes(data[i : i + word_bytes], byteorder=endian, signed=False)
        for i in range(0, len(data), word_bytes)
    ]


def write_coe(words: List[int], output_path: Path, values_per_line: int = 1) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        values = [f"{word:08X}" for word in words]
        for i in range(0, len(values), values_per_line):
            chunk = values[i : i + values_per_line]
            is_last = i + values_per_line >= len(values)
            f.write(", ".join(chunk) + (";\n" if is_last else ",\n"))


def npu_channel_groups(out_ch: int, aligned_out_ch: int, max_channels: int = 8) -> List[Tuple[int, int, int, bool]]:
    groups: List[Tuple[int, int, int, bool]] = []
    start = 0
    while start < aligned_out_ch:
        remaining_valid = max(0, out_ch - start)
        group_size = max_channels if remaining_valid >= max_channels else align_up(max(remaining_valid, 1), 4)
        group_size = min(group_size, aligned_out_ch - start)
        valid_count = max(0, min(out_ch - start, group_size))
        groups.append((start, group_size, valid_count, valid_count < group_size))
        start += group_size
    return groups


def convert_weights(
    flat_values: np.ndarray,
    in_ch: int,
    out_ch: int,
    kernel_h: int,
    kernel_w: int,
) -> tuple[List[int], int, int]:
    if in_ch > 1024:
        raise NotImplementedError("conv input channels > 1024 are not supported by current RCONV1.")

    expected = out_ch * in_ch * kernel_h * kernel_w
    if flat_values.size != expected:
        raise ValueError(
            f"weight count mismatch: file has {flat_values.size}, "
            f"expected {expected} for out_ch={out_ch}, in_ch={in_ch}, kernel={kernel_h}x{kernel_w}"
        )

    check_int8_range(flat_values)
    pad_in_ch = align_up(in_ch, 4)
    original = flat_values.reshape(out_ch, in_ch, kernel_h, kernel_w)
    padded = np.zeros((out_ch, pad_in_ch, kernel_h, kernel_w), dtype=np.int64)
    padded[:out_ch, :in_ch, :, :] = original

    emitted: List[int] = []
    for oc in range(out_ch):
        for ic_base in range(0, pad_in_ch, 4):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    for ic in range(ic_base, ic_base + 4):
                        emitted.append(signed_to_unsigned_twos_complement(int(padded[oc, ic, kh, kw])))
    return emitted, pad_in_ch, out_ch


def iter_weight_tensors(memory_plan: dict) -> list[tuple[int, str, dict]]:
    tensors = memory_plan.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError("memory plan must contain tensors object")

    items: list[tuple[int, str, dict]] = []
    for name, tensor in tensors.items():
        match = LAYER_WEIGHT_RE.match(name)
        if match:
            items.append((int(match.group("layer")), name, tensor))
    return sorted(items)


def weight_shape(tensor_name: str, tensor: dict) -> tuple[int, int, int, int]:
    shape = tensor.get("shape_oihw")
    if not shape or len(shape) != 4:
        raise ValueError(f"{tensor_name} does not contain shape_oihw")
    out_ch, in_ch, kernel_h, kernel_w = [int(value) for value in shape]
    return out_ch, in_ch, kernel_h, kernel_w


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate all parsed layer weight COE files from memory_plan.json."
    )
    parser.add_argument("--memory-plan", type=Path, required=True, help="Input data/memory_plan.json.")
    parser.add_argument("--model-params", type=Path, required=True, help="Directory containing layerN_0_weight.txt files.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for generated layerN_weight.coe files.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    memory_plan = load_json(args.memory_plan)
    weight_tensors = iter_weight_tensors(memory_plan)
    if not weight_tensors:
        raise ValueError(f"No layerN_weight tensors found in {args.memory_plan}")

    for layer, tensor_name, tensor in weight_tensors:
        out_ch, in_ch, kernel_h, kernel_w = weight_shape(tensor_name, tensor)
        weight_path = args.model_params / f"layer{layer}_0_weight.txt"
        output_path = args.out_dir / f"layer{layer}_weight.coe"
        flat_values = read_weight_txt(weight_path)
        units, pad_in_ch, stored_out_ch = convert_weights(flat_values, in_ch, out_ch, kernel_h, kernel_w)
        expected_storage = [out_ch, pad_in_ch, kernel_h, kernel_w]
        storage_shape = tensor.get("storage_shape_oihw")
        if storage_shape is not None and [int(value) for value in storage_shape] != expected_storage:
            raise ValueError(
                f"{tensor_name} storage_shape_oihw={storage_shape}, expected {expected_storage}"
            )
        expected_size = int(tensor.get("size_bytes", len(units)))
        if len(units) != expected_size:
            raise ValueError(f"{tensor_name} size mismatch: generated={len(units)}, memory_plan={expected_size}")
        words = bytes_to_words(units)
        write_coe(words, output_path)

        aligned_out_ch = align_up(out_ch, 4)
        groups = npu_channel_groups(out_ch, aligned_out_ch)
        bytes_per_output_channel = pad_in_ch * kernel_h * kernel_w

        print(f"[OK] weight COE generated: {output_path}")
        print(f"     layer={layer}, tensor={tensor_name}")
        print(f"     shape_oihw={out_ch}x{in_ch}x{kernel_h}x{kernel_w}")
        print(f"     storage_oihw={stored_out_ch}x{pad_in_ch}x{kernel_h}x{kernel_w}")
        print(f"     output_storage_channels={aligned_out_ch} (runtime/bias padding only)")
        print(f"     bytes={len(units)}, words={len(words)}")
        print(f"     first16={' '.join(f'{x:02X}' for x in units[:16])}")
        for idx, (start_channel, group_channels, valid_channels, has_padding) in enumerate(groups):
            offset = start_channel * bytes_per_output_channel
            print(
                f"     group{idx}: start_channel={start_channel}, channels={group_channels}, "
                f"valid_weight_channels={valid_channels}, offset=0x{offset:08X}, has_padding={has_padding}"
            )


if __name__ == "__main__":
    main()
