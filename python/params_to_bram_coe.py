#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate physical per-layer parameter COEs: weights then bias per <=8-channel group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bias_to_bram_coe import bias_values_to_words, parse_numbers
from linear_params_to_bram_coe import convert_linear_weights, read_weight_txt as read_linear_weight_txt
from weight_to_bram_coe import bytes_to_words, convert_weights, read_weight_txt, weight_shape, write_coe


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_layer_bytes(layer_plan: dict, weight_tensor: dict, raw_weights, raw_biases=None) -> list[int]:
    out_ch, in_ch, kernel_h, kernel_w = weight_shape(layer_plan["layer"] + "_weight", weight_tensor)
    weight_units, padded_in_ch, _ = convert_weights(raw_weights, in_ch, out_ch, kernel_h, kernel_w)
    has_bias = bool(layer_plan.get("has_bias", False))
    bias_words = bias_values_to_words(raw_biases, out_ch) if has_bias else []
    bytes_per_output = padded_in_ch * kernel_h * kernel_w
    result: list[int] = []

    for group in layer_plan["splits"]:
        start = int(group["start_channel"])
        valid = int(group["valid_channels"])
        channels = int(group["channels"])
        weight_start = start * bytes_per_output
        weight_end = weight_start + valid * bytes_per_output
        result.extend(weight_units[weight_start:weight_end])
        if has_bias:
            for word in bias_words[start : start + valid]:
                result.extend(word.to_bytes(4, byteorder="little", signed=False))
            result.extend(b"\x00" * ((channels - valid) * 4))

    return result


def generate_conv_params(layer_plan: dict, tensors: dict, model_params: Path, out_dir: Path) -> None:
    layer_name = layer_plan["layer"]
    layer = int(layer_name.removeprefix("layer"))
    bias_path = model_params / f"layer{layer}_0_bias.txt"
    raw_biases = parse_numbers(bias_path) if layer_plan.get("has_bias", False) else None
    units = build_layer_bytes(
        layer_plan,
        tensors[f"{layer_name}_weight"],
        read_weight_txt(model_params / f"layer{layer}_0_weight.txt"),
        raw_biases,
    )
    expected_size = int(tensors[f"{layer_name}_params"]["size_bytes"])
    if len(units) != expected_size:
        raise ValueError(f"{layer_name}: generated {len(units)} bytes, expected {expected_size}")
    output = out_dir / f"{layer_name}_params.coe"
    write_coe(bytes_to_words(units), output)
    print(f"[OK] parameter COE generated: {output} ({len(units)} bytes)")


def int32_bias_word(value: float, index: int) -> int:
    rounded = round(value)
    if abs(value - rounded) > 1e-6:
        raise ValueError(f"linear bias[{index}]={value!r} is not an integer value and cannot be emitted as raw int32")
    if not -(1 << 31) <= rounded <= (1 << 31) - 1:
        raise ValueError(f"linear bias[{index}]={rounded} is out of signed int32 range")
    return int(rounded) & 0xFFFFFFFF


def generate_linear_params(layer_plan: dict, tensors: dict, model_params: Path, out_dir: Path) -> None:
    layer_name = layer_plan["layer"]
    weight_tensor = tensors[f"{layer_name}_weight"]
    out_features, in_features = [int(value) for value in weight_tensor["shape_oi"]]
    _, input_channels, input_height, input_width = [int(value) for value in weight_tensor["input_shape_nchw"]]
    _, aligned_input_channels, aligned_input_height, aligned_input_width = [
        int(value) for value in weight_tensor["input_aligned_shape_nchw"]
    ]
    if (aligned_input_height, aligned_input_width) != (input_height, input_width):
        raise ValueError(
            f"{layer_name}: input_aligned_shape_nchw must keep actual H/W, got "
            f"{aligned_input_height}x{aligned_input_width}, expected {input_height}x{input_width}"
        )
    weight_units = convert_linear_weights(
        read_linear_weight_txt(model_params / f"{layer_name}_weight.txt"),
        out_features=out_features,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        aligned_input_channels=aligned_input_channels,
    )
    input_bytes = aligned_input_channels * input_height * input_width
    has_bias = bool(layer_plan.get("has_bias", False))
    raw_biases = parse_numbers(model_params / f"{layer_name}_bias.txt") if has_bias else []
    if has_bias and len(raw_biases) != out_features:
        raise ValueError(f"{layer_name}: bias count mismatch: file has {len(raw_biases)}, expected {out_features}")

    result: list[int] = []
    for output_index in range(out_features):
        weight_start = output_index * input_bytes
        weight_end = weight_start + input_bytes
        result.extend(weight_units[weight_start:weight_end])
        if has_bias:
            result.extend(int32_bias_word(raw_biases[output_index], output_index).to_bytes(4, "little"))

    expected_size = int(tensors[f"{layer_name}_params"]["size_bytes"])
    if len(result) != expected_size:
        raise ValueError(f"{layer_name}: generated {len(result)} bytes, expected {expected_size}")
    output = out_dir / f"{layer_name}_params.coe"
    write_coe(bytes_to_words(result), output)
    print(f"[OK] parameter COE generated: {output} ({len(result)} bytes)")


PARAMETER_STAGE_BUILDERS = {
    "conv": generate_conv_params,
    "linear": generate_linear_params,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-layer interleaved weight/bias parameter COEs.")
    parser.add_argument("--memory-plan", type=Path, required=True)
    parser.add_argument("--model-params", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    plan = load_json(args.memory_plan)
    tensors = plan["tensors"]
    for layer_plan in plan.get("execution_plan", []):
        op_type = layer_plan.get("op_type", "conv")
        builder = PARAMETER_STAGE_BUILDERS.get(op_type)
        if builder is None:
            continue
        builder(layer_plan, tensors, args.model_params, args.out_dir)


if __name__ == "__main__":
    main()
