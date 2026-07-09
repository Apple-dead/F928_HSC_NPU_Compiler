#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate physical per-layer parameter COEs: weights then bias per <=8-channel group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bias_to_bram_coe import bias_values_to_words, parse_numbers
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


PARAMETER_STAGE_BUILDERS = {
    "conv": generate_conv_params,
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
