#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import npu_config as cfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def literal_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(literal_value(e) for e in node.elts)
    if isinstance(node, ast.List):
        return [literal_value(e) for e in node.elts]
    raise ValueError(f"unsupported non-literal argument: {ast.dump(node)}")


def normalize_pair(value: Any) -> Tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return (int(value[0]), int(value[1]))
    raise ValueError(f"expected int or int pair, got {value!r}")


def call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def resolve_model_py(model_py_arg: str) -> Path:
    path = Path(model_py_arg)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(PROJECT_ROOT / path)
        if len(path.parts) == 1:
            candidates.append(PROJECT_ROOT / "model" / path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"model definition not found: {model_py_arg}; searched: {searched}")


def conv2d_call_to_dict(node: ast.Call) -> Dict[str, Any] | None:
    if call_name(node) != "Conv2d":
        return None

    names = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
    values: Dict[str, Any] = {}
    for idx, arg in enumerate(node.args[: len(names)]):
        values[names[idx]] = literal_value(arg)
    for kw in node.keywords:
        if kw.arg in names:
            values[kw.arg] = literal_value(kw.value)

    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"Conv2d is missing required args: {missing}")

    kernel = normalize_pair(values["kernel_size"])
    stride = normalize_pair(values["stride"])
    padding = normalize_pair(values["padding"])
    return {
        "in_channels": int(values["in_channels"]),
        "out_channels": int(values["out_channels"]),
        "kernel_size": [kernel[0], kernel[1]],
        "stride": [stride[0], stride[1]],
        "padding": [padding[0], padding[1]],
    }


def negative_slope_to_tan(negative_slope: float) -> int:
    if negative_slope == 0:
        return 8
    if negative_slope < 0:
        raise ValueError(f"negative_slope must be non-negative, got {negative_slope}")

    exponent = round(math.log2(1.0 / negative_slope))
    expected = 1.0 / (2 ** exponent)
    if not math.isclose(negative_slope, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"unsupported LeakyReLU negative_slope={negative_slope}; "
            "NPU tan field supports 1/(2^n), n=0..7, or 0."
        )
    if not 0 <= exponent <= 7:
        raise ValueError(
            f"unsupported LeakyReLU negative_slope={negative_slope}; "
            "NPU tan field supports 1, 1/2, ..., 1/128, or 0."
        )
    return exponent


def leaky_relu_call_to_dict(node: ast.Call) -> Dict[str, Any] | None:
    if call_name(node) != "LeakyReLU":
        return None

    negative_slope = 0.01
    if node.args:
        negative_slope = literal_value(node.args[0])
    for kw in node.keywords:
        if kw.arg == "negative_slope":
            negative_slope = literal_value(kw.value)

    negative_slope = float(negative_slope)
    return {
        "mode": "leaky_relu",
        "negative_slope": negative_slope,
        "tan": negative_slope_to_tan(negative_slope),
    }


def find_self_assignment(tree: ast.AST, name: str) -> ast.AST | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == name
            ):
                return node.value
    return None


def find_layer_relu(tree: ast.AST, model_py: Path) -> Dict[str, Any]:
    layer = find_self_assignment(tree, "layer")
    if layer is None:
        raise ValueError(f"could not find self.layer activation pipeline in {model_py}")
    for sub in ast.walk(layer):
        if isinstance(sub, ast.Call):
            relu = leaky_relu_call_to_dict(sub)
            if relu is not None:
                return relu
    raise ValueError(f"could not find LeakyReLU in self.layer in {model_py}")


def parse_all_model_layers(model_py: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tree = ast.parse(model_py.read_text(encoding="utf-8"))
    layers: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                continue
            match = re.fullmatch(r"layer(\d+)", target.attr)
            if not match:
                continue
            conv = None
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Call):
                    conv = conv2d_call_to_dict(sub)
                    if conv is not None:
                        break
            if conv is None:
                continue
            layer_index = int(match.group(1))
            layers.append(
                {
                    "layer_index": layer_index,
                    "layer": f"layer{layer_index}",
                    "conv": conv,
                }
            )

    if not layers:
        raise ValueError(f"could not find any self.layerN Conv2d definitions in {model_py}")
    layers.sort(key=lambda item: item["layer_index"])
    return layers, find_layer_relu(tree, model_py)


def parse_model_layers(model_py: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    layers, relu = parse_all_model_layers(model_py)
    return select_layers(layers), relu


def select_layers(layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mode = int(cfg.INFER_PARSE_MODE)
    if mode == 1:
        return layers
    if mode == 2:
        start = int(cfg.INFER_PARSE_LAYER_START)
        end = int(cfg.INFER_PARSE_LAYER_END)
        if start <= 0 or end <= 0:
            raise ValueError("INFER_PARSE_LAYER_START/END must be positive when INFER_PARSE_MODE=2")
        if start > end:
            raise ValueError("INFER_PARSE_LAYER_START must be <= INFER_PARSE_LAYER_END")
        selected = [layer for layer in layers if start <= layer["layer_index"] <= end]
        expected_count = end - start + 1
        if len(selected) != expected_count:
            found = [layer["layer_index"] for layer in layers]
            raise ValueError(f"requested layers {start}..{end}, but model contains layers {found}")
        return selected
    raise ValueError("INFER_PARSE_MODE must be 1 (full model) or 2 (layer range)")


def conv_output_hw(height: int, width: int, conv: Dict[str, Any]) -> tuple[int, int]:
    kh, kw = conv["kernel_size"]
    sh, sw = conv["stride"]
    ph, pw = conv["padding"]
    out_h = (height + 2 * ph - kh) // sh + 1
    out_w = (width + 2 * pw - kw) // sw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"invalid Conv2d output size for conv={conv}, input={height}x{width}")
    return out_h, out_w

