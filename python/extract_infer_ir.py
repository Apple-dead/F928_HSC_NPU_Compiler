#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import npu_config as cfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PY = PROJECT_ROOT / "model" / "yolov2_14layer_quantized.py"
DEFAULT_MEMORY_PLAN = PROJECT_ROOT / "data" / "memory_plan.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "infer_ir"


def literal_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(literal_value(e) for e in node.elts)
    if isinstance(node, ast.List):
        return [literal_value(e) for e in node.elts]
    raise ValueError(f"unsupported non-literal Conv2d argument: {ast.dump(node)}")


def normalize_pair(value: Any) -> Tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return (int(value[0]), int(value[1]))
    raise ValueError(f"expected int or int pair, got {value!r}")


def conv2d_call_to_dict(node: ast.Call) -> Dict[str, Any] | None:
    func = node.func
    is_conv2d = (
        isinstance(func, ast.Attribute)
        and func.attr == "Conv2d"
        or isinstance(func, ast.Name)
        and func.id == "Conv2d"
    )
    if not is_conv2d:
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
        raise ValueError(f"Conv2d is missing required args for first-stage IR: {missing}")
    return values


def find_layer1_conv(model_py: Path) -> Dict[str, Any]:
    tree = ast.parse(model_py.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "layer1"
            ):
                continue
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Call):
                    conv = conv2d_call_to_dict(sub)
                    if conv is not None:
                        return conv
    raise ValueError(f"could not find self.layer1 Conv2d in {model_py}")


def validate_layer1(conv: Dict[str, Any]) -> None:
    kernel = normalize_pair(conv["kernel_size"])
    stride = normalize_pair(conv["stride"])
    padding = normalize_pair(conv["padding"])
    expected = {
        "in_channels": cfg.LAYER1_IN_CHANNELS,
        "out_channels": cfg.LAYER1_OUT_CHANNELS,
        "kernel_size": (cfg.LAYER1_KERNEL_SIZE, cfg.LAYER1_KERNEL_SIZE),
        "stride": (cfg.LAYER1_STRIDE, cfg.LAYER1_STRIDE),
        "padding": (cfg.LAYER1_PADDING, cfg.LAYER1_PADDING),
    }
    actual = {
        "in_channels": int(conv["in_channels"]),
        "out_channels": int(conv["out_channels"]),
        "kernel_size": kernel,
        "stride": stride,
        "padding": padding,
    }
    if actual != expected:
        raise ValueError(f"unsupported layer1 Conv2d for first-stage compiler: actual={actual}, expected={expected}")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_ir(memory_plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    h = cfg.IMAGE_HEIGHT
    w = cfg.IMAGE_WIDTH
    in_ch = cfg.LAYER1_IN_CHANNELS
    out_ch = cfg.LAYER1_OUT_CHANNELS
    aligned_out = memory_plan["tensors"]["layer1_conv_out"]["aligned_channels"]

    common_feature = {
        "height": h,
        "width": w,
        "channels": out_ch,
        "aligned_channels": aligned_out,
        "dtype": "int8",
    }

    conv = {
        "op": "conv",
        "layer": "layer1",
        "name": "layer1_conv",
        "input": "image",
        "weight": "layer1_weight",
        "output": "layer1_conv_out",
        "input_shape_nchw": [1, in_ch, h, w],
        "output_shape_nchw": [1, out_ch, h, w],
        "input_channels": in_ch,
        "output_channels": out_ch,
        "kernel_size": [cfg.LAYER1_KERNEL_SIZE, cfg.LAYER1_KERNEL_SIZE],
        "stride": [cfg.LAYER1_STRIDE, cfg.LAYER1_STRIDE],
        "padding": [cfg.LAYER1_PADDING, cfg.LAYER1_PADDING],
        "dtype": "int8",
        "start_position": cfg.CONV_START_POSITION,
    }
    madd = {
        "op": "madd",
        "layer": "layer1",
        "name": "layer1_madd",
        "input": "layer1_conv_out",
        "bias": "layer1_bias",
        "output": "layer1_madd_out",
        "feature": common_feature,
        "start_position": cfg.MADD_START_POSITION,
    }
    relu = {
        "op": "relu",
        "layer": "layer1",
        "name": "layer1_relu",
        "input": "layer1_madd_out",
        "output": "layer1_relu_out",
        "feature": common_feature,
        "relu": {"mode": "relu", "tan": 0, "slope": 0},
    }
    return {"layer1_conv": conv, "layer1_madd": madd, "layer1_relu": relu}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate first-stage layer1 infer IR JSON files")
    parser.add_argument("--model-py", default=str(DEFAULT_MODEL_PY))
    parser.add_argument("--memory-plan", default=str(DEFAULT_MEMORY_PLAN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    model_py = Path(args.model_py)
    memory_plan_path = Path(args.memory_plan)
    out_dir = Path(args.out_dir)
    if not model_py.is_file():
        raise FileNotFoundError(f"model definition not found: {model_py}")
    if not memory_plan_path.is_file():
        raise FileNotFoundError(f"memory plan not found: {memory_plan_path}")

    conv = find_layer1_conv(model_py)
    validate_layer1(conv)

    memory_plan = json.loads(memory_plan_path.read_text(encoding="utf-8"))
    irs = build_ir(memory_plan)
    for name, ir in irs.items():
        write_json(out_dir / f"{name}.json", ir)
        print(f"[OK] IR generated: {out_dir / f'{name}.json'}")


if __name__ == "__main__":
    main()

