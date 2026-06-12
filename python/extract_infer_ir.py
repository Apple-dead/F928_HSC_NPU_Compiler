#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import model_parser
import npu_config as cfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_PLAN = PROJECT_ROOT / "data" / "memory_plan.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "infer_ir"


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clean_old_ir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("*.json"):
        path.unlink()


def build_layer_irs(
    layer: Dict[str, Any],
    input_tensor: str,
    input_h: int,
    input_w: int,
    relu_cfg: Dict[str, Any],
    memory_plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    idx = layer["layer_index"]
    layer_name = f"layer{idx}"
    conv_cfg = layer["conv"]
    out_h, out_w = model_parser.conv_output_hw(input_h, input_w, conv_cfg)
    out_ch = conv_cfg["out_channels"]
    aligned_out = memory_plan["tensors"][f"{layer_name}_conv_out"]["aligned_channels"]

    common_feature = {
        "height": out_h,
        "width": out_w,
        "channels": out_ch,
        "aligned_channels": aligned_out,
        "dtype": "int8",
    }

    conv = {
        "order": idx * 3 - 2,
        "op": "conv",
        "layer": layer_name,
        "name": f"{layer_name}_conv",
        "input": input_tensor,
        "weight": f"{layer_name}_weight",
        "output": f"{layer_name}_conv_out",
        "input_shape_nchw": [1, conv_cfg["in_channels"], input_h, input_w],
        "output_shape_nchw": [1, out_ch, out_h, out_w],
        "input_channels": conv_cfg["in_channels"],
        "output_channels": out_ch,
        "kernel_size": conv_cfg["kernel_size"],
        "stride": conv_cfg["stride"],
        "padding": conv_cfg["padding"],
        "dtype": "int8",
        "start_position": cfg.CONV_START_POSITION,
    }
    madd = {
        "order": idx * 3 - 1,
        "op": "madd",
        "layer": layer_name,
        "name": f"{layer_name}_madd",
        "input": f"{layer_name}_conv_out",
        "bias": f"{layer_name}_bias",
        "output": f"{layer_name}_madd_out",
        "feature": common_feature,
        "start_position": cfg.MADD_START_POSITION,
    }
    relu = {
        "order": idx * 3,
        "op": "relu",
        "layer": layer_name,
        "name": f"{layer_name}_relu",
        "input": f"{layer_name}_madd_out",
        "output": f"{layer_name}_relu_out",
        "feature": common_feature,
        "relu": relu_cfg,
    }
    return [conv, madd, relu]


def build_ir_files(model_py: Path, memory_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    layers, relu_cfg = model_parser.parse_model_layers(model_py)
    plan_layers = [layer["layer_index"] for layer in memory_plan["model_layers"]]
    parsed_layers = [layer["layer_index"] for layer in layers]
    if parsed_layers != plan_layers:
        raise ValueError(f"model parse result {parsed_layers} does not match memory_plan layers {plan_layers}")

    all_irs: List[Dict[str, Any]] = []
    input_tensor = "image"
    height = cfg.IMAGE_HEIGHT
    width = cfg.IMAGE_WIDTH
    for layer in layers:
        layer_irs = build_layer_irs(layer, input_tensor, height, width, relu_cfg, memory_plan)
        all_irs.extend(layer_irs)
        height, width = model_parser.conv_output_hw(height, width, layer["conv"])
        input_tensor = f"layer{layer['layer_index']}_relu_out"
    return all_irs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate infer IR JSON files from model structure")
    parser.add_argument("model_py", help="model definition .py path or filename under ./model")
    parser.add_argument("--memory-plan", default=str(DEFAULT_MEMORY_PLAN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    model_py = model_parser.resolve_model_py(args.model_py)
    memory_plan_path = Path(args.memory_plan)
    out_dir = Path(args.out_dir)
    if not memory_plan_path.is_file():
        raise FileNotFoundError(f"memory plan not found: {memory_plan_path}")

    memory_plan = json.loads(memory_plan_path.read_text(encoding="utf-8"))
    irs = build_ir_files(model_py, memory_plan)
    clean_old_ir(out_dir)
    for ir in irs:
        out_path = out_dir / f"{ir['name']}.json"
        write_json(out_path, ir)
        print(f"[OK] IR generated: {out_path}")


if __name__ == "__main__":
    main()

