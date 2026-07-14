#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(ctx) -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    generate_memory_plan = importlib.import_module("generate_memory_plan")
    cfg = importlib.import_module("npu_config")

    old_values = {
        "IMAGE_SOURCE": cfg.IMAGE_SOURCE,
        "IMAGE_BASE_ADDR": cfg.IMAGE_BASE_ADDR,
        "MODEL_FORMAT": cfg.MODEL_FORMAT,
        "MODEL_PATH": cfg.MODEL_PATH,
        "INTR_MOVE_PATH": cfg.INTR_MOVE_PATH,
        "INPUT_HEIGHT": cfg.INPUT_HEIGHT,
        "INPUT_WIDTH": cfg.INPUT_WIDTH,
        "INFER_PARSE_MODE": cfg.INFER_PARSE_MODE,
        "INFER_PARSE_OP_LIMIT": cfg.INFER_PARSE_OP_LIMIT,
    }

    tmp = PROJECT_ROOT / "data/tmp_regression/physical_hw_propagation"
    tmp.mkdir(parents=True, exist_ok=True)
    ir_path = tmp / "model_ir.json"
    ir = {
        "ir_version": 1,
        "frontend": "unit",
        "model_path": "unit",
        "input": {"name": "x", "height": 32, "width": 32, "channels": 1},
        "output": {"name": "avgpool2d_0_out", "producer_op_id": 2, "producer_op": "avgpool2d"},
        "parse": {"mode": 1, "op_limit": None, "is_truncated": False},
        "ops": [
            {
                "id": 0,
                "op": "conv2d",
                "name": "conv2d_0",
                "source_target": "unit.conv2d",
                "input": "x",
                "output": "conv2d_0_out",
                "output_shape": [1, 4, 34, 34],
                "layer_index": 1,
                "param_prefix": "layer1_0",
                "weight_file": "unused",
                "bias_file": None,
                "has_bias": False,
                "in_channels": 1,
                "out_channels": 4,
                "kernel_size": [3, 3],
                "stride": [1, 1],
                "padding": [2, 2],
                "dilation": [1, 1],
                "groups": 1,
            },
            {
                "id": 1,
                "op": "conv2d",
                "name": "conv2d_1",
                "source_target": "unit.conv2d",
                "input": "conv2d_0_out",
                "output": "conv2d_1_out",
                "output_shape": [1, 4, 32, 32],
                "layer_index": 2,
                "param_prefix": "layer2_0",
                "weight_file": "unused",
                "bias_file": None,
                "has_bias": False,
                "in_channels": 4,
                "out_channels": 4,
                "kernel_size": [3, 3],
                "stride": [1, 1],
                "padding": [0, 0],
                "dilation": [1, 1],
                "groups": 1,
            },
            {
                "id": 2,
                "op": "avgpool2d",
                "name": "avgpool2d_0",
                "source_target": "unit.avgpool2d",
                "input": "conv2d_1_out",
                "output": "avgpool2d_0_out",
                "output_shape": [1, 4, 16, 16],
                "kernel_size": [2, 2],
                "stride": [2, 2],
                "padding": [0, 0],
            },
        ],
    }
    ir_path.write_text(json.dumps(ir, indent=2) + "\n", encoding="utf-8")

    try:
        cfg.IMAGE_SOURCE = "external"
        cfg.IMAGE_BASE_ADDR = 0x00100000
        cfg.MODEL_FORMAT = "unit"
        cfg.MODEL_PATH = "unit"
        cfg.INTR_MOVE_PATH = "unit"
        cfg.INPUT_HEIGHT = 32
        cfg.INPUT_WIDTH = 32
        cfg.INFER_PARSE_MODE = 1
        cfg.INFER_PARSE_OP_LIMIT = 999

        plan = generate_memory_plan.build_plan(ir_path)
    finally:
        for key, value in old_values.items():
            setattr(cfg, key, value)

    tensors = plan["tensors"]
    layer1 = tensors["layer1_conv_out"]
    layer2 = tensors["layer2_conv_out"]
    pool = tensors["avgpool2d_0_out"]

    if layer1["shape_nchw"] != [1, 4, 34, 34] or layer1["storage_shape_nchw"] != [1, 4, 40, 40]:
        raise AssertionError(f"layer1 shape mismatch: {layer1}")
    if layer2["shape_nchw"] != [1, 4, 38, 38] or layer2["storage_shape_nchw"] != [1, 4, 40, 40]:
        raise AssertionError(f"layer2 should be computed from physical 40x40 input: {layer2}")
    if pool["shape_nchw"] != [1, 4, 20, 20] or pool["storage_shape_nchw"] != [1, 4, 24, 24]:
        raise AssertionError(f"pool should be computed from physical 40x40 input: {pool}")
