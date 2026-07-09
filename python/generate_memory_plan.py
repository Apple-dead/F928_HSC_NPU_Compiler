#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import merge as coe_merge
import npu_config as cfg
from frontend.ir_schema import BACKEND_SUPPORTED_OPS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "data" / "memory_plan.json"
DEFAULT_MODEL_IR = PROJECT_ROOT / "data" / "model_ir.json"
MIN_RUNTIME_FEATURE_HW = 8


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def align_up(value: int, alignment: int = 4) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def hex_addr(value: int) -> str:
    return f"0x{value:08X}"


def aligned_channels(channels: int, alignment: int = 4) -> int:
    return align_up(channels, alignment)


def region(name: str, addr: int, size: int, **extra: Any) -> Dict[str, Any]:
    if addr % 4 != 0:
        raise ValueError(f"{name} start address is not 4-byte aligned: {hex_addr(addr)}")
    if size < 0:
        raise ValueError(f"{name} size must not be negative")
    item = {
        "name": name,
        "addr": hex_addr(addr),
        "size_bytes": size,
        "end_addr_exclusive": hex_addr(addr + size),
    }
    item.update(extra)
    return item


def add_init_region(plan: Dict[str, Any], name: str, size: int, file: str | None = None) -> Dict[str, Any]:
    addr = align_up(plan["_next_init_addr"], 4)
    item = region(name, addr, size, file=file) if file else region(name, addr, size)
    plan["init_regions"].append(item)
    plan["_next_init_addr"] = addr + size
    return item


def add_runtime_region(plan: Dict[str, Any], name: str, size: int, **extra: Any) -> Dict[str, Any]:
    addr = align_up(plan["_next_runtime_addr"], 4)
    item = region(name, addr, size, **extra)
    plan["runtime_regions"].append(item)
    plan["_next_runtime_addr"] = addr + size
    return item


def validate_no_overlap(a_start: int, a_size: int, b_start: int, b_size: int, a_name: str, b_name: str) -> None:
    a_end = a_start + a_size
    b_end = b_start + b_size
    if a_start < b_end and b_start < a_end:
        raise ValueError(
            f"{a_name} [{hex_addr(a_start)}, {hex_addr(a_end)}) overlaps "
            f"{b_name} [{hex_addr(b_start)}, {hex_addr(b_end)})"
        )


def feature_map_elements(height: int, width: int, channels: int) -> int:
    return height * width * channels


def runtime_storage_hw(height: int, width: int) -> tuple[int, int]:
    return align_up(max(height, MIN_RUNTIME_FEATURE_HW), 8), align_up(max(width, MIN_RUNTIME_FEATURE_HW), 8)


def conv_output_hw(height: int, width: int, conv: Dict[str, Any]) -> tuple[int, int]:
    kh, kw = conv["kernel_size"]
    sh, sw = conv["stride"]
    ph, pw = conv["padding"]
    dh, dw = conv.get("dilation", [1, 1])
    if [dh, dw] != [1, 1]:
        raise ValueError(f"dilation other than 1 is not supported: dilation={[dh, dw]}")
    if int(conv.get("groups", 1)) != 1:
        raise ValueError(f"grouped conv is not supported: groups={conv.get('groups')}")
    out_h = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"invalid Conv2d output size for conv={conv}, input={height}x{width}")
    return out_h, out_w


def channel_group_max_channels(
    *,
    input_h: int,
    input_w: int,
    aligned_in_ch: int,
    output_h: int,
    output_w: int,
    aligned_out_ch: int,
) -> int:
    input_size = feature_map_elements(input_h, input_w, aligned_in_ch)
    output_size = feature_map_elements(output_h, output_w, aligned_out_ch)
    threshold = cfg.CHANNEL_GROUP4_FEATURE_SIZE_THRESHOLD
    return 4 if input_size >= threshold or output_size >= threshold else 8


def instr_count_for_execution_plan(execution_plan: List[Dict[str, Any]]) -> int:
    total = 1
    for stage in execution_plan:
        for split in stage.get("splits", []):
            if "conv" in split:
                total += 9
            if "dsmp" in split:
                total += 6
            if "relu" in split:
                total += 7
        if "pool" in split:
            total += 6
    return total


def layer_needs_dsmp(conv: Dict[str, Any]) -> bool:
    stride = conv["stride"]
    padding = conv["padding"]
    if stride[0] != stride[1]:
        raise ValueError(f"non-square stride is not supported: {stride}")
    if padding[0] != padding[1]:
        raise ValueError(f"non-square padding is not supported: {padding}")
    if stride[0] > 2:
        raise ValueError(f"stride > 2 is not supported: stride={stride}")
    if stride[0] == 2:
        if padding != [0, 0]:
            raise ValueError(f"stride=2 only supports padding=0 for DSMP flow, got padding={padding}")
        return True
    return False


def pool_output_hw(height: int, width: int, pool: Dict[str, Any]) -> tuple[int, int]:
    kh, kw = pool["kernel_size"]
    sh, sw = pool["stride"]
    ph, pw = pool["padding"]
    dh, dw = pool.get("dilation", [1, 1])
    ceil_mode = bool(pool.get("ceil_mode", False))
    if [kh, kw] != [2, 2]:
        raise ValueError(f"pool only supports kernel_size=[2, 2], got {[kh, kw]}")
    if [sh, sw] != [2, 2]:
        raise ValueError(f"pool only supports stride=[2, 2], got {[sh, sw]}")
    if [ph, pw] != [0, 0]:
        raise ValueError(f"pool only supports padding=[0, 0], got {[ph, pw]}")
    if [dh, dw] != [1, 1]:
        raise ValueError(f"pool only supports dilation=[1, 1], got {[dh, dw]}")
    if ceil_mode:
        raise ValueError("pool ceil_mode=True is not supported")
    out_h = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"invalid pool output size for pool={pool}, input={height}x{width}")
    return out_h, out_w


def addr_to_int(addr: str) -> int:
    return int(addr, 16)


def npu_channel_groups(out_ch: int, aligned_out_ch: int, max_channels: int = 8) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    start = 0
    while start < aligned_out_ch:
        remaining_valid = max(0, out_ch - start)
        group_channels = max_channels if remaining_valid >= max_channels else align_up(max(remaining_valid, 1), 4)
        group_channels = min(group_channels, aligned_out_ch - start)
        valid_channels = max(0, min(out_ch - start, group_channels))
        groups.append(
            {
                "group_index": len(groups),
                "start_channel": start,
                "channels": group_channels,
                "valid_channels": valid_channels,
                "has_padding": valid_channels < group_channels,
            }
        )
        start += group_channels
    return groups


def build_layer_execution_plan(
    *,
    conv_index: int,
    layer_name: str,
    in_ch: int,
    out_ch: int,
    aligned_in_ch: int,
    aligned_out_ch: int,
    kh: int,
    kw: int,
    input_h: int,
    input_w: int,
    out_h: int,
    out_w: int,
    input_tensor: Dict[str, Any],
    weight_tensor: Dict[str, Any],
    has_bias: bool,
    has_relu: bool,
    conv_out_tensor: Dict[str, Any],
    dsmp_out_tensor: Dict[str, Any] | None,
    relu_out_tensor: Dict[str, Any] | None,
    conv_out_h: int,
    conv_out_w: int,
    conv_storage_h: int,
    conv_storage_w: int,
    output_storage_h: int,
    output_storage_w: int,
) -> Dict[str, Any]:
    if in_ch > 256:
        raise NotImplementedError(f"{layer_name}: conv input channels > 256 are not supported by current NPU.")

    bytes_per_weight_output_channel = aligned_in_ch * kh * kw
    bytes_per_conv_feature_channel = conv_storage_h * conv_storage_w
    bytes_per_feature_channel = output_storage_h * output_storage_w
    has_dsmp = dsmp_out_tensor is not None
    input_feature_elements = feature_map_elements(input_h, input_w, aligned_in_ch)
    output_feature_elements = feature_map_elements(output_storage_h, output_storage_w, aligned_out_ch)
    max_group_channels = channel_group_max_channels(
        input_h=input_h,
        input_w=input_w,
        aligned_in_ch=aligned_in_ch,
        output_h=output_storage_h,
        output_w=output_storage_w,
        aligned_out_ch=aligned_out_ch,
    )
    splits: List[Dict[str, Any]] = []

    # Physical parameter layout is, per group: valid weights, then padded bias
    # only when the source Conv2d has bias.
    parameter_offset = 0
    for group in npu_channel_groups(out_ch, aligned_out_ch, max_channels=max_group_channels):
        start_channel = group["start_channel"]
        conv_feature_offset = start_channel * bytes_per_conv_feature_channel
        feature_offset = start_channel * bytes_per_feature_channel
        weight_size = group["valid_channels"] * bytes_per_weight_output_channel
        bias_size = group["channels"] * 4 if has_bias else 0
        weight_offset = parameter_offset
        bias_offset = weight_offset + weight_size if has_bias else None
        conv_feature_size = group["channels"] * bytes_per_conv_feature_channel
        feature_size = group["channels"] * bytes_per_feature_channel
        item = dict(group)
        item["offsets_bytes"] = {
            "weight": weight_offset,
            "conv_output": conv_feature_offset,
            "dsmp_output": feature_offset if has_dsmp else None,
            "bias": bias_offset,
            "output": feature_offset,
        }
        item["size_bytes"] = {
            "weight": weight_size,
            "conv_output": conv_feature_size,
            "dsmp_output": feature_size if has_dsmp else None,
            "bias": bias_size,
            "output": feature_size,
        }
        item["conv"] = {
            "input_addr": input_tensor["addr"],
            "weight_addr": hex_addr(addr_to_int(weight_tensor["addr"]) + weight_offset),
            "output_addr": hex_addr(addr_to_int(conv_out_tensor["addr"]) + conv_feature_offset),
            "has_bias": has_bias,
        }
        item["bias_addr"] = hex_addr(addr_to_int(weight_tensor["addr"]) + bias_offset) if has_bias else None
        if has_dsmp:
            item["dsmp"] = {
                "input_addr": hex_addr(addr_to_int(conv_out_tensor["addr"]) + conv_feature_offset),
                "output_addr": hex_addr(addr_to_int(dsmp_out_tensor["addr"]) + feature_offset),
                "image_size": conv_storage_h,
                "channels": group["valid_channels"],
            }
            relu_input_addr = hex_addr(addr_to_int(dsmp_out_tensor["addr"]) + feature_offset)
        else:
            relu_input_addr = hex_addr(addr_to_int(conv_out_tensor["addr"]) + feature_offset)
        if relu_out_tensor is not None:
            item["relu"] = {
                "input_addr": relu_input_addr,
                "output_addr": hex_addr(addr_to_int(relu_out_tensor["addr"]) + feature_offset),
            }
        splits.append(item)
        parameter_offset += weight_size + bias_size

    expected_parameter_size = out_ch * bytes_per_weight_output_channel + (aligned_out_ch * 4 if has_bias else 0)
    if parameter_offset != expected_parameter_size:
        raise ValueError(
            f"{layer_name}: parameter layout size mismatch: "
            f"groups={parameter_offset}, expected={expected_parameter_size}"
        )

    return {
        "conv_index": conv_index,
        "layer": layer_name,
        "npu_max_channels_per_pass": 8,
        "channel_group_max_channels": max_group_channels,
        "channel_group4_feature_size_threshold": cfg.CHANNEL_GROUP4_FEATURE_SIZE_THRESHOLD,
        "input_feature_elements": input_feature_elements,
        "output_feature_elements": output_feature_elements,
        "channel_alignment": 4,
        "input_channels": in_ch,
        "output_channels": out_ch,
        "aligned_input_channels": aligned_in_ch,
        "aligned_output_channels": aligned_out_ch,
        "kernel_size": [kh, kw],
        "has_bias": has_bias,
        "has_relu": has_relu,
        "input_hw": [input_h, input_w],
        "logical_conv_output_hw": [conv_out_h, conv_out_w],
        "conv_output_hw": [conv_storage_h, conv_storage_w],
        "has_dsmp": has_dsmp,
        "logical_output_hw": [out_h, out_w],
        "output_hw": [output_storage_h, output_storage_w],
        "minimum_runtime_feature_hw": MIN_RUNTIME_FEATURE_HW,
        "reserved_regions": {
            "conv_out": {
                "addr": conv_out_tensor["addr"],
                "size_bytes": conv_out_tensor["size_bytes"],
                "layout": "channel_groups_are_tightly_packed",
            },
            **(
                {
                    "dsmp_out": {
                        "addr": dsmp_out_tensor["addr"],
                        "size_bytes": dsmp_out_tensor["size_bytes"],
                        "layout": "channel_groups_are_tightly_packed",
                    }
                }
                if dsmp_out_tensor is not None
                else {}
            ),
            **(
                {
                    "relu_out": {
                        "addr": relu_out_tensor["addr"],
                        "size_bytes": relu_out_tensor["size_bytes"],
                        "layout": "channel_groups_are_tightly_packed",
                    }
                }
                if relu_out_tensor is not None
                else {}
            ),
        },
        "splits": splits,
    }


def build_pool_execution_plan(
    *,
    op_name: str,
    layer_name: str,
    input_channels: int,
    input_h: int,
    input_w: int,
    output_h: int,
    output_w: int,
    output_storage_h: int,
    output_storage_w: int,
    input_tensor: Dict[str, Any],
    output_tensor: Dict[str, Any],
    pool: Dict[str, Any],
) -> Dict[str, Any]:
    if input_h % 8 != 0 or input_w % 8 != 0:
        raise ValueError(f"{layer_name}: pool input storage HW must be divisible by 8, got {input_h}x{input_w}")
    if output_storage_h % 8 != 0 or output_storage_w % 8 != 0:
        raise ValueError(
            f"{layer_name}: pool output storage HW must be divisible by 8, got {output_storage_h}x{output_storage_w}"
        )
    if input_channels % 4 != 0:
        raise ValueError(f"{layer_name}: pool input storage channels must be a multiple of 4, got {input_channels}")

    input_size = input_channels * input_h * input_w
    output_size = input_channels * output_storage_h * output_storage_w
    splits: List[Dict[str, Any]] = [
        {
            "group_index": 0,
            "start_channel": 0,
            "channels": input_channels,
            "valid_channels": input_channels,
            "has_padding": False,
            "offsets_bytes": {
                "input": 0,
                "output": 0,
            },
            "size_bytes": {
                "input": input_size,
                "output": output_size,
            },
            "pool": {
                "input_addr": input_tensor["addr"],
                "output_addr": output_tensor["addr"],
                "feature_size": input_w,
                "channels": input_channels,
            },
        }
    ]

    return {
        "op_type": op_name,
        "layer": layer_name,
        "channel_alignment": 4,
        "pool_channel_unit": "all_input_channels",
        "input_channels": input_channels,
        "output_channels": input_channels,
        "aligned_input_channels": input_channels,
        "aligned_output_channels": input_channels,
        "kernel_size": pool["kernel_size"],
        "stride": pool["stride"],
        "padding": pool["padding"],
        "input_hw": [input_h, input_w],
        "logical_output_hw": [output_h, output_w],
        "output_hw": [output_storage_h, output_storage_w],
        "minimum_runtime_feature_hw": MIN_RUNTIME_FEATURE_HW,
        "reserved_regions": {
            "pool_out": {
                "addr": output_tensor["addr"],
                "size_bytes": output_tensor["size_bytes"],
                "layout": "channel_groups_are_tightly_packed",
            },
        },
        "splits": splits,
    }

def validate_input_coe_size(file_text: str, expected_size: int, input_shape: List[int]) -> None:
    path = resolve_project_path(file_text)
    if not path.is_file():
        raise FileNotFoundError(f"input COE not found: {path}")
    values, _ = coe_merge.read_coe_values(path)
    actual_size = len(values) * 4
    if actual_size != expected_size:
        raise ValueError(
            f"{path}: input COE size mismatch for start-layer input; "
            f"file has {actual_size} bytes ({len(values)} 32-bit words), "
            f"expected {expected_size} bytes for storage_shape_nchw={input_shape}"
        )


def load_model_ir(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"model IR not found: {path}; run python ./python/generate_model_ir.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_conv_unit(ops: List[Dict[str, Any]], index: int, units: List[Dict[str, Any]]) -> tuple[Dict[str, Any], int]:
    op = ops[index]
    conv_index = len([unit for unit in units if unit["op_type"] == "conv"]) + 1
    layer_index = int(op["layer_index"])
    unit = {
        "op_type": "conv",
        "conv_index": conv_index,
        "layer_index": layer_index,
        "layer": f"layer{layer_index}",
        "input": op.get("input"),
        "output": op.get("output"),
        "has_relu": False,
        "conv": {
            "in_channels": int(op["in_channels"]),
            "out_channels": int(op["out_channels"]),
            "kernel_size": [int(op["kernel_size"][0]), int(op["kernel_size"][1])],
            "stride": [int(op["stride"][0]), int(op["stride"][1])],
            "padding": [int(op["padding"][0]), int(op["padding"][1])],
            "dilation": [int(op.get("dilation", [1, 1])[0]), int(op.get("dilation", [1, 1])[1])],
            "groups": int(op.get("groups", 1)),
            "has_bias": bool(op.get("has_bias", False)),
        },
    }
    next_index = index + 1
    if next_index < len(ops) and ops[next_index].get("op") == "relu":
        relu = ops[next_index]
        if relu.get("input") != op.get("output"):
            raise NotImplementedError(
                f"Unsupported relu at op id {relu.get('id')}. "
                "Current backend only supports relu directly after its producer conv2d."
            )
        unit["has_relu"] = True
        unit["relu_output"] = relu.get("output")
        next_index += 1
    return unit, next_index


def parse_pool_unit(ops: List[Dict[str, Any]], index: int, units: List[Dict[str, Any]]) -> tuple[Dict[str, Any], int]:
    op = ops[index]
    op_name = op.get("op")
    return (
        {
            "op_type": "avgpool" if op_name == "avgpool2d" else "maxpool",
            "layer": op.get("name", f"{op_name}_{len(units)}"),
            "input": op.get("input"),
            "output": op.get("output"),
            "pool": {
                "kernel_size": [int(op["kernel_size"][0]), int(op["kernel_size"][1])],
                "stride": [int(op["stride"][0]), int(op["stride"][1])],
                "padding": [int(op["padding"][0]), int(op["padding"][1])],
                "dilation": [int(op.get("dilation", [1, 1])[0]), int(op.get("dilation", [1, 1])[1])],
                "ceil_mode": bool(op.get("ceil_mode", False)),
            },
        },
        index + 1,
    )


UNIT_PARSERS = {
    "conv2d": parse_conv_unit,
    "avgpool2d": parse_pool_unit,
    "maxpool2d": parse_pool_unit,
}


def execution_units_from_ir(ir: Dict[str, Any]) -> List[Dict[str, Any]]:
    ops = ir.get("ops", [])
    units: List[Dict[str, Any]] = []
    index = 0
    while index < len(ops):
        op = ops[index]
        op_name = op.get("op")
        if op_name not in BACKEND_SUPPORTED_OPS:
            raise NotImplementedError(
                f"Unsupported op {op_name} at op id {op.get('id')}. "
                "Current backend only supports conv2d/relu/avgpool2d/maxpool2d path."
            )
        if op_name == "relu":
            raise NotImplementedError(
                f"Unsupported relu at op id {op.get('id')}. "
                "Current backend only supports relu directly after conv2d."
            )
        parser = UNIT_PARSERS.get(op_name)
        if parser is not None:
            unit, index = parser(ops, index, units)
            units.append(unit)
            continue
        raise NotImplementedError(
            f"Unsupported op {op_name} at op id {op.get('id')}. "
            "Current backend only supports conv2d/relu/avgpool2d/maxpool2d path."
        )
    if not units:
        raise ValueError("model IR does not contain any op supported by the backend")
    return units


def layers_from_ir(ir: Dict[str, Any]) -> List[Dict[str, Any]]:
    layers = [unit for unit in execution_units_from_ir(ir) if unit["op_type"] == "conv"]
    if not layers:
        raise ValueError("model IR does not contain any conv2d op supported by the backend")
    return layers


def relu_config_from_ir(ir: Dict[str, Any]) -> Dict[str, Any]:
    for op in ir.get("ops", []):
        if op.get("op") == "relu":
            return {
                "mode": "leaky_relu" if "negative_slope" in op else "relu",
                "negative_slope": op.get("negative_slope", 0),
                "tan": int(op.get("tan", 8)),
            }
    return {"mode": "relu", "tan": 8}


def make_value_shape(
    *,
    height: int,
    width: int,
    storage_height: int,
    storage_width: int,
    channels: int,
    aligned_channels: int,
) -> Dict[str, int]:
    return {
        "height": height,
        "width": width,
        "storage_height": storage_height,
        "storage_width": storage_width,
        "channels": channels,
        "aligned_channels": aligned_channels,
    }


def register_value(
    value_tensors: Dict[str, str],
    value_shapes: Dict[str, Dict[str, int]],
    value_name: Any,
    tensor_name: str,
    *,
    height: int,
    width: int,
    storage_height: int,
    storage_width: int,
    channels: int,
    aligned_channels: int,
) -> None:
    if value_name is None:
        raise ValueError(f"cannot register tensor {tensor_name}: missing IR value name")
    key = str(value_name)
    value_tensors[key] = tensor_name
    value_shapes[key] = make_value_shape(
        height=height,
        width=width,
        storage_height=storage_height,
        storage_width=storage_width,
        channels=channels,
        aligned_channels=aligned_channels,
    )


def require_input_value(
    unit: Dict[str, Any],
    value_tensors: Dict[str, str],
    value_shapes: Dict[str, Dict[str, int]],
) -> tuple[str, str, Dict[str, int]]:
    input_name = str(unit.get("input"))
    if input_name not in value_tensors:
        raise ValueError(f"{unit.get('layer', unit.get('op_type'))}: missing input tensor for IR value {input_name!r}")
    return input_name, value_tensors[input_name], value_shapes[input_name]


def init_value_maps(
    input_info: Dict[str, Any],
    *,
    input_h: int,
    input_w: int,
    input_ch: int,
    input_storage_h: int,
    input_storage_w: int,
    image_aligned_ch: int,
) -> tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    value_tensors: Dict[str, str] = {}
    value_shapes: Dict[str, Dict[str, int]] = {}
    input_names = {str(input_info.get("name", "x")), "x"}
    for name in input_names:
        register_value(
            value_tensors,
            value_shapes,
            name,
            "image",
            height=input_h,
            width=input_w,
            storage_height=input_storage_h,
            storage_width=input_storage_w,
            channels=input_ch,
            aligned_channels=image_aligned_ch,
        )
    return value_tensors, value_shapes


def plan_pool_unit(
    plan: Dict[str, Any],
    unit: Dict[str, Any],
    value_tensors: Dict[str, str],
    value_shapes: Dict[str, Dict[str, int]],
) -> None:
    layer_name = unit["layer"]
    _, input_tensor_name, input_shape = require_input_value(unit, value_tensors, value_shapes)
    input_tensor_info = plan["tensors"][input_tensor_name]
    pool_input_h = int(input_shape["storage_height"])
    pool_input_w = int(input_shape["storage_width"])
    pool_channels = int(input_shape["aligned_channels"])
    pool_out_h, pool_out_w = pool_output_hw(pool_input_h, pool_input_w, unit["pool"])
    pool_storage_h, pool_storage_w = runtime_storage_hw(pool_out_h, pool_out_w)
    pool_output_size = pool_storage_h * pool_storage_w * pool_channels
    tensor_name = f"{layer_name}_out"
    runtime = add_runtime_region(
        plan,
        tensor_name,
        pool_output_size,
        channels=pool_channels,
        aligned_channels=pool_channels,
        shape_nchw=[1, pool_channels, pool_out_h, pool_out_w],
        storage_shape_nchw=[1, pool_channels, pool_storage_h, pool_storage_w],
        minimum_runtime_feature_hw=MIN_RUNTIME_FEATURE_HW,
    )
    plan["tensors"][tensor_name] = runtime
    plan["execution_plan"].append(
        build_pool_execution_plan(
            op_name=unit["op_type"],
            layer_name=layer_name,
            input_channels=pool_channels,
            input_h=pool_input_h,
            input_w=pool_input_w,
            output_h=pool_out_h,
            output_w=pool_out_w,
            output_storage_h=pool_storage_h,
            output_storage_w=pool_storage_w,
            input_tensor=input_tensor_info,
            output_tensor=runtime,
            pool=unit["pool"],
        )
    )
    register_value(
        value_tensors,
        value_shapes,
        unit.get("output"),
        tensor_name,
        height=pool_storage_h,
        width=pool_storage_w,
        storage_height=pool_storage_h,
        storage_width=pool_storage_w,
        channels=pool_channels,
        aligned_channels=pool_channels,
    )


def plan_conv_unit(
    plan: Dict[str, Any],
    unit: Dict[str, Any],
    value_tensors: Dict[str, str],
    value_shapes: Dict[str, Dict[str, int]],
) -> None:
    idx = unit["layer_index"]
    conv = unit["conv"]
    has_relu = bool(unit.get("has_relu", False))
    layer_name = f"layer{idx}"
    _, input_tensor_name, input_shape = require_input_value(unit, value_tensors, value_shapes)
    height = int(input_shape["height"])
    width = int(input_shape["width"])
    in_ch = conv["in_channels"]
    out_ch = conv["out_channels"]
    if in_ch != int(input_shape["channels"]):
        raise ValueError(
            f"{layer_name} in_channels={in_ch} does not match input {unit.get('input')} channels="
            f"{input_shape['channels']}"
        )

    aligned_in_ch = aligned_channels(in_ch)
    aligned_out_ch = aligned_channels(out_ch)
    kh, kw = conv["kernel_size"]
    needs_dsmp = layer_needs_dsmp(conv)
    out_h, out_w = conv_output_hw(height, width, conv)
    conv_out_h, conv_out_w = (height, width) if needs_dsmp else (out_h, out_w)
    conv_storage_h, conv_storage_w = runtime_storage_hw(conv_out_h, conv_out_w)
    output_storage_h, output_storage_w = runtime_storage_hw(out_h, out_w)

    weight_size = out_ch * aligned_in_ch * kh * kw
    has_bias = bool(conv.get("has_bias", False))
    bias_size = aligned_out_ch * 4 if has_bias else 0
    parameter_size = weight_size + bias_size
    conv_output_size = conv_storage_h * conv_storage_w * aligned_out_ch
    output_size = output_storage_h * output_storage_w * aligned_out_ch
    params = add_init_region(plan, f"{layer_name}_params", parameter_size, f"coe/{layer_name}_params.coe")

    weight_tensor = {
        "addr": params["addr"],
        "channels": {"in": in_ch, "out": out_ch},
        "aligned_channels": {"in": aligned_in_ch, "out": aligned_out_ch},
        "shape_oihw": [out_ch, in_ch, kh, kw],
        "storage_shape_oihw": [out_ch, aligned_in_ch, kh, kw],
        "size_bytes": weight_size,
        "parameter_region": f"{layer_name}_params",
    }
    plan["tensors"][f"{layer_name}_weight"] = weight_tensor

    if has_bias:
        bias_tensor = {
            "addr": params["addr"],
            "channels": out_ch,
            "aligned_channels": aligned_out_ch,
            "shape": [out_ch],
            "storage_shape": [aligned_out_ch],
            "size_bytes": bias_size,
            "parameter_region": f"{layer_name}_params",
            "layout": "int32_bias_values_padded_per_output_channel_group",
        }
        plan["tensors"][f"{layer_name}_bias"] = bias_tensor
    plan["tensors"][f"{layer_name}_params"] = {
        "addr": params["addr"],
        "size_bytes": parameter_size,
        "has_bias": has_bias,
        "layout": (
            "weight_then_padded_int32_bias_per_output_channel_group"
            if has_bias
            else "weight_only_per_output_channel_group"
        ),
    }

    runtime_tensors: Dict[str, Dict[str, Any]] = {}
    runtime_specs = [
        ("conv", conv_output_size, conv_out_h, conv_out_w, conv_storage_h, conv_storage_w),
    ]
    if needs_dsmp:
        runtime_specs.append(("dsmp", output_size, out_h, out_w, output_storage_h, output_storage_w))
    if has_relu:
        runtime_specs.append(("relu", output_size, out_h, out_w, output_storage_h, output_storage_w))

    for op_name, region_size, region_h, region_w, storage_h, storage_w in runtime_specs:
        tensor_name = f"{layer_name}_{op_name}_out"
        runtime = add_runtime_region(
            plan,
            tensor_name,
            region_size,
            channels=out_ch,
            aligned_channels=aligned_out_ch,
            shape_nchw=[1, out_ch, region_h, region_w],
            storage_shape_nchw=[1, aligned_out_ch, storage_h, storage_w],
            minimum_runtime_feature_hw=MIN_RUNTIME_FEATURE_HW,
        )
        plan["tensors"][tensor_name] = runtime
        runtime_tensors[op_name] = runtime

    plan["execution_plan"].append(
        build_layer_execution_plan(
            conv_index=int(unit["conv_index"]),
            layer_name=layer_name,
            in_ch=in_ch,
            out_ch=out_ch,
            aligned_in_ch=aligned_in_ch,
            aligned_out_ch=aligned_out_ch,
            kh=kh,
            kw=kw,
            input_h=height,
            input_w=width,
            out_h=out_h,
            out_w=out_w,
            input_tensor=plan["tensors"][input_tensor_name],
            weight_tensor=weight_tensor,
            has_bias=has_bias,
            has_relu=has_relu,
            conv_out_tensor=runtime_tensors["conv"],
            dsmp_out_tensor=runtime_tensors.get("dsmp"),
            relu_out_tensor=runtime_tensors.get("relu"),
            conv_out_h=conv_out_h,
            conv_out_w=conv_out_w,
            conv_storage_h=conv_storage_h,
            conv_storage_w=conv_storage_w,
            output_storage_h=output_storage_h,
            output_storage_w=output_storage_w,
        )
    )

    if has_relu:
        output_tensor_name = f"{layer_name}_relu_out"
        register_value(
            value_tensors,
            value_shapes,
            unit.get("relu_output"),
            output_tensor_name,
            height=output_storage_h,
            width=output_storage_w,
            storage_height=output_storage_h,
            storage_width=output_storage_w,
            channels=out_ch,
            aligned_channels=aligned_out_ch,
        )
    elif needs_dsmp:
        output_tensor_name = f"{layer_name}_dsmp_out"
    else:
        output_tensor_name = f"{layer_name}_conv_out"

    register_value(
        value_tensors,
        value_shapes,
        unit.get("output"),
        output_tensor_name,
        height=output_storage_h,
        width=output_storage_w,
        storage_height=output_storage_h,
        storage_width=output_storage_w,
        channels=out_ch,
        aligned_channels=aligned_out_ch,
    )


UNIT_PLANNERS = {
    "conv": plan_conv_unit,
    "avgpool": plan_pool_unit,
    "maxpool": plan_pool_unit,
}


def build_plan(model_ir_path: Path = DEFAULT_MODEL_IR) -> Dict[str, Any]:
    ir = load_model_ir(model_ir_path)
    units = execution_units_from_ir(ir)
    layers = [unit for unit in units if unit["op_type"] == "conv"]
    relu_cfg = relu_config_from_ir(ir)
    input_info = ir.get("input", {})

    if cfg.IMAGE_SOURCE not in ("coe", "external"):
        raise ValueError('IMAGE_SOURCE must be "coe" or "external"')
    input_h = int(input_info.get("height", cfg.INPUT_HEIGHT))
    input_w = int(input_info.get("width", cfg.INPUT_WIDTH))
    fallback_input_ch = layers[0]["conv"]["in_channels"] if layers else 0
    input_ch = int(input_info.get("channels", fallback_input_ch))
    if input_ch <= 0:
        raise ValueError("could not derive input channel count from IR input or first conv")
    if layers and input_ch != layers[0]["conv"]["in_channels"]:
        raise ValueError(
            f"first conv expects {layers[0]['conv']['in_channels']} input channels, "
            f"but IR input has {input_ch} channels"
        )

    image_aligned_ch = aligned_channels(input_ch)
    input_storage_h, input_storage_w = runtime_storage_hw(input_h, input_w)
    image_size = input_storage_h * input_storage_w * image_aligned_ch
    image_file = getattr(cfg, "IMAGE_PATH", "./coe/image.coe") if cfg.IMAGE_SOURCE == "coe" else None
    image_addr = cfg.INIT_BASE_ADDR if cfg.IMAGE_SOURCE == "coe" else cfg.IMAGE_BASE_ADDR
    input_storage_shape = [1, image_aligned_ch, input_storage_h, input_storage_w]

    plan: Dict[str, Any] = {
        "config": {
            "INIT_BASE_ADDR": hex_addr(cfg.INIT_BASE_ADDR),
            "INIT_LIMIT_ADDR": hex_addr(cfg.INIT_LIMIT_ADDR),
            "RUNTIME_BASE_ADDR": hex_addr(cfg.RUNTIME_BASE_ADDR),
            "IMAGE_BASE_ADDR": hex_addr(cfg.IMAGE_BASE_ADDR),
            "IMAGE_SOURCE": cfg.IMAGE_SOURCE,
            "IMAGE_PATH": image_file,
            "MODEL_FORMAT": cfg.MODEL_FORMAT,
            "MODEL_PATH": cfg.MODEL_PATH,
            "INTR_MOVE_PATH": cfg.INTR_MOVE_PATH,
            "INPUT_HEIGHT": cfg.INPUT_HEIGHT,
            "INPUT_WIDTH": cfg.INPUT_WIDTH,
            "INFER_PARSE_MODE": cfg.INFER_PARSE_MODE,
            "INFER_PARSE_OP_LIMIT": cfg.INFER_PARSE_OP_LIMIT,
            "CHANNEL_GROUP4_FEATURE_SIZE_THRESHOLD": cfg.CHANNEL_GROUP4_FEATURE_SIZE_THRESHOLD,
            "MIN_RUNTIME_FEATURE_HW": MIN_RUNTIME_FEATURE_HW,
            "model_ir": str(model_ir_path),
        },
        "relu": relu_cfg,
        "alignment_bytes": 4,
        "image": {
            "source": cfg.IMAGE_SOURCE,
            "addr": hex_addr(image_addr),
            "input_layer": layers[0]["layer"] if layers else units[0]["layer"],
            "channels": input_ch,
            "aligned_channels": image_aligned_ch,
            "shape_nchw": [1, input_ch, input_h, input_w],
            "storage_shape_nchw": input_storage_shape,
            "size_bytes": image_size,
            "file": image_file,
        },
        "model_layers": layers,
        "model_ops": ir.get("ops", []),
        "tensors": {},
        "execution_plan": [],
        "init_regions": [],
        "runtime_regions": [],
        "_next_init_addr": cfg.INIT_BASE_ADDR,
        "_next_runtime_addr": cfg.RUNTIME_BASE_ADDR,
    }

    if cfg.IMAGE_SOURCE == "coe":
        if cfg.IMAGE_BASE_ADDR != cfg.INIT_BASE_ADDR:
            raise ValueError("IMAGE_SOURCE=coe currently requires IMAGE_BASE_ADDR == INIT_BASE_ADDR")
        validate_input_coe_size(image_file, image_size, input_storage_shape)
        add_init_region(plan, "image", image_size, image_file)
    plan["tensors"]["image"] = plan["image"]

    value_tensors, value_shapes = init_value_maps(
        input_info,
        input_h=input_h,
        input_w=input_w,
        input_ch=input_ch,
        input_storage_h=input_storage_h,
        input_storage_w=input_storage_w,
        image_aligned_ch=image_aligned_ch,
    )

    for unit in units:
        planner = UNIT_PLANNERS.get(unit["op_type"])
        if planner is None:
            raise ValueError(f"unsupported execution unit type: {unit['op_type']}")
        planner(plan, unit, value_tensors, value_shapes)

    instr_size = instr_count_for_execution_plan(plan["execution_plan"]) * cfg.INSTR_WORD_BYTES
    instr = add_init_region(plan, "instr", instr_size, "coe/instr.coe")
    plan["tensors"]["instr"] = instr

    init_end = plan["_next_init_addr"]
    if init_end > cfg.INIT_LIMIT_ADDR:
        raise ValueError(
            f"init region exceeds INIT_LIMIT_ADDR: next={hex_addr(init_end)}, limit={hex_addr(cfg.INIT_LIMIT_ADDR)}"
        )
    if cfg.RUNTIME_BASE_ADDR < cfg.INIT_LIMIT_ADDR:
        raise ValueError(
            f"RUNTIME_BASE_ADDR {hex_addr(cfg.RUNTIME_BASE_ADDR)} overlaps init address window "
            f"ending at {hex_addr(cfg.INIT_LIMIT_ADDR)}"
        )

    if cfg.IMAGE_SOURCE == "external":
        validate_no_overlap(
            cfg.IMAGE_BASE_ADDR,
            image_size,
            cfg.INIT_BASE_ADDR,
            init_end - cfg.INIT_BASE_ADDR,
            "external input",
            "init data",
        )
        validate_no_overlap(
            cfg.IMAGE_BASE_ADDR,
            image_size,
            cfg.RUNTIME_BASE_ADDR,
            plan["_next_runtime_addr"] - cfg.RUNTIME_BASE_ADDR,
            "external input",
            "runtime data",
        )

    plan["init_end_addr_exclusive"] = hex_addr(init_end)
    plan["runtime_end_addr_exclusive"] = hex_addr(plan["_next_runtime_addr"])
    del plan["_next_init_addr"]
    del plan["_next_runtime_addr"]
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NPU memory plan from data/model_ir.json")
    parser.add_argument("--model-ir", type=Path, default=DEFAULT_MODEL_IR)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    plan = build_plan(args.model_ir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] memory plan generated: {out_path}")
    print(f"     layers      = {len(plan['model_layers'])}")
    print(f"     init_end    = {plan['init_end_addr_exclusive']}")
    print(f"     runtime_end = {plan['runtime_end_addr_exclusive']}")


if __name__ == "__main__":
    main()
