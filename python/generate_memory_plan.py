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


def add_output_region_at(plan: Dict[str, Any], name: str, addr: int, size: int, **extra: Any) -> Dict[str, Any]:
    item = region(name, addr, size, **extra)
    plan["output_regions"].append(item)
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


def validate_feature_hw(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError(f"feature HW must be positive, got {height}x{width}")


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
        if stage.get("op_type") == "usmp":
            total += 7 * len(stage.get("splits", []))
            continue
        if stage.get("op_type") == "linear":
            total += 9 * int(stage.get("output_features", 0))
            continue
        if stage.get("op_type", "conv") == "conv":
            for split in stage.get("splits", []):
                if "conv" in split:
                    total += 11
            if "dsmp" in stage:
                total += 7
            if "relu" in stage:
                total += 7
            continue
        for split in stage.get("splits", []):
            if "pool" in split:
                total += 7 if stage.get("op_type") in ("avgpool", "maxpool") else 6
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


def pool_output_hw(height: int, width: int, pool: Dict[str, Any], *, op_type: str) -> tuple[int, int]:
    kh, kw = pool["kernel_size"]
    sh, sw = pool["stride"]
    ph, pw = pool["padding"]
    dh, dw = pool.get("dilation", [1, 1])
    ceil_mode = bool(pool.get("ceil_mode", False))
    if [kh, kw] != [2, 2]:
        raise ValueError(f"pool only supports kernel_size=[2, 2], got {[kh, kw]}")
    if op_type in ("avgpool", "maxpool"):
        if sh != sw or sh not in (1, 2):
            raise ValueError(f"{op_type} only supports square stride 1 or 2, got {[sh, sw]}")
    elif [sh, sw] != [2, 2]:
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
    padding: List[int],
    stride: List[int],
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
) -> Dict[str, Any]:
    if in_ch > 1024:
        raise NotImplementedError(f"{layer_name}: conv input channels > 1024 are not supported by current RCONV1.")

    bytes_per_weight_output_channel = aligned_in_ch * kh * kw
    bytes_per_conv_feature_channel = conv_out_h * conv_out_w
    bytes_per_feature_channel = out_h * out_w
    has_dsmp = dsmp_out_tensor is not None
    input_feature_elements = feature_map_elements(input_h, input_w, aligned_in_ch)
    output_feature_elements = feature_map_elements(out_h, out_w, aligned_out_ch)
    max_group_channels = channel_group_max_channels(
        input_h=input_h,
        input_w=input_w,
        aligned_in_ch=aligned_in_ch,
        output_h=out_h,
        output_w=out_w,
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

        splits.append(item)
        parameter_offset += weight_size + bias_size

    expected_parameter_size = out_ch * bytes_per_weight_output_channel + (aligned_out_ch * 4 if has_bias else 0)
    if parameter_offset != expected_parameter_size:
        raise ValueError(
            f"{layer_name}: parameter layout size mismatch: "
            f"groups={parameter_offset}, expected={expected_parameter_size}"
        )

    layer_plan = {
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
        "padding": padding,
        "stride": stride,
        "has_bias": has_bias,
        "has_relu": has_relu,
        "input_hw": [input_h, input_w],
        "conv_output_hw": [conv_out_h, conv_out_w],
        "has_dsmp": has_dsmp,
        "output_hw": [out_h, out_w],
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
    if has_dsmp:
        if conv_out_h != conv_out_w:
            raise ValueError(f"{layer_name}: dsmp input HW must be square, got {conv_out_h}x{conv_out_w}")
        if conv_out_w > 1024:
            raise NotImplementedError(
                f"{layer_name}: dsmp feature size > 1024 is not supported by current RDSMP."
            )
        if out_ch > 1024:
            raise NotImplementedError(f"{layer_name}: dsmp input channels > 1024 are not supported by current RDSMP.")
        layer_plan["dsmp"] = {
            "input_addr": conv_out_tensor["addr"],
            "output_addr": dsmp_out_tensor["addr"],
            "input_hw": [conv_out_h, conv_out_w],
            "feature_size": conv_out_w,
            "channels": out_ch,
        }
        relu_input_addr = dsmp_out_tensor["addr"]
    else:
        relu_input_addr = conv_out_tensor["addr"]
    if relu_out_tensor is not None:
        layer_plan["relu"] = {
            "input_addr": relu_input_addr,
            "output_addr": relu_out_tensor["addr"],
            "feature_size": out_w,
            "channels": out_ch,
        }
    return layer_plan


def build_pool_execution_plan(
    *,
    op_name: str,
    layer_name: str,
    input_channels: int,
    aligned_input_channels: int,
    input_h: int,
    input_w: int,
    output_h: int,
    output_w: int,
    input_tensor: Dict[str, Any],
    output_tensor: Dict[str, Any],
    pool: Dict[str, Any],
) -> Dict[str, Any]:
    if op_name in ("avgpool", "maxpool") and input_channels > 1024:
        raise NotImplementedError(
            f"{layer_name}: {op_name} input channels > 1024 are not supported by current register format."
        )
    if aligned_input_channels % 4 != 0:
        raise ValueError(
            f"{layer_name}: pool input aligned channels must be a multiple of 4, got {aligned_input_channels}"
        )

    input_size = aligned_input_channels * input_h * input_w
    output_size = aligned_input_channels * output_h * output_w
    splits: List[Dict[str, Any]] = [
        {
            "group_index": 0,
            "start_channel": 0,
            "channels": input_channels,
            "valid_channels": input_channels,
            "aligned_channels": aligned_input_channels,
            "has_padding": aligned_input_channels != input_channels,
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
                "stride": pool["stride"],
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
        "aligned_input_channels": aligned_input_channels,
        "aligned_output_channels": aligned_input_channels,
        "kernel_size": pool["kernel_size"],
        "stride": pool["stride"],
        "padding": pool["padding"],
        "input_hw": [input_h, input_w],
        "output_hw": [output_h, output_w],
        "reserved_regions": {
            "pool_out": {
                "addr": output_tensor["addr"],
                "size_bytes": output_tensor["size_bytes"],
                "layout": "channel_groups_are_tightly_packed",
            },
        },
        "splits": splits,
    }


def build_usmp_execution_plan(
    *,
    layer_name: str,
    input_channels: int,
    aligned_input_channels: int,
    input_h: int,
    input_w: int,
    output_h: int,
    output_w: int,
    input_tensor: Dict[str, Any],
    output_tensor: Dict[str, Any],
    usmp: Dict[str, Any],
) -> Dict[str, Any]:
    if input_channels > 1024:
        raise NotImplementedError(
            f"{layer_name}: usmp input channels > 1024 are not supported by current RUSMP."
        )
    if input_h != input_w:
        raise ValueError(f"{layer_name}: usmp input HW must be square, got {input_h}x{input_w}")
    if output_h != input_h * 2 or output_w != input_w * 2:
        raise ValueError(
            f"{layer_name}: current USMP backend only supports nearest x2, got "
            f"input={input_h}x{input_w}, output={output_h}x{output_w}"
        )

    input_size = aligned_input_channels * input_h * input_w
    output_size = aligned_input_channels * output_h * output_w
    splits: List[Dict[str, Any]] = [
        {
            "group_index": 0,
            "start_channel": 0,
            "channels": input_channels,
            "valid_channels": input_channels,
            "aligned_channels": aligned_input_channels,
            "has_padding": aligned_input_channels != input_channels,
            "offsets_bytes": {
                "input": 0,
                "output": 0,
            },
            "size_bytes": {
                "input": input_size,
                "output": output_size,
            },
            "usmp": {
                "input_addr": input_tensor["addr"],
                "output_addr": output_tensor["addr"],
                "feature_size": input_w,
                "channels": input_channels,
                "scale_factor": usmp["scale_factor"],
                "mode": usmp["mode"],
            },
        }
    ]

    return {
        "op_type": "usmp",
        "layer": layer_name,
        "channel_alignment": 4,
        "usmp_channel_unit": "all_input_channels",
        "input_channels": input_channels,
        "output_channels": input_channels,
        "aligned_input_channels": aligned_input_channels,
        "aligned_output_channels": aligned_input_channels,
        "scale_factor": usmp["scale_factor"],
        "mode": usmp["mode"],
        "input_hw": [input_h, input_w],
        "logical_output_hw": [output_h, output_w],
        "output_hw": [output_h, output_w],
        "reserved_regions": {
            "usmp_out": {
                "addr": output_tensor["addr"],
                "size_bytes": output_tensor["size_bytes"],
                "layout": "channel_groups_are_tightly_packed",
            },
        },
        "splits": splits,
    }


def shape_from_nchw(shape: List[int]) -> Dict[str, int]:
    if len(shape) != 4:
        raise ValueError(f"expected NCHW shape, got {shape}")
    return make_value_shape(
        height=int(shape[2]),
        width=int(shape[3]),
        channels=int(shape[1]),
        aligned_channels=aligned_channels(int(shape[1])),
    )


def output_shape_by_value(ir: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}
    input_info = ir.get("input", {})
    input_name = input_info.get("name")
    if input_name:
        result[str(input_name)] = make_value_shape(
            height=int(input_info["height"]),
            width=int(input_info["width"]),
            channels=int(input_info["channels"]),
            aligned_channels=aligned_channels(int(input_info["channels"])),
        )
    for op in ir.get("ops", []):
        output = op.get("output")
        shape = op.get("output_shape")
        if output is not None and shape is not None and len(shape) == 4:
            result[str(output)] = shape_from_nchw([int(value) for value in shape])
    return result


def model_output_items(ir: Dict[str, Any]) -> List[Dict[str, Any]]:
    outputs = ir.get("outputs")
    if isinstance(outputs, list) and outputs:
        return outputs
    output = ir.get("output", {})
    if output.get("name") is None:
        return []
    return [
        {
            "index": 0,
            "name": output["name"],
            "source": output.get("source"),
            "producer_op_id": output.get("producer_op_id"),
            "producer_op": output.get("producer_op"),
        }
    ]


def build_cat_aliases(ir: Dict[str, Any]) -> Dict[str, Any]:
    shapes = output_shape_by_value(ir)
    input_aliases: Dict[str, Dict[str, Any]] = {}
    output_tensors: Dict[str, Dict[str, Any]] = {}
    for op in ir.get("ops", []):
        if op.get("op") != "cat":
            continue
        if int(op.get("dim", 0)) != 1:
            continue
        cat_output = str(op["output"])
        inputs = [str(value) for value in op.get("input", [])]
        if len(inputs) < 2:
            continue
        cat_shape = shapes.get(cat_output)
        if cat_shape is None:
            continue
        height = int(cat_shape["height"])
        width = int(cat_shape["width"])
        output_channels = int(cat_shape["channels"])
        aligned_output_ch = int(cat_shape["aligned_channels"])
        if output_channels != aligned_output_ch:
            continue

        offset = 0
        input_infos: List[Dict[str, Any]] = []
        can_alias = True
        for input_value in inputs:
            input_shape = shapes.get(input_value)
            if input_shape is None:
                can_alias = False
                break
            channels = int(input_shape["channels"])
            aligned_ch = int(input_shape["aligned_channels"])
            if int(input_shape["height"]) != height or int(input_shape["width"]) != width or channels != aligned_ch:
                can_alias = False
                break
            size = height * width * aligned_ch
            input_infos.append(
                {
                    "value": input_value,
                    "offset_bytes": offset,
                    "size_bytes": size,
                    "segment_height": height,
                    "segment_width": width,
                    "segment_channels": channels,
                    "segment_aligned_channels": aligned_ch,
                }
            )
            offset += size
        if not can_alias or offset != height * width * aligned_output_ch:
            continue

        tensor_name = f"{op.get('name', 'cat')}_out"
        output_info = {
            "cat_output": cat_output,
            "tensor_name": tensor_name,
            "layer": op.get("name", tensor_name),
            "inputs": inputs,
            "input_infos": input_infos,
            "height": height,
            "width": width,
            "channels": output_channels,
            "aligned_channels": aligned_output_ch,
            "size_bytes": offset,
        }
        output_tensors[cat_output] = output_info
        for input_info in input_infos:
            input_aliases[input_info["value"]] = {**output_info, "segment": input_info}
    return {
        "input_aliases": input_aliases,
        "output_tensors": output_tensors,
    }


def ensure_cat_output_tensor(plan: Dict[str, Any], alias: Dict[str, Any]) -> Dict[str, Any]:
    tensor_name = alias["tensor_name"]
    tensor = plan["tensors"].get(tensor_name)
    if tensor is not None:
        return tensor
    runtime = add_runtime_region(
        plan,
        tensor_name,
        int(alias["size_bytes"]),
        channels=int(alias["channels"]),
        aligned_channels=int(alias["aligned_channels"]),
        shape_nchw=[1, int(alias["channels"]), int(alias["height"]), int(alias["width"])],
        aligned_shape_nchw=[1, int(alias["aligned_channels"]), int(alias["height"]), int(alias["width"])],
        layout="channel_concat_alias_4channel_pixel_interleaved",
    )
    plan["tensors"][tensor_name] = runtime
    plan.setdefault("aliases", []).append(
        {
            "type": "cat",
            "output": alias["cat_output"],
            "tensor": tensor_name,
            "inputs": alias["inputs"],
            "layout": "producer_outputs_preplaced_in_channel_order",
        }
    )
    return runtime


def add_feature_region_for_value(
    plan: Dict[str, Any],
    value_name: Any,
    tensor_name: str,
    size: int,
    **extra: Any,
) -> Dict[str, Any]:
    alias = plan.get("_cat_input_aliases", {}).get(str(value_name)) if value_name is not None else None
    if alias is None:
        return add_runtime_region(plan, tensor_name, size, **extra)
    segment = alias["segment"]
    if int(segment["size_bytes"]) != size:
        raise ValueError(
            f"{tensor_name}: cat alias size mismatch for {value_name!r}: "
            f"producer={size}, alias={segment['size_bytes']}"
        )
    cat_tensor = ensure_cat_output_tensor(plan, alias)
    item = region(
        tensor_name,
        addr_to_int(cat_tensor["addr"]) + int(segment["offset_bytes"]),
        size,
        **extra,
        alias_of=alias["tensor_name"],
        alias_offset_bytes=int(segment["offset_bytes"]),
    )
    return item


def replace_execution_plan_addresses(value: Any, replacements: List[tuple[int, int, int]]) -> Any:
    if isinstance(value, dict):
        return {key: replace_execution_plan_addresses(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_execution_plan_addresses(item, replacements) for item in value]
    if isinstance(value, str) and value.startswith("0x"):
        try:
            addr = int(value, 16)
        except ValueError:
            return value
        for old_base, new_base, size in replacements:
            if old_base <= addr < old_base + size:
                return hex_addr(new_base + (addr - old_base))
    return value


def relocate_model_outputs(
    plan: Dict[str, Any],
    outputs: List[Dict[str, Any]],
    value_tensors: Dict[str, str],
    value_shapes: Dict[str, Dict[str, int]],
) -> None:
    if not outputs:
        return

    enable_output_addr = bool(getattr(cfg, "ENABLE_OUTPUT_ADDR", False))
    output_infos: List[Dict[str, Any]] = []
    output_tensor_names: set[str] = set()

    for index, output in enumerate(outputs):
        value_name = str(output["name"])
        tensor_name = value_tensors.get(value_name)
        if tensor_name is None:
            raise ValueError(f"model output {index} references unknown IR value {value_name!r}")
        tensor = plan["tensors"][tensor_name]
        output_infos.append(
            {
                "index": index,
                "value_name": value_name,
                "tensor_name": tensor_name,
                "tensor": tensor,
                "output": output,
            }
        )
        output_tensor_names.add(tensor_name)

    plan["runtime_regions"] = [
        item for item in plan["runtime_regions"] if item.get("name") not in output_tensor_names
    ]
    runtime_tail = cfg.RUNTIME_BASE_ADDR
    if plan["runtime_regions"]:
        runtime_tail = max(addr_to_int(item["end_addr_exclusive"]) for item in plan["runtime_regions"])

    cursor = cfg.OUTPUT_BASE_ADDR if enable_output_addr else align_up(runtime_tail, 4)
    output_regions: List[Dict[str, Any]] = []
    replacements: List[tuple[int, int, int]] = []
    model_outputs: List[Dict[str, Any]] = []

    for info in output_infos:
        index = int(info["index"])
        value_name = str(info["value_name"])
        tensor_name = str(info["tensor_name"])
        tensor = info["tensor"]
        output = info["output"]
        old_base = addr_to_int(tensor["addr"])
        size = int(tensor["size_bytes"])
        cursor = align_up(cursor, 4)
        new_region = add_output_region_at(
            plan,
            f"output{index}_{tensor_name}",
            cursor,
            size,
            is_model_output=True,
            output_index=index,
            output_value=value_name,
            output_tensor=tensor_name,
            source=output.get("source"),
            channels=tensor.get("channels", value_shapes[value_name].get("channels")),
            aligned_channels=tensor.get("aligned_channels", value_shapes[value_name].get("aligned_channels")),
            shape_nchw=tensor.get("shape_nchw"),
            aligned_shape_nchw=tensor.get("aligned_shape_nchw"),
            shape=tensor.get("shape"),
            layout=tensor.get("layout", "channel_groups_are_tightly_packed"),
        )
        replacements.append((old_base, cursor, size))
        tensor.update(
            {
                "addr": new_region["addr"],
                "end_addr_exclusive": new_region["end_addr_exclusive"],
                "is_model_output": True,
                "output_index": index,
                "output_value": value_name,
            }
        )
        output_regions.append(new_region)
        model_outputs.append(
            {
                "index": index,
                "name": value_name,
                "tensor": tensor_name,
                "addr": new_region["addr"],
                "size_bytes": size,
                "end_addr_exclusive": new_region["end_addr_exclusive"],
                "shape": tensor.get("shape_nchw", tensor.get("shape")),
            }
        )
        cursor += size

    plan["execution_plan"] = replace_execution_plan_addresses(plan["execution_plan"], replacements)
    plan["model_outputs"] = model_outputs
    plan["_next_runtime_addr"] = runtime_tail if enable_output_addr else cursor


def validate_input_coe_size(file_text: str, expected_size: int, expected_shape: List[int]) -> None:
    path = resolve_project_path(file_text)
    if not path.is_file():
        raise FileNotFoundError(f"input COE not found: {path}")
    values, _ = coe_merge.read_coe_values(path)
    actual_size = len(values) * 4
    if actual_size != expected_size:
        raise ValueError(
            f"{path}: input COE size mismatch for start-layer input; "
            f"file has {actual_size} bytes ({len(values)} 32-bit words), "
            f"expected {expected_size} bytes for aligned_shape_nchw={expected_shape}"
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


def parse_fused_pad_maxpool_unit(
    ops: List[Dict[str, Any]],
    index: int,
    units: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], int]:
    pad_op = ops[index]
    pool_op = ops[index + 1]
    if pad_op.get("pad") != [0, 1, 0, 1]:
        raise ValueError(f"{pad_op.get('name')}: only pad [0, 1, 0, 1] can be fused into maxpool")
    if pad_op.get("mode") != "constant" or float(pad_op.get("value", 0.0)) != 0.0:
        raise ValueError(
            f"{pad_op.get('name')}: only constant zero pad can be fused into maxpool, "
            f"got mode={pad_op.get('mode')!r}, value={pad_op.get('value')!r}"
        )
    unit, _ = parse_pool_unit(ops, index + 1, units)
    pool = unit["pool"]
    if unit["op_type"] != "maxpool" or pool["kernel_size"] != [2, 2] or pool["stride"] != [1, 1]:
        raise ValueError(
            f"{pad_op.get('name')}: only pad [0, 1, 0, 1] followed by maxpool kernel=2 stride=1 can be fused"
        )
    unit["input"] = pad_op.get("input")
    unit["fused_pad"] = {
        "op": pad_op.get("name"),
        "pad": pad_op.get("pad"),
        "mode": pad_op.get("mode"),
        "value": pad_op.get("value", 0.0),
        "npu_semantics": "maxpool_implicit_right_bottom_zero_boundary",
    }
    return unit, index + 2


def parse_usmp_unit(ops: List[Dict[str, Any]], index: int, units: List[Dict[str, Any]]) -> tuple[Dict[str, Any], int]:
    op = ops[index]
    return (
        {
            "op_type": "usmp",
            "layer": op.get("name", f"usmp_{len(units)}"),
            "input": op.get("input"),
            "output": op.get("output"),
            "usmp": {
                "scale_factor": [int(op["scale_factor"][0]), int(op["scale_factor"][1])],
                "mode": op.get("mode", "nearest"),
            },
        },
        index + 1,
    )


def parse_pad_unit(ops: List[Dict[str, Any]], index: int, units: List[Dict[str, Any]]) -> tuple[Dict[str, Any], int]:
    op = ops[index]
    return (
        {
            "op_type": "pad",
            "layer": op.get("name", f"pad_{len(units)}"),
            "input": op.get("input"),
            "output": op.get("output"),
            "pad": [int(value) for value in op["pad"]],
            "mode": op.get("mode", "constant"),
            "value": float(op.get("value", 0.0)),
        },
        index + 1,
    )


def parse_cat_unit(ops: List[Dict[str, Any]], index: int, units: List[Dict[str, Any]]) -> tuple[Dict[str, Any], int]:
    op = ops[index]
    return (
        {
            "op_type": "cat",
            "layer": op.get("name", f"cat_{len(units)}"),
            "inputs": [str(value) for value in op["input"]],
            "output": op.get("output"),
            "dim": int(op.get("dim", 0)),
        },
        index + 1,
    )


def parse_linear_unit(ops: List[Dict[str, Any]], index: int, units: List[Dict[str, Any]]) -> tuple[Dict[str, Any], int]:
    op = ops[index]
    linear_index = len([unit for unit in units if unit["op_type"] == "linear"]) + 1
    return (
        {
            "op_type": "linear",
            "linear_index": linear_index,
            "layer": f"linear{linear_index}",
            "input": op.get("input"),
            "output": op.get("output"),
            "weight_file": op.get("weight_file"),
            "bias_file": op.get("bias_file"),
            "has_bias": bool(op.get("has_bias", False)),
            "in_features": int(op["in_features"]),
            "out_features": int(op["out_features"]),
        },
        index + 1,
    )


UNIT_PARSERS = {
    "conv2d": parse_conv_unit,
    "avgpool2d": parse_pool_unit,
    "maxpool2d": parse_pool_unit,
    "upsample_nearest2d": parse_usmp_unit,
    "pad": parse_pad_unit,
    "cat": parse_cat_unit,
    "linear": parse_linear_unit,
}


def execution_units_from_ir(ir: Dict[str, Any]) -> List[Dict[str, Any]]:
    ops = ir.get("ops", [])
    units: List[Dict[str, Any]] = []
    index = 0
    while index < len(ops):
        op = ops[index]
        op_name = op.get("op")
        if (
            op_name == "pad"
            and index + 1 < len(ops)
            and ops[index + 1].get("op") == "maxpool2d"
            and ops[index + 1].get("input") == op.get("output")
        ):
            unit, index = parse_fused_pad_maxpool_unit(ops, index, units)
            units.append(unit)
            continue
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
    channels: int,
    aligned_channels: int,
) -> Dict[str, int]:
    return {
        "height": height,
        "width": width,
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
    input_h = int(input_shape["height"])
    input_w = int(input_shape["width"])
    input_channels = int(input_shape["channels"])
    aligned_input_ch = int(input_shape["aligned_channels"])
    validate_feature_hw(input_h, input_w)
    if unit.get("fused_pad") is not None:
        if unit["op_type"] != "maxpool":
            raise ValueError(f"{layer_name}: only maxpool supports fused pad semantics")
        pool_out_h, pool_out_w = input_h, input_w
    else:
        pool_out_h, pool_out_w = pool_output_hw(input_h, input_w, unit["pool"], op_type=unit["op_type"])
    pool_output_size = pool_out_h * pool_out_w * aligned_input_ch
    tensor_name = f"{layer_name}_out"
    runtime = add_feature_region_for_value(
        plan,
        unit.get("output"),
        tensor_name,
        pool_output_size,
        channels=input_channels,
        aligned_channels=aligned_input_ch,
        shape_nchw=[1, input_channels, pool_out_h, pool_out_w],
        aligned_shape_nchw=[1, aligned_input_ch, pool_out_h, pool_out_w],
    )
    plan["tensors"][tensor_name] = runtime
    stage = build_pool_execution_plan(
        op_name=unit["op_type"],
        layer_name=layer_name,
        input_channels=input_channels,
        aligned_input_channels=aligned_input_ch,
        input_h=input_h,
        input_w=input_w,
        output_h=pool_out_h,
        output_w=pool_out_w,
        input_tensor=input_tensor_info,
        output_tensor=runtime,
        pool=unit["pool"],
    )
    if unit.get("fused_pad") is not None:
        stage["fused_pad"] = unit["fused_pad"]
        stage["output_hw_semantics"] = "npu_implicit_right_bottom_zero_boundary"
    plan["execution_plan"].append(stage)
    register_value(
        value_tensors,
        value_shapes,
        unit.get("output"),
        tensor_name,
        height=pool_out_h,
        width=pool_out_w,
        channels=input_channels,
        aligned_channels=aligned_input_ch,
    )


def plan_usmp_unit(
    plan: Dict[str, Any],
    unit: Dict[str, Any],
    value_tensors: Dict[str, str],
    value_shapes: Dict[str, Dict[str, int]],
) -> None:
    layer_name = unit["layer"]
    _, input_tensor_name, input_shape = require_input_value(unit, value_tensors, value_shapes)
    input_tensor_info = plan["tensors"][input_tensor_name]
    input_h = int(input_shape["height"])
    input_w = int(input_shape["width"])
    input_channels = int(input_shape["channels"])
    aligned_input_ch = int(input_shape["aligned_channels"])
    validate_feature_hw(input_h, input_w)

    usmp = unit["usmp"]
    scale = [int(usmp["scale_factor"][0]), int(usmp["scale_factor"][1])]
    if usmp.get("mode") != "nearest":
        raise ValueError(f"{layer_name}: only nearest USMP is supported, got mode={usmp.get('mode')!r}")
    if scale != [2, 2]:
        raise ValueError(f"{layer_name}: only scale_factor=[2, 2] is supported, got {scale}")
    output_h = input_h * 2
    output_w = input_w * 2
    output_size = output_h * output_w * aligned_input_ch
    tensor_name = f"{layer_name}_out"
    runtime = add_feature_region_for_value(
        plan,
        unit.get("output"),
        tensor_name,
        output_size,
        channels=input_channels,
        aligned_channels=aligned_input_ch,
        shape_nchw=[1, input_channels, output_h, output_w],
        aligned_shape_nchw=[1, aligned_input_ch, output_h, output_w],
    )
    plan["tensors"][tensor_name] = runtime
    plan["execution_plan"].append(
        build_usmp_execution_plan(
            layer_name=layer_name,
            input_channels=input_channels,
            aligned_input_channels=aligned_input_ch,
            input_h=input_h,
            input_w=input_w,
            output_h=output_h,
            output_w=output_w,
            input_tensor=input_tensor_info,
            output_tensor=runtime,
            usmp=unit["usmp"],
        )
    )
    register_value(
        value_tensors,
        value_shapes,
        unit.get("output"),
        tensor_name,
        height=output_h,
        width=output_w,
        channels=input_channels,
        aligned_channels=aligned_input_ch,
    )


def plan_pad_unit(
    plan: Dict[str, Any],
    unit: Dict[str, Any],
    value_tensors: Dict[str, str],
    value_shapes: Dict[str, Dict[str, int]],
) -> None:
    raise NotImplementedError(
        f"{unit['layer']}: standalone pad is not supported. "
        "Only constant zero pad [0, 1, 0, 1] directly followed by maxpool2d kernel=2 stride=1 "
        "can be fused into the NPU MAXPOOL boundary semantics."
    )


def plan_cat_unit(
    plan: Dict[str, Any],
    unit: Dict[str, Any],
    value_tensors: Dict[str, str],
    value_shapes: Dict[str, Dict[str, int]],
) -> None:
    layer_name = unit["layer"]
    if int(unit.get("dim", 0)) != 1:
        raise ValueError(f"{layer_name}: only channel-dimension cat (dim=1) is supported, got dim={unit.get('dim')}")

    alias = plan.get("_cat_output_tensors", {}).get(str(unit.get("output")))
    if alias is None:
        raise NotImplementedError(f"{layer_name}: cat cannot be represented as a preplanned channel alias")
    cat_tensor = ensure_cat_output_tensor(plan, alias)
    for segment in alias["input_infos"]:
        value = segment["value"]
        tensor_name = value_tensors.get(value)
        if tensor_name is None:
            raise ValueError(f"{layer_name}: missing preplanned cat input {value!r}")
        tensor = plan["tensors"][tensor_name]
        expected_addr = addr_to_int(cat_tensor["addr"]) + int(segment["offset_bytes"])
        if addr_to_int(tensor["addr"]) != expected_addr:
            raise ValueError(
                f"{layer_name}: cat input {value!r} is not placed at expected address "
                f"{hex_addr(expected_addr)}, got {tensor['addr']}"
            )

    register_value(
        value_tensors,
        value_shapes,
        unit.get("output"),
        alias["tensor_name"],
        height=int(alias["height"]),
        width=int(alias["width"]),
        channels=int(alias["channels"]),
        aligned_channels=int(alias["aligned_channels"]),
    )


def plan_linear_unit(
    plan: Dict[str, Any],
    unit: Dict[str, Any],
    value_tensors: Dict[str, str],
    value_shapes: Dict[str, Dict[str, int]],
) -> None:
    layer_name = unit["layer"]
    linear_index = int(unit["linear_index"])
    _, input_tensor_name, input_shape = require_input_value(unit, value_tensors, value_shapes)
    input_tensor = plan["tensors"][input_tensor_name]

    input_h = int(input_shape["height"])
    input_w = int(input_shape["width"])
    input_ch = int(input_shape["channels"])
    aligned_input_ch = int(input_shape["aligned_channels"])
    in_features = int(unit["in_features"])
    out_features = int(unit["out_features"])
    max_in_features = input_ch * input_h * input_w
    if in_features > max_in_features:
        raise ValueError(
            f"{layer_name}: linear in_features={in_features} exceeds previous physical output "
            f"{input_ch}x{input_h}x{input_w}={max_in_features}"
        )

    input_bytes = aligned_input_ch * input_h * input_w
    if input_bytes % 4 != 0:
        raise ValueError(f"{layer_name}: FULL input byte count must be divisible by 4, got {input_bytes}")
    input_words = input_bytes // 4
    if not 1 <= input_words <= 65536:
        raise ValueError(f"{layer_name}: FULL input words must be in [1, 65536], got {input_words}")

    has_bias = bool(unit.get("has_bias", False))
    bias_bytes_per_output = 4 if has_bias else 0
    bytes_per_output_params = input_bytes + bias_bytes_per_output
    parameter_size = out_features * bytes_per_output_params
    output_size = out_features

    params = add_init_region(plan, f"{layer_name}_params", parameter_size, f"coe/{layer_name}_params.coe")
    output_tensor = add_runtime_region(
        plan,
        f"{layer_name}_out",
        output_size,
        channels=out_features,
        aligned_channels=out_features,
        shape=[out_features],
        layout="signed_int8_linear_outputs_tightly_packed",
    )

    plan["tensors"][f"{layer_name}_weight"] = {
        "addr": params["addr"],
        "shape_oi": [out_features, in_features],
        "input_shape_nchw": [1, input_ch, input_h, input_w],
        "input_aligned_shape_nchw": [1, aligned_input_ch, input_h, input_w],
        "size_bytes": out_features * input_bytes,
        "parameter_region": f"{layer_name}_params",
        "layout": "per_output_4channel_pixel_interleaved",
    }
    if has_bias:
        plan["tensors"][f"{layer_name}_bias"] = {
            "addr": params["addr"],
            "channels": out_features,
            "shape": [out_features],
            "storage_shape": [out_features],
            "size_bytes": out_features * 4,
            "parameter_region": f"{layer_name}_params",
            "layout": "int32_bias_after_each_linear_output_weight_block",
        }
    plan["tensors"][f"{layer_name}_params"] = {
        "addr": params["addr"],
        "size_bytes": parameter_size,
        "has_bias": has_bias,
        "bytes_per_output": bytes_per_output_params,
        "layout": (
            "per_output_padded_weight_then_int32_bias"
            if has_bias
            else "per_output_padded_weight"
        ),
    }
    plan["tensors"][f"{layer_name}_out"] = output_tensor

    splits: List[Dict[str, Any]] = []
    param_base = addr_to_int(params["addr"])
    output_base = addr_to_int(output_tensor["addr"])
    for output_index in range(out_features):
        param_offset = output_index * bytes_per_output_params
        output_offset = output_index
        splits.append(
            {
                "group_index": output_index,
                "output_index": output_index,
                "start_channel": 0,
                "channels": aligned_input_ch,
                "valid_channels": input_ch,
                "has_padding": aligned_input_ch != input_ch,
                "offsets_bytes": {
                    "weight": param_offset,
                    "bias": param_offset + input_bytes if has_bias else None,
                    "output": output_offset,
                },
                "size_bytes": {
                    "weight": input_bytes,
                    "bias": bias_bytes_per_output,
                    "output": 1,
                },
                "full": {
                    "input_addr": input_tensor["addr"],
                    "output_addr": hex_addr(output_base + output_offset),
                    "weight_addr": hex_addr(param_base + param_offset),
                    "has_bias": has_bias,
                },
            }
        )

    plan["execution_plan"].append(
        {
            "op_type": "linear",
            "linear_index": linear_index,
            "layer": layer_name,
            "input": unit.get("input"),
            "output": unit.get("output"),
            "input_tensor": input_tensor_name,
            "output_tensor": f"{layer_name}_out",
            "input_features": in_features,
            "output_features": out_features,
            "input_channels": input_ch,
            "aligned_input_channels": aligned_input_ch,
            "input_hw": [input_h, input_w],
            "input_bytes": input_bytes,
            "input_words": input_words,
            "has_bias": has_bias,
            "output_layout": "signed_int8_tightly_packed",
            "reserved_regions": {
                "linear_out": {
                    "addr": output_tensor["addr"],
                    "size_bytes": output_tensor["size_bytes"],
                    "layout": "signed_int8_tightly_packed",
                },
            },
            "splits": splits,
        }
    )
    register_value(
        value_tensors,
        value_shapes,
        unit.get("output"),
        f"{layer_name}_out",
        height=1,
        width=out_features,
        channels=1,
        aligned_channels=1,
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
    validate_feature_hw(height, width)
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
    if needs_dsmp:
        conv_out_h, conv_out_w = height, width
        out_h, out_w = conv_out_h // 2, conv_out_w // 2
    else:
        out_h, out_w = conv_output_hw(height, width, conv)
        conv_out_h, conv_out_w = out_h, out_w
    validate_feature_hw(conv_out_h, conv_out_w)
    validate_feature_hw(out_h, out_w)

    weight_size = out_ch * aligned_in_ch * kh * kw
    has_bias = bool(conv.get("has_bias", False))
    bias_size = aligned_out_ch * 4 if has_bias else 0
    parameter_size = weight_size + bias_size
    conv_output_size = conv_out_h * conv_out_w * aligned_out_ch
    output_size = out_h * out_w * aligned_out_ch
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
        ("conv", conv_output_size, conv_out_h, conv_out_w),
    ]
    if needs_dsmp:
        runtime_specs.append(("dsmp", output_size, out_h, out_w))
    if has_relu:
        runtime_specs.append(("relu", output_size, out_h, out_w))

    final_runtime_op = "relu" if has_relu else "dsmp" if needs_dsmp else "conv"
    for op_name, region_size, region_h, region_w in runtime_specs:
        tensor_name = f"{layer_name}_{op_name}_out"
        value_name = None
        if op_name == final_runtime_op:
            value_name = unit.get("relu_output") if has_relu else unit.get("output")
        runtime = add_feature_region_for_value(
            plan,
            value_name,
            tensor_name,
            region_size,
            channels=out_ch,
            aligned_channels=aligned_out_ch,
            shape_nchw=[1, out_ch, region_h, region_w],
            aligned_shape_nchw=[1, aligned_out_ch, region_h, region_w],
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
            padding=conv["padding"],
            stride=conv["stride"],
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
        )
    )

    if has_relu:
        output_tensor_name = f"{layer_name}_relu_out"
        register_value(
            value_tensors,
            value_shapes,
            unit.get("relu_output"),
            output_tensor_name,
            height=out_h,
            width=out_w,
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
        height=out_h,
        width=out_w,
        channels=out_ch,
        aligned_channels=aligned_out_ch,
    )


UNIT_PLANNERS = {
    "conv": plan_conv_unit,
    "avgpool": plan_pool_unit,
    "maxpool": plan_pool_unit,
    "usmp": plan_usmp_unit,
    "pad": plan_pad_unit,
    "cat": plan_cat_unit,
    "linear": plan_linear_unit,
}


def build_plan(model_ir_path: Path = DEFAULT_MODEL_IR) -> Dict[str, Any]:
    ir = load_model_ir(model_ir_path)
    units = execution_units_from_ir(ir)
    layers = [unit for unit in units if unit["op_type"] == "conv"]
    relu_cfg = relu_config_from_ir(ir)
    input_info = ir.get("input", {})
    outputs = model_output_items(ir)
    cat_aliases = build_cat_aliases(ir)

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
    validate_feature_hw(input_h, input_w)
    image_size = input_h * input_w * image_aligned_ch
    image_file = getattr(cfg, "IMAGE_PATH", "./coe/image.coe") if cfg.IMAGE_SOURCE == "coe" else None
    image_addr = cfg.INIT_BASE_ADDR if cfg.IMAGE_SOURCE == "coe" else cfg.IMAGE_BASE_ADDR
    input_aligned_shape = [1, image_aligned_ch, input_h, input_w]

    runtime_limit_addr = getattr(cfg, "RUNTIME_LIMIT_ADDR", 0xFFFFFFFF)
    enable_output_addr = bool(getattr(cfg, "ENABLE_OUTPUT_ADDR", False))
    output_base_addr = int(getattr(cfg, "OUTPUT_BASE_ADDR", 0))
    output_limit_addr = int(getattr(cfg, "OUTPUT_LIMIT_ADDR", 0))
    if enable_output_addr and output_base_addr >= output_limit_addr:
        raise ValueError(
            f"OUTPUT_BASE_ADDR must be less than OUTPUT_LIMIT_ADDR when ENABLE_OUTPUT_ADDR=True: "
            f"base={hex_addr(output_base_addr)}, limit={hex_addr(output_limit_addr)}"
        )

    plan: Dict[str, Any] = {
        "config": {
            "INIT_BASE_ADDR": hex_addr(cfg.INIT_BASE_ADDR),
            "INIT_LIMIT_ADDR": hex_addr(cfg.INIT_LIMIT_ADDR),
            "RUNTIME_BASE_ADDR": hex_addr(cfg.RUNTIME_BASE_ADDR),
            "RUNTIME_LIMIT_ADDR": hex_addr(runtime_limit_addr),
            "ENABLE_OUTPUT_ADDR": enable_output_addr,
            "OUTPUT_BASE_ADDR": hex_addr(output_base_addr),
            "OUTPUT_LIMIT_ADDR": hex_addr(output_limit_addr),
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
            "aligned_shape_nchw": input_aligned_shape,
            "size_bytes": image_size,
            "file": image_file,
        },
        "model_layers": layers,
        "model_ops": ir.get("ops", []),
        "model_output_values": outputs,
        "tensors": {},
        "execution_plan": [],
        "init_regions": [],
        "runtime_regions": [],
        "output_regions": [],
        "aliases": [],
        "_cat_input_aliases": cat_aliases["input_aliases"],
        "_cat_output_tensors": cat_aliases["output_tensors"],
        "_next_init_addr": cfg.INIT_BASE_ADDR,
        "_next_runtime_addr": cfg.RUNTIME_BASE_ADDR,
    }

    if cfg.IMAGE_SOURCE == "coe":
        if cfg.IMAGE_BASE_ADDR != cfg.INIT_BASE_ADDR:
            raise ValueError("IMAGE_SOURCE=coe currently requires IMAGE_BASE_ADDR == INIT_BASE_ADDR")
        validate_input_coe_size(image_file, image_size, input_aligned_shape)
        add_init_region(plan, "image", image_size, image_file)
    plan["tensors"]["image"] = plan["image"]

    value_tensors, value_shapes = init_value_maps(
        input_info,
        input_h=input_h,
        input_w=input_w,
        input_ch=input_ch,
        image_aligned_ch=image_aligned_ch,
    )

    for unit_index, unit in enumerate(units):
        planner = UNIT_PLANNERS.get(unit["op_type"])
        if planner is None:
            raise ValueError(f"unsupported execution unit type: {unit['op_type']}")
        planner(plan, unit, value_tensors, value_shapes)

    relocate_model_outputs(plan, outputs, value_tensors, value_shapes)

    instr_size = instr_count_for_execution_plan(plan["execution_plan"]) * cfg.INSTR_WORD_BYTES
    instr = add_init_region(plan, "instr", instr_size, "coe/instr.coe")
    plan["tensors"]["instr"] = instr

    init_end = plan["_next_init_addr"]
    if init_end > cfg.INIT_LIMIT_ADDR:
        raise ValueError(
            f"init region exceeds INIT_LIMIT_ADDR: next={hex_addr(init_end)}, limit={hex_addr(cfg.INIT_LIMIT_ADDR)}"
        )
    runtime_end = plan["_next_runtime_addr"]
    if runtime_end > runtime_limit_addr:
        raise ValueError(
            f"runtime region exceeds RUNTIME_LIMIT_ADDR: next={hex_addr(runtime_end)}, limit={hex_addr(runtime_limit_addr)}"
        )
    output_end = output_base_addr
    if plan["output_regions"]:
        output_end = max(addr_to_int(output_region["end_addr_exclusive"]) for output_region in plan["output_regions"])
    if enable_output_addr:
        if not plan["output_regions"]:
            raise ValueError("ENABLE_OUTPUT_ADDR=True but no model output region was generated")
        if output_end > output_limit_addr:
            raise ValueError(
                f"model output exceeds OUTPUT_LIMIT_ADDR: next={hex_addr(output_end)}, "
                f"limit={hex_addr(output_limit_addr)}"
            )
    if cfg.RUNTIME_BASE_ADDR < cfg.INIT_LIMIT_ADDR:
        raise ValueError(
            f"RUNTIME_BASE_ADDR {hex_addr(cfg.RUNTIME_BASE_ADDR)} overlaps init address window "
            f"ending at {hex_addr(cfg.INIT_LIMIT_ADDR)}"
        )
    if enable_output_addr:
        for output_region in plan["output_regions"]:
            output_start = addr_to_int(output_region["addr"])
            output_size = int(output_region["size_bytes"])
            validate_no_overlap(
                output_start,
                output_size,
                cfg.INIT_BASE_ADDR,
                init_end - cfg.INIT_BASE_ADDR,
                "model output",
                "init data",
            )
            validate_no_overlap(
                output_start,
                output_size,
                cfg.RUNTIME_BASE_ADDR,
                runtime_end - cfg.RUNTIME_BASE_ADDR,
                "model output",
                "runtime data",
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
        if enable_output_addr:
            validate_no_overlap(
                cfg.IMAGE_BASE_ADDR,
                image_size,
                output_base_addr,
                output_end - output_base_addr,
                "external input",
                "model output",
            )

    plan["init_end_addr_exclusive"] = hex_addr(init_end)
    plan["runtime_end_addr_exclusive"] = hex_addr(runtime_end)
    plan["output_end_addr_exclusive"] = hex_addr(output_end) if plan["output_regions"] else None
    del plan["_cat_input_aliases"]
    del plan["_cat_output_tensors"]
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
    if plan.get("output_end_addr_exclusive") is not None:
        print(f"     output_end  = {plan['output_end_addr_exclusive']}")


if __name__ == "__main__":
    main()
