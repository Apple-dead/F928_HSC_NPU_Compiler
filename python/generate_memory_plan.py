#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import merge as coe_merge
import model_parser
import npu_config as cfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "data" / "memory_plan.json"
MIN_RUNTIME_FEATURE_HW = 8


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
    return max(height, MIN_RUNTIME_FEATURE_HW), max(width, MIN_RUNTIME_FEATURE_HW)


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


def instr_count_for_layers(layers: List[Dict[str, Any]], input_h: int, input_w: int) -> int:
    # Per split: conv: 9, optional dsmp: 6, relu: 7. Plus END: 1.
    total = 1
    height = input_h
    width = input_w
    for layer in layers:
        conv = layer["conv"]
        aligned_in_ch = aligned_channels(conv["in_channels"])
        aligned_out_ch = aligned_channels(conv["out_channels"])
        out_h, out_w = model_parser.conv_output_hw(height, width, conv)
        storage_out_h, storage_out_w = runtime_storage_hw(out_h, out_w)
        max_channels = channel_group_max_channels(
            input_h=height,
            input_w=width,
            aligned_in_ch=aligned_in_ch,
            output_h=storage_out_h,
            output_w=storage_out_w,
            aligned_out_ch=aligned_out_ch,
        )
        split_count = len(npu_channel_groups(conv["out_channels"], aligned_out_ch, max_channels=max_channels))
        per_split = 9 + 7
        if layer_needs_dsmp(conv):
            per_split += 6
        total += split_count * per_split
        height, width = storage_out_h, storage_out_w
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
    conv_out_tensor: Dict[str, Any],
    dsmp_out_tensor: Dict[str, Any] | None,
    relu_out_tensor: Dict[str, Any],
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

    # Physical parameter layout is, per group: valid weights then padded bias.
    # A later group must therefore skip the preceding group's bias words.
    parameter_offset = 0
    for group in npu_channel_groups(out_ch, aligned_out_ch, max_channels=max_group_channels):
        start_channel = group["start_channel"]
        conv_feature_offset = start_channel * bytes_per_conv_feature_channel
        feature_offset = start_channel * bytes_per_feature_channel
        weight_size = group["valid_channels"] * bytes_per_weight_output_channel
        bias_size = group["channels"] * 4
        weight_offset = parameter_offset
        bias_offset = weight_offset + weight_size
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
            "has_bias": True,
        }
        item["bias_addr"] = hex_addr(addr_to_int(weight_tensor["addr"]) + bias_offset)
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
        item["relu"] = {
            "input_addr": relu_input_addr,
            "output_addr": hex_addr(addr_to_int(relu_out_tensor["addr"]) + feature_offset),
        }
        splits.append(item)
        parameter_offset += weight_size + bias_size

    expected_parameter_size = out_ch * bytes_per_weight_output_channel + aligned_out_ch * 4
    if parameter_offset != expected_parameter_size:
        raise ValueError(
            f"{layer_name}: parameter layout size mismatch: "
            f"groups={parameter_offset}, expected={expected_parameter_size}"
        )

    return {
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
            "relu_out": {
                "addr": relu_out_tensor["addr"],
                "size_bytes": relu_out_tensor["size_bytes"],
                "layout": "channel_groups_are_tightly_packed",
            },
        },
        "splits": splits,
    }



def selected_input_shape(
    all_layers: List[Dict[str, Any]],
    selected_layers: List[Dict[str, Any]],
    image_h: int,
    image_w: int,
) -> tuple[int, int, int]:
    if not selected_layers:
        raise ValueError("no selected layers")

    start_layer = selected_layers[0]["layer_index"]
    height = image_h
    width = image_w
    channels = all_layers[0]["conv"]["in_channels"]

    for layer in all_layers:
        idx = layer["layer_index"]
        conv = layer["conv"]
        if idx == start_layer:
            if conv["in_channels"] != channels:
                raise ValueError(
                    f"layer{idx} expects {conv['in_channels']} input channels, "
                    f"but previous model output has {channels} channels"
                )
            return height, width, conv["in_channels"]

        if conv["in_channels"] != channels:
            raise ValueError(
                f"layer{idx} expects {conv['in_channels']} input channels, "
                f"but previous model output has {channels} channels"
            )
        height, width = model_parser.conv_output_hw(height, width, conv)
        channels = conv["out_channels"]

    raise ValueError(f"could not derive input shape for layer{start_layer}")


def validate_input_coe_size(file_text: str, expected_size: int, input_shape: List[int]) -> None:
    path = PROJECT_ROOT / file_text
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
def build_plan(model_py: Path) -> Dict[str, Any]:
    all_layers, relu_cfg = model_parser.parse_all_model_layers(model_py)
    layers = model_parser.select_layers(all_layers)

    if cfg.IMAGE_SOURCE not in ("coe", "external"):
        raise ValueError('IMAGE_SOURCE must be "coe" or "external"')
    if cfg.IMAGE_HEIGHT % 8 != 0 or cfg.IMAGE_WIDTH % 8 != 0:
        raise ValueError("image height/width must be divisible by 8")

    input_h, input_w, input_ch = selected_input_shape(all_layers, layers, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
    image_aligned_ch = aligned_channels(input_ch)
    image_size = input_h * input_w * image_aligned_ch
    image_file = "coe/image.coe" if cfg.IMAGE_SOURCE == "coe" else None
    image_addr = cfg.INIT_BASE_ADDR if cfg.IMAGE_SOURCE == "coe" else cfg.IMAGE_BASE_ADDR
    instr_size = instr_count_for_layers(layers, input_h, input_w) * cfg.INSTR_WORD_BYTES
    input_storage_shape = [1, image_aligned_ch, input_h, input_w]

    plan: Dict[str, Any] = {
        "config": {
            "INIT_BASE_ADDR": hex_addr(cfg.INIT_BASE_ADDR),
            "INIT_LIMIT_ADDR": hex_addr(cfg.INIT_LIMIT_ADDR),
            "RUNTIME_BASE_ADDR": hex_addr(cfg.RUNTIME_BASE_ADDR),
            "IMAGE_BASE_ADDR": hex_addr(cfg.IMAGE_BASE_ADDR),
            "IMAGE_SOURCE": cfg.IMAGE_SOURCE,
            "IMAGE_HEIGHT": cfg.IMAGE_HEIGHT,
            "IMAGE_WIDTH": cfg.IMAGE_WIDTH,
            "INFER_PARSE_MODE": cfg.INFER_PARSE_MODE,
            "INFER_PARSE_LAYER_START": cfg.INFER_PARSE_LAYER_START,
            "INFER_PARSE_LAYER_END": cfg.INFER_PARSE_LAYER_END,
            "CHANNEL_GROUP4_FEATURE_SIZE_THRESHOLD": cfg.CHANNEL_GROUP4_FEATURE_SIZE_THRESHOLD,
            "MIN_RUNTIME_FEATURE_HW": MIN_RUNTIME_FEATURE_HW,
            "model_py": str(model_py),
        },
        "relu": relu_cfg,
        "alignment_bytes": 4,
        "image": {
            "source": cfg.IMAGE_SOURCE,
            "addr": hex_addr(image_addr),
            "input_layer": layers[0]["layer"],
            "channels": input_ch,
            "aligned_channels": image_aligned_ch,
            "shape_nchw": [1, input_ch, input_h, input_w],
            "storage_shape_nchw": input_storage_shape,
            "size_bytes": image_size,
            "file": image_file,
        },
        "model_layers": layers,
        "all_model_layers": all_layers,
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

    height = input_h
    width = input_w
    input_tensor = "image"
    for layer in layers:
        idx = layer["layer_index"]
        conv = layer["conv"]
        layer_name = f"layer{idx}"
        in_ch = conv["in_channels"]
        out_ch = conv["out_channels"]
        aligned_in_ch = aligned_channels(in_ch)
        aligned_out_ch = aligned_channels(out_ch)
        kh, kw = conv["kernel_size"]
        needs_dsmp = layer_needs_dsmp(conv)
        out_h, out_w = model_parser.conv_output_hw(height, width, conv)
        conv_out_h, conv_out_w = (height, width) if needs_dsmp else (out_h, out_w)
        conv_storage_h, conv_storage_w = runtime_storage_hw(conv_out_h, conv_out_w)
        output_storage_h, output_storage_w = runtime_storage_hw(out_h, out_w)

        if idx == 1 and input_tensor == "image" and in_ch != plan["image"]["channels"]:
            raise ValueError(f"{layer_name} in_channels does not match image channels")

        weight_size = out_ch * aligned_in_ch * kh * kw
        bias_size = aligned_out_ch * 4
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

        bias_tensor = {
            "addr": params["addr"],
            "channels": out_ch,
            "aligned_channels": aligned_out_ch,
            "shape": [out_ch],
            "storage_shape": [aligned_out_ch],
            "size_bytes": bias_size,
            "parameter_region": f"{layer_name}_params",
            "layout": "int32_bias_values_padded_per_8_channel_weight_group",
        }
        plan["tensors"][f"{layer_name}_bias"] = bias_tensor
        plan["tensors"][f"{layer_name}_params"] = {
            "addr": params["addr"],
            "size_bytes": parameter_size,
            "layout": "weight_then_padded_int32_bias_per_group_of_at_most_8_output_channels",
        }

        runtime_tensors: Dict[str, Dict[str, Any]] = {}
        runtime_specs = [
            ("conv", conv_output_size, conv_out_h, conv_out_w, conv_storage_h, conv_storage_w),
        ]
        if needs_dsmp:
            runtime_specs.append(("dsmp", output_size, out_h, out_w, output_storage_h, output_storage_w))
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
                input_tensor=plan["tensors"][input_tensor],
                weight_tensor=weight_tensor,
                conv_out_tensor=runtime_tensors["conv"],
                dsmp_out_tensor=runtime_tensors.get("dsmp"),
                relu_out_tensor=runtime_tensors["relu"],
                conv_out_h=conv_out_h,
                conv_out_w=conv_out_w,
                conv_storage_h=conv_storage_h,
                conv_storage_w=conv_storage_w,
                output_storage_h=output_storage_h,
                output_storage_w=output_storage_w,
            )
        )

        input_tensor = f"{layer_name}_relu_out"
        height, width = output_storage_h, output_storage_w

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
    parser = argparse.ArgumentParser(description="Generate NPU memory plan from model structure")
    parser.add_argument("model_py", help="model definition .py path or filename under ./model")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    model_py = model_parser.resolve_model_py(args.model_py)
    plan = build_plan(model_py)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] memory plan generated: {out_path}")
    print(f"     layers      = {len(plan['model_layers'])}")
    print(f"     init_end    = {plan['init_end_addr_exclusive']}")
    print(f"     runtime_end = {plan['runtime_end_addr_exclusive']}")


if __name__ == "__main__":
    main()
