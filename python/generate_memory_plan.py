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
OUT_PATH = PROJECT_ROOT / "data" / "memory_plan.json"


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


def instr_count_for_layers(layers: List[Dict[str, Any]]) -> int:
    # conv: 9, madd: 8, relu: 7, END: 1 for the current non-fused pipeline.
    return len(layers) * (9 + 8 + 7) + 1


def build_plan(model_py: Path) -> Dict[str, Any]:
    layers, _relu_cfg = model_parser.parse_model_layers(model_py)

    if cfg.IMAGE_SOURCE not in ("coe", "external"):
        raise ValueError('IMAGE_SOURCE must be "coe" or "external"')
    if cfg.IMAGE_HEIGHT % 8 != 0 or cfg.IMAGE_WIDTH % 8 != 0:
        raise ValueError("image height/width must be divisible by 8")

    image_aligned_ch = aligned_channels(layers[0]["conv"]["in_channels"])
    image_size = cfg.IMAGE_HEIGHT * cfg.IMAGE_WIDTH * image_aligned_ch
    instr_size = instr_count_for_layers(layers) * cfg.INSTR_WORD_BYTES

    plan: Dict[str, Any] = {
        "config": {
            "INIT_BASE_ADDR": hex_addr(cfg.INIT_BASE_ADDR),
            "INIT_LIMIT_ADDR": hex_addr(cfg.INIT_LIMIT_ADDR),
            "RUNTIME_BASE_ADDR": hex_addr(cfg.RUNTIME_BASE_ADDR),
            "IMAGE_BASE_ADDR": hex_addr(cfg.IMAGE_BASE_ADDR),
            "IMAGE_SOURCE": cfg.IMAGE_SOURCE,
            "INFER_PARSE_MODE": cfg.INFER_PARSE_MODE,
            "INFER_PARSE_LAYER_LIMIT": cfg.INFER_PARSE_LAYER_LIMIT,
            "model_py": str(model_py),
        },
        "alignment_bytes": 4,
        "image": {
            "source": cfg.IMAGE_SOURCE,
            "addr": hex_addr(cfg.IMAGE_BASE_ADDR),
            "channels": layers[0]["conv"]["in_channels"],
            "aligned_channels": image_aligned_ch,
            "shape_nchw": [1, layers[0]["conv"]["in_channels"], cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH],
            "storage_shape_nchw": [1, image_aligned_ch, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH],
            "size_bytes": image_size,
            "file": "coe/image.coe" if cfg.IMAGE_SOURCE == "coe" else None,
        },
        "model_layers": layers,
        "tensors": {},
        "init_regions": [],
        "runtime_regions": [],
        "_next_init_addr": cfg.INIT_BASE_ADDR,
        "_next_runtime_addr": cfg.RUNTIME_BASE_ADDR,
    }

    if cfg.IMAGE_SOURCE == "coe":
        if cfg.IMAGE_BASE_ADDR != cfg.INIT_BASE_ADDR:
            raise ValueError("IMAGE_SOURCE=coe currently requires IMAGE_BASE_ADDR == INIT_BASE_ADDR")
        add_init_region(plan, "image", image_size, "coe/image.coe")
    else:
        validate_no_overlap(cfg.IMAGE_BASE_ADDR, image_size, cfg.INIT_BASE_ADDR, instr_size, "external image", "init data")

    plan["tensors"]["image"] = plan["image"]

    height = cfg.IMAGE_HEIGHT
    width = cfg.IMAGE_WIDTH
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
        out_h, out_w = model_parser.conv_output_hw(height, width, conv)

        if idx == 1 and input_tensor == "image" and in_ch != plan["image"]["channels"]:
            raise ValueError(f"{layer_name} in_channels does not match image channels")

        weight_size = aligned_out_ch * aligned_in_ch * kh * kw
        bias_size = out_h * out_w * aligned_out_ch
        output_size = out_h * out_w * aligned_out_ch
        weight = add_init_region(plan, f"{layer_name}_weight", weight_size, f"coe/{layer_name}_weight.coe")
        bias = add_init_region(plan, f"{layer_name}_bias", bias_size, f"coe/{layer_name}_bias.coe")

        plan["tensors"][f"{layer_name}_weight"] = {
            "addr": weight["addr"],
            "channels": {"in": in_ch, "out": out_ch},
            "aligned_channels": {"in": aligned_in_ch, "out": aligned_out_ch},
            "shape_oihw": [out_ch, in_ch, kh, kw],
            "storage_shape_oihw": [aligned_out_ch, aligned_in_ch, kh, kw],
            "size_bytes": weight_size,
        }
        plan["tensors"][f"{layer_name}_bias"] = {
            "addr": bias["addr"],
            "channels": out_ch,
            "aligned_channels": aligned_out_ch,
            "shape_nchw": [1, out_ch, out_h, out_w],
            "storage_shape_nchw": [1, aligned_out_ch, out_h, out_w],
            "size_bytes": bias_size,
        }

        for op_name in ("conv", "madd", "relu"):
            tensor_name = f"{layer_name}_{op_name}_out"
            runtime = add_runtime_region(
                plan,
                tensor_name,
                output_size,
                channels=out_ch,
                aligned_channels=aligned_out_ch,
                shape_nchw=[1, out_ch, out_h, out_w],
                storage_shape_nchw=[1, aligned_out_ch, out_h, out_w],
            )
            plan["tensors"][tensor_name] = runtime

        input_tensor = f"{layer_name}_relu_out"
        height, width = out_h, out_w

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

