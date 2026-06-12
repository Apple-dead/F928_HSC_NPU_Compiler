#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

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


def build_plan() -> Dict[str, Any]:
    h = cfg.IMAGE_HEIGHT
    w = cfg.IMAGE_WIDTH
    in_ch = cfg.LAYER1_IN_CHANNELS
    out_ch = cfg.LAYER1_OUT_CHANNELS
    kernel = cfg.LAYER1_KERNEL_SIZE

    if cfg.IMAGE_SOURCE not in ("coe", "external"):
        raise ValueError('IMAGE_SOURCE must be "coe" or "external"')
    if h % 8 != 0 or w % 8 != 0:
        raise ValueError("image height/width must be divisible by 8")

    aligned_in_ch = aligned_channels(in_ch)
    aligned_out_ch = aligned_channels(out_ch)

    image_size = h * w * aligned_in_ch
    weight_size = aligned_out_ch * aligned_in_ch * kernel * kernel
    bias_size = h * w * aligned_out_ch
    output_size = h * w * aligned_out_ch
    instr_size = cfg.FIRST_STAGE_INSTR_COUNT * cfg.INSTR_WORD_BYTES

    plan: Dict[str, Any] = {
        "config": {
            "INIT_BASE_ADDR": hex_addr(cfg.INIT_BASE_ADDR),
            "INIT_LIMIT_ADDR": hex_addr(cfg.INIT_LIMIT_ADDR),
            "RUNTIME_BASE_ADDR": hex_addr(cfg.RUNTIME_BASE_ADDR),
            "IMAGE_BASE_ADDR": hex_addr(cfg.IMAGE_BASE_ADDR),
            "IMAGE_SOURCE": cfg.IMAGE_SOURCE,
        },
        "alignment_bytes": 4,
        "image": {
            "source": cfg.IMAGE_SOURCE,
            "addr": hex_addr(cfg.IMAGE_BASE_ADDR),
            "channels": in_ch,
            "aligned_channels": aligned_in_ch,
            "shape_nchw": [1, in_ch, h, w],
            "storage_shape_nchw": [1, aligned_in_ch, h, w],
            "size_bytes": image_size,
            "file": "coe/image.coe" if cfg.IMAGE_SOURCE == "coe" else None,
        },
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
        validate_no_overlap(
            cfg.IMAGE_BASE_ADDR,
            image_size,
            cfg.INIT_BASE_ADDR,
            weight_size + bias_size + instr_size,
            "external image",
            "init data",
        )

    weight = add_init_region(plan, "layer1_weight", weight_size, "coe/layer1_weight.coe")
    bias = add_init_region(plan, "layer1_bias", bias_size, "coe/layer1_bias.coe")
    instr = add_init_region(plan, "instr", instr_size, "coe/instr.coe")

    conv_out = add_runtime_region(
        plan,
        "layer1_conv_out",
        output_size,
        channels=out_ch,
        aligned_channels=aligned_out_ch,
        shape_nchw=[1, out_ch, h, w],
        storage_shape_nchw=[1, aligned_out_ch, h, w],
    )
    madd_out = add_runtime_region(
        plan,
        "layer1_madd_out",
        output_size,
        channels=out_ch,
        aligned_channels=aligned_out_ch,
        shape_nchw=[1, out_ch, h, w],
        storage_shape_nchw=[1, aligned_out_ch, h, w],
    )
    relu_out = add_runtime_region(
        plan,
        "layer1_relu_out",
        output_size,
        channels=out_ch,
        aligned_channels=aligned_out_ch,
        shape_nchw=[1, out_ch, h, w],
        storage_shape_nchw=[1, aligned_out_ch, h, w],
    )

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

    plan["tensors"] = {
        "image": plan["image"],
        "layer1_weight": {
            "addr": weight["addr"],
            "channels": {"in": in_ch, "out": out_ch},
            "aligned_channels": {"in": aligned_in_ch, "out": aligned_out_ch},
            "shape_oihw": [out_ch, in_ch, kernel, kernel],
            "storage_shape_oihw": [aligned_out_ch, aligned_in_ch, kernel, kernel],
            "size_bytes": weight_size,
        },
        "layer1_bias": {
            "addr": bias["addr"],
            "channels": out_ch,
            "aligned_channels": aligned_out_ch,
            "shape_nchw": [1, out_ch, h, w],
            "storage_shape_nchw": [1, aligned_out_ch, h, w],
            "size_bytes": bias_size,
        },
        "layer1_conv_out": conv_out,
        "layer1_madd_out": madd_out,
        "layer1_relu_out": relu_out,
        "instr": instr,
    }
    plan["init_end_addr_exclusive"] = hex_addr(init_end)
    plan["runtime_end_addr_exclusive"] = hex_addr(plan["_next_runtime_addr"])

    del plan["_next_init_addr"]
    del plan["_next_runtime_addr"]
    return plan


def main() -> None:
    plan = build_plan()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] memory plan generated: {OUT_PATH}")
    print(f"     init_end    = {plan['init_end_addr_exclusive']}")
    print(f"     runtime_end = {plan['runtime_end_addr_exclusive']}")


if __name__ == "__main__":
    main()

