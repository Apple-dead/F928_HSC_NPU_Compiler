#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List


STRIDE_TO_STEP_CODE = {
    (1, 1): 0,
    (2, 2): 1,
}


def parse_addr(value: str | int) -> int:
    return value if isinstance(value, int) else int(value, 16)


def cfg_addr(reg: str, addr: int) -> List[str]:
    return [
        f"CFG_REGISTER {reg}, LOW,  0x{addr & 0xFFFF:04X}",
        f"CFG_REGISTER {reg}, HIGH, 0x{(addr >> 16) & 0xFFFF:04X}",
    ]


def encode_rmaxpool(width: int, channels: int, stride: List[int]) -> int:
    stride_key = tuple(stride)
    if stride_key not in STRIDE_TO_STEP_CODE:
        raise ValueError(f"unsupported maxpool stride: {stride}")
    if not 1 <= width <= 1024:
        raise ValueError(f"maxpool feature width must be in [1, 1024], got {width}")
    if not 1 <= channels <= 1024:
        raise ValueError(f"maxpool input channels must be in [1, 1024], got {channels}")
    return ((width - 1) << 22) | ((channels - 1) << 12) | (STRIDE_TO_STEP_CODE[stride_key] << 11)


def compile_op(op_plan: Dict[str, Any], memory_plan: Dict[str, Any]) -> List[str]:
    if op_plan.get("op") != "maxpool":
        raise ValueError(f"maxpool compiler received non-maxpool op: {op_plan.get('op')}")

    input_addr = parse_addr(op_plan["input_addr"])
    output_addr = parse_addr(op_plan["output_addr"])
    rmaxpool = encode_rmaxpool(
        width=int(op_plan["feature_size"]),
        channels=int(op_plan["channels"]),
        stride=op_plan["stride"],
    )
    low16 = rmaxpool & 0xFFFF
    high16 = (rmaxpool >> 16) & 0xFFFF

    asm: List[str] = [
        f"; {op_plan['layer']} group{op_plan['group_index']} maxpool",
        f"; input=0x{input_addr:08X} output=0x{output_addr:08X} RMAXPOOL=0x{rmaxpool:08X}",
    ]
    asm += cfg_addr("R1", input_addr)
    asm += cfg_addr("R2", output_addr)
    asm += [
        f"CFG_REGISTER MAXPOOL_P_1, 0x{low16:04X}",
        f"CFG_REGISTER MAXPOOL_P, 0x{high16:04X}",
        "MAXPOOL R1, R2",
    ]
    return asm
