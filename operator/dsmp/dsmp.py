#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List


def parse_addr(value: str | int) -> int:
    return value if isinstance(value, int) else int(value, 16)


def cfg_addr(reg: str, addr: int) -> List[str]:
    return [
        f"CFG_REGISTER {reg}, LOW,  0x{addr & 0xFFFF:04X}",
        f"CFG_REGISTER {reg}, HIGH, 0x{(addr >> 16) & 0xFFFF:04X}",
    ]


def encode_rdsmp(feature_size: int, channels: int) -> int:
    if not 1 <= feature_size <= 1024:
        raise ValueError(f"dsmp feature size must be in [1, 1024], got {feature_size}")
    if not 1 <= channels <= 1024:
        raise ValueError(f"dsmp input channels must be in [1, 1024], got {channels}")
    return ((feature_size - 1) << 22) | ((channels - 1) << 12)


def compile_op(op_plan: Dict[str, Any], memory_plan: Dict[str, Any]) -> List[str]:
    if op_plan.get("op") != "dsmp":
        raise ValueError(f"dsmp compiler received non-dsmp op: {op_plan.get('op')}")

    input_addr = parse_addr(op_plan["input_addr"])
    output_addr = parse_addr(op_plan["output_addr"])
    feature_size = int(op_plan.get("feature_size", op_plan.get("image_size")))
    rdsmp = encode_rdsmp(feature_size=feature_size, channels=int(op_plan["channels"]))
    dsmp_low16 = rdsmp & 0xFFFF
    dsmp_high16 = (rdsmp >> 16) & 0xFFFF

    asm: List[str] = [
        f"; {op_plan['layer']} group{op_plan['group_index']} dsmp",
        f"; input=0x{input_addr:08X} output=0x{output_addr:08X} RDSMP=0x{rdsmp:08X}",
    ]
    asm += cfg_addr("R1", input_addr)
    asm += cfg_addr("R2", output_addr)
    asm += [
        f"CFG_REGISTER DSMP_P_1, 0x{dsmp_low16:04X}",
        f"CFG_REGISTER DSMP_P, 0x{dsmp_high16:04X}",
        "DSMP R1, R2",
    ]
    return asm
