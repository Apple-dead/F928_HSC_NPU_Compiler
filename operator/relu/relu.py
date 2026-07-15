#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def parse_addr(value: str | int) -> int:
    return value if isinstance(value, int) else int(value, 16)


def cfg_addr(reg: str, addr: int) -> List[str]:
    return [
        f"CFG_REGISTER {reg}, LOW,  0x{addr & 0xFFFF:04X}",
        f"CFG_REGISTER {reg}, HIGH, 0x{(addr >> 16) & 0xFFFF:04X}",
    ]


def split_u32(value: int) -> Tuple[int, int]:
    return value & 0xFFFF, (value >> 16) & 0xFFFF


def encode_rrelu(feature_size: int, channels: int, tan: int) -> int:
    if not 1 <= feature_size <= 1024:
        raise ValueError(f"relu feature size must be in [1, 1024], got {feature_size}")
    if feature_size % 8 != 0:
        raise ValueError(f"relu feature size must be divisible by 8, got {feature_size}")
    if not 1 <= channels <= 256:
        raise ValueError(f"relu input channels must be in [1, 256], got {channels}")
    if not 0 <= tan <= 0xFF:
        raise ValueError(f"relu tan must fit in 8 bits, got {tan}")
    return ((feature_size - 1) << 22) | ((channels - 1) << 14) | (tan << 6)


def compile_op(op_plan: Dict[str, Any], memory_plan: Dict[str, Any]) -> List[str]:
    if op_plan.get("op") != "relu":
        raise ValueError(f"relu compiler received non-relu op: {op_plan.get('op')}")

    input_addr = parse_addr(op_plan["input_addr"])
    output_addr = parse_addr(op_plan["output_addr"])
    tan = int(memory_plan.get("relu", {}).get("tan", 0))
    rrelu = encode_rrelu(feature_size=int(op_plan["feature_size"]), channels=int(op_plan["channels"]), tan=tan)
    low16, high16 = split_u32(rrelu)

    asm: List[str] = [
        f"; {op_plan['layer']} group{op_plan['group_index']} relu",
        f"; input=0x{input_addr:08X} output=0x{output_addr:08X} RRELU=0x{rrelu:08X}",
    ]
    asm += cfg_addr("R1", input_addr)
    asm += cfg_addr("R2", output_addr)
    asm += [
        f"CFG_REGISTER RELU_P_1, 0x{low16:04X}",
        f"CFG_REGISTER RELU_P_2, 0x{high16:04X}",
        "RELU R1, R2",
    ]
    return asm

