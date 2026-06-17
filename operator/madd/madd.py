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


def encode_rmadd(channels: int, width: int, start_position: int) -> int:
    if not 1 <= channels <= 8:
        raise ValueError(f"madd channels per pass must be <= 8, got {channels}")
    if width % 8 != 0:
        raise ValueError(f"madd width must be divisible by 8, got {width}")
    if not 0 <= start_position <= 31:
        raise ValueError(f"madd start_position must fit in 5 bits, got {start_position}")
    block_image = width // 8
    if not 1 <= block_image <= 64:
        raise ValueError(f"madd block_image must be in [1, 64], got {block_image}")
    return ((channels - 1) << 27) | ((block_image - 1) << 21) | (start_position << 16)


def compile_op(op_plan: Dict[str, Any], memory_plan: Dict[str, Any]) -> List[str]:
    if op_plan.get("op") != "madd":
        raise ValueError(f"madd compiler received non-madd op: {op_plan.get('op')}")

    input_addr = parse_addr(op_plan["input_addr"])
    bias_addr = parse_addr(op_plan["bias_addr"])
    output_addr = parse_addr(op_plan["output_addr"])
    rmadd = encode_rmadd(
        channels=int(op_plan["channels"]),
        width=int(op_plan["feature_size"]),
        start_position=int(op_plan["start_position"]),
    )
    low16, high16 = split_u32(rmadd)
    if low16 != 0:
        raise ValueError("RMADD low16 reserve bits must be zero")

    asm: List[str] = [
        f"; {op_plan['layer']} group{op_plan['group_index']} madd",
        f"; input=0x{input_addr:08X} bias=0x{bias_addr:08X} output=0x{output_addr:08X} RMADD=0x{rmadd:08X}",
    ]
    asm += cfg_addr("R1", input_addr)
    asm += cfg_addr("R2", output_addr)
    asm += cfg_addr("R3", bias_addr)
    asm += [
        f"CFG_REGISTER MADD_P, 0x{high16:04X}",
        "MADD R1, R2, R3",
    ]
    return asm
