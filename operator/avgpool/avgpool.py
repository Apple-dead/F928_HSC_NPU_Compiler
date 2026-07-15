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


def encode_ravgpool(width: int, channels: int) -> int:
    if not 1 <= channels <= 256:
        raise ValueError(f"avgpool input channels must be in [1, 256], got {channels}")
    if channels % 4 != 0:
        raise ValueError(f"avgpool input channels must be a multiple of 4, got {channels}")
    if width % 8 != 0:
        raise ValueError(f"avgpool width must be divisible by 8, got {width}")
    block_image = width // 8
    if not 1 <= block_image <= 64:
        raise ValueError(f"avgpool block_image must be in [1, 64], got {block_image}")
    return ((block_image - 1) << 26) | ((channels - 1) << 18)


def compile_op(op_plan: Dict[str, Any], memory_plan: Dict[str, Any]) -> List[str]:
    if op_plan.get("op") != "avgpool":
        raise ValueError(f"avgpool compiler received non-avgpool op: {op_plan.get('op')}")

    input_addr = parse_addr(op_plan["input_addr"])
    output_addr = parse_addr(op_plan["output_addr"])
    ravgpool = encode_ravgpool(width=int(op_plan["feature_size"]), channels=int(op_plan["channels"]))
    high16 = (ravgpool >> 16) & 0xFFFF

    asm: List[str] = [
        f"; {op_plan['layer']} group{op_plan['group_index']} avgpool",
        f"; input=0x{input_addr:08X} output=0x{output_addr:08X} RAVGPOOL=0x{ravgpool:08X}",
    ]
    asm += cfg_addr("R1", input_addr)
    asm += cfg_addr("R2", output_addr)
    asm += [
        f"CFG_REGISTER AVGPOOL_P, 0x{high16:04X}",
        "AVGPOOL R1, R2",
    ]
    return asm
