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


def encode_rdsmp(block_image: int, channels: int) -> int:
    if not 1 <= block_image <= 64:
        raise ValueError(f"dsmp block_image must be in [1, 64], got {block_image}")
    if not 1 <= channels <= 256:
        raise ValueError(f"dsmp input channels must be in [1, 256], got {channels}")
    return ((block_image - 1) << 26) | ((channels - 1) << 18)


def compile_op(op_plan: Dict[str, Any], memory_plan: Dict[str, Any]) -> List[str]:
    if op_plan.get("op") != "dsmp":
        raise ValueError(f"dsmp compiler received non-dsmp op: {op_plan.get('op')}")

    input_addr = parse_addr(op_plan["input_addr"])
    output_addr = parse_addr(op_plan["output_addr"])
    if "block_image" in op_plan:
        block_image = int(op_plan["block_image"])
    else:
        image_size = int(op_plan["image_size"])
        if image_size % 8 != 0:
            raise ValueError(f"dsmp image_size must be divisible by 8, got {image_size}")
        block_image = image_size // 8
    rdsmp = encode_rdsmp(block_image=block_image, channels=int(op_plan["channels"]))
    dsmp_high16 = (rdsmp >> 16) & 0xFFFF

    asm: List[str] = [
        f"; {op_plan['layer']} group{op_plan['group_index']} dsmp",
        f"; input=0x{input_addr:08X} output=0x{output_addr:08X} RDSMP=0x{rdsmp:08X}",
    ]
    asm += cfg_addr("R1", input_addr)
    asm += cfg_addr("R2", output_addr)
    asm += [
        f"CFG_REGISTER DSMP_P, 0x{dsmp_high16:04X}",
        "DSMP R1, R2",
    ]
    return asm

