#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Tuple


KERNEL_SIZE_TO_CODE = {
    (1, 1): 0,
    (2, 2): 1,
    (3, 3): 2,
    (5, 5): 3,
}


def parse_addr(value: str | int) -> int:
    return value if isinstance(value, int) else int(value, 16)


def cfg_addr(reg: str, addr: int) -> List[str]:
    return [
        f"CFG_REGISTER {reg}, LOW,  0x{addr & 0xFFFF:04X}",
        f"CFG_REGISTER {reg}, HIGH, 0x{(addr >> 16) & 0xFFFF:04X}",
    ]


def split_u32(value: int) -> Tuple[int, int]:
    return value & 0xFFFF, (value >> 16) & 0xFFFF


def encode_rconv(kernel_size: List[int], width: int, start_position: int, input_channels: int, output_channels: int) -> int:
    kernel_key = tuple(kernel_size)
    if kernel_key not in KERNEL_SIZE_TO_CODE:
        raise ValueError(f"unsupported conv kernel_size: {kernel_size}")
    if width % 8 != 0:
        raise ValueError(f"conv feature width must be divisible by 8, got {width}")
    if not 1 <= input_channels <= 4:
        raise ValueError(f"conv input channels must be <= 4, got {input_channels}")
    if not 1 <= output_channels <= 8:
        raise ValueError(f"conv output channels per pass must be <= 8, got {output_channels}")
    if not 0 <= start_position <= 31:
        raise ValueError(f"conv start_position must fit in 5 bits, got {start_position}")

    block_image = width // 8
    if not 1 <= block_image <= 64:
        raise ValueError(f"conv block_image must be in [1, 64], got {block_image}")

    return (
        (KERNEL_SIZE_TO_CODE[kernel_key] << 30)
        | ((block_image - 1) << 24)
        | (start_position << 19)
        | ((input_channels - 1) << 14)
        | ((output_channels - 1) << 9)
    )


def compile_op(op_plan: Dict[str, Any], memory_plan: Dict[str, Any]) -> List[str]:
    if op_plan.get("op") != "conv":
        raise ValueError(f"conv compiler received non-conv op: {op_plan.get('op')}")

    input_addr = parse_addr(op_plan["input_addr"])
    output_addr = parse_addr(op_plan["output_addr"])
    weight_addr = parse_addr(op_plan["weight_addr"])
    rconv = encode_rconv(
        kernel_size=op_plan["kernel_size"],
        width=int(op_plan["feature_size"]),
        start_position=int(op_plan["start_position"]),
        input_channels=int(op_plan["input_channels"]),
        output_channels=int(op_plan["output_channels"]),
    )
    low16, high16 = split_u32(rconv)

    asm: List[str] = [
        f"; {op_plan['layer']} group{op_plan['group_index']} conv",
        f"; input=0x{input_addr:08X} output=0x{output_addr:08X} weight=0x{weight_addr:08X} RCONV=0x{rconv:08X}",
    ]
    asm += cfg_addr("R1", input_addr)
    asm += cfg_addr("R2", output_addr)
    asm += cfg_addr("R3", weight_addr)
    asm += [
        f"CFG_REGISTER CONV_P_1, 0x{low16:04X}",
        f"CFG_REGISTER CONV_P_2, 0x{high16:04X}",
        "CONV R1, R2, R3",
    ]
    return asm
