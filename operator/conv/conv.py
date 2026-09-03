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

PADDING_TO_CODE = {
    (0, 0): 0,
    (1, 1): 1,
    (2, 2): 2,
}

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


def split_u32(value: int) -> Tuple[int, int]:
    return value & 0xFFFF, (value >> 16) & 0xFFFF


def encode_rconv(
    kernel_size: List[int],
    padding: List[int],
    stride: List[int],
    width: int,
    start_position: int,
    input_channels: int,
    output_channels: int,
    has_bias: bool,
) -> Tuple[int, int]:
    kernel_key = tuple(kernel_size)
    if kernel_key not in KERNEL_SIZE_TO_CODE:
        raise ValueError(f"unsupported conv kernel_size: {kernel_size}")
    padding_key = tuple(padding)
    if padding_key not in PADDING_TO_CODE:
        raise ValueError(f"unsupported conv padding: {padding}")
    stride_key = tuple(stride)
    if stride_key not in STRIDE_TO_STEP_CODE:
        raise ValueError(f"unsupported conv stride: {stride}")
    if not 1 <= width <= 1024:
        raise ValueError(f"conv feature width must be in [1, 1024], got {width}")
    if not 1 <= input_channels <= 1024:
        raise ValueError(f"conv input channels must be <= 1024, got {input_channels}")
    if not 1 <= output_channels <= 8:
        raise ValueError(f"conv output channels per pass must be <= 8, got {output_channels}")
    if not 0 <= start_position <= 31:
        raise ValueError(f"conv start_position must fit in 5 bits, got {start_position}")

    rconv1 = (
        (KERNEL_SIZE_TO_CODE[kernel_key] << 30)
        | (PADDING_TO_CODE[padding_key] << 28)
        | (STRIDE_TO_STEP_CODE[stride_key] << 27)
        | ((width - 1) << 17)
        | ((input_channels - 1) << 7)
        | ((output_channels - 1) << 2)
        | ((1 if has_bias else 0) << 1)
    )
    rconv2 = (start_position & 0x1F) << 16
    return rconv1, rconv2


def compile_op(op_plan: Dict[str, Any], memory_plan: Dict[str, Any]) -> List[str]:
    if op_plan.get("op") != "conv":
        raise ValueError(f"conv compiler received non-conv op: {op_plan.get('op')}")

    input_addr = parse_addr(op_plan["input_addr"])
    output_addr = parse_addr(op_plan["output_addr"])
    weight_addr = parse_addr(op_plan["weight_addr"])
    rconv1, rconv2 = encode_rconv(
        kernel_size=op_plan["kernel_size"],
        padding=op_plan["padding"],
        stride=op_plan["stride"],
        width=int(op_plan["feature_size"]),
        start_position=int(op_plan["start_position"]),
        input_channels=int(op_plan["input_channels"]),
        output_channels=int(op_plan["output_channels"]),
        has_bias=bool(op_plan.get("has_bias", False)),
    )
    rconv1_low16, rconv1_high16 = split_u32(rconv1)
    rconv2_low16, rconv2_high16 = split_u32(rconv2)

    asm: List[str] = [
        f"; {op_plan['layer']} group{op_plan['group_index']} conv",
        (
            f"; input=0x{input_addr:08X} output=0x{output_addr:08X} "
            f"weight=0x{weight_addr:08X} RCONV1=0x{rconv1:08X} RCONV2=0x{rconv2:08X}"
        ),
    ]
    asm += cfg_addr("R1", input_addr)
    asm += cfg_addr("R2", output_addr)
    asm += cfg_addr("R3", weight_addr)
    asm += [
        f"CFG_REGISTER CONV_P_1, 0x{rconv1_low16:04X}",
        f"CFG_REGISTER CONV_P_2, 0x{rconv1_high16:04X}",
        f"CFG_REGISTER CONV_P_3, 0x{rconv2_low16:04X}",
        f"CFG_REGISTER CONV_P_4, 0x{rconv2_high16:04X}",
        "CONV R1, R2, R3",
    ]
    return asm
