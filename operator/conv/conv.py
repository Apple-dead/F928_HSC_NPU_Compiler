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
    if isinstance(value, int):
        return value
    return int(value, 16)


def cfg_addr(reg: str, addr: int) -> List[str]:
    return [
        f"CFG_REGISTER {reg}, LOW,  0x{addr & 0xFFFF:04X}",
        f"CFG_REGISTER {reg}, HIGH, 0x{(addr >> 16) & 0xFFFF:04X}",
    ]


def check_u5(name: str, value: int) -> None:
    if not 0 <= value <= 31:
        raise ValueError(f"{name} must fit in 5 bits, got {value}")


def split_u32(value: int) -> Tuple[int, int]:
    return value & 0xFFFF, (value >> 16) & 0xFFFF


def encode_rconv(
    kernel_size: List[int],
    width: int,
    start_position: int,
    input_channels: int,
    output_channels: int,
) -> int:
    kernel_key = tuple(kernel_size)
    if kernel_key not in KERNEL_SIZE_TO_CODE:
        raise ValueError(f"unsupported conv kernel_size: {kernel_size}")
    if width % 8 != 0:
        raise ValueError(f"conv width must be divisible by 8, got {width}")
    if input_channels not in (3, 4) or output_channels not in (3, 4):
        raise ValueError("first-stage conv only supports 3 or 4 input channels and 3 or 4 output channels")

    block_image = width // 8
    if not 1 <= block_image <= 64:
        raise ValueError(f"conv block_image must be in [1, 64], got {block_image}")
    check_u5("conv start_position", start_position)

    return (
        (KERNEL_SIZE_TO_CODE[kernel_key] << 30)
        | ((block_image - 1) << 24)
        | (start_position << 19)
        | ((input_channels - 1) << 14)
        | ((output_channels - 1) << 9)
    )


def compile_op(ir: Dict[str, Any], memory_plan: Dict[str, Any]) -> List[str]:
    if ir.get("op") != "conv":
        raise ValueError(f"conv compiler received non-conv IR: {ir.get('op')}")

    kernel = ir["kernel_size"]
    stride = ir["stride"]
    padding = ir["padding"]
    h = ir["input_shape_nchw"][2]
    w = ir["input_shape_nchw"][3]
    in_ch = ir["input_channels"]
    out_ch = ir["output_channels"]
    start_position = ir["start_position"]

    if in_ch not in (3, 4) or out_ch not in (3, 4):
        raise ValueError("first-stage conv only supports 3 or 4 input channels and 3 or 4 output channels")
    if kernel != [3, 3] or stride != [1, 1] or padding != [1, 1]:
        raise ValueError("first-stage conv only supports kernel=3, stride=1, padding=1")
    if h != 256 or w != 256 or h % 8 != 0 or w % 8 != 0:
        raise ValueError("first-stage conv only supports 256x256 feature maps divisible by 8")
    rconv = encode_rconv(kernel, w, int(start_position), int(in_ch), int(out_ch))
    rconv_low16, rconv_high16 = split_u32(rconv)

    tensors = memory_plan["tensors"]
    input_addr = parse_addr(tensors[ir["input"]]["addr"])
    output_addr = parse_addr(tensors[ir["output"]]["addr"])
    weight_addr = parse_addr(tensors[ir["weight"]]["addr"])

    asm: List[str] = [
        "; layer1 conv",
        f"; input_addr=0x{input_addr:08X} output_addr=0x{output_addr:08X} weight_addr=0x{weight_addr:08X}",
        f"; RCONV=0x{rconv:08X} kernel=3 block_image={w // 8} start_position={start_position} in_ch={in_ch} out_ch={out_ch}",
    ]
    asm += cfg_addr("R1", input_addr)
    asm += cfg_addr("R2", output_addr)
    asm += cfg_addr("R3", weight_addr)
    asm += [
        f"CFG_REGISTER CONV_P_1, 0x{rconv_low16:04X}",
        f"CFG_REGISTER CONV_P_2, 0x{rconv_high16:04X}",
        "CONV R1, R2, R3",
    ]
    return asm
