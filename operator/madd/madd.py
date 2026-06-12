#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Tuple


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


def encode_rmadd(channels: int, width: int, start_position: int) -> int:
    if not 1 <= channels <= 32:
        raise ValueError(f"madd channels must be in [1, 32], got {channels}")
    if width % 8 != 0:
        raise ValueError(f"madd width must be divisible by 8, got {width}")
    block_image = width // 8
    if not 1 <= block_image <= 64:
        raise ValueError(f"madd block_image must be in [1, 64], got {block_image}")
    check_u5("madd start_position", start_position)

    return ((channels - 1) << 27) | ((block_image - 1) << 21) | (start_position << 16)


def compile_op(ir: Dict[str, Any], memory_plan: Dict[str, Any]) -> List[str]:
    if ir.get("op") != "madd":
        raise ValueError(f"madd compiler received non-madd IR: {ir.get('op')}")

    feature = ir["feature"]
    channels = int(feature["channels"])
    height = int(feature["height"])
    width = int(feature["width"])
    if channels > 32:
        raise ValueError("first-stage madd only supports channels <= 32")
    if channels != 3 or height != 256 or width != 256:
        raise ValueError("first-stage madd only supports 256x256x3")
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError("madd feature height/width must be divisible by 8")
    start_position = int(ir.get("start_position", 0))
    rmadd = encode_rmadd(channels, width, start_position)
    rmadd_low16, rmadd_high16 = split_u32(rmadd)
    if rmadd_low16 != 0:
        raise ValueError("current assembler only supports RMADD low16 reserve=0")

    tensors = memory_plan["tensors"]
    input_addr = parse_addr(tensors[ir["input"]]["addr"])
    bias_addr = parse_addr(tensors[ir["bias"]]["addr"])
    output_addr = parse_addr(tensors[ir["output"]]["addr"])

    asm: List[str] = [
        "; layer1 madd",
        f"; input_addr=0x{input_addr:08X} bias_addr=0x{bias_addr:08X} output_addr=0x{output_addr:08X}",
        f"; RMADD=0x{rmadd:08X} feature={height}x{width} channels={channels} start_position={start_position}",
    ]
    asm += cfg_addr("R1", input_addr)
    asm += cfg_addr("R2", bias_addr)
    asm += cfg_addr("R3", output_addr)
    asm += [
        f"CFG_REGISTER MADD_P, 0x{rmadd_high16:04X}",
        "MADD R1, R2, R3",
    ]
    return asm
