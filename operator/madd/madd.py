#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List


MADD_P_LAYER1_256_3CH = 0x13E0


def parse_addr(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return int(value, 16)


def cfg_addr(reg: str, addr: int) -> List[str]:
    return [
        f"CFG_REGISTER {reg}, LOW,  0x{addr & 0xFFFF:04X}",
        f"CFG_REGISTER {reg}, HIGH, 0x{(addr >> 16) & 0xFFFF:04X}",
    ]


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

    tensors = memory_plan["tensors"]
    input_addr = parse_addr(tensors[ir["input"]]["addr"])
    bias_addr = parse_addr(tensors[ir["bias"]]["addr"])
    output_addr = parse_addr(tensors[ir["output"]]["addr"])

    asm: List[str] = [
        "; layer1 madd",
        f"; input_addr=0x{input_addr:08X} bias_addr=0x{bias_addr:08X} output_addr=0x{output_addr:08X}",
        f"; feature={height}x{width} channels={channels}",
    ]
    asm += cfg_addr("R1", input_addr)
    asm += cfg_addr("R2", bias_addr)
    asm += cfg_addr("R3", output_addr)
    asm += [
        f"CFG_REGISTER MADD_P, 0x{MADD_P_LAYER1_256_3CH:04X}",
        "MADD R1, R2, R3",
    ]
    return asm
