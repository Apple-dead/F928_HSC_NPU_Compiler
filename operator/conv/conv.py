#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List


CONV_P1_LAYER1_3X3_256_START0 = 0x8400
CONV_P2_LAYER1_3_IN_3_OUT = 0x9F48


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

    if [in_ch, out_ch] != [3, 3] or kernel != [3, 3] or stride != [1, 1] or padding != [1, 1]:
        raise ValueError("first-stage conv only supports Conv2d(3, 3, kernel=3, stride=1, padding=1)")
    if h != 256 or w != 256 or h % 8 != 0 or w % 8 != 0:
        raise ValueError("first-stage conv only supports 256x256 feature maps divisible by 8")
    if start_position != 0:
        raise ValueError("first-stage conv only supports start_position=0")

    tensors = memory_plan["tensors"]
    input_addr = parse_addr(tensors[ir["input"]]["addr"])
    output_addr = parse_addr(tensors[ir["output"]]["addr"])
    weight_addr = parse_addr(tensors[ir["weight"]]["addr"])

    asm: List[str] = [
        "; layer1 conv",
        f"; input_addr=0x{input_addr:08X} output_addr=0x{output_addr:08X} weight_addr=0x{weight_addr:08X}",
        f"; kernel=3 block_image={w // 8} start_position={start_position} in_ch={in_ch} out_ch={out_ch}",
    ]
    asm += cfg_addr("R1", input_addr)
    asm += cfg_addr("R2", output_addr)
    asm += cfg_addr("R3", weight_addr)
    asm += [
        f"CFG_REGISTER CONV_P_1, 0x{CONV_P1_LAYER1_3X3_256_START0:04X}",
        f"CFG_REGISTER CONV_P_2, 0x{CONV_P2_LAYER1_3_IN_3_OUT:04X}",
        "CONV R1, R2, R3",
    ]
    return asm
