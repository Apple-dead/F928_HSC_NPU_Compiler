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


def split_u32(value: int) -> tuple[int, int]:
    return value & 0xFFFF, (value >> 16) & 0xFFFF


def encode_rfull(input_words: int, start_position: int, has_bias: bool) -> int:
    if not 1 <= input_words <= 65536:
        raise ValueError(f"FULL input_words must be in [1, 65536], got {input_words}")
    if not 0 <= start_position <= 31:
        raise ValueError(f"FULL start_position must fit in 5 bits, got {start_position}")
    condition_bias = 1 if has_bias else 0
    return ((input_words - 1) << 16) | (start_position << 11) | (condition_bias << 10)


def compile_op(op_plan: Dict[str, Any], memory_plan: Dict[str, Any]) -> List[str]:
    if op_plan.get("op") != "full":
        raise ValueError(f"full compiler received non-full op: {op_plan.get('op')}")

    input_addr = parse_addr(op_plan["input_addr"])
    output_addr = parse_addr(op_plan["output_addr"])
    weight_addr = parse_addr(op_plan["weight_addr"])
    rfull = encode_rfull(
        input_words=int(op_plan["input_words"]),
        start_position=int(op_plan["start_position"]),
        has_bias=bool(op_plan.get("has_bias", False)),
    )
    low16, high16 = split_u32(rfull)

    asm: List[str] = [
        f"; {op_plan['layer']} output{op_plan['output_index']} full",
        (
            f"; input=0x{input_addr:08X} output=0x{output_addr:08X} "
            f"weight=0x{weight_addr:08X} RFULL=0x{rfull:08X}"
        ),
    ]
    asm += cfg_addr("R1", input_addr)
    asm += cfg_addr("R2", output_addr)
    asm += cfg_addr("R3", weight_addr)
    asm += [
        f"CFG_REGISTER FULL_P_1, 0x{low16:04X}",
        f"CFG_REGISTER FULL_P_2, 0x{high16:04X}",
        "FULL R1, R2, R3",
    ]
    return asm
