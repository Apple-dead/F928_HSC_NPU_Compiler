#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def split_u32(value: int) -> Tuple[int, int]:
    return value & 0xFFFF, (value >> 16) & 0xFFFF


def encode_rrelu(feature_size: int, channels: int, tan: int) -> int:
    if not 1 <= feature_size <= 1024:
        raise ValueError(f"relu feature size must be in [1, 1024], got {feature_size}")
    if not 1 <= channels <= 1024:
        raise ValueError(f"relu input channels must be in [1, 1024], got {channels}")
    if not 0 <= tan <= 0xFF:
        raise ValueError(f"relu tan must fit in 8 bits, got {tan}")
    return ((feature_size - 1) << 22) | ((channels - 1) << 12) | (tan << 4)


def compile_op(op_plan: Dict[str, Any], memory_plan: Dict[str, Any]) -> List[str]:
    if op_plan.get("op") != "relu":
        raise ValueError(f"relu compiler received non-relu op: {op_plan.get('op')}")

    tan = int(op_plan.get("tan", memory_plan.get("relu", {}).get("tan", 0)))
    rrelu = encode_rrelu(feature_size=int(op_plan["feature_size"]), channels=int(op_plan["channels"]), tan=tan)
    low16, high16 = split_u32(rrelu)

    asm: List[str] = [
        f"; {op_plan['layer']} fused relu configuration RRELU=0x{rrelu:08X}",
    ]
    asm += [
        f"CFG_REGISTER RRELU_LOW, 0x{low16:04X}",
        f"CFG_REGISTER RRELU_HIGH, 0x{high16:04X}",
    ]
    return asm
