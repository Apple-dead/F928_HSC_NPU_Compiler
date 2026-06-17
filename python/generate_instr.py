#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import assembler
import npu_config as cfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_PLAN = PROJECT_ROOT / "data" / "memory_plan.json"
DEFAULT_INTR_MOVE = PROJECT_ROOT / "data" / "intr_move.json"
DEFAULT_ASM = PROJECT_ROOT / "data" / "instr.asm"
DEFAULT_TXT = PROJECT_ROOT / "data" / "instr.txt"

OPERATOR_PATHS = {
    "conv": PROJECT_ROOT / "operator" / "conv" / "conv.py",
    "dsmp": PROJECT_ROOT / "operator" / "dsmp" / "dsmp.py",
    "madd": PROJECT_ROOT / "operator" / "madd" / "madd.py",
    "relu": PROJECT_ROOT / "operator" / "relu" / "relu.py",
}


def load_operator(op: str):
    path = OPERATOR_PATHS[op]
    if not path.is_file():
        raise FileNotFoundError(f"operator compiler not found for {op}: {path}")
    spec = importlib.util.spec_from_file_location(f"npu_operator_{op}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load operator module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "compile_op"):
        raise AttributeError(f"{path} does not define compile_op(op_plan, memory_plan)")
    return module


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def move_to_start_position(move: int | float, *, layer: int, op: str) -> int:
    value = int(move)
    if value != move:
        raise ValueError(f"{op} move for layer {layer} must be an integer, got {move!r}")
    if value == 0:
        return 0
    if value < 0 or value & (value - 1):
        raise ValueError(f"{op} move for layer {layer} must be a positive power of 2, got {move!r}")
    start_position = int(math.log2(value))
    if not 0 <= start_position <= 31:
        raise ValueError(f"{op} start_position for layer {layer} must fit in 5 bits, got {start_position}")
    return start_position


def load_intr_moves(path: Path) -> Dict[str, Dict[int, int]]:
    data = read_json(path)
    result: Dict[str, Dict[int, int]] = {}
    for field in ("CONV_MOVE_BY_LAYER", "MADD_MOVE_BY_LAYER"):
        raw = data.get(field)
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain object field {field}")
        result[field] = {int(layer): int(move) for layer, move in raw.items()}
    return result


def layer_number(layer_name: str) -> int:
    if not layer_name.startswith("layer"):
        raise ValueError(f"unexpected layer name: {layer_name}")
    return int(layer_name.removeprefix("layer"))


def get_start_position(intr_moves: Dict[str, Dict[int, int]], field: str, layer: int, op: str) -> int:
    try:
        move = intr_moves[field][layer]
    except KeyError as exc:
        raise KeyError(f"missing {field} entry for layer {layer}") from exc
    return move_to_start_position(move, layer=layer, op=op)


def build_op_plan(
    layer_plan: Dict[str, Any],
    split: Dict[str, Any],
    op: str,
    intr_moves: Dict[str, Dict[int, int]],
) -> Dict[str, Any]:
    layer = layer_number(layer_plan["layer"])
    common = {
        "op": op,
        "layer": layer_plan["layer"],
        "group_index": split["group_index"],
        "start_channel": split["start_channel"],
        "channels": split["channels"],
        "valid_channels": split["valid_channels"],
        "has_padding": split["has_padding"],
    }

    if op == "conv":
        common.update(
            split["conv"]
            | {
                "kernel_size": layer_plan["kernel_size"],
                "feature_size": layer_plan["conv_output_hw"][1],
                "input_channels": layer_plan["input_channels"],
                "output_channels": split["channels"],
                "start_position": get_start_position(intr_moves, "CONV_MOVE_BY_LAYER", layer, "conv"),
            }
        )
        return common

    if op == "dsmp":
        common.update(split["dsmp"])
        return common

    if op == "madd":
        common.update(
            split["madd"]
            | {
                "feature_size": layer_plan["output_hw"][1],
                "channels": split["channels"],
                "start_position": get_start_position(intr_moves, "MADD_MOVE_BY_LAYER", layer, "madd"),
            }
        )
        return common

    if op == "relu":
        common.update(
            split["relu"]
            | {
                "feature_size": layer_plan["output_hw"][1],
                "channels": split["channels"],
            }
        )
        return common

    raise ValueError(f"unsupported op: {op}")


def build_asm(memory_plan: Dict[str, Any], intr_moves: Dict[str, Dict[int, int]]) -> List[str]:
    operators = {op: load_operator(op) for op in OPERATOR_PATHS}
    asm: List[str] = [
        "; Auto-generated NPU instruction assembly.",
        "; Source: data/memory_plan.json",
        "",
    ]

    for layer_plan in memory_plan.get("execution_plan", []):
        asm.append(f"; ===== {layer_plan['layer']} =====")
        for split in layer_plan.get("splits", []):
            asm.append(f"; -- group{split['group_index']} ch{split['start_channel']}+{split['channels']} --")
            ops = ["conv"]
            if layer_plan.get("has_dsmp"):
                ops.append("dsmp")
            ops.extend(["madd", "relu"])
            for op in ops:
                op_plan = build_op_plan(layer_plan, split, op, intr_moves)
                asm.extend(operators[op].compile_op(op_plan, memory_plan))
                asm.append("")

    asm.append("END")
    return asm


def write_asm(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_instr_size(memory_plan: Dict[str, Any], word_count: int) -> None:
    expected_size = int(memory_plan["tensors"]["instr"]["size_bytes"])
    actual_size = word_count * cfg.INSTR_WORD_BYTES
    if actual_size != expected_size:
        raise ValueError(f"instruction size mismatch: memory_plan={expected_size} bytes, generated={actual_size} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NPU instruction asm/txt from memory_plan.json")
    parser.add_argument("--memory-plan", default=str(DEFAULT_MEMORY_PLAN))
    parser.add_argument("--intr-move", default=str(DEFAULT_INTR_MOVE))
    parser.add_argument("--asm-out", default=str(DEFAULT_ASM))
    parser.add_argument("--txt-out", default=str(DEFAULT_TXT))
    args = parser.parse_args()

    memory_plan = read_json(Path(args.memory_plan))
    intr_moves = load_intr_moves(Path(args.intr_move))
    asm_lines = build_asm(memory_plan, intr_moves)
    words = assembler.assemble_lines(asm_lines)
    validate_instr_size(memory_plan, len(words))

    write_asm(Path(args.asm_out), asm_lines)
    assembler.write_instr_txt(Path(args.txt_out), words)
    print(f"[OK] asm generated: {args.asm_out}")
    print(f"[OK] instr generated: {args.txt_out}")
    print(f"     instruction_count = {len(words)}")
    print(f"     instruction_bytes = 0x{len(words) * cfg.INSTR_WORD_BYTES:08X}")


if __name__ == "__main__":
    main()
