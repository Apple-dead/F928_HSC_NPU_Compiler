#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

import assembler
import npu_config as cfg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_PLAN = PROJECT_ROOT / "data" / "memory_plan.json"
DEFAULT_IR_DIR = PROJECT_ROOT / "data" / "infer_ir"
DEFAULT_ASM = PROJECT_ROOT / "data" / "instr.asm"
DEFAULT_TXT = PROJECT_ROOT / "data" / "instr.txt"

OPERATOR_PATHS = {
    "conv": PROJECT_ROOT / "operator" / "conv" / "conv.py",
    "madd": PROJECT_ROOT / "operator" / "madd" / "madd.py",
    "relu": PROJECT_ROOT / "operator" / "relu" / "relu.py",
}

IR_ORDER = [
    "layer1_conv.json",
    "layer1_madd.json",
    "layer1_relu.json",
]


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
        raise AttributeError(f"{path} does not define compile_op(ir, memory_plan)")
    return module


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_asm(memory_plan: Dict[str, Any], ir_dir: Path) -> List[str]:
    asm: List[str] = [
        "; Auto-generated NPU instruction assembly.",
        "; Source: data/memory_plan.json + data/infer_ir/*.json",
        "",
    ]
    for filename in IR_ORDER:
        ir = read_json(ir_dir / filename)
        op = ir["op"]
        module = load_operator(op)
        if asm and asm[-1] != "":
            asm.append("")
        asm.extend(module.compile_op(ir, memory_plan))
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
        raise ValueError(
            f"instruction size mismatch: memory_plan={expected_size} bytes, generated={actual_size} bytes. "
            "Update FIRST_STAGE_INSTR_COUNT in python/npu_config.py if the instruction sequence changed."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NPU instruction asm/txt from memory plan and IR")
    parser.add_argument("--memory-plan", default=str(DEFAULT_MEMORY_PLAN))
    parser.add_argument("--ir-dir", default=str(DEFAULT_IR_DIR))
    parser.add_argument("--asm-out", default=str(DEFAULT_ASM))
    parser.add_argument("--txt-out", default=str(DEFAULT_TXT))
    args = parser.parse_args()

    memory_plan = read_json(Path(args.memory_plan))
    asm_lines = build_asm(memory_plan, Path(args.ir_dir))
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

