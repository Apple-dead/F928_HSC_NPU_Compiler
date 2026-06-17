#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List


CFG_OPCODE = 0x04000000
END_WORD = 0xFC000000

GENERAL_REGISTER_ID = {f"R{i}": i for i in range(1, 13)}

SPECIAL_REGISTER_CODE = {
    "CONV_P_1": 0x02,
    "CONV_P_2": 0x03,
    "DSMP_P": 0x04,
    "RELU_P_1": 0x05,
    "RELU_P_2": 0x06,
    "MADD_P": 0x09,
}

COMPUTE_OPCODE = {
    "CONV": 0b000011,
    "DSMP": 0b000100,
    "RELU": 0b000101,
    "MADD": 0b000111,
}

COMPUTE_OPERAND_COUNT = {
    "CONV": 3,
    "DSMP": 2,
    "RELU": 2,
    "MADD": 3,
}


def strip_comment(line: str) -> str:
    return line.split(";", 1)[0].split("#", 1)[0].split("//", 1)[0].strip()


def split_operands(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_int(text: str) -> int:
    t = text.strip().replace("_", "")
    if re.fullmatch(r"0[xX][0-9a-fA-F]+", t):
        return int(t, 16)
    if re.fullmatch(r"0[bB][01]+", t):
        return int(t, 2)
    if re.fullmatch(r"\d+", t):
        return int(t, 10)
    raise ValueError(f"invalid integer literal: {text!r}")


def encode_cfg(operands: List[str]) -> int:
    if len(operands) == 3:
        reg, half, imm_text = operands
        reg = reg.upper()
        half = half.upper()
        if reg not in GENERAL_REGISTER_ID:
            raise ValueError(f"CFG_REGISTER address form only supports R1-R12, got {reg}")
        if half not in ("LOW", "HIGH"):
            raise ValueError(f"address CFG half must be LOW or HIGH, got {half}")
        imm = parse_int(imm_text)
        if not 0 <= imm <= 0xFFFF:
            raise ValueError(f"CFG_REGISTER immediate out of uint16 range: {imm_text}")
        rd = GENERAL_REGISTER_ID[reg]
        subop = 1 if half == "HIGH" else 0
        return CFG_OPCODE | (rd << 21) | (subop << 16) | imm

    if len(operands) == 2:
        reg, imm_text = operands
        reg = reg.upper()
        if reg not in SPECIAL_REGISTER_CODE:
            raise ValueError(f"unsupported special CFG_REGISTER: {reg}")
        imm = parse_int(imm_text)
        if not 0 <= imm <= 0xFFFF:
            raise ValueError(f"CFG_REGISTER immediate out of uint16 range: {imm_text}")
        return CFG_OPCODE | (SPECIAL_REGISTER_CODE[reg] << 16) | imm

    raise ValueError(f"CFG_REGISTER expects 2 or 3 operands, got {operands}")


def encode_compute(mnemonic: str, operands: List[str]) -> int:
    expected_count = COMPUTE_OPERAND_COUNT[mnemonic]
    if len(operands) != expected_count:
        raise ValueError(f"{mnemonic} expects {expected_count} operand(s), got {operands}")

    regs = [op.upper() for op in operands]
    for reg in regs:
        if reg not in GENERAL_REGISTER_ID:
            raise ValueError(f"{mnemonic} only supports R1-R12 operands, got {reg}")

    rs0 = GENERAL_REGISTER_ID[regs[0]]
    rs1 = GENERAL_REGISTER_ID[regs[1]]
    rs2 = GENERAL_REGISTER_ID[regs[2]] if len(regs) == 3 else 0
    dtype = 0  # INT8, per current compiler pipeline and instruction manual.
    return (COMPUTE_OPCODE[mnemonic] << 26) | (rs0 << 21) | (rs1 << 16) | (rs2 << 11) | (dtype << 8)


def assemble_line(line: str) -> int | None:
    cleaned = strip_comment(line)
    if not cleaned:
        return None

    if " " in cleaned:
        mnemonic, rest = cleaned.split(None, 1)
        operands = split_operands(rest)
    else:
        mnemonic, operands = cleaned, []

    mnemonic = mnemonic.upper()
    if mnemonic == "CFG_REGISTER":
        return encode_cfg(operands)
    if mnemonic == "END":
        if operands:
            raise ValueError("END does not take operands")
        return END_WORD
    if mnemonic in ("CONV", "DSMP", "MADD", "RELU"):
        return encode_compute(mnemonic, operands)

    raise ValueError(f"unsupported instruction: {mnemonic}")


def assemble_lines(lines: Iterable[str]) -> List[int]:
    words: List[int] = []
    for line_no, line in enumerate(lines, start=1):
        try:
            word = assemble_line(line)
        except Exception as exc:
            raise ValueError(f"assembly failed at line {line_no}: {line.rstrip()!r}") from exc
        if word is not None:
            words.append(word)
    if not words:
        raise ValueError("assembly input contains no instructions")
    return words


def write_instr_txt(path: Path, words: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"0x{word:08X}\n" for word in words), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble NPU asm into 32-bit machine code txt")
    parser.add_argument("asm_path")
    parser.add_argument("out_txt")
    args = parser.parse_args()

    asm_path = Path(args.asm_path)
    out_txt = Path(args.out_txt)
    if not asm_path.is_file():
        raise FileNotFoundError(f"asm file not found: {asm_path}")

    words = assemble_lines(asm_path.read_text(encoding="utf-8").splitlines())
    write_instr_txt(out_txt, words)
    print(f"[OK] assembled {len(words)} instruction(s): {out_txt}")


if __name__ == "__main__":
    main()
