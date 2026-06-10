#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
instr_txt_to_bram_coe.py

将指定 txt 文件中的 32-bit NPU 指令转换成 Xilinx BRAM 初始化 .coe 文件。

默认输出 32-bit BRAM word 粒度 COE：
  1 个 memory_initialization_vector 项 = 4 byte = 32 bit
  连续 4 个低到高地址 byte 打包成 1 个 word，低地址 byte 放在低 8 位/最右边。

例如原 byte 序列为：
  20, 22, 1F, 00
默认 32-bit COE 输出为：
  20221F00

注意：
  指令先按 --instr-endian 拆成 byte 序列，再按 --word-endian 打包为 BRAM word。
  为了兼容原脚本的 byte 顺序，--instr-endian 默认仍为 big。

支持输入：
  0x04200000
  04200000
  32'h04200000
  00000100001000000000000000000000
  000001 00001 00000 0000000000000000
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List


def strip_comment(line: str) -> str:
    line = line.split("//", 1)[0]
    line = line.split("#", 1)[0]
    line = line.split(";", 1)[0]
    return line.strip()


def parse_number_token(token: str, default_radix: int = 16) -> int:
    """解析数字 token；不会把 COE 中的 0D/0B/0C 误判为前缀。"""
    t = token.strip().strip(",").strip()
    if not t:
        raise ValueError("empty token")

    # Verilog 风格：32'h04200000 / 8'h0D / 32'b0001
    m = re.fullmatch(r"(?:\d+)?'([hHbBdD])([0-9a-fA-F_xXzZ]+)", t)
    if m:
        base_ch = m.group(1).lower()
        digits = m.group(2).replace("_", "")
        digits = digits.replace("x", "0").replace("X", "0").replace("z", "0").replace("Z", "0")
        base = {"h": 16, "b": 2, "d": 10}[base_ch]
        return int(digits, base)

    if re.fullmatch(r"0[xX][0-9a-fA-F_]+", t):
        return int(t[2:].replace("_", ""), 16)
    if re.fullmatch(r"0[bB][01_]+", t):
        return int(t[2:].replace("_", ""), 2)
    if re.fullmatch(r"0[dD][0-9_]+", t):
        return int(t[2:].replace("_", ""), 10)

    return int(t.replace("_", ""), default_radix)


def read_instruction_words(instr_txt: Path) -> List[int]:
    words: List[int] = []
    for line_no, raw in enumerate(instr_txt.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = strip_comment(raw)
        if not line:
            continue

        compact = re.sub(r"[\s_]+", "", line)
        if re.fullmatch(r"[01]{32}", compact):
            words.append(int(compact, 2))
            continue

        tokens = re.split(r"[,\s]+", line)
        tokens = [t for t in tokens if t.strip()]
        if len(tokens) != 1:
            raise ValueError(
                f"{instr_txt} 第 {line_no} 行无法判断为一条 32-bit 指令: {raw!r}\n"
                f"建议每行只写一条指令，例如 0x04200000。"
            )

        try:
            val = parse_number_token(tokens[0], default_radix=16)
        except Exception as e:
            raise ValueError(f"{instr_txt} 第 {line_no} 行解析失败: {raw!r}") from e

        if not 0 <= val <= 0xFFFFFFFF:
            raise ValueError(f"{instr_txt} 第 {line_no} 行不是合法 32-bit 指令: {raw!r}")
        words.append(val)

    if not words:
        raise ValueError(f"{instr_txt} 中没有解析到任何指令")
    return words


def bytes_to_words(data: bytes, word_bytes: int, word_endian: str) -> List[int]:
    """连续 byte 按指定端序打包成 BRAM word。"""
    if word_bytes <= 0:
        raise ValueError("word_bytes 必须为正整数")
    if len(data) % word_bytes != 0:
        raise ValueError(f"数据长度 {len(data)} 不能被 word_bytes={word_bytes} 整除")

    return [
        int.from_bytes(data[i : i + word_bytes], byteorder=word_endian, signed=False)
        for i in range(0, len(data), word_bytes)
    ]


def format_values(values: Iterable[int], radix: int, width_hex: int) -> List[str]:
    if radix == 16:
        return [f"{v:0{width_hex}X}" for v in values]
    if radix == 10:
        return [str(v) for v in values]
    if radix == 2:
        return [format(v, f"0{width_hex * 4}b") for v in values]
    raise ValueError("radix 仅支持 2、10、16")


def write_coe(output_path: Path, values: List[int], radix: int, width_hex: int, values_per_line: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formatted = format_values(values, radix=radix, width_hex=width_hex)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"memory_initialization_radix={radix};\n")
        f.write("memory_initialization_vector=\n")
        for i in range(0, len(formatted), values_per_line):
            chunk = formatted[i:i + values_per_line]
            is_last = i + values_per_line >= len(formatted)
            f.write(", ".join(chunk) + (";\n" if is_last else ",\n"))


def main() -> None:
    parser = argparse.ArgumentParser(description="将 32-bit NPU 指令 txt 转换为 BRAM .coe 文件")
    parser.add_argument("instr_txt", help="输入指令 txt，每行一条 32-bit 指令")
    parser.add_argument("output_coe", help="输出 instruction.coe")
    parser.add_argument(
        "--mode",
        choices=["word", "byte"],
        default="word",
        help="word: 默认，32-bit BRAM word 输出；byte: 兼容旧版，每个 COE 项为 1 byte。",
    )
    parser.add_argument(
        "--instr-endian",
        choices=["big", "little"],
        default="big",
        help="指令拆成 byte 的端序。默认 big，保持旧脚本 0x04200000 -> 04,20,00,00 的 byte 顺序。",
    )
    parser.add_argument(
        "--word-bytes",
        type=int,
        default=4,
        help="mode=word 时每个 BRAM word 包含几个 byte。默认 4，即 32-bit。",
    )
    parser.add_argument(
        "--word-endian",
        choices=["little", "big"],
        default="big",
        help="mode=word 时 byte 打包端序。默认 big：保持 byte 序列原顺序输出。",
    )
    parser.add_argument("--radix", type=int, choices=[2, 10, 16], default=16, help="输出 COE radix，默认 16")
    parser.add_argument("--values-per-line", type=int, default=None, help="每行输出多少个初始化项。默认 word=1，byte=16")
    args = parser.parse_args()

    instr_txt = Path(args.instr_txt)
    output_coe = Path(args.output_coe)
    if not instr_txt.is_file():
        raise FileNotFoundError(f"找不到指令文件：{instr_txt}")

    words = read_instruction_words(instr_txt)
    raw_bytes = b"".join(w.to_bytes(4, byteorder=args.instr_endian, signed=False) for w in words)

    if args.mode == "byte":
        values = list(raw_bytes)
        width_hex = 2
        size_bytes = len(values)
        values_per_line = args.values_per_line or 16
    else:
        values = bytes_to_words(raw_bytes, word_bytes=args.word_bytes, word_endian=args.word_endian)
        width_hex = args.word_bytes * 2
        size_bytes = len(raw_bytes)
        values_per_line = args.values_per_line or 1

    write_coe(output_coe, values, radix=args.radix, width_hex=width_hex, values_per_line=values_per_line)

    print("[OK] 指令 COE 生成完成")
    print(f"     输入指令 : {instr_txt}")
    print(f"     输出 COE : {output_coe}")
    print(f"     输出模式 : {args.mode}")
    print(f"     指令条数 : {len(words)}")
    print(f"     指令大小 : 0x{size_bytes:08X} ({size_bytes}) bytes")
    if args.mode == "word":
        print(f"     BRAM word: {len(values)} words, {args.word_bytes} byte(s)/word")
        print(f"     打包端序 : {args.word_endian}")
    else:
        print(f"     拆字节序 : {args.instr_endian}")
        print(f"     示例     : 0x{words[0]:08X} -> " + " ".join(f"{b:02X}" for b in words[0].to_bytes(4, byteorder=args.instr_endian)))


if __name__ == "__main__":
    main()
