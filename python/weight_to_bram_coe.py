#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weight_to_bram_coe.py

将第一层卷积权重 txt 转换为 NPU 需要的 4 输入通道 x 4 输出通道格式，并生成 Xilinx BRAM 初始化 .coe 文件。

默认假设输入权重 txt 来自 PyTorch Conv2d weight 的 arr.reshape(-1)，即原始布局为：
  [out_channel, in_channel, kernel_row, kernel_col]

对 3 输入通道、3 输出通道、3x3 卷积：
  原始权重数量 = 3 * 3 * 3 * 3 = 81
  补零后数量   = 4 * 4 * 3 * 3 = 144

输出排列方式：
  for oc in 0..3:
    for kh in 0..2:
      for kw in 0..2:
        for ic in 0..3:
          emit weight[oc][ic][kh][kw]

默认输出 32-bit BRAM word 粒度 COE：
  每 4 个连续权重 byte 打成一个 32-bit word；低地址 byte 放在低 8 位/最右边。
  例如 byte 序列 20,22,1F,00 输出为 001F2220。

负数按 signed int8 的二进制补码写入 .coe，例如 -1 作为 byte 为 FF。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np


NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def read_weight_txt(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    nums = [float(x) for x in NUMBER_RE.findall(text)]
    if not nums:
        raise ValueError(f"权重文件中没有解析到数字：{path}")

    arr_float = np.asarray(nums, dtype=np.float64)
    arr_int = np.rint(arr_float).astype(np.int64)

    if not np.allclose(arr_float, arr_int, atol=1e-6):
        bad_idx = np.where(~np.isclose(arr_float, arr_int, atol=1e-6))[0][:10]
        bad_vals = arr_float[bad_idx].tolist()
        raise ValueError(
            "检测到非整数权重值；当前脚本按 int8 量化权重生成 COE。"
            f" 示例索引/数值：{list(zip(bad_idx.tolist(), bad_vals))}"
        )

    return arr_int


def check_int_range(values: np.ndarray, bit_width: int) -> None:
    min_v = -(1 << (bit_width - 1))
    max_v = (1 << (bit_width - 1)) - 1
    if values.size and (values.min() < min_v or values.max() > max_v):
        raise ValueError(
            f"权重超出 signed int{bit_width} 范围 [{min_v}, {max_v}]："
            f" min={values.min()}, max={values.max()}"
        )


def signed_to_unsigned_twos_complement(v: int, bit_width: int) -> int:
    mask = (1 << bit_width) - 1
    return int(v) & mask


def pack_units_to_words(units: List[int], unit_bits: int, word_bytes: int, endian: str) -> List[int]:
    """先把 int8 单元变为 byte，再按 word_bytes 打包成 .coe word。"""
    if unit_bits != 8:
        raise ValueError("当前脚本只支持 unit_bits=8，也就是 int8 权重")
    if word_bytes <= 0:
        raise ValueError("word_bytes 必须为正整数")
    if len(units) % word_bytes != 0:
        raise ValueError(
            f"权重字节数 {len(units)} 不能被 word_bytes={word_bytes} 整除；"
            "当前默认 32-bit word 要求字节数是 4 的倍数。"
        )

    data = bytes(units)
    words: List[int] = []
    for i in range(0, len(data), word_bytes):
        words.append(int.from_bytes(data[i : i + word_bytes], byteorder=endian, signed=False))
    return words


def format_words(words: Iterable[int], radix: int, word_bytes: int) -> List[str]:
    if radix == 16:
        return [f"{w:0{word_bytes * 2}X}" for w in words]
    if radix == 10:
        return [str(w) for w in words]
    raise ValueError("目前仅支持 radix=16 或 radix=10")


def write_coe(words: List[int], output_path: Path, radix: int, word_bytes: int, values_per_line: int) -> None:
    values = format_words(words, radix=radix, word_bytes=word_bytes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"memory_initialization_radix={radix};\n")
        f.write("memory_initialization_vector=\n")
        for i in range(0, len(values), values_per_line):
            chunk = values[i : i + values_per_line]
            is_last = i + values_per_line >= len(values)
            f.write(", ".join(chunk) + (";\n" if is_last else ",\n"))


def convert_weights(
    flat_values: np.ndarray,
    in_ch: int,
    out_ch: int,
    kernel_h: int,
    kernel_w: int,
    pad_in_ch: int,
    pad_out_ch: int,
    bit_width: int,
) -> List[int]:
    expected = out_ch * in_ch * kernel_h * kernel_w
    if flat_values.size != expected:
        raise ValueError(
            f"权重数量不匹配：当前解析到 {flat_values.size} 个数，"
            f"但按 out_ch={out_ch}, in_ch={in_ch}, kernel={kernel_h}x{kernel_w} 应为 {expected} 个。"
        )

    if pad_in_ch < in_ch or pad_out_ch < out_ch:
        raise ValueError("pad_in_ch/pad_out_ch 不能小于原始 in_ch/out_ch")

    check_int_range(flat_values, bit_width=bit_width)

    original = flat_values.reshape(out_ch, in_ch, kernel_h, kernel_w)
    padded = np.zeros((pad_out_ch, pad_in_ch, kernel_h, kernel_w), dtype=np.int64)
    padded[:out_ch, :in_ch, :, :] = original

    emitted_signed: List[int] = []
    for oc in range(pad_out_ch):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                for ic in range(pad_in_ch):
                    emitted_signed.append(int(padded[oc, ic, kh, kw]))

    return [signed_to_unsigned_twos_complement(v, bit_width=bit_width) for v in emitted_signed]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将第一层 Conv 权重 txt 转换成补齐 4 通道后的 32-bit BRAM 初始化 .coe 文件"
    )
    parser.add_argument("weight_txt", help="输入权重 txt 路径，例如 layer1_0_weight.txt")
    parser.add_argument("output_coe", help="输出 .coe 文件路径")
    parser.add_argument("--in-ch", type=int, default=3, help="原始输入通道数，默认 3")
    parser.add_argument("--out-ch", type=int, default=3, help="原始输出通道数，默认 3")
    parser.add_argument("--kernel-h", type=int, default=3, help="卷积核高度，默认 3")
    parser.add_argument("--kernel-w", type=int, default=3, help="卷积核宽度，默认 3")
    parser.add_argument("--pad-in-ch", type=int, default=4, help="补齐后的输入通道数，默认 4")
    parser.add_argument("--pad-out-ch", type=int, default=4, help="补齐后的输出通道数，默认 4")
    parser.add_argument("--bit-width", type=int, default=8, help="权重量化位宽，默认 signed int8")
    parser.add_argument("--radix", type=int, choices=[10, 16], default=16, help=".coe radix，默认 16")
    parser.add_argument(
        "--word-bytes",
        type=int,
        default=4,
        help="每个 BRAM word 包含几个权重字节。默认 4，即 32-bit。",
    )
    parser.add_argument(
        "--word-endian",
        choices=["little", "big"],
        default="little",
        help="word 打包端序。默认 little：第一个/低地址权重 byte 放在低 8 位/最右边。",
    )
    parser.add_argument("--values-per-line", type=int, default=1, help="每行输出多少个 word，默认 1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weight_path = Path(args.weight_txt)
    output_coe = Path(args.output_coe)

    if not weight_path.is_file():
        raise FileNotFoundError(f"找不到权重文件：{weight_path}")

    flat_values = read_weight_txt(weight_path)
    units = convert_weights(
        flat_values=flat_values,
        in_ch=args.in_ch,
        out_ch=args.out_ch,
        kernel_h=args.kernel_h,
        kernel_w=args.kernel_w,
        pad_in_ch=args.pad_in_ch,
        pad_out_ch=args.pad_out_ch,
        bit_width=args.bit_width,
    )
    words = pack_units_to_words(
        units, unit_bits=args.bit_width, word_bytes=args.word_bytes, endian=args.word_endian
    )
    write_coe(words, output_coe, radix=args.radix, word_bytes=args.word_bytes, values_per_line=args.values_per_line)

    print("[OK] 权重转换完成")
    print(f"     输入权重 : {weight_path}")
    print(f"     原始数量 : {flat_values.size} = {args.out_ch} * {args.in_ch} * {args.kernel_h} * {args.kernel_w}")
    print(
        f"     补齐数量 : {len(units)} = {args.pad_out_ch} * {args.pad_in_ch} * {args.kernel_h} * {args.kernel_w}"
    )
    print(f"     数值范围 : signed int{args.bit_width}, 写入 COE 时采用二进制补码")
    print(f"     BRAM word: {len(words)} words, {args.word_bytes} byte(s)/word")
    print(f"     打包规则 : 低地址 byte 在右边/低 8 位")
    print(f"     前 16 个输出字节(hex): {' '.join(f'{x:02X}' for x in units[:16])}")
    print(f"     输出 COE : {output_coe}")


if __name__ == "__main__":
    main()
