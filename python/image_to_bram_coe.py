#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_to_bram_coe.py

将 256x256 输入图像整理为 NPU 需要的 4 通道字节排列，并生成 Xilinx BRAM 初始化 .coe 文件。

支持的图像模式：
  - RGB  : 按每个像素 R,G,B 后补 0，形成 R,G,B,0
  - RGBA : 直接使用图像原始 R,G,B,A 四字节
  - RGBX : 直接使用图像原始 R,G,B,X 四字节

默认 .coe 以 32-bit BRAM word 为一个 memory_initialization_vector 项输出，radix=16。
连续 4 个 byte 低地址到高地址排列为 b0,b1,b2,b3 时，默认输出 word 为 b3b2b1b0，
即低地址 byte 在右边/低 8 位。例如 20,22,1F,00 -> 001F2220。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "[ERROR] 缺少 Pillow。请先安装：pip install pillow"
    ) from exc


VALID_SIZE = (256, 256)
RGB888_MODES = {"RGB"}
RGB8888_MODES = {"RGBA", "RGBX"}


def bytes_to_words(data: bytes, word_bytes: int, endian: str) -> List[int]:
    """把连续字节按 word_bytes 打包成 .coe 中的一个个 memory word。"""
    if word_bytes <= 0:
        raise ValueError("word_bytes 必须为正整数")
    if len(data) % word_bytes != 0:
        raise ValueError(
            f"数据长度 {len(data)} 不能被 word_bytes={word_bytes} 整除，无法整字打包"
        )

    words: List[int] = []
    for i in range(0, len(data), word_bytes):
        chunk = data[i : i + word_bytes]
        words.append(int.from_bytes(chunk, byteorder=endian, signed=False))
    return words


def format_words(words: Iterable[int], radix: int, word_bytes: int) -> List[str]:
    """根据 radix 把 word 格式化成 .coe 初始化向量字符串。"""
    if radix == 16:
        width = word_bytes * 2
        return [f"{w:0{width}X}" for w in words]
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


def image_to_rgba_like_bytes(image_path: Path) -> tuple[bytes, str]:
    """读取图像，并转换/保持为 NPU 所需的每像素 4 字节排列。"""
    with Image.open(image_path) as img:
        if img.size != VALID_SIZE:
            raise ValueError(f"输入图像尺寸必须为 {VALID_SIZE[0]}x{VALID_SIZE[1]}，当前为 {img.size}")

        mode = img.mode
        if mode in RGB8888_MODES:
            # 文档要求 RGB8888 直接用原始四通道数据，不额外改动第四通道。
            return img.tobytes(), mode

        if mode in RGB888_MODES:
            rgb = img.tobytes()
            out = bytearray()
            for i in range(0, len(rgb), 3):
                out.extend(rgb[i : i + 3])
                out.append(0)
            return bytes(out), mode

        raise ValueError(
            f"不支持的图像模式：{mode}。要求图像解码后为 RGB/RGBA/RGBX，即 RGB888 或 RGB8888。"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 256x256 RGB888/RGB8888 图像转换成 32-bit BRAM 初始化 .coe 文件"
    )
    parser.add_argument("image_path", help="输入图像路径，要求 256x256，模式为 RGB/RGBA/RGBX")
    parser.add_argument("output_coe", help="输出 .coe 文件路径")
    parser.add_argument("--radix", type=int, choices=[10, 16], default=16, help=".coe radix，默认 16")
    parser.add_argument(
        "--word-bytes",
        type=int,
        default=4,
        help="每个 BRAM word 包含的字节数。默认 4，即 32-bit。",
    )
    parser.add_argument(
        "--word-endian",
        choices=["little", "big"],
        default="little",
        help="word 打包端序。默认 little：低地址 byte 放在低 8 位/最右边。",
    )
    parser.add_argument("--values-per-line", type=int, default=1, help="每行输出多少个 word，默认 1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_path)
    output_coe = Path(args.output_coe)

    if not image_path.is_file():
        raise FileNotFoundError(f"找不到输入图像：{image_path}")
    if args.values_per_line <= 0:
        raise ValueError("--values-per-line 必须大于 0")

    data, mode = image_to_rgba_like_bytes(image_path)
    expected_bytes = 256 * 256 * 4
    if len(data) != expected_bytes:
        raise RuntimeError(f"内部错误：输出字节数应为 {expected_bytes}，实际为 {len(data)}")

    words = bytes_to_words(data, word_bytes=args.word_bytes, endian=args.word_endian)
    write_coe(words, output_coe, radix=args.radix, word_bytes=args.word_bytes, values_per_line=args.values_per_line)

    print("[OK] 图像转换完成")
    print(f"     输入图像 : {image_path}")
    print(f"     图像模式 : {mode}")
    print(f"     输出字节 : {len(data)} bytes = 256 * 256 * 4")
    print(f"     BRAM word: {len(words)} words, {args.word_bytes} byte(s)/word")
    print("     打包规则 : 低地址 byte 在右边/低 8 位")
    print(f"     输出 COE : {output_coe}")


if __name__ == "__main__":
    main()
