#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge.py

用途：
  将任意多个 .coe 文件按命令行输入顺序拼接成一个总 .coe，并生成 map 文件。
  最后一个位置参数作为输出总 .coe，前面的所有位置参数都作为待拼接输入 .coe。

典型用法：
  python merge.py image.coe weight.coe bias.coe instr.coe target.coe

含义：
  将 image.coe、weight.coe、bias.coe、instr.coe 按顺序拼接到 target.coe 中。

对齐规则：
  - 第 1 个输入 COE 从地址 0x00000000 开始；
  - 从第 2 个输入 COE 开始，每个 COE 区域的起始地址都按 --align 对齐；
  - 默认 --align=4，即每个区域起始地址都凑到 4 byte 的倍数；
  - 如果当前位置不是 4 byte 的倍数，会自动在前一个区域后补 0x00。

默认假设：
  - 输入 COE 默认是 32-bit word 粒度：1 个 memory_initialization_vector 项 = 4 byte；
  - 输出总 COE 默认也是 32-bit word 粒度：1 个 memory_initialization_vector 项 = 4 byte；
  - word 内部默认 little-endian 解释/打包，即低地址 byte 在右边/低 8 位。

例如：
  输入 COE 项 001F2220 会被解释回 byte 序列 20,22,1F,00；
  多个输入按 byte 地址顺序拼接后，再按 20,22,1F,00 -> 001F2220 的规则输出。

如果你仍要拼接旧版 byte 粒度 COE：
  python merge.py old_byte.coe another_byte.coe target.coe --input-word-bytes 1 --output-word-bytes 4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple


COE_HEADER_RE = re.compile(r"memory_initialization_radix\s*=\s*(\d+)\s*;?", re.IGNORECASE)
COE_VECTOR_RE = re.compile(r"memory_initialization_vector\s*=\s*(.*)", re.IGNORECASE | re.DOTALL)


def strip_comment(line: str) -> str:
    """去掉 COE 中常见注释。"""
    line = line.split("//", 1)[0]
    line = line.split("#", 1)[0]
    line = line.split(";", 1)[0]
    return line.strip()


def parse_number_token(token: str, default_radix: int) -> int:
    """
    严格按 COE 文件头 memory_initialization_radix 解析裸 token。

    不支持 Python 风格 0x/0b/0d 前缀，也不支持 Verilog 风格
    32'h.../32'b.../32'd... 前缀，避免把 radix=16 下的普通 word
    例如 0D010413 误判为十进制前缀。
    """
    t = token.strip().strip(",").strip().rstrip(";").strip()
    if not t:
        raise ValueError("empty token")

    return int(t.replace("_", ""), default_radix)


def read_coe_values(path: Path) -> Tuple[List[int], int]:
    """读取 COE 初始化向量，返回数值列表和 radix。"""
    text = path.read_text(encoding="utf-8", errors="ignore")

    radix_match = COE_HEADER_RE.search(text)
    if not radix_match:
        raise ValueError(f"{path}: 未找到 memory_initialization_radix")
    radix = int(radix_match.group(1))

    vector_match = COE_VECTOR_RE.search(text)
    if not vector_match:
        raise ValueError(f"{path}: 未找到 memory_initialization_vector")
    vector_text = vector_match.group(1)

    # COE vector 到第一个分号结束
    if ";" in vector_text:
        vector_text = vector_text.split(";", 1)[0]

    cleaned_lines = []
    for line in vector_text.splitlines():
        cleaned = strip_comment(line)
        if cleaned:
            cleaned_lines.append(cleaned)

    raw_tokens = re.split(r"[,\s]+", "\n".join(cleaned_lines))
    tokens = [t.strip() for t in raw_tokens if t.strip()]

    values: List[int] = []
    for tok in tokens:
        try:
            values.append(parse_number_token(tok, default_radix=radix))
        except Exception as e:
            raise ValueError(f"{path}: 解析 COE token 失败: {tok!r}, radix={radix}") from e

    return values, radix


def coe_values_to_bytes(path: Path, word_bytes: int, word_endian: str) -> Tuple[List[int], int, int]:
    """
    将 COE 初始化项转成 byte 列表。

    word_bytes=4：
      每个初始化项可在 0..0xFFFFFFFF，并按 word_endian 拆成 4 byte。
      默认 little，可将 001F2220 还原成 20,22,1F,00。

    word_bytes=1：
      兼容旧版 byte COE，每个初始化项必须在 0..255。
    """
    if word_bytes <= 0:
        raise ValueError("word_bytes 必须大于 0")

    values, radix = read_coe_values(path)
    max_value = (1 << (word_bytes * 8)) - 1

    out: List[int] = []
    for v in values:
        if not 0 <= v <= max_value:
            raise ValueError(
                f"{path}: 发现值 0x{v:X} 超出 word_bytes={word_bytes} 可表示范围 0x{max_value:X}。\n"
                f"如果该 COE 是旧版 byte 粒度，请指定 --input-word-bytes 1；"
                f"如果每个输入粒度不同，请用 --word-bytes-list。"
            )
        out.extend(v.to_bytes(word_bytes, byteorder=word_endian, signed=False))

    return out, radix, len(values)


def align_up(value: int, alignment: int) -> int:
    """向上对齐到 alignment 的整数倍。"""
    if alignment <= 0:
        raise ValueError("alignment 必须大于 0")
    return ((value + alignment - 1) // alignment) * alignment


def bytes_to_words(data: List[int], word_bytes: int, word_endian: str) -> List[int]:
    if word_bytes <= 0:
        raise ValueError("word_bytes 必须大于 0")
    if len(data) % word_bytes != 0:
        raise ValueError(f"总 byte 数 {len(data)} 不能被 output_word_bytes={word_bytes} 整除")
    raw = bytes(data)
    return [
        int.from_bytes(raw[i : i + word_bytes], byteorder=word_endian, signed=False)
        for i in range(0, len(raw), word_bytes)
    ]


def format_word(v: int, radix: int, word_bytes: int) -> str:
    """输出 word 粒度 COE 的单个初始化项。"""
    if radix == 16:
        return f"{v:0{word_bytes * 2}X}"
    if radix == 10:
        return str(v)
    if radix == 2:
        return f"{v:0{word_bytes * 8}b}"
    raise ValueError("输出 radix 仅支持 2、10、16")


def write_word_coe(path: Path, data: List[int], radix: int, word_bytes: int, word_endian: str, values_per_line: int) -> None:
    """写出 word 粒度总 COE。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    words = bytes_to_words(data, word_bytes=word_bytes, word_endian=word_endian)
    formatted = [format_word(v, radix, word_bytes) for v in words]

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"memory_initialization_radix={radix};\n")
        f.write("memory_initialization_vector=\n")

        for i in range(0, len(formatted), values_per_line):
            chunk = formatted[i : i + values_per_line]
            is_last = (i + values_per_line) >= len(formatted)
            f.write(", ".join(chunk) + (";\n" if is_last else ",\n"))


def parse_word_bytes_list(text: str, expected_len: int) -> List[int]:
    values = [int(x.strip(), 0) for x in text.split(",") if x.strip()]
    if len(values) != expected_len:
        raise ValueError(
            f"--word-bytes-list 的数量必须等于输入 COE 数量。"
            f"当前输入 COE 数量={expected_len}，但 word-bytes-list 数量={len(values)}。"
        )
    if any(v <= 0 for v in values):
        raise ValueError("--word-bytes-list 中的每个值都必须大于 0")
    return values


def parse_word_endian_list(text: str, expected_len: int) -> List[str]:
    values = [x.strip().lower() for x in text.split(",") if x.strip()]
    if len(values) != expected_len:
        raise ValueError(
            f"--word-endian-list 的数量必须等于输入 COE 数量。"
            f"当前输入 COE 数量={expected_len}，但 word-endian-list 数量={len(values)}。"
        )
    for v in values:
        if v not in ("little", "big"):
            raise ValueError("--word-endian-list 只能包含 little 或 big")
    return values


def resolve_plan_file(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base_dir / path


def input_paths_from_memory_plan(memory_plan_path: Path, base_dir: Optional[Path]) -> Tuple[List[Path], List[dict]]:
    """Read init_regions from memory_plan.json and return COE paths in planned order."""
    plan = json.loads(memory_plan_path.read_text(encoding="utf-8"))
    init_regions = plan.get("init_regions")
    if not isinstance(init_regions, list) or not init_regions:
        raise ValueError(f"{memory_plan_path}: missing non-empty init_regions")

    plan_base_dir = base_dir if base_dir is not None else Path.cwd()
    input_paths: List[Path] = []
    region_infos: List[dict] = []
    for idx, region in enumerate(init_regions):
        if not isinstance(region, dict):
            raise ValueError(f"{memory_plan_path}: init_regions[{idx}] is not an object")
        file_text = region.get("file")
        if not file_text:
            raise ValueError(f"{memory_plan_path}: init_regions[{idx}] missing file")
        input_paths.append(resolve_plan_file(str(file_text), plan_base_dir))
        region_infos.append(region)

    return input_paths, region_infos


def write_map(
    path: Path,
    input_paths: List[Path],
    output_path: Path,
    regions: List[dict],
    total_size: int,
    align: int,
    pad_value: int,
    output_radix: int,
    output_word_bytes: int,
    output_word_endian: str,
    final_padding: int,
) -> None:
    """写出每个输入 COE 在总 COE 中的地址范围。"""
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("多 COE 拼接地址映射说明\n")
        f.write("=" * 80 + "\n")
        f.write(f"说明：总 COE 输出为 {output_word_bytes}-byte word 粒度，即 1 个 memory_initialization_vector 项 = {output_word_bytes} byte。\n")
        f.write(f"说明：word 输出端序 = {output_word_endian}。little 表示低地址 byte 在右边/低 8 位。\n")
        f.write(f"说明：每个输入 COE 区域起始地址按 {align} byte 对齐。\n")
        f.write(f"说明：对齐填充值 pad_value = 0x{pad_value:02X}。\n")
        f.write(f"说明：总 COE 输出 radix = {output_radix}。\n\n")

        f.write("输入 COE 文件顺序:\n")
        for idx, p in enumerate(input_paths):
            f.write(f"  [{idx}] {p}\n")
        f.write(f"\n输出总 COE 文件:\n  {output_path}\n\n")

        f.write("区域地址映射:\n")
        for r in regions:
            idx = r["index"]
            start = r["start"]
            size = r["size"]
            end = r["end"]
            pad_before = r["pad_before"]
            item_count = r["item_count"]
            word_bytes = r["word_bytes"]
            word_endian = r["word_endian"]
            radix = r["input_radix"]
            path_str = r["path"]
            name = r.get("name")
            plan_addr = r.get("plan_addr")
            plan_size = r.get("plan_size")
            plan_end = r.get("plan_end")

            f.write("-" * 80 + "\n")
            f.write(f"region_index        = {idx}\n")
            if name is not None:
                f.write(f"region_name         = {name}\n")
            f.write(f"input_file          = {path_str}\n")
            f.write(f"input_radix         = {radix}\n")
            f.write(f"input_item_count    = 0x{item_count:08X} ({item_count})\n")
            f.write(f"input_word_bytes    = {word_bytes}\n")
            f.write(f"input_word_endian   = {word_endian}\n")
            if plan_addr is not None:
                f.write(f"plan_start_addr     = {plan_addr}\n")
            if plan_size is not None:
                f.write(f"plan_size_bytes     = {plan_size}\n")
            if plan_end is not None:
                f.write(f"plan_end_exclusive  = {plan_end}\n")
            f.write(f"padding_before      = 0x{pad_before:08X} ({pad_before})\n")
            f.write(f"start_addr          = 0x{start:08X} ({start})\n")
            f.write(f"size_bytes          = 0x{size:08X} ({size})\n")
            f.write(f"end_addr            = 0x{end:08X} ({end})\n")
        f.write("-" * 80 + "\n")
        f.write(f"final_padding_bytes = 0x{final_padding:08X} ({final_padding})\n")
        f.write(f"total_size_bytes    = 0x{total_size:08X} ({total_size})\n")
        f.write(f"total_output_words  = 0x{(total_size // output_word_bytes):08X} ({total_size // output_word_bytes})\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "将任意多个输入 COE 按命令行顺序拼接成一个 32-bit word 粒度总 COE。"
            "用法：python merge.py xx1.coe xx2.coe ... xxn.coe target.coe"
        )
    )
    parser.add_argument(
        "coe_files",
        nargs="*",
        help=(
            "手动模式：前 N-1 个为输入 COE，最后 1 个为输出总 COE；"
            "memory-plan 模式：只传输出总 COE，输入 COE 从 init_regions 读取。"
        ),
    )
    parser.add_argument(
        "--memory-plan",
        type=Path,
        default=None,
        help="读取 memory_plan.json 的 init_regions 顺序作为输入 COE 顺序。",
    )
    parser.add_argument(
        "--memory-plan-base-dir",
        type=Path,
        default=None,
        help="解析 init_regions[*].file 相对路径的基准目录，默认使用当前工作目录。",
    )
    parser.add_argument("--map-out", default=None, help="输出 map 文件路径。默认 <target.coe>.map.txt")
    parser.add_argument("--align", type=int, default=4, help="每个输入 COE 起始地址对齐字节数，默认 4")
    parser.add_argument("--pad-value", type=lambda x: int(x, 0), default=0, help="对齐填充值，默认 0x00")
    parser.add_argument("--output-radix", type=int, choices=[2, 10, 16], default=16, help="总 COE 输出 radix，默认 16")
    parser.add_argument("--values-per-line", type=int, default=1, help="总 COE 每行输出多少个 word，默认 1")

    parser.add_argument(
        "--input-word-bytes",
        type=int,
        default=4,
        help="所有输入 COE 的每个初始化项包含几个 byte，默认 4，即输入为 32-bit word COE。旧版 byte COE 请设为 1。",
    )
    parser.add_argument(
        "--input-word-endian",
        choices=["little", "big"],
        default="little",
        help="所有输入 COE 的 word 拆 byte 端序，默认 little：001F2220 -> 20,22,1F,00。",
    )
    parser.add_argument(
        "--output-word-bytes",
        type=int,
        default=4,
        help="输出 COE 的每个初始化项包含几个 byte，默认 4，即输出 32-bit word COE。",
    )
    parser.add_argument(
        "--output-word-endian",
        choices=["little", "big"],
        default="little",
        help="输出 COE 的 word 打包端序，默认 little：20,22,1F,00 -> 001F2220。",
    )
    parser.add_argument(
        "--word-bytes-list",
        default=None,
        help="为每个输入 COE 单独指定 word_bytes，例如 4,4,1。数量必须等于输入 COE 数量。",
    )
    parser.add_argument(
        "--word-endian-list",
        default=None,
        help="为每个输入 COE 单独指定 word endian，例如 little,little,big。数量必须等于输入 COE 数量。",
    )

    args = parser.parse_args()

    if args.memory_plan is not None and len(args.coe_files) != 1:
        raise ValueError(
            "使用 --memory-plan 时，位置参数只需要传输出 COE。\n"
            "例如：python merge.py --memory-plan data/memory_plan.json target/all.coe"
        )
    if args.memory_plan is None and len(args.coe_files) < 2:
        raise ValueError(
            "至少需要 2 个位置参数：一个或多个输入 COE + 一个输出 COE。\n"
            "例如：python merge.py xx1.coe xx2.coe target.coe"
        )
    if not 0 <= args.pad_value <= 0xFF:
        raise ValueError("--pad-value 必须在 0~255 范围内")
    if args.values_per_line <= 0:
        raise ValueError("--values-per-line 必须大于 0")
    if args.input_word_bytes <= 0:
        raise ValueError("--input-word-bytes 必须大于 0")
    if args.output_word_bytes <= 0:
        raise ValueError("--output-word-bytes 必须大于 0")

    region_infos: Optional[List[dict]] = None
    if args.memory_plan is not None:
        input_paths, region_infos = input_paths_from_memory_plan(args.memory_plan, args.memory_plan_base_dir)
        output_path = Path(args.coe_files[0])
    else:
        input_paths = [Path(p) for p in args.coe_files[:-1]]
        output_path = Path(args.coe_files[-1])
    map_path = Path(args.map_out) if args.map_out else Path(str(output_path) + ".map.txt")

    for p in input_paths:
        if not p.is_file():
            raise FileNotFoundError(f"找不到输入 COE 文件：{p}")

    input_count = len(input_paths)

    if args.word_bytes_list:
        word_bytes_list = parse_word_bytes_list(args.word_bytes_list, input_count)
    else:
        word_bytes_list = [args.input_word_bytes] * input_count

    if args.word_endian_list:
        word_endian_list = parse_word_endian_list(args.word_endian_list, input_count)
    else:
        word_endian_list = [args.input_word_endian] * input_count

    total_data: List[int] = []
    regions: List[dict] = []
    current_addr = 0

    for idx, path in enumerate(input_paths):
        word_bytes = word_bytes_list[idx]
        word_endian = word_endian_list[idx]

        data_bytes, input_radix, item_count = coe_values_to_bytes(path, word_bytes, word_endian)

        aligned_start = align_up(current_addr, args.align)
        pad_before = aligned_start - current_addr
        if pad_before:
            total_data.extend([args.pad_value] * pad_before)

        start = aligned_start
        size = len(data_bytes)
        end = start + size - 1 if size > 0 else start

        total_data.extend(data_bytes)
        current_addr = start + size

        regions.append(
            {
                "index": idx,
                "name": region_infos[idx].get("name") if region_infos else None,
                "plan_addr": region_infos[idx].get("addr") if region_infos else None,
                "plan_size": region_infos[idx].get("size_bytes") if region_infos else None,
                "plan_end": region_infos[idx].get("end_addr_exclusive") if region_infos else None,
                "path": str(path),
                "input_radix": input_radix,
                "item_count": item_count,
                "word_bytes": word_bytes,
                "word_endian": word_endian,
                "pad_before": pad_before,
                "start": start,
                "size": size,
                "end": end,
            }
        )

    final_padding = (-len(total_data)) % args.output_word_bytes
    if final_padding:
        total_data.extend([args.pad_value] * final_padding)

    write_word_coe(
        output_path,
        total_data,
        radix=args.output_radix,
        word_bytes=args.output_word_bytes,
        word_endian=args.output_word_endian,
        values_per_line=args.values_per_line,
    )
    write_map(
        path=map_path,
        input_paths=input_paths,
        output_path=output_path,
        regions=regions,
        total_size=len(total_data),
        align=args.align,
        pad_value=args.pad_value,
        output_radix=args.output_radix,
        output_word_bytes=args.output_word_bytes,
        output_word_endian=args.output_word_endian,
        final_padding=final_padding,
    )

    print("[OK] 多 COE 拼接完成")
    print(f"     输入 COE 数量 : {input_count}")
    for r in regions:
        print(
            f"     [{r['index']}] start=0x{r['start']:08X}, "
            f"size=0x{r['size']:08X}, end=0x{r['end']:08X}, "
            f"pad_before={r['pad_before']}, file={r['path']}"
        )
    print("-" * 72)
    print(f"     输出 COE      : {output_path}")
    print(f"     输出 map      : {map_path}")
    print(f"     总大小        : 0x{len(total_data):08X} ({len(total_data)}) bytes")
    print(f"     输出 word     : {len(total_data) // args.output_word_bytes} words, {args.output_word_bytes} byte(s)/word")
    print("     打包规则      : 低地址 byte 在右边/低 8 位")


if __name__ == "__main__":
    main()
