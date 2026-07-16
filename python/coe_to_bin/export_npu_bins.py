#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export NPU DDR binary images from generated COE files.

The original build flow emits COE files for BRAM initialization. This script is
an extra export stage for DDR loading:

  target/npu_params.bin = init regions from INIT_BASE_ADDR up to, but excluding, instr
  target/npu_instr.bin  = instr region only

All COE regions are exported with little-endian word interpretation: the least
significant byte of each 32-bit COE word is placed at the lowest address.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


COE_HEADER_RE = re.compile(r"memory_initialization_radix\s*=\s*(\d+)\s*;?", re.IGNORECASE)
COE_VECTOR_RE = re.compile(r"memory_initialization_vector\s*=\s*(.*)", re.IGNORECASE | re.DOTALL)


def strip_comment(line: str) -> str:
    line = line.split("//", 1)[0]
    line = line.split("#", 1)[0]
    line = line.split(";", 1)[0]
    return line.strip()


def parse_number_token(token: str, default_radix: int) -> int:
    text = token.strip().strip(",").strip().rstrip(";").strip()
    if not text:
        raise ValueError("empty token")
    return int(text.replace("_", ""), default_radix)


def read_coe_values(path: Path) -> tuple[list[int], int]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    radix_match = COE_HEADER_RE.search(text)
    if not radix_match:
        raise ValueError(f"{path}: missing memory_initialization_radix")
    radix = int(radix_match.group(1))

    vector_match = COE_VECTOR_RE.search(text)
    if not vector_match:
        raise ValueError(f"{path}: missing memory_initialization_vector")
    vector_text = vector_match.group(1)
    if ";" in vector_text:
        vector_text = vector_text.split(";", 1)[0]

    cleaned_lines = []
    for line in vector_text.splitlines():
        cleaned = strip_comment(line)
        if cleaned:
            cleaned_lines.append(cleaned)

    tokens = [token for token in re.split(r"[,\s]+", "\n".join(cleaned_lines)) if token.strip()]
    values: list[int] = []
    for token in tokens:
        try:
            values.append(parse_number_token(token, radix))
        except Exception as exc:
            raise ValueError(f"{path}: failed to parse COE token {token!r}, radix={radix}") from exc
    return values, radix


def coe_to_bytes(path: Path, *, word_bytes: int, word_endian: str) -> bytes:
    values, _ = read_coe_values(path)
    max_value = (1 << (word_bytes * 8)) - 1
    out = bytearray()
    for value in values:
        if not 0 <= value <= max_value:
            raise ValueError(f"{path}: value 0x{value:X} exceeds {word_bytes}-byte word")
        out.extend(value.to_bytes(word_bytes, byteorder=word_endian, signed=False))
    return bytes(out)


def addr_to_int(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return int(value, 16)


def resolve_plan_file(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base_dir / path


def region_name(region: dict) -> str:
    name = region.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"init region missing valid name: {region!r}")
    return name


def find_instr_index(init_regions: list[dict]) -> int:
    for index, region in enumerate(init_regions):
        if region_name(region) == "instr":
            return index
    raise ValueError("memory plan init_regions does not contain an instr region")


def append_region(
    image: bytearray,
    *,
    region: dict,
    base_addr: int,
    coe_base_dir: Path,
    word_endian: str,
) -> None:
    start_addr = addr_to_int(region["addr"])
    offset = start_addr - base_addr
    if offset < 0:
        raise ValueError(f"{region_name(region)} starts before base address")
    if len(image) > offset:
        raise ValueError(f"{region_name(region)} overlaps previous exported data")
    if len(image) < offset:
        image.extend(b"\x00" * (offset - len(image)))

    file_text = region.get("file")
    if not file_text:
        raise ValueError(f"{region_name(region)} is missing file")
    coe_path = resolve_plan_file(str(file_text), coe_base_dir)
    data = coe_to_bytes(coe_path, word_bytes=4, word_endian=word_endian)

    expected_size = int(region["size_bytes"])
    if len(data) != expected_size:
        raise ValueError(
            f"{region_name(region)} size mismatch: {coe_path} has {len(data)} bytes, "
            f"memory_plan expects {expected_size}"
        )
    image.extend(data)


def export_params_bin(
    *,
    init_regions: list[dict],
    instr_index: int,
    init_base_addr: int,
    coe_base_dir: Path,
) -> bytes:
    image = bytearray()
    for region in init_regions[:instr_index]:
        append_region(
            image,
            region=region,
            base_addr=init_base_addr,
            coe_base_dir=coe_base_dir,
            word_endian="little",
        )
    instr_addr = addr_to_int(init_regions[instr_index]["addr"])
    expected_size = instr_addr - init_base_addr
    if len(image) > expected_size:
        raise ValueError("parameter export exceeds instr start address")
    if len(image) < expected_size:
        image.extend(b"\x00" * (expected_size - len(image)))
    return bytes(image)


def export_instr_bin(*, instr_region: dict, coe_base_dir: Path) -> bytes:
    file_text = instr_region.get("file")
    if not file_text:
        raise ValueError("instr region is missing file")
    coe_path = resolve_plan_file(str(file_text), coe_base_dir)
    data = coe_to_bytes(coe_path, word_bytes=4, word_endian="little")
    expected_size = int(instr_region["size_bytes"])
    if len(data) != expected_size:
        raise ValueError(
            f"instr size mismatch: {coe_path} has {len(data)} bytes, "
            f"memory_plan expects {expected_size}"
        )
    return data


def write_binary(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NPU params/instr DDR binary images from generated COE files.")
    parser.add_argument("--memory-plan", type=Path, default=Path("./data/memory_plan.json"))
    parser.add_argument("--memory-plan-base-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("./target"))
    parser.add_argument("--params-out", default="npu_params.bin")
    parser.add_argument("--instr-out", default="npu_instr.bin")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    plan = json.loads(args.memory_plan.read_text(encoding="utf-8"))
    init_regions = plan.get("init_regions")
    if not isinstance(init_regions, list) or not init_regions:
        raise ValueError(f"{args.memory_plan}: missing non-empty init_regions")

    config = plan.get("config", {})
    if "INIT_BASE_ADDR" not in config:
        raise ValueError(f"{args.memory_plan}: config.INIT_BASE_ADDR is required")
    init_base_addr = addr_to_int(config["INIT_BASE_ADDR"])

    instr_index = find_instr_index(init_regions)
    params_data = export_params_bin(
        init_regions=init_regions,
        instr_index=instr_index,
        init_base_addr=init_base_addr,
        coe_base_dir=args.memory_plan_base_dir,
    )
    instr_data = export_instr_bin(instr_region=init_regions[instr_index], coe_base_dir=args.memory_plan_base_dir)

    params_path = args.out_dir / args.params_out
    instr_path = args.out_dir / args.instr_out
    write_binary(params_path, params_data)
    write_binary(instr_path, instr_data)

    print("[OK] NPU DDR binary export complete")
    print(f"     params bin : {params_path} ({len(params_data)} bytes), load_addr=0x{init_base_addr:08X}")
    print(
        f"     instr bin  : {instr_path} ({len(instr_data)} bytes), "
        f"planned_addr={init_regions[instr_index]['addr']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
