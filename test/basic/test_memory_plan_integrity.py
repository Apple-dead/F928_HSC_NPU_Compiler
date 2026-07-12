#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def addr_to_int(value: str) -> int:
    return int(value, 16)


def assert_non_overlapping(regions: list[dict], name: str) -> None:
    ranges = []
    for region in regions:
        start = addr_to_int(region["addr"])
        end = addr_to_int(region["end_addr_exclusive"])
        if end < start:
            raise AssertionError(f"{name} region {region.get('name')} has end before start")
        if end - start != int(region["size_bytes"]):
            raise AssertionError(f"{name} region {region.get('name')} size does not match address range")
        ranges.append((start, end, region.get("name", "<unnamed>")))

    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            raise AssertionError(
                f"{name} regions overlap: {previous[2]} ends at 0x{previous[1]:08X}, "
                f"{current[2]} starts at 0x{current[0]:08X}"
            )


def run(ctx) -> None:
    ctx.build_with_config_patch({})
    plan = ctx.read_json("data/memory_plan.json")
    assert_non_overlapping(plan.get("init_regions", []), "init")
    assert_non_overlapping(plan.get("runtime_regions", []), "runtime")

    instr_size = int(plan["tensors"]["instr"]["size_bytes"])
    instr_lines = [line for line in ctx.read_text("data/instr.txt").splitlines() if line.strip()]
    config_word_bytes = int(plan["config"].get("INSTR_WORD_BYTES", 4))
    actual_size = len(instr_lines) * config_word_bytes
    if actual_size != instr_size:
        raise AssertionError(f"instr size mismatch: memory_plan={instr_size}, instr.txt={actual_size}")

    init_by_name = {region["name"]: region for region in plan.get("init_regions", [])}
    map_text = ctx.read_text("target/all.coe.map.txt")
    for name, region in init_by_name.items():
        if f"region_name         = {name}" not in map_text:
            raise AssertionError(f"target map is missing init region {name}")
        if f"plan_size_bytes     = {int(region['size_bytes'])}" not in map_text:
            raise AssertionError(f"target map size for init region {name} does not match memory_plan")
