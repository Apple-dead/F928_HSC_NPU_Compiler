#!/usr/bin/env python3
# -*- coding: utf-8 -*-


GENERATED_FILES = (
    "data/model_ir.json",
    "data/memory_plan.json",
    "data/instr.asm",
    "data/instr.txt",
    "coe/instr.coe",
    "target/all.coe",
    "target/all.coe.map.txt",
)


def run(ctx) -> None:
    ctx.build_with_config_patch({})
    for relative_path in GENERATED_FILES:
        ctx.require_file(relative_path)

    ctx.clean()
    for relative_path in GENERATED_FILES:
        ctx.require_absent(relative_path)
