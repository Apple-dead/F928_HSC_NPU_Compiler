#!/usr/bin/env python3
# -*- coding: utf-8 -*-


PASSTHROUGH_OPS = {"flatten", "clamp", "to", "floor"}
PASSTHROUGH_TARGET_FRAGMENTS = ("flatten", "clamp", "to.dtype", "floor")


def run(ctx) -> None:
    ctx.build_with_config_patch({"INFER_PARSE_MODE": 1})
    ir = ctx.read_json("data/model_ir.json")
    for op in ir.get("ops", []):
        op_name = str(op.get("op", ""))
        source_target = str(op.get("source_target", ""))
        if op_name in PASSTHROUGH_OPS:
            raise AssertionError(f"passthrough op should not appear in IR: {op_name}")
        if any(fragment in source_target for fragment in PASSTHROUGH_TARGET_FRAGMENTS):
            raise AssertionError(f"passthrough target should not appear in IR: {source_target}")
