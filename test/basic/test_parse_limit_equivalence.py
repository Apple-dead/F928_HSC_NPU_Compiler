#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def run(ctx) -> None:
    ctx.build_with_config_patch({"INFER_PARSE_MODE": 1})
    full_hash = ctx.sha256("target/all.coe")
    op_count = len(ctx.read_json("data/model_ir.json").get("ops", []))
    if op_count <= 0:
        raise AssertionError("mode=1 build produced no IR ops")

    ctx.build_with_config_patch({"INFER_PARSE_MODE": 2, "INFER_PARSE_OP_LIMIT": op_count})
    limited_hash = ctx.sha256("target/all.coe")

    if limited_hash != full_hash:
        raise AssertionError(
            "mode=2 with INFER_PARSE_OP_LIMIT equal to effective op count "
            "must match mode=1 target/all.coe"
        )
