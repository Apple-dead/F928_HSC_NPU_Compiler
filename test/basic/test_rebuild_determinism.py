#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def run(ctx) -> None:
    ctx.build_with_config_patch({})
    first_hash = ctx.sha256("target/all.coe")

    ctx.build_with_config_patch({})
    second_hash = ctx.sha256("target/all.coe")

    if second_hash != first_hash:
        raise AssertionError("two identical builds must produce identical target/all.coe")
