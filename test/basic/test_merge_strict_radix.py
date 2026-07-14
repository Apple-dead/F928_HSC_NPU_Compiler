#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_coe(path: Path, values: list[str]) -> None:
    path.write_text(
        "memory_initialization_radix=16;\n"
        "memory_initialization_vector=\n"
        + ",\n".join(values)
        + ";\n",
        encoding="utf-8",
        newline="\n",
    )


def run_merge(input_path: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python", "python/merge.py", str(input_path), str(output_path)],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def run(ctx) -> None:
    work = PROJECT_ROOT / "data/tmp_regression/merge_strict_radix"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    good_input = work / "good.coe"
    good_output = work / "good_all.coe"
    write_coe(good_input, ["0D010413", "0B010203", "0D040202"])

    good = run_merge(good_input, good_output)
    if good.returncode != 0:
        raise AssertionError(f"merge.py rejected strict radix tokens:\n{good.stdout}\n{good.stderr}")

    output_text = good_output.read_text(encoding="utf-8")
    for expected in ("0D010413", "0B010203", "0D040202"):
        if expected not in output_text:
            raise AssertionError(f"{expected} was not preserved in merged COE")
    for wrong in ("000028AD", "00009D0A"):
        if wrong in output_text:
            raise AssertionError(f"merge.py still mis-parsed a radix=16 token as decimal: {wrong}")

    bad_input = work / "bad_verilog.coe"
    bad_output = work / "bad_verilog_all.coe"
    write_coe(bad_input, ["32'h0D010413"])

    bad = run_merge(bad_input, bad_output)
    if bad.returncode == 0:
        raise AssertionError("merge.py unexpectedly accepted Verilog-style COE token")
