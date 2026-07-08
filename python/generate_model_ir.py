#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import npu_config as cfg
from frontend.registry import run_frontend


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IR_OUT = PROJECT_ROOT / "data" / "model_ir.json"
DEFAULT_PARAMS_DIR = PROJECT_ROOT / "data" / "model_params"


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standard model IR and parameter text files.")
    parser.add_argument("--out", type=Path, default=DEFAULT_IR_OUT)
    parser.add_argument("--params-dir", type=Path, default=DEFAULT_PARAMS_DIR)
    args = parser.parse_args()

    model_format = getattr(cfg, "MODEL_FORMAT", None)
    model_path = resolve_project_path(getattr(cfg, "MODEL_PATH", ""))
    ir = run_frontend(
        model_format=model_format,
        model_path=model_path,
        ir_out=args.out,
        params_dir=args.params_dir,
        config=cfg,
    )
    print(f"[OK] model IR generated: {args.out}")
    print(f"[OK] model params generated: {args.params_dir}")
    print(f"     model_format = {model_format}")
    print(f"     model_path   = {model_path}")
    print(f"     parse        = {ir['parse']}")
    print(f"     output       = {ir['output']}")
    print(f"     ops          = {[op['op'] for op in ir['ops']]}")
    print(json.dumps({"input": ir["input"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
