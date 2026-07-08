#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from . import pt2_frontend


def run_frontend(
    *,
    model_format: str,
    model_path: Path,
    ir_out: Path,
    params_dir: Path,
    config: Any,
) -> Dict[str, Any]:
    if model_format != "pt2":
        raise ValueError(f'Unsupported MODEL_FORMAT {model_format!r}. Current compiler only supports "pt2".')
    return pt2_frontend.export_model_ir(
        model_path=model_path,
        ir_out=ir_out,
        params_dir=params_dir,
        config=config,
    )
