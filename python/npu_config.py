#!/usr/bin/env python3
# -*- coding: utf-8 -*-

INIT_BASE_ADDR = 0x00000000
INIT_LIMIT_ADDR = 0x00200000
RUNTIME_BASE_ADDR = 0x00200000

IMAGE_BASE_ADDR = 0x00000000
IMAGE_SOURCE = "coe"

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# 1: parse the full model inference graph from the model .py.
# 2: parse only the first INFER_PARSE_LAYER_LIMIT model layer(s).
INFER_PARSE_MODE = 2
INFER_PARSE_LAYER_LIMIT = 1

INSTR_WORD_BYTES = 4
