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
INFER_PARSE_LAYER_LIMIT = 3

INSTR_WORD_BYTES = 4

# If either a layer input feature map or output feature map has at least this
# many stored elements (width * height * aligned_channels), split output
# channels only in 4-channel groups instead of the default 8-then-4 grouping.
CHANNEL_GROUP4_FEATURE_SIZE_THRESHOLD = 256 * 256 * 4
