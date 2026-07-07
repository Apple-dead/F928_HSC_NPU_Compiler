#!/usr/bin/env python3
# -*- coding: utf-8 -*-

INIT_BASE_ADDR = 0x00000000
INIT_LIMIT_ADDR = 0x00200000
RUNTIME_BASE_ADDR = 0x00200000

IMAGE_BASE_ADDR = 0x00000000

# IMAGE_SOURCE:
#   "coe"      : use ./coe/image.coe as initialized input data. The input
#                starts at INIT_BASE_ADDR, so IMAGE_BASE_ADDR must match it.
#   "external" : do not merge input data into target/all.coe. The input feature
#                map is already available at IMAGE_BASE_ADDR, and initialized
#                params/instructions start at INIT_BASE_ADDR.
IMAGE_SOURCE = "coe"

# Original model input size. When INFER_PARSE_LAYER_START > 1, the compiler
# derives the selected start layer input size from this and the full model
# structure; these are not the current start layer feature-map dimensions.
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# 1: parse the full model inference graph from the model .py.
# 2: parse only INFER_PARSE_LAYER_START..INFER_PARSE_LAYER_END.
INFER_PARSE_MODE = 2
INFER_PARSE_LAYER_START = 1
INFER_PARSE_LAYER_END = 3

INSTR_WORD_BYTES = 4

# If either a layer input feature map or output feature map has at least this
# many stored elements (width * height * aligned_channels), split output
# channels only in 4-channel groups instead of the default 8-then-4 grouping.
CHANNEL_GROUP4_FEATURE_SIZE_THRESHOLD = 256 * 256 * 4
