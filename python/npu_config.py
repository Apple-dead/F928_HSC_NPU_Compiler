#!/usr/bin/env python3
# -*- coding: utf-8 -*-

INIT_BASE_ADDR = 0x00010000
INIT_LIMIT_ADDR = 0x00200000
RUNTIME_BASE_ADDR = 0x00200000

MODEL_FORMAT = "pt2"
MODEL_PATH = "./model/lenet_0707_ptq_int8.pt2"
INTR_MOVE_PATH = "./model/intr_move.json"

IMAGE_BASE_ADDR = 0x00000000

# IMAGE_SOURCE:
#   "coe"      : use IMAGE_PATH as initialized input data. The input starts
#                at INIT_BASE_ADDR, so IMAGE_BASE_ADDR must match it.
#   "external" : do not merge input data into target/all.coe. The input feature
#                map is already available at IMAGE_BASE_ADDR, and initialized
#                params/instructions start at INIT_BASE_ADDR.
IMAGE_SOURCE = "external"
IMAGE_PATH = "./coe/image.coe"

# Fallback input size only. The PT2 frontend and memory planner prefer input
# shape metadata exported in the PT2 graph. These values are used only when
# that metadata is missing.
INPUT_HEIGHT = 28
INPUT_WIDTH = 28

# 1: export the full PT2 inference graph.
# 2: export the first INFER_PARSE_OP_LIMIT effective IR ops. Quantization
#    helper nodes such as clamp/to/floor are passthrough and do not count.
INFER_PARSE_MODE = 2
INFER_PARSE_OP_LIMIT = 4

INSTR_WORD_BYTES = 4

# If either a layer input feature map or output feature map has at least this
# many stored elements (width * height * aligned_channels), split output
# channels only in 4-channel groups instead of the default 8-then-4 grouping.
CHANNEL_GROUP4_FEATURE_SIZE_THRESHOLD = 256 * 256 * 4
