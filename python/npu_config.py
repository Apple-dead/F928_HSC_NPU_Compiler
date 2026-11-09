#!/usr/bin/env python3
# -*- coding: utf-8 -*-

INIT_BASE_ADDR = 0x10000000
INIT_LIMIT_ADDR = 0x1FFFFFFF
RUNTIME_BASE_ADDR = 0x24000000
RUNTIME_LIMIT_ADDR = 0x4BFFFFFF 

# False: final output is allocated in runtime space. True: place it at OUTPUT_BASE_ADDR.
ENABLE_OUTPUT_ADDR = True
OUTPUT_BASE_ADDR = 0x4D000000
OUTPUT_LIMIT_ADDR = 0x50000000

MODEL_FORMAT = "pt2"
MODEL_PATH = "./model/handmade_int8_clean.pt2"
INTR_MOVE_PATH = "./model/intr_move.json"

IMAGE_BASE_ADDR = 0x10000000

# IMAGE_SOURCE:
#   "coe"      : use IMAGE_PATH as initialized input data. The input starts
#                at INIT_BASE_ADDR, so IMAGE_BASE_ADDR must match it.
#   "external" : do not merge input data into target/all.coe. The input feature
#                map is already available at IMAGE_BASE_ADDR, and initialized
#                params/instructions start at INIT_BASE_ADDR.
IMAGE_SOURCE = "coe"
# Input COE is a 32-bit intermediate file ordered as channel group -> H -> W
# -> eight lanes: all pixels of ch1..8 precede all pixels of ch9..16.
# Pad only the last channel group. External feature maps use the same layout.
# Only the final target/all.coe is packed into 512-bit entries.
IMAGE_PATH = "./coe/image.coe"

# Fallback input size only. The PT2 frontend and memory planner prefer input
# shape metadata exported in the PT2 graph. These values are used only when
# that metadata is missing.
INPUT_HEIGHT = 28
INPUT_WIDTH = 28

# 1: export the full PT2 inference graph.
# 2: export the first INFER_PARSE_OP_LIMIT effective IR ops. Quantization
#    helper nodes such as clamp/to/floor are passthrough and do not count.
INFER_PARSE_MODE = 1
INFER_PARSE_OP_LIMIT = 6

# True: if a graph output is produced by cat, export the cat inputs as model
# outputs instead of emitting the final cat op into IR.
EXPAND_FINAL_CAT_OUTPUTS = True

INSTR_WORD_BYTES = 4

# CONV emits at most 8 output channels per pass (16 outputs -> 8 + 8).
# Each pass writes one contiguous H x W x 8 block; the final pass pads to 8.
# Parameter records contain only valid outputs; no dummy output weights/biases.
