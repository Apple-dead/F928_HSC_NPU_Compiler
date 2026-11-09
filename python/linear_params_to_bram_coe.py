#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Linear OI weights to the NPU FULL physical parameter layout."""

from __future__ import annotations

from typing import List

import numpy as np
from memory_layout import CHANNEL_ALIGNMENT, FEATURE_LAYOUT

from weight_to_bram_coe import check_int8_range, read_weight_txt


def convert_linear_weights(
    flat_values: np.ndarray,
    *,
    out_features: int,
    input_channels: int,
    input_height: int,
    input_width: int,
    aligned_input_channels: int,
    input_layout: str = FEATURE_LAYOUT,
) -> List[int]:
    if input_layout != FEATURE_LAYOUT:
        raise ValueError(f"FULL requires {FEATURE_LAYOUT}, got {input_layout!r}; regenerate the memory plan")
    actual_features = input_channels * input_height * input_width
    expected_max = out_features * actual_features
    if flat_values.size % out_features != 0:
        raise ValueError(
            f"linear weight count mismatch: file has {flat_values.size}, "
            f"not divisible by out_features={out_features}"
        )
    in_features = flat_values.size // out_features
    if in_features > actual_features:
        raise ValueError(
            f"linear weight count mismatch: file has {flat_values.size}, max expected {expected_max} "
            f"for out_features={out_features}, input={input_channels}x{input_height}x{input_width}"
        )
    if aligned_input_channels < input_channels:
        raise ValueError(
            f"aligned_input_channels={aligned_input_channels} must be >= input_channels={input_channels}"
        )
    if aligned_input_channels % CHANNEL_ALIGNMENT != 0:
        raise ValueError(f"FULL input channels must be padded to a multiple of 8, got {aligned_input_channels}")

    check_int8_range(flat_values)
    original = flat_values.reshape(out_features, in_features)
    padded = np.zeros(
        (out_features, aligned_input_channels, input_height, input_width),
        dtype=np.int64,
    )
    logical = np.zeros((out_features, actual_features), dtype=np.int64)
    logical[:, :in_features] = original
    padded[:, :input_channels, :, :] = logical.reshape(
        out_features, input_channels, input_height, input_width
    )

    # Every producer uses the same grouping, including pooling/DSMP/USMP/image.
    return (padded.reshape(out_features, aligned_input_channels // CHANNEL_ALIGNMENT,
                           CHANNEL_ALIGNMENT, input_height, input_width)
            .transpose(0, 1, 3, 4, 2).astype(np.int8).view(np.uint8).reshape(-1).tolist())


__all__ = ["convert_linear_weights", "read_weight_txt"]
