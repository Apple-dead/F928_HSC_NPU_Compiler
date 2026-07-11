#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Linear OI weights to the NPU FULL physical parameter layout."""

from __future__ import annotations

from typing import List

import numpy as np

from weight_to_bram_coe import check_int8_range, read_weight_txt, signed_to_unsigned_twos_complement


def convert_linear_weights(
    flat_values: np.ndarray,
    *,
    out_features: int,
    input_channels: int,
    input_height: int,
    input_width: int,
    aligned_input_channels: int,
    input_storage_height: int,
    input_storage_width: int,
) -> List[int]:
    expected = out_features * input_channels * input_height * input_width
    if flat_values.size != expected:
        raise ValueError(
            f"linear weight count mismatch: file has {flat_values.size}, expected {expected} "
            f"for out_features={out_features}, input={input_channels}x{input_height}x{input_width}"
        )
    if aligned_input_channels < input_channels:
        raise ValueError(
            f"aligned_input_channels={aligned_input_channels} must be >= input_channels={input_channels}"
        )
    if input_storage_height < input_height or input_storage_width < input_width:
        raise ValueError(
            f"input storage HW {input_storage_height}x{input_storage_width} must contain "
            f"logical HW {input_height}x{input_width}"
        )
    if aligned_input_channels % 4 != 0:
        raise ValueError(f"FULL input channels must be padded to a multiple of 4, got {aligned_input_channels}")

    check_int8_range(flat_values)
    original = flat_values.reshape(out_features, input_channels, input_height, input_width)
    padded = np.zeros(
        (out_features, aligned_input_channels, input_storage_height, input_storage_width),
        dtype=np.int64,
    )
    padded[:, :input_channels, :input_height, :input_width] = original

    emitted: List[int] = []
    for output_index in range(out_features):
        for channel_base in range(0, aligned_input_channels, 4):
            for row in range(input_storage_height):
                for col in range(input_storage_width):
                    for channel in range(channel_base, channel_base + 4):
                        emitted.append(
                            signed_to_unsigned_twos_complement(
                                int(padded[output_index, channel, row, col])
                            )
                        )
    return emitted


__all__ = ["convert_linear_weights", "read_weight_txt"]
