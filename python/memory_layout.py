"""Shared physical feature-map layout; addresses remain byte addresses."""

CHANNEL_ALIGNMENT = 8
FEATURE_LAYOUT = "channel_group_then_h_then_w_then_8_channels"
INTERMEDIATE_COE_WORD_BYTES = 4
BRAM_WORD_BYTES = 64
CONV_MAX_OUTPUT_CHANNELS = 8
# Kept as an alias for callers using the earlier CONV-specific name.
CONV_FEATURE_LAYOUT = FEATURE_LAYOUT


def conv_channel_groups(out_ch: int, aligned_out_ch: int, max_channels: int = CONV_MAX_OUTPUT_CHANNELS):
    """Return (first channel, stored lanes, valid lanes, has padding) per pass."""
    if aligned_out_ch != aligned_channels(out_ch):
        raise ValueError(f"expected 8-channel padding for {out_ch}, got {aligned_out_ch}")
    if max_channels != CONV_MAX_OUTPUT_CHANNELS:
        raise ValueError("CONV passes must use eight stored channel lanes")
    return [(start, 8, min(8, out_ch - start), out_ch - start < 8)
            for start in range(0, out_ch, 8)]


def aligned_channels(channels: int) -> int:
    if channels <= 0:
        raise ValueError(f"channel count must be positive, got {channels}")
    return ((channels + CHANNEL_ALIGNMENT - 1) // CHANNEL_ALIGNMENT) * CHANNEL_ALIGNMENT


def pack_feature_hwc(data: bytes, *, height: int, width: int, channels: int) -> bytes:
    """Pack logical HWC bytes as channel-group -> H -> W -> eight lanes.

    Padding lanes belong to the final group at each spatial position. Byte
    values (including signed-int8 two's complement) are preserved unchanged.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"feature HW must be positive, got {height}x{width}")
    stored_channels = aligned_channels(channels)
    if len(data) != height * width * channels:
        raise ValueError(f"HWC byte count {len(data)} does not match {height}x{width}x{channels}")
    result = bytearray(height * width * stored_channels)
    target = 0
    for start in range(0, channels, CHANNEL_ALIGNMENT):
        valid = min(CHANNEL_ALIGNMENT, channels - start)
        for pixel in range(height * width):
            source = pixel * channels + start
            result[target:target + valid] = data[source:source + valid]
            target += CHANNEL_ALIGNMENT
    return bytes(result)
