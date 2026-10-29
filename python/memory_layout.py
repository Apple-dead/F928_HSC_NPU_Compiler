"""Shared physical feature-map layout; addresses remain byte addresses."""

CHANNEL_ALIGNMENT = 8
FEATURE_LAYOUT = "pixel_interleaved_all_channels_padded_to_8"
INTERMEDIATE_COE_WORD_BYTES = 4
BRAM_WORD_BYTES = 64
CONV_MAX_OUTPUT_CHANNELS = 32


def aligned_channels(channels: int) -> int:
    if channels <= 0:
        raise ValueError(f"channel count must be positive, got {channels}")
    return ((channels + CHANNEL_ALIGNMENT - 1) // CHANNEL_ALIGNMENT) * CHANNEL_ALIGNMENT
