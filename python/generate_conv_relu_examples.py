#!/usr/bin/env python3
"""Generate standalone BRAM images and integer references for fused CONV.

Uses the compiler's memory planner, parameter packer, assembler and COE writer.
Does not modify the active model/configuration or use the test/ directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np

import assembler
import generate_instr
import generate_memory_plan
import merge
import npu_config
import params_to_bram_coe
from memory_layout import BRAM_WORD_BYTES, aligned_channels


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEIGHT = WIDTH = 16
INPUT_CHANNELS, OUTPUT_CHANNELS = 16, 8
KERNEL = 3
SEED = 20261020
INSTRUCTION_WORDS = 14  # 6 address CFG + 4 RCONV CFG + 2 RRELU CFG + CONV + END.


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_mem(path: Path, data: bytes, word_bytes: int) -> None:
    if len(data) % word_bytes:
        raise ValueError(f"{path}: incomplete memory word")
    path.write_text("".join(
        f"{int.from_bytes(data[i:i + word_bytes], 'little'):0{word_bytes * 2}X}\n"
        for i in range(0, len(data), word_bytes)
    ), encoding="ascii")


def pack_input_hwc8(values: np.ndarray) -> bytes:
    """Channel group -> row -> column -> eight channel lanes; byte order unchanged."""
    height, width, channels = values.shape
    assert channels % 8 == 0
    return values.reshape(height, width, channels // 8, 8).transpose(2, 0, 1, 3).tobytes()


def reference(x: np.ndarray, weights: np.ndarray, bias: np.ndarray):
    """Cross-correlation, smallest safe arithmetic shift, then LeakyReLU."""
    height, width, input_channels = x.shape
    output_channels, kernel_h, kernel_w, _ = weights.shape
    padded = np.pad(x.astype(np.int64), ((1, 1), (1, 1), (0, 0)))
    accumulator = np.zeros((height, width, output_channels), dtype=np.int64)
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            accumulator += np.einsum(
                "hwc,oc->hwo", padded[kh:kh + height, kw:kw + width],
                weights[:, kh, kw, :].astype(np.int64),
            )
    accumulator += bias.astype(np.int64)

    # Independently check the vector calculation with explicit scalar indexing.
    for h in range(height):
        for w in range(width):
            for oc in range(output_channels):
                value = int(bias[oc])
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        ih, iw = h + kh - 1, w + kw - 1
                        if 0 <= ih < height and 0 <= iw < width:
                            for ic in range(input_channels):
                                value += int(x[ih, iw, ic]) * int(weights[oc, kh, kw, ic])
                if value != accumulator[h, w, oc]:
                    raise AssertionError(f"reference mismatch at {(h, w, oc)}")

    scale = 0
    while int(accumulator.min()) >> scale < -128 or int(accumulator.max()) >> scale > 127:
        scale += 1
    shifted = accumulator >> scale
    truncated = ((shifted + 128) % 256 - 128).astype(np.int64)
    assert np.array_equal(truncated, shifted)
    output = np.where(truncated < 0, truncated >> 3, truncated).astype(np.int8)
    assert np.any(truncated < 0) and np.any(truncated > 0)
    assert np.any((accumulator < -128) | (accumulator > 127))
    assert np.any((truncated < 0) & (truncated % 8 != 0))
    return accumulator, truncated, output, scale


def make_ir(height: int, width: int, input_channels: int, output_channels: int) -> dict:
    return {
        "input": {"name": "x", "height": height, "width": width, "channels": input_channels},
        "ops": [
            {"id": 1, "op": "conv2d", "name": "conv", "layer_index": 1,
             "input": "x", "output": "conv_out", "in_channels": input_channels,
             "out_channels": output_channels, "kernel_size": [KERNEL, KERNEL],
             "padding": [1, 1], "stride": [1, 1], "dilation": [1, 1],
             "groups": 1, "has_bias": True,
             "output_shape": [1, output_channels, height, width]},
            {"id": 2, "op": "relu", "name": "leaky_relu", "input": "conv_out",
             "output": "y", "negative_slope": 0.125, "tan": 3,
             "output_shape": [1, output_channels, height, width]},
        ],
        "outputs": [{"index": 0, "name": "y"}],
    }


def make_plan(case_dir: Path, input_addr: int, output_addr: int, region_alignment: int) -> dict:
    # Isolate this example's addresses from the user's active compiler config.
    config = {name: getattr(npu_config, name) for name in dir(npu_config) if name.isupper()}
    input_info = json.loads((case_dir / "model_ir.json").read_text(encoding="utf-8"))["input"]
    config.update(
        INIT_BASE_ADDR=input_addr, INIT_LIMIT_ADDR=0x10000,
        IMAGE_BASE_ADDR=input_addr, IMAGE_SOURCE="coe", IMAGE_PATH=str(case_dir / "input.coe"),
        RUNTIME_BASE_ADDR=0x10000, RUNTIME_LIMIT_ADDR=0x20000,
        ENABLE_OUTPUT_ADDR=True, OUTPUT_BASE_ADDR=output_addr, OUTPUT_LIMIT_ADDR=0x10000,
        INPUT_HEIGHT=input_info["height"], INPUT_WIDTH=input_info["width"], INFER_PARSE_MODE=1,
        MODEL_FORMAT="ir", MODEL_PATH="model_ir.json", INTR_MOVE_PATH="intr_move.json",
    )
    original_config = generate_memory_plan.cfg
    try:
        generate_memory_plan.cfg = SimpleNamespace(**config)
        plan = generate_memory_plan.build_plan(case_dir / "model_ir.json")
    finally:
        generate_memory_plan.cfg = original_config
    # Keep this example-specific policy separate from the compiler's normal
    # 4-byte region alignment. Padded input/parameter footprints already meet
    # the selected region alignment in these profiles; align the instruction entry.
    # Rebase parameter aliases and execution addresses when example regions need
    # stronger alignment than the compiler's normal four-byte allocation.
    params = plan["tensors"]["layer1_params"]
    old_base = int(params["addr"], 16)
    delta = generate_memory_plan.align_up(old_base, region_alignment) - old_base
    if delta:
        for tensor_name, tensor in plan["tensors"].items():
            if tensor_name == "layer1_params" or tensor.get("parameter_region") == "layer1_params" or tensor_name == "instr":
                for key in ("addr", "end_addr_exclusive"):
                    if key in tensor:
                        tensor[key] = f"0x{int(tensor[key], 16) + delta:08X}"
        for split in plan["execution_plan"][0]["splits"]:
            split["conv"]["weight_addr"] = f"0x{int(split['conv']['weight_addr'], 16) + delta:08X}"
            if split.get("bias_addr"):
                split["bias_addr"] = f"0x{int(split['bias_addr'], 16) + delta:08X}"
        for region in plan["init_regions"]:
            if region["name"] in ("layer1_params", "instr"):
                region.update(plan["tensors"][region["name"]])
    instr = plan["tensors"]["instr"]
    instr_addr = generate_memory_plan.align_up(int(instr["addr"], 16), region_alignment)
    instr_end = instr_addr + int(instr["size_bytes"])
    if instr_end > output_addr:
        raise ValueError("aligned instruction region overlaps the output region")
    instr.update(addr=f"0x{instr_addr:08X}", end_addr_exclusive=f"0x{instr_end:08X}")
    for item in plan["init_regions"]:
        if item["name"] == "instr":
            item.update(instr)
        if int(item["addr"], 16) % region_alignment:
            raise ValueError(f"{item['name']} is not aligned to {region_alignment} bytes")
    plan["init_end_addr_exclusive"] = f"0x{instr_end:08X}"
    plan["alignment_bytes"] = region_alignment
    plan["config"]["EXAMPLE_INIT_REGION_ALIGNMENT_BYTES"] = region_alignment
    return plan


def region_map(name: str, start: int, size: int) -> dict:
    return {
        "name": name, "start_addr": f"0x{start:08X}", "size_bytes": size,
        "end_addr_exclusive": f"0x{start + size:08X}",
        "bram_word_index": start // BRAM_WORD_BYTES,
        "byte_offset_in_bram_word": start % BRAM_WORD_BYTES,
    }


def build_case(out_dir: Path, name: str, input_addr: int, x, weights, bias, stages,
               region_alignment: int = 4) -> dict:
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    accumulator, truncated, expected, scale = stages
    height, width, input_channels = x.shape
    output_channels = weights.shape[0]
    stored_input_channels = aligned_channels(input_channels)
    stored_output_channels = aligned_channels(output_channels)
    stored_input = np.zeros((height, width, stored_input_channels), dtype=np.int8)
    stored_input[:, :, :input_channels] = x
    stored_weights = np.zeros((output_channels, KERNEL, KERNEL, stored_input_channels), dtype=np.int8)
    stored_weights[:, :, :, :input_channels] = weights
    stored_bias = np.zeros(stored_output_channels, dtype=np.int64)
    stored_bias[:output_channels] = bias
    stored_output = np.zeros((height, width, stored_output_channels), dtype=np.int8)
    stored_output[:, :, :output_channels] = expected
    stored_accumulator = np.zeros((height, width, stored_output_channels), dtype=np.int64)
    stored_accumulator[:, :, :output_channels] = accumulator
    stored_truncated = np.zeros_like(stored_accumulator)
    stored_truncated[:, :, :output_channels] = truncated
    input_bytes = pack_input_hwc8(stored_input)
    weight_bytes = stored_weights.tobytes(order="C")
    record_weight_bytes = KERNEL * KERNEL * stored_input_channels
    record_stride = record_weight_bytes + 8
    expected_parameters = b"".join(
        weight_bytes[c * record_weight_bytes:(c + 1) * record_weight_bytes]
        + int(bias[c]).to_bytes(4, "little", signed=True) + bytes(4)
        for c in range(output_channels))
    weight_addr = generate_memory_plan.align_up(input_addr + len(input_bytes), region_alignment)
    bias_addr = weight_addr + record_weight_bytes
    instr_addr = generate_memory_plan.align_up(weight_addr + len(expected_parameters), region_alignment)
    instruction_padding = instr_addr - (weight_addr + len(expected_parameters))
    output_addr = (instr_addr + INSTRUCTION_WORDS * 4 + 7) // 8 * 8
    expected_bytes = stored_output.tobytes(order="C")

    merge.write_word_coe(case_dir / "input.coe", list(input_bytes), 16, 4, "little", 1)
    write_json(case_dir / "model_ir.json", make_ir(height, width, input_channels, output_channels))
    moves = {"CONV_MOVE_BY_INDEX": {1: 1 << scale}, "FULL_MOVE_BY_INDEX": {}}
    write_json(case_dir / "intr_move.json", {key: {str(k): v for k, v in values.items()} for key, values in moves.items()})
    plan = make_plan(case_dir, input_addr, output_addr, region_alignment)
    plan["tensors"]["image"]["layout"] = "channel_group_then_h_then_w_then_8_channels"
    plan["image"]["layout"] = "channel_group_then_h_then_w_then_8_channels"
    layer_plan = plan["execution_plan"][0]
    assert layer_plan["splits"][0]["conv"]["weight_addr"] == f"0x{weight_addr:08X}"
    assert plan["tensors"]["instr"]["addr"] == f"0x{instr_addr:08X}"

    # Convert the logical OIHW weights using exactly the compiler's packer.
    logical_weights = weights.transpose(0, 3, 1, 2).reshape(-1).astype(np.int64)
    parameter_bytes = bytes(params_to_bram_coe.build_layer_bytes(
        layer_plan, plan["tensors"]["layer1_weight"], logical_weights, bias.astype(np.int64).tolist(),
    ))
    assert parameter_bytes == expected_parameters
    merge.write_word_coe(case_dir / "params.coe", list(parameter_bytes), 16, 4, "little", 1)

    asm = generate_instr.build_asm(plan, moves)
    words = assembler.assemble_lines(asm)
    generate_instr.validate_instr_size(plan, len(words))
    assert len(words) == INSTRUCTION_WORDS
    # This preserves the existing final merger's byte order for instruction COE:
    # each 32-bit machine word's LSB is stored at its lowest byte address.
    instruction_bytes = b"".join(word.to_bytes(4, "little") for word in words)
    generate_instr.write_asm(case_dir / "instr.asm", generate_instr.add_instruction_line_numbers(asm))
    assembler.write_instr_txt(case_dir / "instr.txt", words)
    merge.write_word_coe(case_dir / "instr.coe", list(instruction_bytes), 16, 4, "little", 1)
    write_csv(case_dir / "instructions.csv",
              ["index", "byte_addr", "machine_word", "assembly"],
              ((i, f"0x{instr_addr + i * 4:08X}", f"{word:08X}", line)
               for i, (word, line) in enumerate(zip(words, (a for a in asm if assembler.assemble_line(a) is not None)))))

    init_end = instr_addr + len(instruction_bytes)
    total_bytes = (init_end + BRAM_WORD_BYTES - 1) // BRAM_WORD_BYTES * BRAM_WORD_BYTES
    runtime_end = output_addr + len(expected_bytes)
    required_depth = (runtime_end + BRAM_WORD_BYTES - 1) // BRAM_WORD_BYTES
    image = bytearray(total_bytes)
    regions = [
        ("input", input_addr, input_bytes), ("parameters", weight_addr, parameter_bytes),
        ("instructions", instr_addr, instruction_bytes),
    ]
    for _, addr, data in regions:
        image[addr:addr + len(data)] = data
    merge.write_word_coe(case_dir / "all.coe", list(image), 16, BRAM_WORD_BYTES, "little", 1)
    write_mem(case_dir / "all.mem", bytes(image), BRAM_WORD_BYTES)
    (case_dir / "all.bin").write_bytes(image)
    (case_dir / "expected_output.bin").write_bytes(expected_bytes)
    write_mem(case_dir / "expected_output.mem", expected_bytes, 1)
    expected_coe_padding = (-len(expected_bytes)) % BRAM_WORD_BYTES
    merge.write_word_coe(case_dir / "expected_output.coe", list(expected_bytes) + [0] * expected_coe_padding,
                         16, BRAM_WORD_BYTES, "little", 1)

    write_csv(case_dir / "input_hwc.csv", ["h", "w"] + [f"ch{c}" for c in range(stored_input_channels)],
              ([h, w] + [int(v) for v in stored_input[h, w]] for h in range(height) for w in range(width)))
    write_csv(case_dir / "weights_ohwi.csv", ["out_ch", "kh", "kw"] + [f"in_ch{c}" for c in range(stored_input_channels)],
              ([o, kh, kw] + [int(v) for v in stored_weights[o, kh, kw]]
               for o in range(output_channels) for kh in range(KERNEL) for kw in range(KERNEL)))
    write_csv(case_dir / "bias.csv", ["out_ch", "signed_int8_value", "int32_word", "byte_addr"],
              ([c, int(stored_bias[c]), f"{int(stored_bias[c]) & 0xFFFFFFFF:08X}", f"0x{bias_addr + c * record_stride:08X}"]
               for c in range(output_channels)))
    write_csv(case_dir / "expected_hwc.csv", ["h", "w"] + [f"ch{c}" for c in range(stored_output_channels)],
              ([h, w] + [int(v) for v in stored_output[h, w]] for h in range(height) for w in range(width)))
    write_csv(case_dir / "reference.csv",
              ["h", "w", "out_ch", "output_byte_addr", "conv_sum_before_bias", "bias", "accumulator_with_bias",
               "scaled_int8_signed", "expected_signed", "expected_hex", "is_padding_channel"],
              ([h, w, c, f"0x{output_addr + (h * width + w) * stored_output_channels + c:08X}",
                int(stored_accumulator[h,w,c]) - int(stored_bias[c]), int(stored_bias[c]), int(stored_accumulator[h,w,c]),
                int(stored_truncated[h,w,c]), int(stored_output[h,w,c]), f"{int(stored_output[h,w,c]) & 255:02X}",
                int(c >= output_channels)]
               for h in range(height) for w in range(width) for c in range(stored_output_channels)))

    rconv1, rconv2 = generate_instr.load_operator("conv").encode_rconv(
        [3, 3], [1, 1], [1, 1], width, scale, input_channels, output_channels, True, "relu",
    )
    rrelu = generate_instr.load_operator("relu").encode_rrelu(width, output_channels, 3)
    summary = {
        "case": name, "seed": SEED, "memory_base_addr": "0x00000000",
        "init_region_alignment_bytes": region_alignment,
        "parameter_addr": f"0x{weight_addr:08X}",
        "parameter_size_bytes": len(parameter_bytes),
        "zero_padding_before_instructions_bytes": instruction_padding,
        "input_addr": f"0x{input_addr:08X}", "instruction_entry_addr": f"0x{instr_addr:08X}",
        "output_addr": f"0x{output_addr:08X}", "output_size_bytes": len(expected_bytes),
        "bram_word_bits": 512, "bram_depth_words": required_depth,
        "initialized_bram_words": total_bytes // BRAM_WORD_BYTES,
        "initialized_size_bytes": total_bytes,
        "required_bram_size_bytes": required_depth * BRAM_WORD_BYTES,
        "logical_init_end_exclusive": f"0x{init_end:08X}",
        "initialized_end_addr_exclusive": f"0x{total_bytes:08X}",
        "final_padding_bytes": total_bytes - init_end,
        "memory_size_bytes": total_bytes, "instruction_count": len(words),
        "input_shape_hwc": [height, width, input_channels],
        "input_storage_shape_hwc": [height, width, stored_input_channels],
        "output_shape_hwc": [height, width, output_channels],
        "output_storage_shape_hwc": [height, width, stored_output_channels],
        "bias_valid_count": output_channels, "bias_storage_count": output_channels,
        "parameter_record_stride_bytes": record_stride, "first_bias_addr": f"0x{bias_addr:08X}",
        "bias_padding_bytes_per_value": 4,
        "expected_output_coe_final_padding_bytes": expected_coe_padding,
        "kernel": [3, 3], "padding": [1, 1], "stride": [1, 1], "condition_bias": 1,
        "input_weight_bias_random_value_dtype": "signed_int8", "bias_storage_dtype": "signed_int32_le",
        "slope": "1/8", "tan": 3, "start_bit": scale,
        "input_layout": "channel_group_then_h_then_w_then_8_channels",
        "accumulator_range": [int(accumulator.min()), int(accumulator.max())],
        "scaled_range": [int(truncated.min()), int(truncated.max())],
        "compute_order": "int32_sum_plus_bias -> arithmetic_shift_right_scale -> signed_int8 -> negative_arithmetic_shift_right_3",
        "saturation": False, "instruction_word_endian": "little",
        "registers": {"RCONV1": f"0x{rconv1:08X}", "RCONV2": f"0x{rconv2:08X}", "RRELU": f"0x{rrelu:08X}"},
        "output_buffer_exported": False,
        "output_bytes_overlapping_final_zero_padding": max(0, total_bytes - output_addr),
        "expected_output_sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "regions": [dict(region_map(region_name, addr, len(data)), included_in_initialization=True)
                    for region_name, addr, data in regions]
                   + [dict(region_map("output", output_addr, len(expected_bytes)), included_in_initialization=False)],
    }
    write_json(case_dir / "memory_map.json", summary)
    lines = [f"case: {name}", "All addresses are BYTE addresses; all.coe starts at byte address 0.",
             f"INSTRUCTION_ENTRY = 0x{instr_addr:08X}", f"INPUT_BASE = 0x{input_addr:08X}",
             f"OUTPUT_BASE = 0x{output_addr:08X}", f"OUTPUT_SIZE = {len(expected_bytes)} bytes",
             f"BRAM_WIDTH = 512 bits, BRAM_DEPTH >= {required_depth} words (including runtime output)",
             f"INITIALIZED_WORDS = {total_bytes // BRAM_WORD_BYTES}",
             f"INIT_REGION_ALIGNMENT = {region_alignment} bytes",
             f"ZERO_PADDING_BEFORE_INSTRUCTIONS = {instruction_padding} bytes after parameters",
             f"FINAL_ZERO_PADDING = {total_bytes - init_end} bytes after END", "",
             "region        start       end_exclusive  bytes   BRAM_index  byte_in_word"]
    for reg in summary["regions"]:
        lines.append(f"{reg['name']:<13} {reg['start_addr']}  {reg['end_addr_exclusive']}  "
                     f"{reg['size_bytes']:5d}   {reg['bram_word_index']:5d}       {reg['byte_offset_in_bram_word']:2d}")
    lines += ["", f"RCONV1=0x{rconv1:08X}  RCONV2=0x{rconv2:08X}  RRELU=0x{rrelu:08X}",
              f"INPUT_CHANNELS = {input_channels}, STORED_INPUT_CHANNELS = {stored_input_channels}",
              f"OUTPUT_CHANNELS = {output_channels}, STORED_OUTPUT_CHANNELS = {stored_output_channels}",
              "bias random values are int8, sign-extended to int32 storage.",
              "Each 512-bit COE value: lowest byte address is bits [7:0] (rightmost hex byte).",
              "Read instruction word from its four little-endian bytes; entry is NOT a BRAM word index.",
              "The output region is runtime-only; all.coe ends after END plus final word zero padding.",
              "Final zero padding can overlap the beginning of the output region; the full output is NOT exported.",
              "Expected output is in H,W,C order; human-readable matrices are in ../reference.txt."]
    (case_dir / "memory_map.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    portable_plan = copy.deepcopy(plan)
    portable_plan["config"].update(IMAGE_PATH="input.coe", model_ir="model_ir.json")
    portable_plan["image"]["file"] = portable_plan["tensors"]["image"]["file"] = "input.coe"
    names = {"image": "input.coe", "layer1_params": "params.coe", "instr": "instr.coe"}
    for reg in portable_plan["init_regions"]:
        reg["file"] = names[reg["name"]]
    write_json(case_dir / "memory_plan.json", portable_plan)
    validate_case(case_dir, image, regions, words, summary, expected_bytes)
    from export_output_ddr import export_case
    export_case(case_dir, summary)
    return summary


def validate_case(case_dir, image, regions, words, summary, expected_bytes):
    """Read exported files back, including the instruction words inside wide BRAM."""
    decoded, _, _ = merge.coe_values_to_bytes(case_dir / "all.coe", 64, "little")
    assert bytes(decoded) == bytes(image)
    mem = [int(line, 16) for line in (case_dir / "all.mem").read_text().splitlines()]
    assert b"".join(word.to_bytes(64, "little") for word in mem) == image
    for _, addr, data in regions:
        assert bytes(decoded[addr:addr + len(data)]) == data
    pc = int(summary["instruction_entry_addr"], 16)
    for key in ("input_addr", "parameter_addr", "instruction_entry_addr"):
        assert int(summary[key], 16) % summary["init_region_alignment_bytes"] == 0
    params_end = int(summary["parameter_addr"], 16) + summary["parameter_size_bytes"]
    assert decoded[params_end:pc] == [0] * summary["zero_padding_before_instructions_bytes"]
    for i, word in enumerate(words):
        assert int.from_bytes(decoded[pc + i * 4:pc + i * 4 + 4], "little") == word
    assert sum(word >> 26 == 3 for word in words) == 1
    assert not any(word >> 26 == 5 for word in words)
    assert words[-1] == assembler.END_WORD
    init_end = int(summary["logical_init_end_exclusive"], 16)
    assert len(decoded) == (init_end + 63) // 64 * 64
    assert len(decoded) < int(summary["output_addr"], 16) + len(expected_bytes)
    assert not any(decoded[init_end:])
    cfg = {((word >> 16) & 31): word & 0xFFFF
           for word in words if word >> 26 == 1 and (word >> 21) & 31 == 0}
    assert (cfg[3] << 16) | cfg[2] == int(summary["registers"]["RCONV1"], 16)
    assert (cfg[5] << 16) | cfg[4] == int(summary["registers"]["RCONV2"], 16)
    assert (cfg[9] << 16) | cfg[8] == int(summary["registers"]["RRELU"], 16)
    assert bytes(int(line, 16) for line in (case_dir / "expected_output.mem").read_text().splitlines()) == expected_bytes
    expected_coe, _, _ = merge.coe_values_to_bytes(case_dir / "expected_output.coe", 64, "little")
    assert bytes(expected_coe[:len(expected_bytes)]) == expected_bytes
    assert expected_coe[len(expected_bytes):] == [0] * summary["expected_output_coe_final_padding_bytes"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("16x8", "15x5"), default="16x8",
                        help="16x8: original two 16x16 cases; 15x5: one 13x13 case, all regions aligned to 8 bytes")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    padded_case = args.preset == "15x5"
    height, width, input_channels, output_channels = (13, 13, 15, 5) if padded_case else (16, 16, 16, 8)
    default_name = "conv_relu_15in5out_13x13" if padded_case else "conv_relu"
    out_dir = (args.out_dir or PROJECT_ROOT / "examples" / default_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = rng.integers(-128, 128, (height, width, input_channels), dtype=np.int16).astype(np.int8)
    weights = rng.integers(-128, 128, (output_channels, KERNEL, KERNEL, input_channels), dtype=np.int16).astype(np.int8)
    bias = rng.integers(-128, 128, output_channels, dtype=np.int16).astype(np.int8)
    stages = reference(x, weights, bias)
    cases = (("aligned64", 0x40, 64), ("unaligned64", 0x08, 8))
    summaries = [build_case(out_dir, name, base, x, weights, bias, stages, region_alignment=alignment)
                 for name, base, alignment in cases]
    assert summaries[0]["expected_output_sha256"] == summaries[1]["expected_output_sha256"]
    assert int(summaries[0]["input_addr"], 16) % 64 == 0
    assert int(summaries[1]["input_addr"], 16) % 64 == 8
    write_json(out_dir / "summary.json", summaries)
    write_matrix_reference(out_dir / "reference.txt", x, weights, bias, stages[2], summaries)
    write_layout_readme(out_dir, summaries)
    files = sorted(path for path in out_dir.rglob("*") if path.is_file())
    (out_dir / "SHA256SUMS.txt").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(out_dir).as_posix()}\n"
        for path in files if path.name != "SHA256SUMS.txt"
    ), encoding="ascii")
    archive = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(out_dir.parent))
    for summary in summaries:
        print(f"{summary['case']}: input={summary['input_addr']} instr={summary['instruction_entry_addr']} "
              f"output={summary['output_addr']} depth={summary['bram_depth_words']}x512")
    print(f"Validated {len(summaries)} case(s); reference SHA256={summaries[0]['expected_output_sha256']}")
    print(f"Archive: {archive}")


def write_matrix_reference(path: Path, x, weights, bias, expected, summaries: list[dict]) -> None:
    """Show logical per-channel matrices, independent of physical HWC byte order."""
    height, width, input_channels = x.shape
    output_channels = expected.shape[2]
    lines = [
        "Conv + LeakyReLU 仿真参考矩阵",
        f"随机种子：{SEED}；所有数值使用有符号十进制。",
        "本 TXT 中通道、行、列均从 1 开始编号；输入内存按通道组→行→列→组内8通道排列，输出布局不变。",
        f"输入：{input_channels} 个 {height}×{width} 矩阵；输出：{output_channels} 个 {height}×{width} 矩阵；权重：{output_channels}×{input_channels} 个 3×3 卷积核。",
        f"物理输入通道补到 {aligned_channels(input_channels)}，物理输出通道补到 {aligned_channels(output_channels)}；补齐通道全为 0，不额外重复列出零矩阵。",
        f"kernel=3×3，padding=1，stride=1，bias 启用，scale={summaries[0]['start_bit']}，slope=1/8。",
        f"累加范围：{summaries[0]['accumulator_range']}；移位后范围：{summaries[0]['scaled_range']}。",
        "计算顺序：乘加+bias → 按scale算术右移后取8位 → 负数算术右移3位。",
    ]
    for item in summaries:
        lines.append(f"{item['case']}：输入 {item['input_addr']}，指令 {item['instruction_entry_addr']}，输出 {item['output_addr']}。")

    def matrix(title: str, values) -> None:
        rows, columns = values.shape
        lines.extend(["", title, "列号 " + " ".join(f"{c + 1:5d}" for c in range(columns))])
        for r in range(rows):
            lines.append(f"行{r + 1:02d} " + " ".join(f"{int(value):5d}" for value in values[r]))

    lines.extend(["", "=" * 72, f"一、输入矩阵（按输入通道 ch01 → ch{input_channels:02d}）", "=" * 72])
    for channel in range(input_channels):
        matrix(f"输入矩阵 ch{channel + 1:02d}（{height}×{width}）", x[:, :, channel])
    lines.extend(["", "=" * 72, f"二、输出矩阵（最终激活结果，按输出通道 ch01 → ch{output_channels:02d}）", "=" * 72])
    for channel in range(output_channels):
        matrix(f"输出矩阵 ch{channel + 1:02d}（{height}×{width}）", expected[:, :, channel])
    lines.extend(["", "=" * 72, "三、权重（先按输出通道，再按输入通道；每个卷积核为 3×3）", "=" * 72])
    for oc in range(output_channels):
        lines.extend(["", f"输出通道 ch{oc + 1:02d}：bias = {int(bias[oc])}（signed int8 值，存储为 int32）"])
        for ic in range(input_channels):
            matrix(f"权重 out_ch{oc + 1:02d} ← in_ch{ic + 1:02d}（3×3）", weights[oc, :, :, ic])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_layout_readme(out_dir: Path, summaries: list[dict]) -> None:
    lines = ["# Conv + LeakyReLU simulation examples", "",
             "Both versions share identical input, weights and expected output. Addresses are byte addresses.",
             "aligned64 aligns input, parameters and instructions to 64 bytes; unaligned64 uses 8-byte alignment.", "",
             "Each valid output channel: all its weights -> int32 little-endian bias -> four ZERO bytes.",
             "Weights: kh -> kw -> input channel (input channels padded to a multiple of 8).",
             "No dummy output-channel parameter records. Output features are still padded to 8 channels.", "",
             "|Case|Input|Parameters|First bias|Instruction entry|Output|Parameter bytes|Minimum BRAM depth|",
             "|---|---|---|---|---|---|---|---|"]
    for m in summaries:
        lines.append(f"|{m['case']}|{m['input_addr']}|{m['parameter_addr']}|{m['first_bias_addr']}|{m['instruction_entry_addr']}|{m['output_addr']}|{m['parameter_size_bytes']}|{m['bram_depth_words']}|")
    lines += ["", "For 15 inputs / 5 outputs / 3x3: each record is 144 weight bytes + 8 bias-slot bytes = 152 bytes; total 760 bytes.",
              "bias.csv lists actual bias addresses. stride_bytes in memory_plan.json describes interleaved tensors.",
              "all.coe/all.mem/all.bin contain the same 512-bit BRAM image, loaded at address 0: input, parameters and instructions.",
              "expected_output.bin/.mem/.coe contain the reference output. reference.txt shows matrices; reference.csv shows scalar results.",
              "In each 512-bit COE word, the rightmost byte has the lowest address. Instructions are 32-bit little-endian.",
              "Input bytes: channel group -> row -> column -> eight channels. All pixels of channels 1..8 precede channels 9..16.",
              f"Conv scale={summaries[0]['start_bit']}; accumulator range={summaries[0]['accumulator_range']}; scaled range={summaries[0]['scaled_range']}.",
              "Compute acc=sum(input*weight)+bias, arithmetic-shift right by scale, take 8 bits as int8, then shift negative values right by 3.",
              "Reference validated with vector and independent scalar calculations. RTL simulation has not been run."]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
