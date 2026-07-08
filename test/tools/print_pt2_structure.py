#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print a human-readable summary of a torch.export PT2 model."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import torch  # noqa: E402
from frontend.pt2_frontend import (  # noqa: E402
    PASSTHROUGH_TARGETS,
    TARGET_TO_OP,
    fake_shape,
    normalize_pair,
    parameter_name_by_placeholder,
    target_name,
    user_input_name,
)


def load_exported_program(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"PT2 file does not exist: {path}")

    logging.getLogger("torch.export").setLevel(logging.ERROR)
    original_torch_load = torch.load

    def torch_load_cpu(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("map_location", "cpu")
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = torch_load_cpu
        return torch.export.load(str(path))
    finally:
        torch.load = original_torch_load


def node_name(value: Any) -> str | None:
    return getattr(value, "name", None)


def format_shape(shape: Any) -> str:
    if shape is None:
        return "?"
    return "[" + ", ".join(str(int(dim)) for dim in shape) + "]"


def format_value(value: Any) -> str:
    name = node_name(value)
    if name is not None:
        return name
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    return repr(value)


def get_arg(node: Any, index: int, key: str, default: Any = None) -> Any:
    if key in node.kwargs:
        return node.kwargs[key]
    if len(node.args) > index:
        return node.args[index]
    return default


def tensor_summary(tensor: Any) -> str:
    if tensor is None:
        return "none"
    shape = tuple(int(dim) for dim in getattr(tensor, "shape", ()))
    dtype = getattr(tensor, "dtype", "?")
    return f"shape={list(shape)}, dtype={dtype}"


def placeholder_kind_by_name(ep: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for spec in ep.graph_signature.input_specs:
        arg_name = getattr(getattr(spec, "arg", None), "name", None)
        if arg_name:
            result[str(arg_name)] = str(getattr(spec, "kind", ""))
    return result


def graph_outputs(graph: Any) -> list[str]:
    for node in graph.nodes:
        if node.op == "output":
            if not node.args:
                return []
            out = node.args[0]
            if isinstance(out, (list, tuple)):
                return [format_value(item) for item in out]
            return [format_value(out)]
    return []


def op_attrs(node: Any, op: str, state_dict: dict[str, Any], placeholder_to_param: dict[str, str]) -> list[str]:
    attrs: list[str] = []
    if op == "conv2d":
        weight_node = get_arg(node, 1, "weight")
        bias_node = get_arg(node, 2, "bias")
        weight_param = placeholder_to_param.get(format_value(weight_node))
        bias_param = placeholder_to_param.get(format_value(bias_node))
        weight = state_dict.get(weight_param) if weight_param else None
        bias = state_dict.get(bias_param) if bias_param else None
        attrs.extend(
            [
                f"weight={weight_param or format_value(weight_node)} ({tensor_summary(weight)})",
                f"bias={bias_param or 'none'} ({tensor_summary(bias)})",
                f"stride={normalize_pair(get_arg(node, 3, 'stride'), default=[1, 1])}",
                f"padding={normalize_pair(get_arg(node, 4, 'padding'), default=[0, 0])}",
                f"dilation={normalize_pair(get_arg(node, 5, 'dilation'), default=[1, 1])}",
                f"groups={int(get_arg(node, 6, 'groups', 1))}",
            ]
        )
    elif op == "relu":
        if target_name(node.target) == "aten.leaky_relu.default":
            attrs.append(f"negative_slope={get_arg(node, 1, 'negative_slope', 0.01)}")
    elif op == "avgpool2d":
        kernel_size = normalize_pair(get_arg(node, 1, "kernel_size"))
        stride = normalize_pair(get_arg(node, 2, "stride"), default=[0, 0])
        attrs.extend(
            [
                f"kernel_size={kernel_size}",
                f"stride={kernel_size if stride == [0, 0] else stride}",
                f"padding={normalize_pair(get_arg(node, 3, 'padding'), default=[0, 0])}",
            ]
        )
    elif op == "flatten":
        attrs.extend(
            [
                f"start_dim={int(get_arg(node, 1, 'start_dim', 0))}",
                f"end_dim={int(get_arg(node, 2, 'end_dim', -1))}",
            ]
        )
    elif op == "linear":
        weight_node = get_arg(node, 1, "weight")
        bias_node = get_arg(node, 2, "bias")
        weight_param = placeholder_to_param.get(format_value(weight_node))
        bias_param = placeholder_to_param.get(format_value(bias_node))
        weight = state_dict.get(weight_param) if weight_param else None
        bias = state_dict.get(bias_param) if bias_param else None
        attrs.extend(
            [
                f"weight={weight_param or format_value(weight_node)} ({tensor_summary(weight)})",
                f"bias={bias_param or 'none'} ({tensor_summary(bias)})",
            ]
        )
    return attrs


def print_structure(model_path: Path, *, show_all_nodes: bool) -> None:
    ep = load_exported_program(model_path)
    graph = ep.graph_module.graph
    state_dict = dict(ep.state_dict)
    placeholder_to_param = parameter_name_by_placeholder(ep)
    placeholder_kinds = placeholder_kind_by_name(ep)
    user_input = user_input_name(ep)

    print(f"Model: {model_path}")
    print(f"Frontend: torch.export PT2")
    print()
    print("Inputs:")
    for node in graph.nodes:
        if node.op != "placeholder":
            continue
        kind = placeholder_kinds.get(node.name, "UNKNOWN")
        shape = format_shape(fake_shape(node))
        mapped_param = placeholder_to_param.get(node.name)
        suffix = f" -> {mapped_param}" if mapped_param else ""
        marker = "user input" if node.name == user_input else kind.split(".")[-1].lower()
        print(f"  - {node.name}: {marker}, shape={shape}{suffix}")

    outputs = graph_outputs(graph)
    if outputs:
        print()
        print("Outputs:")
        for output in outputs:
            print(f"  - {output}")

    print()
    print("Inference Flow:")
    effective_index = 0
    stats: Counter[str] = Counter()
    passthrough_count = 0
    unsupported_count = 0

    for node in graph.nodes:
        if node.op != "call_function":
            if show_all_nodes and node.op not in ("placeholder", "output"):
                print(f"  - {node.op}: {node.name}")
            continue

        source_target = target_name(node.target)
        op = TARGET_TO_OP.get(source_target)
        out_shape = format_shape(fake_shape(node))
        input_text = ", ".join(format_value(arg) for arg in node.args)

        if op is None:
            if source_target in PASSTHROUGH_TARGETS:
                passthrough_count += 1
                if show_all_nodes:
                    print(f"  helper {node.name}: {source_target}")
                    print(f"    args: {input_text}")
                    print(f"    output: {node.name}, shape={out_shape}")
                continue
            unsupported_count += 1
            print(f"  [unsupported] {node.name}: {source_target}")
            print(f"    args: {input_text}")
            print(f"    output: {node.name}, shape={out_shape}")
            continue

        stats[op] += 1
        logical_name = f"{op}_{stats[op] - 1}"
        print(f"  [{effective_index:02d}] {logical_name}: {op} ({source_target})")
        print(f"    input : {input_text}")
        print(f"    output: {node.name}, shape={out_shape}")
        attrs = op_attrs(node, op, state_dict, placeholder_to_param)
        if attrs:
            print("    attrs : " + "; ".join(attrs))
        effective_index += 1

    print()
    print("Operator Summary:")
    if stats:
        for name in sorted(stats):
            print(f"  - {name}: {stats[name]}")
    else:
        print("  - no supported inference operators found")
    print(f"  - passthrough helpers: {passthrough_count}")
    print(f"  - unsupported call_function nodes: {unsupported_count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a human-readable PT2 model inference structure.",
        usage="python ./test/tools/print_pt2_structure.py MODEL.pt2",
    )
    parser.add_argument("model", type=Path, help="path to a torch.export .pt2 model")
    parser.add_argument(
        "--all-nodes",
        action="store_true",
        help="also show passthrough helper nodes such as clamp/to/floor",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        print_structure(args.model, show_all_nodes=args.all_nodes)
    except Exception as exc:
        print(f"print_pt2_structure.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
