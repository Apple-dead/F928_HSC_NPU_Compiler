#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_TO_OP = {
    "aten.conv2d.default": "conv2d",
    "aten.relu.default": "relu",
    "aten.leaky_relu.default": "relu",
    "aten.avg_pool2d.default": "avgpool2d",
    "aten.max_pool2d.default": "maxpool2d",
    "aten.linear.default": "linear",
}

PASSTHROUGH_TARGETS = {
    "aten.clamp.default",
    "aten.to.dtype",
    "aten.floor.default",
    "aten.flatten.using_ints",
}


def normalize_pair(value: Any, *, default: Iterable[int] | None = None) -> list[int]:
    if value is None:
        if default is None:
            raise ValueError("missing pair value")
        value = list(default)
    if isinstance(value, int):
        return [value, value]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [int(value[0]), int(value[1])]
    raise ValueError(f"expected int or int pair, got {value!r}")


def node_name(value: Any) -> str | None:
    return getattr(value, "name", None)


def fake_shape(node: Any) -> list[int] | None:
    val = getattr(node, "meta", {}).get("val")
    shape = getattr(val, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def parameter_name_by_placeholder(ep: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for spec in ep.graph_signature.input_specs:
        if str(getattr(spec, "kind", "")).endswith("PARAMETER"):
            arg_name = getattr(getattr(spec, "arg", None), "name", None)
            target = getattr(spec, "target", None)
            if arg_name and target:
                mapping[str(arg_name)] = str(target)
    return mapping


def user_input_name(ep: Any) -> str:
    for spec in ep.graph_signature.input_specs:
        if str(getattr(spec, "kind", "")).endswith("USER_INPUT"):
            name = getattr(getattr(spec, "arg", None), "name", None)
            if name:
                return str(name)
    return "x"


def tensor_to_integer_tensor(tensor: torch.Tensor, *, kind: str, min_value: int, max_value: int) -> torch.Tensor:
    if tensor.is_quantized:
        values = tensor.int_repr().detach().cpu().to(torch.int64)
    else:
        values_float = tensor.detach().cpu().to(torch.float64)
        rounded = torch.round(values_float)
        if not torch.allclose(values_float, rounded, atol=1e-6, rtol=0):
            bad = torch.nonzero(~torch.isclose(values_float, rounded, atol=1e-6, rtol=0), as_tuple=False)
            first = bad[:10].tolist()
            raise ValueError(f"{kind} contains non-integer values at indexes {first}")
        values = rounded.to(torch.int64)
    if values.numel() and (int(values.min()) < min_value or int(values.max()) > max_value):
        raise ValueError(
            f"{kind} exceeds range [{min_value}, {max_value}]: "
            f"min={int(values.min())}, max={int(values.max())}"
        )
    return values


def write_integer_tensor(path: Path, tensor: torch.Tensor, *, kind: str, min_value: int, max_value: int) -> None:
    values = tensor_to_integer_tensor(tensor, kind=kind, min_value=min_value, max_value=max_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = values.reshape(-1).tolist()
    path.write_text("".join(f"{int(value)}\n" for value in flat), encoding="utf-8")


def ir_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clear_param_outputs(params_dir: Path) -> None:
    params_dir.mkdir(parents=True, exist_ok=True)
    for path in params_dir.glob("*.txt"):
        path.unlink()


def get_param_tensor(state_dict: dict[str, torch.Tensor], placeholder_to_param: dict[str, str], placeholder: Any) -> torch.Tensor:
    placeholder_name = node_name(placeholder)
    if placeholder_name is None or placeholder_name not in placeholder_to_param:
        raise ValueError(f"could not resolve parameter placeholder: {placeholder!r}")
    param_name = placeholder_to_param[placeholder_name]
    if param_name not in state_dict:
        raise KeyError(f"parameter {param_name!r} from graph signature is not in state_dict")
    return state_dict[param_name]


def maybe_get_param_tensor(
    state_dict: dict[str, torch.Tensor],
    placeholder_to_param: dict[str, str],
    placeholder: Any,
) -> torch.Tensor | None:
    if placeholder is None:
        return None
    name = node_name(placeholder)
    if name is None:
        return None
    return get_param_tensor(state_dict, placeholder_to_param, placeholder)


def target_name(target: Any) -> str:
    return str(target).replace("torch.ops.", "")


def negative_slope_to_tan(negative_slope: float) -> int:
    if negative_slope == 0:
        return 8
    if negative_slope < 0:
        raise ValueError(f"negative_slope must be non-negative, got {negative_slope}")
    exponent = round(torch.log2(torch.tensor(1.0 / negative_slope)).item())
    expected = 1.0 / (2 ** exponent)
    if abs(negative_slope - expected) > 1e-12:
        raise ValueError(f"unsupported LeakyReLU negative_slope={negative_slope}")
    if not 0 <= exponent <= 7:
        raise ValueError(f"unsupported LeakyReLU negative_slope={negative_slope}")
    return int(exponent)


def infer_op_limit(config: Any) -> int | None:
    mode = int(getattr(config, "INFER_PARSE_MODE", 1))
    if mode == 1:
        return None
    if mode == 2:
        limit = int(getattr(config, "INFER_PARSE_OP_LIMIT", 0))
        if limit <= 0:
            raise ValueError("INFER_PARSE_OP_LIMIT must be positive when INFER_PARSE_MODE=2")
        return limit
    raise ValueError("INFER_PARSE_MODE must be 1 (full graph) or 2 (first N ops)")


def export_model_ir(*, model_path: Path, ir_out: Path, params_dir: Path, config: Any) -> Dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")

    logging.getLogger("torch.export").setLevel(logging.ERROR)
    original_torch_load = torch.load

    def torch_load_cpu(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("map_location", "cpu")
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = torch_load_cpu
        ep = torch.export.load(ir_path(model_path))
    finally:
        torch.load = original_torch_load
    graph = ep.graph_module.graph
    placeholder_to_param = parameter_name_by_placeholder(ep)
    state_dict = dict(ep.state_dict)
    clear_param_outputs(params_dir)

    user_input = user_input_name(ep)
    input_shape = None
    for node in graph.nodes:
        if node.op == "placeholder" and node.name == user_input:
            input_shape = fake_shape(node)
            break
    if input_shape is None or len(input_shape) != 4:
        input_shape = [1, 0, int(config.INPUT_HEIGHT), int(config.INPUT_WIDTH)]

    ops: list[dict[str, Any]] = []
    counters = {"conv2d": 0, "relu": 0, "avgpool2d": 0, "maxpool2d": 0, "linear": 0}
    value_names = {user_input: user_input}
    op_limit = infer_op_limit(config)

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        source_target = target_name(node.target)
        if source_target in PASSTHROUGH_TARGETS:
            input_node_name = node_name(node.args[0]) if node.args else None
            value_names[node.name] = value_names.get(input_node_name, input_node_name)
            continue
        if op_limit is not None and len(ops) >= op_limit:
            break
        op = TARGET_TO_OP.get(source_target)
        if op is None:
            raise NotImplementedError(f"Unsupported PT2 aten target {source_target!r} at node {node.name}")

        counters[op] += 1
        op_index = counters[op] - 1
        logical_name = f"{op}_{op_index}"
        logical_output = f"{logical_name}_out"
        input_node_name = node_name(node.args[0]) if node.args else None
        item: dict[str, Any] = {
            "id": len(ops),
            "op": op,
            "name": logical_name,
            "source_target": source_target,
            "input": value_names.get(input_node_name, input_node_name),
            "output": logical_output,
        }
        output_shape = fake_shape(node)
        if output_shape is not None:
            item["output_shape"] = output_shape

        if op == "conv2d":
            layer_index = counters[op]
            weight_node = node.args[1]
            bias_node = node.kwargs.get("bias", node.args[2] if len(node.args) > 2 else None)
            weight = get_param_tensor(state_dict, placeholder_to_param, weight_node)
            bias = maybe_get_param_tensor(state_dict, placeholder_to_param, bias_node)
            if weight.ndim != 4:
                raise ValueError(f"conv2d weight must be OIHW, got shape={tuple(weight.shape)}")
            prefix = f"layer{layer_index}_0"
            weight_file = params_dir / f"{prefix}_weight.txt"
            bias_file = params_dir / f"{prefix}_bias.txt"
            write_integer_tensor(weight_file, weight, kind=f"{prefix}_weight", min_value=-128, max_value=127)
            if bias is not None:
                write_integer_tensor(bias_file, bias, kind=f"{prefix}_bias", min_value=-(1 << 31), max_value=(1 << 31) - 1)
            out_ch, in_ch, kh, kw = [int(value) for value in weight.shape]
            item.update(
                {
                    "layer_index": layer_index,
                    "param_prefix": prefix,
                    "weight_file": ir_path(weight_file),
                    "bias_file": ir_path(bias_file) if bias is not None else None,
                    "has_bias": bias is not None,
                    "in_channels": in_ch,
                    "out_channels": out_ch,
                    "kernel_size": [kh, kw],
                    "stride": normalize_pair(node.kwargs.get("stride", node.args[3] if len(node.args) > 3 else None), default=[1, 1]),
                    "padding": normalize_pair(node.kwargs.get("padding", node.args[4] if len(node.args) > 4 else None), default=[0, 0]),
                    "dilation": normalize_pair(node.kwargs.get("dilation", node.args[5] if len(node.args) > 5 else None), default=[1, 1]),
                    "groups": int(node.kwargs.get("groups", node.args[6] if len(node.args) > 6 else 1)),
                }
            )

        elif op == "avgpool2d":
            item.update(
                {
                    "kernel_size": normalize_pair(node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size")),
                    "stride": normalize_pair(node.kwargs.get("stride", node.args[2] if len(node.args) > 2 else None), default=[0, 0]),
                    "padding": normalize_pair(node.kwargs.get("padding", node.args[3] if len(node.args) > 3 else None), default=[0, 0]),
                }
            )
            if item["stride"] == [0, 0]:
                item["stride"] = item["kernel_size"]

        elif op == "maxpool2d":
            item.update(
                {
                    "kernel_size": normalize_pair(node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size")),
                    "stride": normalize_pair(node.kwargs.get("stride", node.args[2] if len(node.args) > 2 else None), default=[0, 0]),
                    "padding": normalize_pair(node.kwargs.get("padding", node.args[3] if len(node.args) > 3 else None), default=[0, 0]),
                    "dilation": normalize_pair(node.kwargs.get("dilation", node.args[4] if len(node.args) > 4 else None), default=[1, 1]),
                    "ceil_mode": bool(node.kwargs.get("ceil_mode", node.args[5] if len(node.args) > 5 else False)),
                }
            )
            if item["stride"] == [0, 0]:
                item["stride"] = item["kernel_size"]

        elif op == "linear":
            linear_index = counters[op]
            weight_node = node.args[1]
            bias_node = node.args[2] if len(node.args) > 2 else None
            weight = get_param_tensor(state_dict, placeholder_to_param, weight_node)
            bias = maybe_get_param_tensor(state_dict, placeholder_to_param, bias_node)
            if weight.ndim != 2:
                raise ValueError(f"linear weight must be OI, got shape={tuple(weight.shape)}")
            prefix = f"linear{linear_index}"
            weight_file = params_dir / f"{prefix}_weight.txt"
            bias_file = params_dir / f"{prefix}_bias.txt"
            write_integer_tensor(weight_file, weight, kind=f"{prefix}_weight", min_value=-128, max_value=127)
            if bias is not None:
                write_integer_tensor(bias_file, bias, kind=f"{prefix}_bias", min_value=-(1 << 31), max_value=(1 << 31) - 1)
            out_features, in_features = [int(value) for value in weight.shape]
            item.update(
                {
                    "param_prefix": prefix,
                    "weight_file": ir_path(weight_file),
                    "bias_file": ir_path(bias_file) if bias is not None else None,
                    "has_bias": bias is not None,
                    "in_features": in_features,
                    "out_features": out_features,
                }
            )

        if source_target == "aten.relu.default":
            item["tan"] = 8
        elif source_target == "aten.leaky_relu.default":
            item["negative_slope"] = float(node.kwargs.get("negative_slope", node.args[1] if len(node.args) > 1 else 0.01))
            item["tan"] = negative_slope_to_tan(item["negative_slope"])

        ops.append(item)
        value_names[node.name] = logical_output

    if not ops:
        raise ValueError("PT2 frontend did not export any executable op")

    input_channels = int(input_shape[1]) if input_shape[1] else next(
        (int(op["in_channels"]) for op in ops if op["op"] == "conv2d"),
        0,
    )
    if input_channels <= 0:
        raise ValueError("could not derive input channel count from PT2 graph")

    ir = {
        "ir_version": 1,
        "frontend": "pt2",
        "model_path": ir_path(model_path),
        "input": {
            "name": user_input,
            "height": int(input_shape[2]),
            "width": int(input_shape[3]),
            "channels": input_channels,
        },
        "output": {
            "name": ops[-1]["output"],
            "producer_op_id": ops[-1]["id"],
            "producer_op": ops[-1]["op"],
        },
        "parse": {
            "mode": int(getattr(config, "INFER_PARSE_MODE", 1)),
            "op_limit": op_limit,
            "is_truncated": op_limit is not None,
        },
        "ops": ops,
    }

    ir_out.parent.mkdir(parents=True, exist_ok=True)
    ir_out.write_text(json.dumps(ir, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ir
