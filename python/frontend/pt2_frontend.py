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
    "aten.upsample_nearest2d.vec": "upsample_nearest2d",
    "aten.pad.default": "pad",
    "aten.cat.default": "cat",
    "aten.linear.default": "linear",
}

REPEAT_INTERLEAVE_TARGET = "aten.repeat_interleave.self_int"

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


def output_node_values(graph: Any) -> list[Any]:
    for node in graph.nodes:
        if node.op != "output":
            continue
        if not node.args:
            return []
        output = node.args[0]
        if isinstance(output, (list, tuple)):
            return list(output)
        return [output]
    return []


def output_node_names(graph: Any) -> set[str]:
    return {name for name in (node_name(value) for value in output_node_values(graph)) if name is not None}


def non_output_users(node: Any) -> list[Any]:
    return [user for user in getattr(node, "users", {}) if getattr(user, "op", None) != "output"]


def repeat_interleave_repeats_dim(node: Any) -> tuple[int, int]:
    repeats = node.args[1] if len(node.args) > 1 else node.kwargs.get("repeats")
    dim = node.kwargs.get("dim", node.args[2] if len(node.args) > 2 else None)
    if not isinstance(repeats, int):
        raise ValueError(f"{node.name}: repeat_interleave repeats must be integer 2 for USMP, got {repeats!r}")
    if not isinstance(dim, int):
        raise ValueError(f"{node.name}: repeat_interleave dim must be integer H/W dim for USMP, got {dim!r}")
    return int(repeats), int(dim)


def normalize_dim(dim: int, rank: int, *, node_label: str) -> int:
    result = dim + rank if dim < 0 else dim
    if not 0 <= result < rank:
        raise ValueError(f"{node_label}: repeat_interleave dim {dim} is out of range for rank {rank}")
    return result


def match_repeat_interleave_hw_x2(node: Any) -> dict[str, Any]:
    users = non_output_users(node)
    if len(users) != 1:
        raise NotImplementedError(
            f"{node.name}: repeat_interleave can map to USMP only when two H/W repeat=2 nodes form a chain"
        )
    next_node = users[0]
    if getattr(next_node, "op", None) != "call_function" or target_name(next_node.target) != REPEAT_INTERLEAVE_TARGET:
        raise NotImplementedError(
            f"{node.name}: repeat_interleave can map to USMP only when followed by another repeat_interleave"
        )
    if len(next_node.args) < 1 or next_node.args[0] is not node:
        raise NotImplementedError(
            f"{node.name}: repeat_interleave USMP pattern requires the second node to consume the first node"
        )

    input_node = node.args[0] if node.args else node.kwargs.get("input")
    input_shape = fake_shape(input_node)
    mid_shape = fake_shape(node)
    output_shape = fake_shape(next_node)
    if input_shape is None or mid_shape is None or output_shape is None:
        raise ValueError(f"{node.name}: repeat_interleave USMP pattern requires static tensor shapes")
    if len(input_shape) != 4 or len(mid_shape) != 4 or len(output_shape) != 4:
        raise ValueError(f"{node.name}: repeat_interleave USMP pattern only supports 4D NCHW tensors")

    repeats0, dim0 = repeat_interleave_repeats_dim(node)
    repeats1, dim1 = repeat_interleave_repeats_dim(next_node)
    dim0 = normalize_dim(dim0, 4, node_label=node.name)
    dim1 = normalize_dim(dim1, 4, node_label=next_node.name)
    if repeats0 != 2 or repeats1 != 2 or {dim0, dim1} != {2, 3}:
        raise NotImplementedError(
            f"{node.name}: repeat_interleave USMP pattern only supports repeat=2 on dims 2 and 3, "
            f"got repeats={[repeats0, repeats1]}, dims={[dim0, dim1]}"
        )

    n, channels, height, width = input_shape
    expected_output = [n, channels, height * 2, width * 2]
    if output_shape != expected_output:
        raise ValueError(
            f"{node.name}: repeat_interleave USMP pattern output shape must be {expected_output}, got {output_shape}"
        )

    return {
        "input_node": input_node,
        "second_node": next_node,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "dims": [dim0, dim1],
        "repeats": [repeats0, repeats1],
    }


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
    counters = {
        "conv2d": 0,
        "relu": 0,
        "avgpool2d": 0,
        "maxpool2d": 0,
        "upsample_nearest2d": 0,
        "pad": 0,
        "cat": 0,
        "linear": 0,
    }
    value_names = {user_input: user_input}
    op_limit = infer_op_limit(config)
    graph_output_names = output_node_names(graph)
    expand_final_cat_outputs = bool(getattr(config, "EXPAND_FINAL_CAT_OUTPUTS", False))
    final_cat_expansions: dict[str, list[dict[str, Any]]] = {}
    skip_nodes: set[str] = set()
    is_truncated = False

    for node in graph.nodes:
        if node.name in skip_nodes or node.op != "call_function":
            continue
        source_target = target_name(node.target)
        if source_target in PASSTHROUGH_TARGETS:
            input_node_name = node_name(node.args[0]) if node.args else None
            value_names[node.name] = value_names.get(input_node_name, input_node_name)
            continue
        if op_limit is not None and len(ops) >= op_limit:
            op_for_limit = TARGET_TO_OP.get(source_target)
            is_final_output_cat = (
                op_for_limit == "cat"
                and expand_final_cat_outputs
                and node.name in graph_output_names
                and not non_output_users(node)
            )
            if not is_final_output_cat:
                is_truncated = True
                break
        if source_target == REPEAT_INTERLEAVE_TARGET:
            pattern = match_repeat_interleave_hw_x2(node)
            op = "upsample_nearest2d"
            counters[op] += 1
            op_index = counters[op] - 1
            logical_name = f"{op}_{op_index}"
            logical_output = f"{logical_name}_out"
            input_node_name = node_name(pattern["input_node"])
            item = {
                "id": len(ops),
                "op": op,
                "name": logical_name,
                "source_target": f"{REPEAT_INTERLEAVE_TARGET}+{REPEAT_INTERLEAVE_TARGET}",
                "input": value_names.get(input_node_name, input_node_name),
                "output": logical_output,
                "output_shape": pattern["output_shape"],
                "scale_factor": [2, 2],
                "mode": "nearest",
                "source_pattern": {
                    "op": "repeat_interleave_hw_x2",
                    "nodes": [node.name, pattern["second_node"].name],
                    "dims": pattern["dims"],
                    "repeats": pattern["repeats"],
                },
            }
            ops.append(item)
            value_names[pattern["second_node"].name] = logical_output
            skip_nodes.add(pattern["second_node"].name)
            continue

        op = TARGET_TO_OP.get(source_target)
        if op is None:
            raise NotImplementedError(f"Unsupported PT2 aten target {source_target!r} at node {node.name}")

        if (
            op == "cat"
            and expand_final_cat_outputs
            and node.name in graph_output_names
            and not non_output_users(node)
        ):
            inputs = node.args[0] if node.args else node.kwargs.get("tensors")
            if not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
                raise ValueError(f"{node.name}: final cat requires at least two input tensors")
            dim = int(node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0))
            final_cat_expansions[node.name] = [
                {
                    "source_name": node_name(input_node),
                    "ir_name": value_names.get(node_name(input_node), node_name(input_node)),
                    "cat_input_index": input_index,
                    "cat_dim": dim,
                }
                for input_index, input_node in enumerate(inputs)
            ]
            continue

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

        elif op == "upsample_nearest2d":
            output_size = node.args[1] if len(node.args) > 1 else node.kwargs.get("output_size")
            scale_factor = node.args[2] if len(node.args) > 2 else node.kwargs.get("scale_factors")
            if output_size is not None:
                item["output_size"] = normalize_pair(output_size)
            if scale_factor is None:
                raise ValueError(
                    f"{logical_name}: USMP only supports nn.Upsample(scale_factor=2, mode='nearest'); "
                    "output_size-only upsample is not supported"
                )
            if isinstance(scale_factor, (int, float)):
                scale_values = [float(scale_factor), float(scale_factor)]
            elif isinstance(scale_factor, (list, tuple)) and len(scale_factor) == 2:
                scale_values = [float(scale_factor[0]), float(scale_factor[1])]
            else:
                raise ValueError(
                    f"{logical_name}: USMP only supports nn.Upsample(scale_factor=2, mode='nearest'), "
                    f"got scale_factor={scale_factor!r}"
                )
            if scale_values != [2.0, 2.0]:
                raise ValueError(
                    f"{logical_name}: USMP only supports nn.Upsample(scale_factor=2, mode='nearest'), "
                    f"got scale_factor={scale_factor!r}"
                )
            item.update(
                {
                    "scale_factor": [2, 2],
                    "mode": "nearest",
                }
            )

        elif op == "pad":
            pad = node.args[1] if len(node.args) > 1 else node.kwargs.get("pad")
            mode = node.kwargs.get("mode", node.args[2] if len(node.args) > 2 else "constant")
            value = node.kwargs.get("value", node.args[3] if len(node.args) > 3 else 0.0)
            if not isinstance(pad, (list, tuple)):
                raise ValueError(f"{logical_name}: pad must be a list/tuple, got {pad!r}")
            item.update(
                {
                    "pad": [int(value) for value in pad],
                    "mode": str(mode),
                    "value": float(value),
                }
            )

        elif op == "cat":
            inputs = node.args[0] if node.args else node.kwargs.get("tensors")
            if not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
                raise ValueError(f"{logical_name}: cat requires at least two input tensors")
            dim = int(node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0))
            item.update(
                {
                    "input": [value_names.get(node_name(input_node), node_name(input_node)) for input_node in inputs],
                    "dim": dim,
                }
            )

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

    output_items: list[dict[str, Any]] = []
    if not is_truncated:
        producer_by_output = {op["output"]: op for op in ops}
        for index, output_value in enumerate(output_node_values(graph)):
            source_name = node_name(output_value)
            if source_name in final_cat_expansions:
                for expansion in final_cat_expansions[source_name]:
                    ir_name = expansion["ir_name"]
                    if ir_name is None:
                        raise ValueError(f"could not resolve final cat input for graph output {index}: {source_name}")
                    producer = producer_by_output.get(ir_name)
                    output_items.append(
                        {
                            "index": len(output_items),
                            "name": ir_name,
                            "source": expansion["source_name"],
                            "producer_op_id": producer.get("id") if producer else None,
                            "producer_op": producer.get("op") if producer else None,
                            "source_kind": "final_cat_input",
                            "from_cat": source_name,
                            "cat_output_index": index,
                            "cat_input_index": expansion["cat_input_index"],
                            "cat_dim": expansion["cat_dim"],
                        }
                    )
                continue
            ir_name = value_names.get(source_name, source_name)
            if ir_name is None:
                raise ValueError(f"could not resolve graph output {index}: {output_value!r}")
            producer = producer_by_output.get(ir_name)
            output_items.append(
                {
                    "index": index,
                    "name": ir_name,
                    "source": source_name,
                    "producer_op_id": producer.get("id") if producer else None,
                    "producer_op": producer.get("op") if producer else None,
                }
            )
    if not output_items:
        output_items = [
            {
                "index": 0,
                "name": ops[-1]["output"],
                "source": None,
                "producer_op_id": ops[-1]["id"],
                "producer_op": ops[-1]["op"],
            }
        ]

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
            "name": output_items[-1]["name"],
            "producer_op_id": output_items[-1]["producer_op_id"],
            "producer_op": output_items[-1]["producer_op"],
        },
        "outputs": output_items,
        "parse": {
            "mode": int(getattr(config, "INFER_PARSE_MODE", 1)),
            "op_limit": op_limit,
            "is_truncated": is_truncated,
        },
        "ops": ops,
    }

    ir_out.parent.mkdir(parents=True, exist_ok=True)
    ir_out.write_text(json.dumps(ir, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ir
