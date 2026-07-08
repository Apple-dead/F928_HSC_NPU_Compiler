# generate_model_ir.py 说明

## 1. 总体功能

`python/generate_model_ir.py` 是当前编译器的模型前端入口。

它读取 `python/npu_config.py` 中的：

```python
MODEL_FORMAT = "pt2"
MODEL_PATH = "./model.pt2"
INPUT_HEIGHT = 28
INPUT_WIDTH = 28
INFER_PARSE_MODE = 1
INFER_PARSE_OP_LIMIT = 0
```

`INPUT_HEIGHT` / `INPUT_WIDTH` 只是 fallback 输入尺寸。PT2 前端会优先读取 PT2 graph 中导出的输入 shape metadata；只有 PT2 缺少该 metadata 时，才会使用这两个配置值。

当前只支持 `MODEL_FORMAT = "pt2"`。脚本会调用 `torch.export.load(MODEL_PATH)` 读取 `ExportedProgram`，再生成：

```text
data/model_ir.json
data/model_params/
```

## 2. 标准 IR

`data/model_ir.json` 记录标准推理过程。当前 PT2 前端支持以下 aten target：

```text
aten.conv2d.default        -> conv2d
aten.relu.default          -> relu
aten.leaky_relu.default    -> relu
aten.avg_pool2d.default    -> avgpool2d
aten.flatten.using_ints    -> flatten
aten.linear.default        -> linear
```

以下量化辅助节点会透传，不写入 IR：

```text
aten.clamp.default
aten.to.dtype
aten.floor.default
```

`aten.leaky_relu.default` 会写成 IR 的 `relu`，并记录 `negative_slope` 和 `tan`。

当前根目录 `model.pt2` 会导出为：

```text
conv2d -> relu -> avgpool2d -> flatten -> linear
```

即使后端暂不支持 `avgpool2d`、`flatten`、`linear`，前端也必须把它们完整写入 IR，后续 `generate_memory_plan.py` 再负责明确报错。

## 3. 解析范围控制

解析范围由 `python/npu_config.py` 控制：

```python
INFER_PARSE_MODE = 1
INFER_PARSE_OP_LIMIT = 0
```

模式含义：

```text
INFER_PARSE_MODE = 1  全量导出 PT2 图中的所有受支持 op
INFER_PARSE_MODE = 2  只导出前 INFER_PARSE_OP_LIMIT 个可执行 op
```

例如当前模型顺序为：

```text
conv2d -> relu -> avgpool2d -> flatten -> linear
```

当 `INFER_PARSE_MODE = 2` 且 `INFER_PARSE_OP_LIMIT = 1` 时，导出的 IR 只包含：

```text
conv2d
```

并把 `conv2d_0_out` 作为当前 IR 输出。

当 `INFER_PARSE_OP_LIMIT = 2` 时，导出的 IR 为：

```text
conv2d -> relu
```

并把 `relu_0_out` 作为当前 IR 输出。

对于当前 YOLO PT2 这类包含 `leaky_relu/floor` 量化辅助节点的图，`floor` 等辅助节点不计入 limit，但 `conv2d` 和 `leaky_relu` 都会计入。因此 `INFER_PARSE_OP_LIMIT = 6` 会导出：

```text
conv2d -> relu -> conv2d -> relu -> conv2d -> relu
```

## 4. 参数导出

参数继续导出到：

```text
data/model_params/
```

Conv 参数命名：

```text
data/model_params/layer1_0_weight.txt
data/model_params/layer1_0_bias.txt    # 仅有 bias 时生成
```

Linear 参数命名：

```text
data/model_params/linear1_weight.txt
data/model_params/linear1_bias.txt     # 仅有 bias 时生成
```

参数文件保持逻辑布局：

```text
Conv weight   : OIHW = out_channel, in_channel, kernel_h, kernel_w
Linear weight : OI = out_features, in_features
```

前端只导出一维整数文本，不做 NPU BRAM 物理重排。物理排列仍由 `params_to_bram_coe.py` 处理。

## 5. 数值校验

weight 必须是 signed int8 整数：

```text
[-128, 127]
```

bias 若存在，必须是 signed int32 整数：

```text
[-2147483648, 2147483647]
```

如果 PT2 中 tensor 是 quantized tensor，前端使用 `int_repr()`；如果是普通 tensor，则要求所有值在数值上已经是整数。非整数或越界值会直接报错。

## 6. 无 bias 行为

当 `Conv2d` 或 `Linear` 无 bias 时：

```text
不生成 bias txt 文件
IR 中 has_bias = false
IR 中 bias_file = null
```

这对后端很重要：无 bias conv 的参数区只包含 weight，不能补写 bias 0。
