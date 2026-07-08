# generate_memory_plan.py 说明

## 1. 总体功能

`python/generate_memory_plan.py` 从标准 IR 生成：

```text
data/memory_plan.json
```

默认输入：

```text
data/model_ir.json
python/npu_config.py
IMAGE_PATH 指向的输入 COE
```

显式命令：

```bash
python ./python/generate_memory_plan.py --model-ir ./data/model_ir.json --out ./data/memory_plan.json
```

该脚本负责把 IR 中后端可支持的 `conv2d/relu` 路径转换为 NPU 地址规划、参数区规划和分组执行计划。

## 2. 后端支持范围

当前 memory plan 阶段只支持：

```text
conv2d
relu
```

如果 IR 中出现暂不支持的 op，会直接报错，不会静默跳过。例如：

```text
Unsupported op avgpool2d at op id 2. Current backend only supports conv2d/relu path.
```

## 3. 主要配置

`npu_config.py` 中与本阶段相关的配置包括：

```text
INIT_BASE_ADDR
INIT_LIMIT_ADDR
RUNTIME_BASE_ADDR
IMAGE_BASE_ADDR
IMAGE_SOURCE
IMAGE_PATH
MODEL_FORMAT
MODEL_PATH
INTR_MOVE_PATH
INPUT_HEIGHT
INPUT_WIDTH
INFER_PARSE_MODE
INFER_PARSE_OP_LIMIT
CHANNEL_GROUP4_FEATURE_SIZE_THRESHOLD
```

`INPUT_HEIGHT` / `INPUT_WIDTH` 只作为 PT2 输入 metadata 缺失时的 fallback。正常情况下，输入尺寸和通道数来自 `data/model_ir.json`。

## 4. 顶层字段

### config

记录生成 plan 时使用的地址、模型、输入和截位配置路径。`IMAGE_PATH` 和 `INTR_MOVE_PATH` 会写入该字段，便于追踪本次构建使用的输入 COE 和截位配置文件。

### image

描述当前输入 feature map：

```text
addr
channels
aligned_channels
shape_nchw
storage_shape_nchw
size_bytes
file
```

`IMAGE_SOURCE = "coe"` 时，脚本会校验 `IMAGE_PATH` 指向的 COE 文件的 32-bit word 数是否等于 `image.size_bytes / 4`。输入 feature map 的逻辑尺寸来自 IR；物理存储尺寸会按至少 `8x8` 且 H/W 为 8 的倍数预留。

### model_ops

保存从 IR 读取的完整 op 列表，便于排查 unsupported op。

### model_layers

保存后端可处理的 conv 层信息。每个 conv 都带有：

```text
conv_index
layer_index
layer
has_relu
conv
```

`conv_index` 表示当前 IR 中第几次 conv，从 1 开始；`layer_index/layer` 仅用于参数文件命名和调试，不用于截位查表。

## 5. tensors

常见 tensor：

```text
image
layer1_weight
layer1_params
layer1_conv_out
layer1_relu_out
instr
```

有 bias 的 conv 还会包含：

```text
layer1_bias
```

无 bias 的 conv 不创建 `layerN_bias` tensor。

## 6. 参数区布局

每层使用单一初始化区：

```text
layerN_params
```

有 bias：

```text
每个 group = valid weights + padded int32 bias
layerN_params.size_bytes = weight_size + padded_bias_size
split.conv.has_bias = true
split.bias_addr 存在
```

无 bias：

```text
每个 group = valid weights
layerN_params.size_bytes = weight_size
layerN_bias tensor 不存在
split.conv.has_bias = false
split.bias_addr = null
split.offsets_bytes.bias = null
split.size_bytes.bias = 0
```

## 7. execution_plan

每个 layer plan 包含：

```text
conv_index
layer
input_channels
output_channels
kernel_size
input_hw
logical_conv_output_hw
conv_output_hw
logical_output_hw
output_hw
has_bias
has_relu
has_dsmp
reserved_regions
splits
```

每个 split 包含：

```text
group_index
start_channel
channels
valid_channels
offsets_bytes
size_bytes
conv
relu
dsmp    # 仅 stride=2,padding=0 时存在
```

`conv_index` 会传给 `generate_instr.py`，用于按 `CONV_MOVE_BY_INDEX` 查找该 conv 的 move。

当 IR 只截断到 conv，例如 `INFER_PARSE_MODE=2` 且 `INFER_PARSE_OP_LIMIT=1`，memory plan 不会自动补 ReLU：

```text
has_relu = false
reserved_regions 中没有 relu_out
split 中没有 relu
generate_instr.py 只生成 CONV 和 END
```

当 IR 包含 `conv2d -> relu`，memory plan 才会规划 ReLU 阶段。

## 8. 通道和尺寸规划

- 数据按 4 通道对齐。
- 单次 NPU pass 最多处理 8 个输出通道。
- 大 feature map 层可按 4 通道 group 拆分。
- conv 输入通道数当前最大支持 256。
- 输入和运行时 feature map 最小按 `8x8` 预留，且 H/W 按 8 对齐。
- `stride=1` 规划为 `conv -> relu`。
- `stride=2,padding=0` 规划为 `conv(stride=1) -> dsmp -> relu`。
- `stride>2` 或 `stride=2,padding!=0` 会报错。
