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

该脚本负责把 IR 中后端可支持的 `conv2d/relu/avgpool2d/maxpool2d/linear` 路径转换为 NPU 地址规划、参数区规划和分组执行计划。

## 2. 后端支持范围

当前 memory plan 阶段只支持：

```text
conv2d
relu
avgpool2d
maxpool2d
linear
```

如果 IR 中出现暂不支持的 op，会直接报错，不会静默跳过。`flatten` 不属于后端 op，已在前端透传忽略。

```text
Unsupported op xxx at op id N.
```

## 3. 主要配置

`npu_config.py` 中与本阶段相关的配置包括：

```text
INIT_BASE_ADDR
INIT_LIMIT_ADDR
RUNTIME_BASE_ADDR
RUNTIME_LIMIT_ADDR
ENABLE_OUTPUT_ADDR
OUTPUT_BASE_ADDR
OUTPUT_LIMIT_ADDR
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

`ENABLE_OUTPUT_ADDR = False` 时，最终输出仍按原逻辑从 runtime 区分配，`RUNTIME_LIMIT_ADDR` 是 runtime feature/output 区的半开上限，memory plan 会检查 `runtime_end_addr_exclusive <= RUNTIME_LIMIT_ADDR`。

`ENABLE_OUTPUT_ADDR = True` 时，最后一次计算的模型输出强制放到 `OUTPUT_BASE_ADDR`，并检查 `output_end_addr_exclusive <= OUTPUT_LIMIT_ADDR`；此时 `RUNTIME_LIMIT_ADDR` 只约束最后一次计算之前的中间 runtime 数据。

`INPUT_HEIGHT` / `INPUT_WIDTH` 只作为 PT2 输入 shape metadata 缺失时的 fallback。正常情况下，输入尺寸和通道数来自 `data/model_ir.json`，而 `data/model_ir.json` 的 input 字段来自 PT2 graph metadata；只要 PT2 中存在该 metadata，memory plan 就不会使用 `INPUT_HEIGHT` / `INPUT_WIDTH`。

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
```

`conv_index` 会传给 `generate_instr.py`，用于按 `CONV_MOVE_BY_INDEX` 查找该 conv 的 move。

当 IR 只截断到 conv，例如 `INFER_PARSE_MODE=2` 且 `INFER_PARSE_OP_LIMIT=1`，memory plan 不会自动补 ReLU：

```text
has_relu = false
reserved_regions 中没有 relu_out
layer plan 中没有 relu
generate_instr.py 只生成 CONV 和 END
```

当 IR 包含 `conv2d -> relu`，memory plan 才会规划 ReLU 阶段。

## 8. 通道和尺寸规划

- 数据按 4 通道对齐。
- 单次 NPU pass 最多处理 8 个输出通道。
- 大 feature map 层可按 4 通道 group 拆分。
- conv 输入通道数当前最大支持 256。
- 输入和运行时 feature map 最小按 `8x8` 预留，且 H/W 按 8 对齐。
- 后端 H/W 传播采用 NPU 物理执行语义：每层使用上一层的物理 storage H/W 作为本层有效输入边长来计算输出 H/W；本层输出 H/W 再向上对齐到 8 的倍数作为下一层 storage H/W。
- `stride=1` 规划为 `conv -> relu`。
- `stride=2,padding=0` 规划为 `conv(stride=1) -> dsmp -> relu`。
- DSMP 的输入和输出都按物理边长处理；例如输入 storage 为 `40x40` 时，DSMP 输出 H/W 为 `20x20`，再按 8 对齐为 storage。DSMP/ReLU 的通道数按实际输入通道数配置，范围为 1 到 256，不再按 conv group 拆分。
- `stride>2` 或 `stride=2,padding!=0` 会报错。

## 9. Linear / FULL 规划规则

当前 memory plan 支持 `linear`，并将其规划为 `op_type = "linear"` 的 execution stage。`flatten` 已在前端透传，不会进入 `model_ir.json` 和 `memory_plan.json`。

规划时会读取上一层输出的逻辑 shape 与物理 storage shape：

```text
逻辑输入特征数 = channels * height * width
FULL 输入字节数 = aligned_channels * storage_height * storage_width
FULL 输入字数 = FULL 输入字节数 / 4
```

`linear.in_features` 必须不大于后端有效输入特征数；若超过则 memory plan 阶段报错。参数区大小按物理 storage footprint 计算，原始 `in_features` 之外的位置由 `linear_params_to_bram_coe.py` 补 0 权重：

```text
linearN_params.size = out_features * (FULL 输入字节数 + optional int32 bias)
```

每个输出元素对应一个 split 和一条 FULL 指令。split 中输入地址保持为上一层输出 tensor 的基址；权重地址按 `bytes_per_output` 递增；输出地址按 byte 递增，因此 `linearN_out` 是 signed int8 紧密排列。

当上一层逻辑通道数不是 4 的倍数时，编译器不会报错，而是在全连接权重的 padded channel 位置补 0。例如上一层输出为 1 通道时，FULL 参数布局按 4 通道处理，额外 3 个通道的权重矩阵全为 0。
