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

该脚本负责把 IR 中后端可支持的 `conv2d/relu/avgpool2d/maxpool2d/linear/upsample_nearest2d/pad/cat` 路径转换为 NPU 地址规划、参数区规划和分组执行计划。

## 2. 后端支持范围

当前 memory plan 阶段只支持：

```text
conv2d
relu
avgpool2d
maxpool2d
upsample_nearest2d
pad
cat
linear
```

如果 IR 中出现暂不支持的 op，会直接报错，不会静默跳过。`flatten` 不属于后端 op，已在前端透传忽略。`pad` 和 `cat` 当前只支持能被规划为零拷贝/融合语义的模式，不能独立生成通用搬移指令。

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

`ENABLE_OUTPUT_ADDR = False` 时，模型输出从 runtime 区分配。memory plan 会先排除 graph outputs 原本占用的普通 runtime 区，再按 graph output 顺序把输出连续放到非输出 runtime 数据之后；如果某个输出本来已经位于尾部，则会复用原地址，不额外后移。`RUNTIME_LIMIT_ADDR` 是 runtime feature/output 区的半开上限，memory plan 会检查 `runtime_end_addr_exclusive <= RUNTIME_LIMIT_ADDR`。

`ENABLE_OUTPUT_ADDR = True` 时，graph outputs 按顺序从 `OUTPUT_BASE_ADDR` 开始连续放置，并检查 `output_end_addr_exclusive <= OUTPUT_LIMIT_ADDR`；此时 `RUNTIME_LIMIT_ADDR` 只约束模型输出之外的中间 runtime 数据。

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
aligned_shape_nchw
size_bytes
file
```

`IMAGE_SOURCE = "coe"` 时，脚本会校验 `IMAGE_PATH` 指向的 COE 文件的 32-bit word 数是否等于 `image.size_bytes / 4`。输入 feature map 的 H/W 尺寸来自 IR，并按实际大小预留；不会补零到 8 的倍数。`aligned_shape_nchw` 只表示通道按 4 对齐后的形状，H/W 与 `shape_nchw` 相同。

### model_ops

保存从 IR 读取的完整 op 列表，便于排查 unsupported op。

### model_output_values / model_outputs / output_regions

`model_output_values` 保存前端从 PT2 graph output 解析出的 IR value 顺序。规划完成后，`model_outputs` 和 `output_regions` 记录每个模型输出的 tensor、起始地址、大小和 shape。

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
conv_output_hw
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

`upsample_nearest2d` 会规划为 `op_type = "usmp"` 的 execution stage。`cat` 在满足通道维拼接和 4 通道对齐条件时不生成 execution stage，而是在生产者输出时预先把各输入放到同一个连续 `cat_out` tensor 的不同 offset。`pad -> maxpool2d` 在满足 `[0,1,0,1]` 常量 0 padding、`kernel_size=[2,2]`、`stride=[1,1]` 时融合为一条 MAXPOOL stage，依赖 NPU 的右/下边界隐式 0 语义，不单独分配 pad 输出。

## 8. 通道和尺寸规划

- 数据按 4 通道对齐。
- 单次 NPU pass 最多处理 8 个输出通道。
- 大 feature map 层可按 4 通道 group 拆分。
- conv 输入通道数当前最大支持 1024。
- 所有参与编译的 feature map H/W 都按实际矩阵大小规划和配置，不再把 H/W 换算成 `/8` 后的块数，也不会把非 8 倍数输出补零为下一层有效输入尺寸。
- 后端 H/W 传播采用实际尺寸语义：例如 CONV 输出为 `13x13` 时，下一次 CONV/AVGPOOL/MAXPOOL/ReLU/DSMP 直接按 `13x13` 看待并给到 NPU。
- `stride=1` 规划为 `conv -> relu`。
- `stride=2,padding=0` 规划为 `conv(stride=1) -> dsmp -> relu`。
- DSMP 的输入和输出按实际矩阵边长处理；例如输入为 `13x13` 时，DSMP 输出 H/W 为 `6x6`，下一层按实际 `6x6` 配置。DSMP 的通道数按实际输入通道数配置，范围为 1 到 1024，不再按 conv group 拆分。
- `stride>2` 或 `stride=2,padding!=0` 会报错。
- AVGPOOL/MAXPOOL 仅支持 `stride=[1,1]` 或 `stride=[2,2]`，并分别写入 RAVGPOOL/RMAXPOOL 的 step 字段：`stride=[1,1] -> step=0`，`stride=[2,2] -> step=1`。
- AVGPOOL/MAXPOOL 默认不做任何 H/W 补零，输出 H/W 按普通 2x2、padding=0 池化公式规划：`out = floor((in - 2) / stride) + 1`。因此 `stride=[1,1]` 时输出为输入 H/W 各减 1，`stride=[2,2]` 时输出为普通 2 倍下采样结果。唯一例外是已融合的 `pad -> maxpool2d`，该模式按 NPU 边界语义使输出 H/W 与 pad 前输入一致。

## 9. Pad / Cat / USMP 规划规则

`upsample_nearest2d` 当前只支持 `mode="nearest"` 且 `scale_factor=[2,2]`，输入 H/W 必须为正方形，通道数范围为 1 到 1024。规划后生成 USMP 指令，输出 H/W 为输入的 2 倍。

`pad` 当前不作为独立算子执行，只支持紧跟 `maxpool2d` 的右/下各补 1、常量 0 场景；该模式会融合进后续 MAXPOOL，并将 MAXPOOL 的输入尺寸配置为 pad 前尺寸。

`cat` 当前只支持 `dim=1` 的通道维拼接。memory plan 会提前创建完整 `cat_out` 区域，并要求各输入生产者按通道顺序直接输出到对应 offset，后续消费者读取 `cat_out`，因此不会生成 MOVE 指令。

## 10. Linear / FULL 规划规则

当前 memory plan 支持 `linear`，并将其规划为 `op_type = "linear"` 的 execution stage。`flatten` 已在前端透传，不会进入 `model_ir.json` 和 `memory_plan.json`。

规划时会读取上一层输出的实际 shape 与通道对齐信息：

```text
实际输入特征数 = channels * height * width
FULL 输入字节数 = aligned_channels * height * width
FULL 输入字数 = FULL 输入字节数 / 4
```

`linear.in_features` 必须不大于后端实际输入特征数；若超过则 memory plan 阶段报错。参数区大小按实际 H/W 和通道对齐后的 footprint 计算，原始 `in_features` 之外的位置由 `linear_params_to_bram_coe.py` 补 0 权重；不会因为 H/W 对齐额外补行或补列：

```text
linearN_params.size = out_features * (FULL 输入字节数 + optional int32 bias)
```

每个输出元素对应一个 split 和一条 FULL 指令。split 中输入地址保持为上一层输出 tensor 的基址；权重地址按 `bytes_per_output` 递增；输出地址按 byte 递增，因此 `linearN_out` 是 signed int8 紧密排列。

当上一层逻辑通道数不是 4 的倍数时，编译器不会报错，而是在全连接权重的 padded channel 位置补 0。例如上一层输出为 1 通道时，FULL 参数布局按 4 通道处理，额外 3 个通道的权重矩阵全为 0。
