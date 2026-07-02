# generate_memory_plan.py 说明

## 1. 总体功能

`python/generate_memory_plan.py` 根据模型结构和 `python/npu_config.py` 生成 `data/memory_plan.json`。

它负责把模型中的卷积层转换成 NPU 数据构建和后续指令生成需要的统一规划信息，包括：

- 模型解析配置；
- 输入图像尺寸和存储布局；
- 每层 weight、bias、指令等初始化数据的地址和大小；
- 每层 conv / 可选 dsmp / relu 运行时输出缓冲区的地址和大小；
- 每层在当前 NPU 通道能力下的拆分执行计划。

当前 NPU 卷积和矩阵计算单次最大处理通道数为 8，数据仍按 4 通道对齐。默认情况下，若某层输出通道数大于 8，`memory_plan.json` 会将该层按先 8 后 4 的方式拆成多个通道段，例如 12 输出通道拆成 8 + 4。若该层输入或输出 feature map 的存储元素数（宽 * 高 * 按 4 对齐后的通道数）大于等于 `CHANNEL_GROUP4_FEATURE_SIZE_THRESHOLD`，则该层输出通道只按 4 通道 group 拆分，例如 12 输出通道拆成 4 + 4 + 4。

当前卷积输入通道数最大支持 256，生成 memory plan 时若超过 256 会直接报错。

当前 `generate_memory_plan.py` 的通道支持范围可以概括为：

- 输入通道数 `<= 256`、输出通道数 `<= 8`：直接生成单个输出通道 split。
- 输入通道数 `<= 256`、输出通道数 `> 8`：支持生成。默认按 NPU 单次最多 8 输出通道拆分，例如 12 输出通道拆成 8 + 4；当该层输入或输出 feature map 存储元素数大于等于 `CHANNEL_GROUP4_FEATURE_SIZE_THRESHOLD` 时，只按 4 输出通道拆分，例如 12 输出通道拆成 4 + 4 + 4。
- 输入通道数 `> 256`：不支持，直接报错，不生成近似或不完整的 memory plan。

当前 stride / padding 支持范围为：

- `stride = 1`：按普通 `conv -> relu` 规划，bias 由 CONV 指令自动处理。
- `stride = 2, padding = 0`：按 `conv(stride=1) -> dsmp -> relu` 规划，bias 由 CONV 指令自动处理。
- `stride > 2`：暂不支持，直接报错。
- `stride = 2, padding != 0`：暂不支持，直接报错。

## 2. 主要输入

```bash
python ./python/generate_memory_plan.py yolov2_14layer_quantized.py
```

模型文件可以直接给文件名；脚本会在 `model/` 下查找。

主要配置来自 `python/npu_config.py`：

```text
IMAGE_HEIGHT / IMAGE_WIDTH
INFER_PARSE_MODE
INFER_PARSE_LAYER_START / INFER_PARSE_LAYER_END
CHANNEL_GROUP4_FEATURE_SIZE_THRESHOLD
INIT_BASE_ADDR / INIT_LIMIT_ADDR
RUNTIME_BASE_ADDR
```

## 3. 顶层字段

### config

记录生成 memory plan 时使用的配置，例如：

```text
INIT_BASE_ADDR
INIT_LIMIT_ADDR
RUNTIME_BASE_ADDR
IMAGE_BASE_ADDR
IMAGE_SOURCE
INFER_PARSE_MODE
INFER_PARSE_LAYER_START / INFER_PARSE_LAYER_END
model_py
```

该字段用于追踪当前 JSON 是按什么配置生成的。

### alignment_bytes

当前初始化区和运行时区地址按 4 字节对齐。

### image

描述当前推理起始层输入 feature map 在内存中的信息。`INFER_PARSE_LAYER_START = 1` 时它是原始输入图像；从中间层开始时，它是该起始层的输入 feature map：

```text
addr
channels
aligned_channels
shape_nchw
storage_shape_nchw
size_bytes
file
```

其中 `storage_shape_nchw` 是按 4 通道对齐后的实际存储形状。`IMAGE_SOURCE = "coe"` 时，`generate_memory_plan.py` 会检查 `coe/image.coe` 的 32-bit word 数是否等于 `size_bytes / 4`；`IMAGE_SOURCE = "external"` 时，输入不进入 `init_regions`，地址来自 `IMAGE_BASE_ADDR`。

### model_layers

记录从模型 `.py` 中解析出的卷积层参数：

```text
layer_index
layer
conv.in_channels
conv.out_channels
conv.kernel_size
conv.stride
conv.padding
```

该字段描述模型原始结构，不包含内存地址。

## 4. tensors

`tensors` 是后续脚本读取最多的字段，保存所有关键数据对象的地址、形状和大小。

常见条目包括：

```text
image
layer1_weight
layer1_bias
layer1_conv_out
layer2_dsmp_out
layer1_relu_out
layer2_weight
layer2_bias
...
instr
```

### layerN_weight

描述第 N 层卷积权重：

```text
addr
channels.in / channels.out
aligned_channels.in / aligned_channels.out
shape_oihw
storage_shape_oihw
size_bytes
```

`shape_oihw` 是模型原始权重形状，`storage_shape_oihw` 是 weight COE 的实际存储形状：输出通道保持有效输出通道数，输入通道补齐到 4 的倍数。`aligned_output_channels` 仍用于 bias、运行时输出和 split 对齐。

### layerN_bias

描述第 N 层 bias 数据：

```text
addr
channels
aligned_channels
shape
storage_shape
size_bytes
```

bias 不再展开为矩阵。当前由 `params_to_bram_coe.py` 读取通道数，把 bias 补 0 到 4 的倍数后按 signed int32 word 写入对应的 `coe/layerN_params.coe`。

### layerN_conv_out / layerN_dsmp_out / layerN_relu_out

描述第 N 层每个运行时阶段的输出缓冲区：

```text
addr
size_bytes
channels
aligned_channels
shape_nchw
storage_shape_nchw
```

这些地址是后续生成 conv、dsmp、relu 指令时需要使用的输入输出地址。

对于 `stride=2,padding=0` 的层，会额外生成：

```text
layerN_dsmp_out
```

此时 `layerN_conv_out` 是 NPU stride=1 卷积后的完整尺寸中间结果，`layerN_dsmp_out` 才是模型语义上的 stride=2 输出结果。后续 `relu` 接在 `dsmp_out` 后面。

## 5. execution_plan

`execution_plan` 是为后续指令生成准备的分段执行计划。

每一层有一个执行计划条目：

```text
layer
npu_max_channels_per_pass
channel_alignment
input_channels
output_channels
aligned_input_channels
aligned_output_channels
kernel_size
output_hw
reserved_regions
splits
```

### stride=2 的 DSMP 规划

当模型层满足：

```text
stride = 2
padding = 0
```

时，memory plan 会为该层规划额外的 DSMP 阶段：

```text
conv(stride=1) -> dsmp -> relu
```

例如 layer2 输入为 `256 x 256`，模型输出应为 `128 x 128`：

```text
layer2_conv_out:
  shape = [1, 12, 256, 256]

layer2_dsmp_out:
  shape = [1, 12, 128, 128]

layer2_relu_out:
  shape = [1, 12, 128, 128]
```

其中 `conv_out` 是 NPU stride=1 产生的中间结果，`dsmp_out` 是下采样后的结果。

### reserved_regions

`reserved_regions` 记录该层在运行时区中一次性预留好的完整输出空间：

```text
conv_out
dsmp_out   # 仅 stride=2,padding=0 的层存在
relu_out
```

每个区域包含：

```text
addr
size_bytes
layout
```

规划方式是先为该层完整的卷积输出、可选下采样输出和 ReLU 输出分别预留总空间，然后每个 split 按通道段连续写入对应区域。

例如第二层输出通道为 12，输入 feature map 存储大小为 `256 * 256 * 4`，达到默认阈值 `CHANNEL_GROUP4_FEATURE_SIZE_THRESHOLD = 256 * 256 * 4`，因此按 4 + 4 + 4 拆成三个 group：

```text
layer2_conv_out 总空间:
  base = 0x00280000
  size = 12 * 256 * 256 = 0x000C0000 bytes

group0 conv output:
  start = 0x00280000
  size  = 4 * 256 * 256 = 0x00040000 bytes

group1 conv output:
  start = 0x002C0000
  size  = 4 * 256 * 256 = 0x00040000 bytes

group2 conv output:
  start = 0x00300000
  size  = 4 * 256 * 256 = 0x00040000 bytes
```

因此 group1 的 conv 数据紧跟 group0 之后。`dsmp_out` 和 `relu_out` 使用同样的紧密排列方式，但它们的空间尺寸是下采样后的 `128 x 128`。

### splits

`splits` 描述该层被拆成几个 NPU pass。

对于输出通道数小于等于 8 的层，通常只有一个 split。对于 `3 -> 12` 的第二层，由于输入 feature map 存储元素数达到阈值，会拆成三个 split：

```text
group0: output channel 0..3
group1: output channel 4..7
group2: output channel 8..11
```

每个 split 包含：

```text
group_index
start_channel
channels
valid_channels
has_padding
offsets_bytes
size_bytes
conv
dsmp   # 仅 stride=2,padding=0 的层存在
relu
```

### offsets_bytes

记录该 split 在对应数据区中的相对字节偏移：

```text
weight
conv_output
dsmp_output
bias
output
```

例如第二层 `3 -> 12, 2x2`：

```text
group0.weight = 0x00000000
group1.weight = 0x00000080

group0.bias   = 0x00000000
group1.bias   = 0x00000020
```

对于 stride=2 的层：

```text
conv_output 使用 NPU stride=1 的中间输出尺寸计算偏移。
dsmp_output / output 使用下采样后的真实输出尺寸计算偏移。
bias 使用 int32 bias word 计算偏移。
```

### size_bytes

记录该 split 实际占用的字节数：

```text
weight
conv_output
dsmp_output
bias
output
```

其中 `conv_output` 按 NPU stride=1 中间图尺寸计算，`dsmp_output` 和 `output` 按下采样后的输出矩阵大小计算，`bias` 按 32-bit bias word 计算。对于第二层：

```text
group0.conv_output = 8 * 256 * 256 = 0x00080000 bytes
group1.conv_output = 4 * 256 * 256 = 0x00040000 bytes

group0.output      = 8 * 128 * 128 = 0x00020000 bytes
group1.output      = 4 * 128 * 128 = 0x00010000 bytes
```

该字段与 `offsets_bytes` 配合，可以确认多个 group 在同一总预留区域中是紧密连续排列的。

### conv / dsmp / relu

记录每个 split 对应的实际指令地址：

```text
conv.input_addr
conv.weight_addr
conv.output_addr

dsmp.input_addr
dsmp.output_addr
dsmp.image_size
dsmp.channels

relu.input_addr
relu.output_addr
```

这些字段使后续 `generate_instr.py` 可以直接知道每一次卷积、下采样和激活要读写哪些地址。

## 6. init_regions 和 runtime_regions

### init_regions

记录初始化数据区中的连续布局顺序：

```text
image
layer1_weight
layer1_bias
layer2_weight
layer2_bias
instr
```

每个区域包含：

```text
name
addr
size_bytes
end_addr_exclusive
file
```

### runtime_regions

记录运行时输出缓冲区布局：

```text
layer1_conv_out
layer1_relu_out
layer2_conv_out
layer2_dsmp_out
layer2_relu_out
```

每个区域包含地址、大小、通道数和 shape 信息。

## 7. 结束地址

```text
init_end_addr_exclusive
runtime_end_addr_exclusive
```

这两个字段分别表示初始化区和运行时区当前规划后的结束地址，用于检查是否越界或与其他区域冲突。

## 8. 参数区分组布局（当前实现）

`init_regions` 对每层使用单一的 `layerN_params` 项，替代原先分离的 `layerN_weight` 和 `layerN_bias` 项。其地址为原 weight 起始地址，大小为原 weight+bias 之和。

`execution_plan.splits` 中的 `offsets_bytes.weight` 和 `conv.weight_addr` 使用参数区物理偏移：每个输出通道 group（默认最多 8 通道，大 feature map 层为 4 通道）先存有效 weight，再存补齐到 4 通道边界的 int32 bias。因此后续 group 的 weight 偏移会跨过前一 group 的 bias；`bias_addr` 记录该 group 自动 bias 读取的起始位置。

运行时 `conv_out`、`dsmp_out`、`relu_out` 仍按 output channel 的 feature-map 大小计算偏移，group 的矩阵在预留区中按通道顺序连续存放。
