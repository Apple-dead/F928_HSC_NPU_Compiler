# NPU 特性与数据排列规则

## 1. 当前通道处理能力

当前 NPU 的卷积和矩阵计算单次最大处理通道数为 8。

虽然配置寄存器字段中为通道数预留了 32 通道范围，但当前数据构建和指令生成阶段应按“单次最多 8 通道”处理。若某一层输出通道数超过 8，需要拆成多个通道段记录地址偏移，例如：

```text
3 -> 12 conv:
  group0: output channel 0..7
  group1: output channel 8..11
```

第二段的起始地址偏移需要记录下来，后续卷积指令可据此选择对应的 weight / bias 起始地址。

## 1.1 结构信息来源

weight COE 和 bias COE 生成脚本不直接解析 PyTorch 模型源码，而是读取 `data/memory_plan.json` 中已经整理好的张量结构信息。

`memory_plan.json` 由 `python/generate_memory_plan.py` 生成。生成时，脚本会调用 `python/model_parser.py` 解析模型定义 `.py` 文件，例如 `model/yolov2_14layer_quantized.py`。模型文件路径来自 `generate_memory_plan.py` 的命令行参数；在 `build.sh` 中对应：

```bash
MODEL_PY_NAME="yolov2_14layer_quantized.py"
python ./python/generate_memory_plan.py "$MODEL_PY_NAME"
```

`model_parser.py` 会扫描 `self.layerN` 中的 `nn.Conv2d` 定义，并提取以下结构信息：

```text
in_channels
out_channels
kernel_size
stride
padding
```

同时还会从 `self.layer` 中提取 `LeakyReLU` 的 `negative_slope`，用于后续 ReLU 指令参数生成。

`generate_memory_plan.py` 会基于上述信息继续计算并写入：

```text
weight:
  shape_oihw          = [out_channel, in_channel, kernel_h, kernel_w]
  storage_shape_oihw  = [aligned_out_channel, aligned_in_channel, kernel_h, kernel_w]
  size_bytes
  addr

bias:
  shape_nchw          = [1, out_channel, output_h, output_w]
  storage_shape_nchw  = [1, aligned_out_channel, output_h, output_w]
  size_bytes
  addr

execution_plan:
  channel group split
  weight/bias/runtime output offset
  conv/dsmp/madd/relu address
```

因此，`weight_to_bram_coe.py` 主要从 `memory_plan.json` 的 `layerN_weight.shape_oihw` 获取 `out_ch / in_ch / kernel_h / kernel_w`，再结合 `data/model_params/layerN_0_weight.txt` 中的权重数值生成 COE。

`bias_to_bram_coe.py` 主要从 `memory_plan.json` 的 `layerN_bias.shape_nchw` 获取 `out_ch / output_h / output_w`，再结合 `data/model_params/layerN_0_bias.txt` 和 `data/bias_move.json` 中的缩放配置生成 COE。

## 1.2 stride=2 卷积与下采样

当前 NPU 的卷积单元实际执行时 stride 固定为 1。

因此当模型中出现：

```text
stride = 2
padding = 0
```

的软件卷积层时，不能只规划一次普通卷积输出。正确流程应为：

```text
conv(stride=1) -> dsmp -> madd -> relu
```

也就是说，普通卷积先输出与输入 feature map 相同空间尺寸的中间结果，然后使用 DSMP 下采样得到模型语义上的 stride=2 输出。

以 layer2 为例，输入 feature map 为 `256 x 256`，模型定义为 `kernel=2, stride=2, padding=0`：

```text
NPU conv 输出尺寸  = 256 x 256
DSMP 输出尺寸      = 128 x 128
madd/relu 输入尺寸 = 128 x 128
```

DSMP 和 conv/madd/relu 一样按通道 group 执行，当前单次最多处理 8 通道。

当前暂不支持：

```text
stride > 2
stride = 2 且 padding != 0
```

遇到这些情况时，memory plan 生成阶段应直接报错。

## 2. 4 通道对齐规则

NPU 数据在内存中统一按 4 通道为一组进行对齐。

无论是图像、权重还是 bias，只要有效通道数不足 4 的倍数，都需要补 0 到 4 的倍数：

```text
3 channels  -> 4 channels
12 channels -> 12 channels
13 channels -> 16 channels
```

补齐通道只用于内存对齐，不参与有效计算。

## 3. 组内元素排列规则

每组 4 个矩阵在内存中采用元素交错排列，而不是把一个矩阵完整写完后再写下一个矩阵。

对于 4 个矩阵：

```text
M0, M1, M2, M3
```

内存排列应为：

```text
M0[0], M1[0], M2[0], M3[0],
M0[1], M1[1], M2[1], M3[1],
M0[2], M1[2], M2[2], M3[2],
...
```

如果当前 4 个 byte 为：

```text
07 06 03 00
```

并以 32-bit little-endian COE word 输出，则 COE 中显示为：

```text
00030607
```

## 4. image 排列规则

输入图像当前按 RGB 图像处理。由于 NPU 数据按 4 通道对齐，写入内存时需要补出第 4 个通道：

```text
R, G, B -> R, G, B, 0
```

图像采用像素优先的交错排列，而不是先写完整 R 平面、再写完整 G 平面、再写完整 B 平面。

内存 byte 顺序为：

```text
pixel0: R0, G0, B0, 00
pixel1: R1, G1, B1, 00
pixel2: R2, G2, B2, 00
...
```

对于宽度为 `W` 的图像，任意像素通道地址为：

```text
image_addr(x, y, c) = image_start_addr + 4 * (y * W + x) + c
```

其中：

```text
c = 0 -> R
c = 1 -> G
c = 2 -> B
c = 3 -> PAD0
```

例如当前默认输入为 `256 x 256`，则：

```text
image_size_bytes = 256 * 256 * 4 = 262144 = 0x00040000
```

## 5. weight 排列规则

卷积权重原始格式来自 PyTorch Conv2d：

```text
[out_channel, in_channel, kernel_h, kernel_w]
```

生成 COE 时先将输入通道和输出通道分别补齐到 4 的倍数。

当前输入通道数大于 4 的情况先保留为后续扩展接口。当前 `generate_memory_plan.py` 和 `weight_to_bram_coe.py` 都会在 `in_channels > 4` 时直接报错，不会生成可用的 memory plan 或 weight COE。

对于输入通道小于 4 的情况，每个输出卷积核补足到 4 个输入卷积核。例如 `3 -> 12` 的第二层卷积，每个输出通道有 3 个有效输入卷积核，需要补 1 个全 0 卷积核。

对于输出通道不是 4 的倍数的情况，不仅要在逻辑上补齐输出通道，还必须把补出来的全 0 输出卷积核真实写入 weight COE。

例如 `in_channels = 3`、`out_channels = 3`、`kernel = 3x3` 时，原始模型只有：

```text
3 output channels * 3 input channels = 9 个有效输入卷积核
```

但内存中需要按如下形状存储：

```text
padded_weight[4][4][3][3]
```

也就是说，每个有效输出通道都要补 1 个全 0 输入卷积核，同时还要额外补 1 个全 0 输出通道。最终 COE 中应包含：

```text
4 output channels * 4 input channels = 16 组卷积核
```

补齐后的输出通道只用于内存布局占位，不参与有效计算，但它对应的全 0 卷积核数据必须出现在 COE 中，保证后续通道组地址偏移和 memory plan 中的 `storage_shape_oihw` 一致。

每个输出卷积核内部按如下顺序写入：

```text
for kh:
  for kw:
    for ic in padded_input_channels:
      emit weight[oc][ic][kh][kw]
```

当输出通道大于 8 时，按先 8 后 4 的方式记录通道段起始偏移。对于 `3 -> 12`：

```text
group0: output channel 0..7,  offset = 0
group1: output channel 8..11, offset = 8 * padded_in_channel * kernel_h * kernel_w
```

## 6. bias 排列规则

bias 个数等于卷积输出通道数，生成 COE 时先补齐到 4 的倍数。

每 4 个 bias 作为一组，对输出 feature map 的每个空间位置重复写入这 4 个 bias：

```text
pixel0: bias0, bias1, bias2, bias3
pixel1: bias0, bias1, bias2, bias3
pixel2: bias0, bias1, bias2, bias3
...
```

当 bias 数量大于 4 时，也必须按每 4 个 bias 矩阵为一组进行排列。每一组内部采用元素交错排列；一组完整写完后，再写下一组。

例如 `bias = 8` 时，应先生成前 4 个 bias 对应的矩阵数据：

```text
pixel0: bias0, bias1, bias2, bias3
pixel1: bias0, bias1, bias2, bias3
pixel2: bias0, bias1, bias2, bias3
...
```

当前 4 个 bias 矩阵完整排列完以后，再生成后 4 个 bias 对应的矩阵数据：

```text
pixel0: bias4, bias5, bias6, bias7
pixel1: bias4, bias5, bias6, bias7
pixel2: bias4, bias5, bias6, bias7
...
```

不能把 8 个 bias 矩阵一起按空间位置交错成：

```text
pixel0: bias0, bias1, bias2, bias3, bias4, bias5, bias6, bias7
pixel1: bias0, bias1, bias2, bias3, bias4, bias5, bias6, bias7
...
```

也就是说，bias COE 的通道展开粒度固定为 4，不是一次展开全部输出通道。

如果 bias 数量超过 8，也按先 8 后 4 的方式记录通道段起始偏移。对于 12 个 bias：

```text
group0: bias channel 0..7,  offset = 0
group1: bias channel 8..11, offset = 8 * output_h * output_w
```

## 7. 配置寄存器尺寸语义

NPU 的不同算子对 image size / block image 字段的语义不完全相同。生成指令时不能把所有算子都统一写成 `feature_width / 8`。

### 7.1 CONV

CONV 使用三个地址寄存器：

```text
R1 = 输入 feature map 起始地址
R2 = 卷积输出起始地址
R3 = weight 起始地址
```

CONV 配置寄存器中的图像大小字段写入 `feature_width / 8`。例如输入宽度为 256 时：

```text
block_image = 256 / 8 = 32
```

CONV 配置中的通道数字段写真实模型通道数或当前 group 的有效执行通道数，不写补齐后的存储通道数。

当前 `operator/conv/conv.py` 会检查 `width % 8 == 0`，然后写入：

```text
block_image = width // 8
```

### 7.2 MADD

MADD 使用三个地址寄存器：

```text
R1 = 输入 feature map 起始地址
R2 = 输出 feature map 起始地址
R3 = bias 矩阵起始地址
```

MADD 配置寄存器中的图像大小字段同样写入 `feature_width / 8`。

当前 `operator/madd/madd.py` 会检查 `width % 8 == 0`，然后写入：

```text
block_image = width // 8
```

### 7.3 DSMP

DSMP 使用两个地址寄存器：

```text
R1 = 下采样输入起始地址
R2 = 下采样输出起始地址
```

DSMP 配置寄存器中的 image size 写实际 feature map 边长，不除以 8。

当前 `operator/dsmp/dsmp.py` 直接使用 `image_size` 编码：

```text
RDSMP.image_size = image_size
```

### 7.4 ReLU

ReLU 使用两个地址寄存器：

```text
R1 = ReLU 输入起始地址
R2 = ReLU 输出起始地址
```

ReLU 比较特殊：ReLU 配置寄存器中的 image size 写实际 feature map 边长，不除以 8。

例如 ReLU 输入是 `128 x 128`，配置寄存器里写语义值 `128`，不是 `16`。

当前 `operator/relu/relu.py` 直接使用 `feature_size` 编码：

```text
RRELU.feature_size = feature_size
```

因此当前 ReLU 生成代码符合该规则。
