# FPGA NPU 卷积特性与数据布局说明（AI 可读版）

## 0. NPU 整体简介

该 NPU 是部署在 FPGA 上的轻量级神经网络加速器，主要通过“配置寄存器 + 计算指令”的方式执行卷积、下采样、ReLU、矩阵累加等算子。

对于卷积算子，软件侧只需要完成输入数据、权重数据、输出缓冲区的内存布局准备，并通过指令配置输入地址、输出地址、权重地址和卷积参数；NPU 会根据配置自动从内存 / DCache 中读取所需数据并完成计算。

---

## 1. NPU 对卷积的处理方式

### 1.1 卷积执行的基本思想

NPU 执行卷积时，不需要软件逐像素、逐通道下发计算任务。软件只需要告诉 NPU：

1. 输入特征图数据的起始地址；
2. 卷积输出结果的写回地址；
3. 卷积核权重数据的起始地址；
4. 卷积核大小；
5. 输入图像块大小；
6. 起始比特位；
7. 实际输入通道数；
8. 实际输出通道数；
9. 数据类型，例如 INT8。

在卷积指令开始执行后，NPU 会根据这些配置自动读取输入特征图的各通道数据和卷积核权重数据，完成第一层卷积计算，并将结果写回到指定输出地址。

---

### 1.2 通道数配置原则

该 NPU 的硬件内部按 4 通道对齐方式处理数据。

但是在卷积配置寄存器中，通道数应填写模型的真实通道数，而不是补齐后的通道数。

例如 YOLOv2 第一层卷积为：

```text
Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
```

则卷积参数中应配置：

```text
input_channel  = 3
output_channel = 3
```

虽然内存中的图像和权重会按 4 通道对齐进行补零，但配置寄存器仍然使用真实通道数。NPU 会在内部自动按 4 输入 / 4 输出的方式进行对齐处理。

常见错误：

```text
错误：因为内存补齐为 4 通道，所以把 input_channel/output_channel 配成 4。
正确：配置真实通道数 3，内存布局仍然按 4 通道补齐。
```

---

### 1.3 图像块大小配置规则

卷积配置寄存器中的“图像块大小”不是实际输入图像边长，而是实际边长除以 8 后的块大小。

换算关系为：

```text
actual_image_width = block_image * 8
block_image = actual_image_width / 8
```

例如当前第一层输入图像为：

```text
256 x 256
```

则应配置：

```text
block_image = 256 / 8 = 32
```

如果手工填写二进制字段，需要注意指令设计文档中的字段编码通常是 N-1 编码：

```text
语义值 block_image = 32
字段编码值 = 31
二进制 = 011111
```

因此在汇编层可以理解为“配置图像块大小为 32”，但在手工拼机器码时应按指令设计文档的具体字段编码填写。

---

### 1.4 起始比特位配置规则

对于当前第一层卷积测试，卷积配置寄存器中的起始比特位直接配置为 0。

```text
start_position = 0
```

如果手工填写二进制字段，则为：

```text
00000
```

---

### 1.5 卷积核大小配置规则

卷积核大小由卷积配置参数中的 kernel_size 字段决定。

常见编码如下，具体位宽和位置以同时上传的《NPU 指令设计》文档为准：

```text
00 -> 1 x 1
01 -> 2 x 2
10 -> 3 x 3
11 -> 5 x 5
```

当前 YOLOv2 第一层卷积核大小为：

```text
kernel_size = 3 x 3
```

因此 kernel_size 字段应选择：

```text
10
```

---

## 2. 卷积指令应如何配置

### 2.1 卷积指令涉及的核心寄存器

一次卷积计算通常需要至少使用三个通用寄存器保存地址：

```text
R1 = 输入图像 / 输入特征图起始地址
R2 = 卷积输出结果写回地址
R3 = 卷积核权重起始地址
```

随后通过卷积参数配置指令配置卷积专用寄存器，例如：

```text
CONV_P_1 = 卷积核大小 + 图像块大小 + 起始比特位
CONV_P_2 = 实际输入通道数 + 实际输出通道数
```

最后执行：

```text
CONV R1, R2, R3
```

含义为：

```text
从 R1 指向的输入数据开始读取特征图；
从 R3 指向的权重数据开始读取卷积核；
将卷积结果写回 R2 指向的输出地址。
```

---

### 2.2 第一层卷积的推荐配置语义

针对当前 YOLOv2 第一层：

```text
输入图像尺寸: 256 x 256
真实输入通道: 3
真实输出通道: 3
硬件对齐通道: 4
卷积核大小: 3 x 3
stride: 1
padding: 1
start_position: 0
数据类型: INT8
```

配置语义为：

```text
input_start_addr  = image_start_addr
output_start_addr = 自行指定的输出缓冲区地址
weight_start_addr = weight_start_addr

kernel_size       = 3 x 3
block_image       = 32
start_position    = 0
input_channel     = 3
output_channel    = 3
dtype             = INT8
```

---

### 2.3 第一层卷积的汇编配置模板

以下模板用于说明配置流程。具体 opcode、subop、寄存器编号和二进制拼接方式，应参考同时上传的《NPU 指令设计》文档。

```asm
; 1. 配置输入图像起始地址
CFG_REGISTER R1, LOW,  input_addr_low16
CFG_REGISTER R1, HIGH, input_addr_high16

; 2. 配置输出结果写回地址
CFG_REGISTER R2, LOW,  output_addr_low16
CFG_REGISTER R2, HIGH, output_addr_high16

; 3. 配置卷积核权重起始地址
CFG_REGISTER R3, LOW,  weight_addr_low16
CFG_REGISTER R3, HIGH, weight_addr_high16

; 4. 配置卷积参数 1
; kernel_size = 3x3
; block_image = 32
; start_position = 0
CFG_REGISTER CONV_P_1, conv_p1_value

; 5. 配置卷积参数 2
; actual input_channel = 3
; actual output_channel = 3
CFG_REGISTER CONV_P_2, conv_p2_value

; 6. 执行卷积
CONV R1, R2, R3

; 7. 结束
END
```

---

## 3. NPU 要求的输入图像内存排列方式

### 3.1 图像通道对齐要求

当前测试输入图像为普通 RGB 图像：

```text
R, G, B
```

但 NPU 内部按 4 通道对齐处理，因此输入图像写入内存时，需要补出第 4 个虚拟通道。

第 4 个通道固定补 0：

```text
R, G, B -> R, G, B, 0
```

这个格式可以理解为 RGB0 或 RGBX，其中第 4 个字节只是为了硬件 4 通道对齐，不参与有效颜色计算。

---

### 3.2 图像数据的内存排列顺序

输入图像采用像素优先的交错排列方式，而不是先存完整 R 平面、再存完整 G 平面、再存完整 B 平面。

对于每个像素，4 个通道连续存放：

```text
pixel0: R0, G0, B0, 0
pixel1: R1, G1, B1, 0
pixel2: R2, G2, B2, 0
...
```

即内存线性排列为：

```text
R0, G0, B0, 00,
R1, G1, B1, 00,
R2, G2, B2, 00,
...
```

对于 256 x 256 图像：

```text
pixel_num = 256 * 256 = 65536
bytes_per_pixel = 4
image_size_bytes = 65536 * 4 = 262144 = 0x00040000
```

---

### 3.3 图像地址公式

设：

```text
image_start_addr = 输入图像起始地址
W = 256
x = 像素横坐标，范围 0..255
y = 像素纵坐标，范围 0..255
c = 通道编号
```

通道编号为：

```text
c = 0 -> R
c = 1 -> G
c = 2 -> B
c = 3 -> PAD0
```

则任意像素通道地址为：

```text
image_addr(x, y, c) = image_start_addr + 4 * (y * W + x) + c
```

当前样例中，如果：

```text
image_start_addr = 0x00000000
```

则开头几个字节含义为：

```text
0x00000000: pixel(0,0).R
0x00000001: pixel(0,0).G
0x00000002: pixel(0,0).B
0x00000003: pixel(0,0).PAD0

0x00000004: pixel(0,1).R
0x00000005: pixel(0,1).G
0x00000006: pixel(0,1).B
0x00000007: pixel(0,1).PAD0
```

---

## 4. NPU 要求的卷积权重内存排列方式

### 4.1 原始权重逻辑格式

YOLOv2 第一层卷积原始权重形状为：

```text
[out_channel = 3, in_channel = 3, kernel_h = 3, kernel_w = 3]
```

也就是：

```text
weight[oc][ic][kh][kw]
```

原始权重总数为：

```text
3 * 3 * 3 * 3 = 81
```

其中：

```text
oc = 输出通道编号，范围 0..2
ic = 输入通道编号，范围 0..2
kh = 卷积核行编号，范围 0..2
kw = 卷积核列编号，范围 0..2
```

---

### 4.2 权重补零规则

虽然卷积配置中填写真实通道数 3 输入 / 3 输出，但权重在内存中仍然要按 4 输入 / 4 输出补齐。

补齐后的逻辑形状为：

```text
padded_weight[4][4][3][3]
```

补零规则：

```text
1. 对于原始输出通道 0、1、2：
   原始输入通道 0、1、2 的权重保持不变；
   额外补出的输入通道 3 的 3x3 权重全部为 0。

2. 对于额外补出的输出通道 3：
   其输入通道 0、1、2、3 对应的所有 3x3 权重全部为 0。
```

补齐后权重总数为：

```text
4 * 4 * 3 * 3 = 144 bytes
```

---

### 4.3 权重实际写入内存的顺序

NPU 要求权重不是简单按：

```text
OC -> IC -> KH -> KW
```

写入，而是按如下顺序写入：

```text
for oc in 0..3:
  for kh in 0..2:
    for kw in 0..2:
      for ic in 0..3:
        emit padded_weight[oc][ic][kh][kw]
```

也就是说，每连续 4 个权重字节表示：

```text
同一个输出通道 oc、
同一个卷积核空间位置 [kh, kw]、
4 个输入通道 IC0, IC1, IC2, IC3 的权重。
```

可以理解为：

```text
OC0, kernel[0,0]: IC0, IC1, IC2, IC3
OC0, kernel[0,1]: IC0, IC1, IC2, IC3
OC0, kernel[0,2]: IC0, IC1, IC2, IC3
OC0, kernel[1,0]: IC0, IC1, IC2, IC3
...
OC1, kernel[0,0]: IC0, IC1, IC2, IC3
...
OC2, ...
OC3, 全部为 0
```

---

### 4.4 权重地址公式

设：

```text
weight_start_addr = 权重区起始地址
oc = 输出通道编号，范围 0..3
kh = 卷积核行编号，范围 0..2
kw = 卷积核列编号，范围 0..2
ic = 输入通道编号，范围 0..3
```

则权重地址为：

```text
weight_addr(oc, kh, kw, ic)
  = weight_start_addr
    + (((oc * 3 + kh) * 3 + kw) * 4 + ic)
```

例如当前样例中：

```text
weight_start_addr = 0x00040000
```

则：

```text
0x00040000: padded_weight[0][0][0][0]
0x00040001: padded_weight[0][1][0][0]
0x00040002: padded_weight[0][2][0][0]
0x00040003: padded_weight[0][3][0][0] = 0

0x00040004: padded_weight[0][0][0][1]
0x00040005: padded_weight[0][1][0][1]
0x00040006: padded_weight[0][2][0][1]
0x00040007: padded_weight[0][3][0][1] = 0
```

---

### 4.5 权重数值编码方式

权重按 signed int8 量化值处理。

有效范围为：

```text
-128 <= weight <= 127
```

写入 COE 或 BRAM 初始化文件时，负数使用 8-bit 二进制补码表示。

例如：

```text
-1  -> 0xFF
-32 -> 0xE0
0   -> 0x00
68  -> 0x44
```

---

### 4.6 bias 数据矩阵生成与内存排列方式

当前 `bias_to_bram.py` 只支持 bias 个数 `<= 4` 的情况；如果 bias 个数大于 4，暂时保留为后续开发接口，不生成近似数据。

bias 原始值先按构建脚本中的 `BIAS_1_MOVE` 做缩放：

```text
processed_bias = floor(raw_bias / BIAS_1_MOVE)
```

随后把处理后的 bias 转为 signed int8 的 8-bit 补码字节。若 bias 个数不足 4，则在末尾补 0，形成一个 4 通道 bias 组：

```text
[bias0, bias1, bias2, bias3]
```

其中补齐后的通道只作为 4 通道对齐占位，不参与有效输出通道计算。

与旧规则不同，bias 不再按“每个 bias 单独生成一个 `L x L` 矩阵，然后多个矩阵依次排列”的方式写入。新的规则是：对 `L x L` 中的每一个空间位置，都连续写入同一组 4 通道 bias 字节。

也就是说，内存中的 byte 顺序为：

```text
pixel0: bias0, bias1, bias2, bias3
pixel1: bias0, bias1, bias2, bias3
pixel2: bias0, bias1, bias2, bias3
...
```

如果当前 3 个有效 bias 经过缩放后分别为：

```text
bias0 = 0x07
bias1 = 0x06
bias2 = 0x03
bias3 = 0x00   # padding
```

则内存 byte 排列应为：

```text
07 06 03 00, 07 06 03 00, 07 06 03 00, ...
```

由于当前 COE 以 32-bit word 输出，并采用 little-endian 打包，即低地址 byte 位于 word 的低 8 位、显示在 32-bit 十六进制字符串右侧，因此 COE 中看到的 word 为：

```text
00030607,
00030607,
00030607,
...
```

对于 `L = 256` 的情况，bias 区总字节数仍为：

```text
256 * 256 * 4 = 262144 bytes = 0x00040000
```

---

## 5. 当前第一层卷积样例的内存布局

当前样例中，合并后的 BRAM 地址空间如下：

```text
BRAM_BASE = 0x00000000

图像数据区:
  image_start_addr = 0x00000000
  image_size_bytes = 0x00040000
  image_end_addr   = 0x0003FFFF

权重数据区:
  weight_start_addr = 0x00040000
  weight_size_bytes = 0x00000090
  weight_end_addr   = 0x0004008F

当前合并数据结束后的下一个空地址:
  next_free_addr = 0x00040090
```

如果需要把第一层卷积输出也写回同一片 BRAM，可以临时选择：

```text
output_start_addr = 0x00040090
```

实际工程中，输出地址也可以由系统软件或测试平台另行指定，只要保证不覆盖输入图像和权重区域即可。

---

## 6. AI 生成卷积指令时的检查清单

当 AI 根据该 NPU 生成某一层卷积指令时，应按以下步骤检查：

```text
1. 从模型定义中读取真实 in_channels、out_channels、kernel_size、stride、padding。
2. 图像 / 特征图在内存中按 4 通道对齐组织。
3. 权重在内存中按 4 输入 / 4 输出对齐组织。
4. 卷积配置寄存器中的通道数填写真实通道数，不填写补齐后的 4 通道数。
5. 图像块大小填写 actual_image_width / 8。
6. 若手工拼二进制字段，注意字段是否使用 N-1 编码。
7. 当前第一层 start_position 直接置 0。
8. 设置 R1 为输入地址，R2 为输出地址，R3 为权重地址。
9. 配置 CONV_P_1。
10. 配置 CONV_P_2。
11. 执行 CONV R1, R2, R3。
12. 根据需要追加 END 指令。
```

---

## 7. 当前 YOLOv2 第一层卷积的机器可读摘要

```yaml
npu:
  type: FPGA_NPU
  execution_model: config_registers_plus_compute_instruction
  conv_behavior:
    input_address: start_address_only
    output_address: start_address_only
    weight_address: start_address_only
    auto_read_channels_and_weights: true
    internal_channel_alignment: 4
    conv_config_channels: actual_model_channels

layer:
  name: yolo_v2_layer1_conv
  input_size: [256, 256]
  kernel_size: [3, 3]
  stride: [1, 1]
  padding: [1, 1]
  actual_input_channels: 3
  actual_output_channels: 3
  aligned_input_channels_in_memory: 4
  aligned_output_channels_in_memory: 4
  dtype: int8
  start_position: 0
  block_image_semantic_value: 32
  actual_image_width_formula: block_image * 8

image_memory:
  layout: pixel_interleaved
  channel_order: [R, G, B, PAD0]
  pixel_layout: [R, G, B, 0]
  image_start_addr: 0x00000000
  image_size_bytes: 0x00040000
  image_end_addr: 0x0003FFFF
  address_formula: image_start_addr + 4 * (y * 256 + x) + channel_index

weight_memory:
  original_shape: [3, 3, 3, 3]
  padded_shape: [4, 4, 3, 3]
  layout: [OC, KH, KW, IC]
  emit_loop: for oc in 0..3, kh in 0..2, kw in 0..2, ic in 0..3
  value_type: signed_int8
  encoding: 8bit_twos_complement_hex
  weight_start_addr: 0x00040000
  weight_size_bytes: 0x00000090
  weight_end_addr: 0x0004008F
  address_formula: weight_start_addr + (((oc * 3 + kh) * 3 + kw) * 4 + ic)

instruction_generation:
  R1: input_start_addr
  R2: output_start_addr
  R3: weight_start_addr
  CONV_P_1:
    kernel_size: 3x3
    block_image: 32
    start_position: 0
  CONV_P_2:
    input_channel: 3
    output_channel: 3
  compute_instruction: CONV R1, R2, R3
```
