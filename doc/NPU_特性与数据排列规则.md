# NPU 特性与数据排列规则

## 1. 本文档范围

本文档记录当前编译器仍需保持不变的 NPU 物理布局规则。PT2 前端改造只改变模型输入和中间层来源，不改变 image、weight、bias、instruction 和 runtime feature map 的内存排列方式。

## 2. 前端到后端的边界

当前前端输出：

```text
data/model_ir.json
data/model_params/*.txt
```

参数文本保持逻辑布局：

```text
Conv weight   : OIHW
Linear weight : OI
Bias          : 一维 int32
```

NPU 物理布局从 `params_to_bram_coe.py` 开始处理。

## 3. image 布局

输入数据 COE 由 `npu_config.py` 中的 `IMAGE_PATH` 指定。

RGB 输入按 4 通道对齐：

```text
pixel0: R, G, B, 0
pixel1: R, G, B, 0
...
```

如果输入是单通道或其他通道数，内存占用仍按 4 通道对齐。输入 H/W 按实际尺寸预留和校验，不会补零到 8 的倍数；例如 28x28x1 的输入，对应 28x28x4 的通道对齐 footprint。`generate_memory_plan.py` 会根据 `data/model_ir.json` 中的输入 shape 和通道对齐规则校验 `IMAGE_PATH` 指向的 COE word 数。

## 4. weight 布局

Conv weight 逻辑格式：

```text
[out_channel, in_channel, kernel_h, kernel_w]
```

物理写入时：

```text
输出通道按 group 拆分
每个有效输出通道内部：
  输入通道补齐到 4 的倍数
  每 4 个输入通道为一组
  组内按 kh -> kw -> ic 交错写入
```

例如输入通道为 12 时，单个输出通道内部依次写入：

```text
ic0..ic3 的所有 kernel 位置
ic4..ic7 的所有 kernel 位置
ic8..ic11 的所有 kernel 位置
```

输入通道补齐产生的 0 kernel 必须真实写入 weight COE。

## 5. bias 布局

有 bias 时，参数区每个 group 的布局为：

```text
valid weights + padded int32 bias
```

bias 按 signed int32 word 原样写入，按 group 补到 4 通道边界。

无 bias 时，参数区每个 group 的布局为：

```text
valid weights
```

无 bias 时不生成 bias txt，不创建 `layerN_bias` tensor，不写 padded bias，后续 group 的 weight 偏移也不会跨过 bias 空间。

## 6. RCONV1 condition_bias

`operator/conv/conv.py` 根据 op plan 中的 `has_bias` 写入 RCONV1：

```text
condition_bias = 1  有 bias，NPU 从卷积核后读取 bias
condition_bias = 0  无 bias，NPU 不读取 bias
```

该字段来自：

```text
model_ir.json -> memory_plan.execution_plan.splits[].conv.has_bias -> generate_instr.py -> operator/conv
```

## 7. runtime feature map

runtime feature map 按输出通道 group 紧密排列：

```text
group0 feature map
group1 feature map
...
```

每个 feature map 的通道数按 4 对齐。所有参与编译的 runtime feature map H/W 都按实际尺寸规划和配置：若输出为 `13x13`，后续 CONV/AVGPOOL/MAXPOOL/ReLU/DSMP 继续按 `13x13` 作为输入尺寸，不再补零到 8 的倍数。`shape_nchw` 保存实际通道和实际 H/W，`aligned_shape_nchw` 保存通道对齐后的形状，其中 H/W 与实际 H/W 相同。

## 8. 通道拆分

当前单次 NPU pass 最多处理 8 个输出通道。

默认拆分示例：

```text
12 output channels -> 8 + 4
16 output channels -> 8 + 8
```

当输入或输出 feature map 存储元素数达到 `CHANNEL_GROUP4_FEATURE_SIZE_THRESHOLD` 时，按 4 通道 group 拆分：

```text
12 output channels -> 4 + 4 + 4
```

## 9. 指令和合并布局

`instr_txt_to_bram_coe.py` 仍把每条 32-bit 指令写成 BRAM COE word。

`merge.py` 仍按 `data/memory_plan.json` 中的 `init_regions` 顺序合并：

```text
image
layerN_params
instr
```

`merge.py` 解析输入 COE 时只按文件头 `memory_initialization_radix`
解释裸 token。当前编译器生成的 COE 均使用 `radix=16`，因此类似
`0D010413`、`0B010203` 的 32-bit word 都必须按十六进制原样解析。
合并阶段不支持 Python 风格 `0x`/`0b`/`0d` 前缀，也不支持 Verilog
风格 `32'h...`/`32'b...`/`32'd...` 前缀，避免与普通十六进制 word
发生歧义。

最终输出：

```text
target/all.coe
target/all.coe.map.txt
```

## 10. Linear / FULL 参数与输出布局

前端仍按 PyTorch 逻辑布局导出全连接参数：

```text
linear weight: [out_features, in_features]
linear bias  : [out_features]，仅有 bias 时生成，int32
```

`flatten` 不写入 IR 和 memory plan。FULL 直接读取上一层 NPU 输出的物理存储数据。编译器在生成 `linearN_params.coe` 时，把每个输出神经元的权重重排为上一层输出的物理 footprint：

```text
out0 padded/interleaved weights
out0 bias (optional int32)
out1 padded/interleaved weights
out1 bias (optional int32)
...
```

单个输出的权重先扩展到：

```text
[aligned_input_channels, input_height, input_width]
```

其中输入通道不足 4 的倍数时补 0 通道；H/W 不做任何补零。写入顺序为每 4 通道一组，组内按同一空间位置交织：

```text
group0 row0 col0 ch0, ch1, ch2, ch3
group0 row0 col1 ch0, ch1, ch2, ch3
...
group1 row0 col0 ch4, ch5, ch6, ch7
...
```

FULL 输出是 signed int8，`linearN_out` runtime 区按 byte 紧密排列：第 0 个输出在 base，第 1 个输出在 base+1，以此类推。
