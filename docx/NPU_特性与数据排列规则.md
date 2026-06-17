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

## 1.1 stride=2 卷积与下采样

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

## 4. weight 排列规则

卷积权重原始格式来自 PyTorch Conv2d：

```text
[out_channel, in_channel, kernel_h, kernel_w]
```

生成 COE 时先将输入通道和输出通道分别补齐到 4 的倍数。

当前输入通道数大于 4 的情况先保留为后续扩展接口。

对于输入通道小于 4 的情况，每个输出卷积核补足到 4 个输入卷积核。例如 `3 -> 12` 的第二层卷积，每个输出通道有 3 个有效输入卷积核，需要补 1 个全 0 卷积核。

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

## 5. bias 排列规则

bias 个数等于卷积输出通道数，生成 COE 时先补齐到 4 的倍数。

每 4 个 bias 作为一组，对输出 feature map 的每个空间位置重复写入这 4 个 bias：

```text
pixel0: bias0, bias1, bias2, bias3
pixel1: bias0, bias1, bias2, bias3
pixel2: bias0, bias1, bias2, bias3
...
```

如果 bias 数量超过 8，也按先 8 后 4 的方式记录通道段起始偏移。对于 12 个 bias：

```text
group0: bias channel 0..7,  offset = 0
group1: bias channel 8..11, offset = 8 * output_h * output_w
```
