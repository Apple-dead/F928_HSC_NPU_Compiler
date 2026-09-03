# operator 目录说明

## 1. 总体职责

`operator/` 下的代码负责把单个 NPU 算子的执行描述翻译成汇编片段。

当前目录包含：

```text
operator/conv/conv.py
operator/dsmp/dsmp.py
operator/relu/relu.py
operator/avgpool/avgpool.py
operator/maxpool/maxpool.py
operator/full/full.py
```

这些文件不再负责全局内存规划，也不再判断某一层应如何拆分。通道拆分、地址偏移、DSMP 是否需要插入、FULL 是否按输出元素展开等信息由 `generate_memory_plan.py` 统一写入 `memory_plan.json`。

## 2. 输入

每个 operator 的入口函数都是：

```python
compile_op(op_plan, memory_plan)
```

其中 `op_plan` 由 `generate_instr.py` 根据当前 layer/group 组装。

operator 主要读取：

```text
input_addr
output_addr
weight_addr / bias_addr
feature_size
channels
kernel_size
start_position
```

## 3. 输出

每个 operator 输出一组汇编字符串，例如：

```asm
CFG_REGISTER R1, LOW,  ...
CFG_REGISTER R1, HIGH, ...
CFG_REGISTER ...
CONV R1, R2, R3
```

这些汇编片段会被 `generate_instr.py` 合并，最终交给 `assembler.py` 转为机器码。

## 4. 合法性检查

operator 只做和当前指令编码直接相关的检查，例如：

```text
当前指令字段取值是否在可表达范围内
feature size 是否在寄存器可表达范围内
feature width 是否能被当前指令寄存器表达
kernel size 是否是 NPU 支持的编码
start_position 是否能放入字段
```

conv/maxpool/relu 输入通道是否超过 1024、conv 输出通道如何拆分、地址空间如何分配等全局问题已经移交给 `generate_memory_plan.py`。

## 5. CONV 通道配置

`operator/conv/conv.py` 写入 RCONV 时：

```text
RCONV1.kernel_size    = 卷积核尺寸编码
RCONV1.padding_size   = 实际 padding 编码，当前为预留接口，NPU 暂不使用
RCONV1.step_size      = 实际 stride 编码，当前为预留接口，NPU 暂不使用
RCONV1.feature        = 当前卷积输入矩阵实际边长，不再 /8
RCONV1.input_channel  = 当前层实际输入通道数，范围 1 到 1024，不按 channel group 拆分
RCONV1.output_channel = 当前 conv pass 的有效输出通道数，当前每条 CONV 仍最多 8 通道
RCONV1.condition_bias = 是否有 bias
RCONV2.scale          = 截位起始比特位，寄存器保留 16 bit，当前只使用 0 到 31
```

也就是说，当前后端只拆分 conv 输出通道；conv 输入通道字段始终配置为该层实际输入通道数。

## 6. bias 处理约定

卷积层的 bias 不再通过 MADD 指令单独相加。`operator/conv/conv.py` 会根据 op plan 中的 `has_bias` 写入 RCONV 的 `condition_bias` bit：

```text
0 = 无 bias
1 = 有 bias
```

当该 bit 为 1 时，NPU 自动从卷积核数据后面读取按 int32 排列的 bias 数据并完成相加。

## 7. ReLU

`operator/relu/relu.py` 写入 RRELU 时：

```text
RRELU.feature      = 当前 ReLU 输入矩阵实际边长，不再 /8
RRELU.channel_size = 当前 ReLU 实际输入通道数，范围 1 到 1024
RRELU.tan          = 斜率编码，例如 00000001 表示 1/2
```

ReLU 的计算指令格式不变，仍为：

```asm
CFG_REGISTER RELU_P_1, ...
CFG_REGISTER RELU_P_2, ...
RELU R1, R2
```

## 8. DSMP

`operator/dsmp/dsmp.py` 负责生成下采样指令：

```asm
CFG_REGISTER DSMP_P_1, ...
CFG_REGISTER DSMP_P,   ...
DSMP R1, R2
```

其中：

```text
R1 = 下采样输入地址
R2 = 下采样输出地址
DSMP_P_1 / DSMP_P = 下采样输入矩阵实际边长 + 实际输入通道数
```

RDSMP 的图像字段按输入矩阵实际边长配置，不再写 `/8` 后的块数；通道字段支持 1 到 1024 通道。conv 仍可按输出通道 group 拆分，但 DSMP 不再按 conv group 拆分；同一层所有 conv group 完成后，编译器只生成一条 DSMP，通道数配置为 DSMP 的实际输入通道数，也就是该层 conv 的实际输出通道数。若上一层实际输出不是 8 的倍数，编译器仍按实际 H/W 看待并直接给到 NPU。

## 9. AVGPOOL / MAXPOOL

`operator/avgpool/avgpool.py` 和 `operator/maxpool/maxpool.py` 负责生成专用池化指令：

```asm
CFG_REGISTER AVGPOOL_P_1, ...
CFG_REGISTER AVGPOOL_P,   ...
AVGPOOL R1, R2

CFG_REGISTER MAXPOOL_P_1, ...
CFG_REGISTER MAXPOOL_P,   ...
MAXPOOL R1, R2
```

其中：
```text
R1 = 池化输入地址
R2 = 池化输出地址
AVGPOOL_P_1 / AVGPOOL_P = 输入矩阵实际边长 + 实际输入通道数 + 步长 step
MAXPOOL_P_1 / MAXPOOL_P = 输入矩阵实际边长 + 实际输入通道数 + 步长 step
```

池化指令由编译器配置输入通道数，NPU 根据该通道数一次处理完整输入通道，不再由编译器按 4 通道拆分。内存占用仍按 4 通道对齐，H/W 保持实际尺寸。

当前 AVGPOOL 指令语义固定为2x2池化窗口、dilation=1、ceil_mode=False；stride 从 IR 中解析得到并写入 RAVGPOOL 的 step 字段，只支持：

```text
stride = [1, 1]  -> step = 0
stride = [2, 2]  -> step = 1
```

当前 MAXPOOL 指令语义固定为2x2池化窗口、dilation=1、ceil_mode=False；stride 从 IR 中解析得到并写入 RMAXPOOL 的 step 字段，只支持：

```text
stride = [1, 1]  -> step = 0
stride = [2, 2]  -> step = 1
```

AVGPOOL/MAXPOOL 不做任何 H/W 补零，编译器只按实际输入边长配置寄存器，并按普通 2x2、padding=0 池化公式规划输出尺寸：`out = floor((in - 2) / stride) + 1`。因此 stride=1 时输出 H/W 为输入各减 1，stride=2 时为普通 2 倍下采样结果。

其他池化配置会在 memory plan 阶段报错。

## 10. FULL

`operator/full/full.py` 负责生成全连接指令片段：

```asm
CFG_REGISTER R1, LOW,  ...
CFG_REGISTER R1, HIGH, ...
CFG_REGISTER R2, LOW,  ...
CFG_REGISTER R2, HIGH, ...
CFG_REGISTER R3, LOW,  ...
CFG_REGISTER R3, HIGH, ...
CFG_REGISTER FULL_P_1, ...
CFG_REGISTER FULL_P_2, ...
FULL R1, R2, R3
```

其中：

```text
R1 = 上一层输出数据起始地址
R2 = 当前全连接输出元素写回地址
R3 = 当前输出元素对应的权重起始地址
RFULL = input_words + start_position + condition_bias
```

一个 FULL operator 只计算一个输出元素。linear stage 的展开规则和参数重排规则分别见 `generate_instr说明.md`、`generate_memory_plan说明.md` 和 `NPU_特性与数据排列规则.md`。
