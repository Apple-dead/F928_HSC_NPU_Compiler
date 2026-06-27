# operator 目录说明

## 1. 总体职责

`operator/` 下的代码负责把单个 NPU 算子的执行描述翻译成汇编片段。

当前目录包含：

```text
operator/conv/conv.py
operator/dsmp/dsmp.py
operator/relu/relu.py
```

这些文件不再负责全局内存规划，也不再判断某一层应如何拆分。通道拆分、地址偏移、DSMP 是否需要插入等信息由 `generate_memory_plan.py` 统一写入 `memory_plan.json`。

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
单次处理通道数是否 <= 8
feature size 是否在寄存器可表达范围内
feature width 是否能被 8 整除
kernel size 是否是 NPU 支持的编码
start_position 是否能放入字段
```

输入通道大于 4、输出通道拆分、地址空间分配等全局问题已经移交给 `generate_memory_plan.py`。

## 5. bias 处理约定

卷积层的 bias 不再通过 MADD 指令单独相加。`operator/conv/conv.py` 会根据 op plan 中的 `has_bias` 写入 RCONV 的 `condition_bias` bit：

```text
0 = 无 bias
1 = 有 bias
```

当该 bit 为 1 时，NPU 自动从卷积核数据后面读取按 int32 排列的 bias 数据并完成相加。

## 6. DSMP

`operator/dsmp/dsmp.py` 负责生成下采样指令：

```asm
CFG_REGISTER DSMP_P, ...
DSMP R1, R2
```

其中：

```text
R1 = 下采样输入地址
R2 = 下采样输出地址
DSMP_P = 图像尺寸 + 通道数
```

DSMP 当前同样按每次最多 8 通道处理。
