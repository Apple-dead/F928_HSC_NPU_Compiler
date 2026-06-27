# generate_instr.py 说明

## 1. 总体功能

`python/generate_instr.py` 根据 `data/memory_plan.json` 直接生成 NPU 汇编和机器码文本：

```text
data/instr.asm
data/instr.txt
```

当前脚本不再读取 `data/infer_ir/*.json`。所有层、group、地址、DSMP 规划和通道拆分信息都来自 `memory_plan.json` 的 `execution_plan` 字段。

## 2. 输入和输出

默认输入：

```text
data/memory_plan.json
```

默认输出：

```text
data/instr.asm
data/instr.txt
```

命令：

```bash
python ./python/generate_instr.py
```

### intr_move.json

默认还会读取：

```text
data/intr_move.json
```

也可以显式指定起始比特位配置文件：

```bash
python ./python/generate_instr.py --intr-move ./data/intr_move.json
```

`intr_move.json` 用于配置每层 conv 的 move 值：

```json
{
  "CONV_MOVE_BY_LAYER": {
    "1": 512,
    "2": 512
  }
}
```

生成指令时，脚本会把 move 值转换为配置寄存器中的起始比特位：

```text
start_position = log2(move)
```

例如：

```text
move = 512 = 2^9
start_position = 9
```

该值会传给 `operator/conv/conv.py`，由 operator 编码进 `RCONV` 配置寄存器。`move=0` 时起始比特位直接取 0；非 0 的 move 必须是正的 2 的幂，否则脚本会报错。

## 3. 执行流程

脚本读取 `memory_plan.json` 后，按 `execution_plan` 的层顺序处理：

```text
layer1
layer2
...
```

每一层再按 `splits` 中的 group 顺序处理：

```text
group0
group1
...
```

对于普通层，每个 group 生成：

```text
conv -> relu
```

对于 `stride=2,padding=0` 的层，每个 group 生成：

```text
conv -> dsmp -> relu
```

其中 DSMP 的输入地址、输出地址、图像尺寸和通道数都来自 `memory_plan.json`。

## 4. operator 调用

`generate_instr.py` 不直接拼具体寄存器配置细节，而是按算子调用：

```text
operator/conv/conv.py
operator/dsmp/dsmp.py
operator/relu/relu.py
```

每个 operator 接收当前 group 的执行描述，完成合法性检查和汇编片段生成。

## 5. 指令大小校验

生成汇编后，脚本调用 `assembler.py` 将汇编转成 32-bit 指令 word。

随后会检查：

```text
generated_instruction_bytes == memory_plan["tensors"]["instr"]["size_bytes"]
```

如果不一致，说明 memory plan 中的指令空间估计和实际生成的指令数量不一致，需要修正规划或生成逻辑。
