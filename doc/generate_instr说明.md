# generate_instr.py 说明

## 1. 总体功能

`python/generate_instr.py` 根据 `data/memory_plan.json` 生成 NPU 汇编和机器码文本：

```text
data/instr.asm
data/instr.txt
```

脚本不再读取旧的 `data/infer_ir/*.json`。层、group、地址、DSMP/ReLU/AVGPOOL/MAXPOOL 规划和通道拆分信息均来自 `memory_plan.json` 的 `execution_plan` 字段。

## 2. 输入输出

默认输入：

```text
data/memory_plan.json
./intr_move.json
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

`intr_move.json` 的默认路径由 `python/npu_config.py` 配置：

```python
INTR_MOVE_PATH = "./intr_move.json"
```

也可以临时覆盖：

```bash
python ./python/generate_instr.py --intr-move ./intr_move.json
```

## 3. CONV_MOVE_BY_INDEX

`intr_move.json` 用于配置当前 IR 中第 N 次 conv 的截位 move：

```json
{
  "CONV_MOVE_BY_INDEX": {
    "1": 512,
    "2": 64,
    "3": 256
  }
}
```

`CONV_MOVE_BY_INDEX["1"]` 表示当前参与编译的第一条 conv，`CONV_MOVE_BY_INDEX["2"]` 表示第二条 conv。它不再表示 `layer1/layer2` 这种模型层名。

生成指令时，同一个 conv 的所有 channel group 共用该 `conv_index` 对应的 move。脚本会把 move 转换为配置寄存器中的起始比特位：

```text
start_position = log2(move)
```

例如：

```text
move = 512 = 2^9
start_position = 9
```

`move=0` 时起始比特位为 0；非 0 的 move 必须是正的 2 的幂，否则脚本直接报错。若当前 `memory_plan.execution_plan` 中存在某个 `conv_index`，但 `intr_move.json` 没有对应条目，也会直接报错。

## 4. 执行流程

脚本按 `execution_plan` 的顺序处理每个 layer plan，再按每个 layer plan 中的 `splits` 顺序处理 group。

普通层生成：

```text
conv -> relu
```

当 IR 只截断到 conv 时，只生成：

```text
conv
```

`stride=2,padding=0` 的层生成：

```text
conv -> dsmp -> relu
```

其中 DSMP 的输入地址、输出地址、图像尺寸和通道数都来自 `memory_plan.json`。

AVGPOOL/MAXPOOL 作为独立 stage 写入 `execution_plan`，`op_type` 分别为 `avgpool` / `maxpool`。每个池化 stage 由单个 split 覆盖完整输入通道，`generate_instr.py` 会调用对应 operator 生成：

```asm
CFG_REGISTER AVGPOOL_P, ...
AVGPOOL R1, R2
```

或：

```asm
CFG_REGISTER MAXPOOL_P, ...
MAXPOOL R1, R2
```

## 5. operator 调用

`generate_instr.py` 不直接拼具体寄存器配置，而是调用：

```text
operator/conv/conv.py
operator/dsmp/dsmp.py
operator/relu/relu.py
```

每个 operator 接收当前 group 的执行描述，完成合法性检查和汇编片段生成。conv 的 `start_position` 来自 `CONV_MOVE_BY_INDEX`。

## 6. 指令大小校验

生成汇编后，脚本调用 `assembler.py` 转成 32-bit 指令 word，并检查：

```text
generated_instruction_bytes == memory_plan["tensors"]["instr"]["size_bytes"]
```

如果不一致，说明 memory plan 中的指令空间估计和实际生成的指令数量不一致，需要修正规划或生成逻辑。
