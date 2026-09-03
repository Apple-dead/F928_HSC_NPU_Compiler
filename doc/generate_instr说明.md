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

脚本按 `execution_plan` 的顺序处理每个 layer plan。conv stage 会先按 `splits` 生成所有 CONV group，再生成 layer 级 DSMP/ReLU。

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

其中 DSMP 的输入地址、输出地址、`feature_size`（输入矩阵实际边长）和通道数都来自 `memory_plan.json` 的 layer 级 `dsmp` 字段；ReLU 同理来自 layer 级 `relu` 字段。DSMP/ReLU 的通道数都配置为该层实际输入通道数，不再按 conv group 拆分；当前 DSMP 和 ReLU 通道范围均为 1 到 1024。

AVGPOOL/MAXPOOL 作为独立 stage 写入 `execution_plan`，`op_type` 分别为 `avgpool` / `maxpool`。每个池化 stage 由单个 split 覆盖完整输入通道，`generate_instr.py` 会调用对应 operator 生成：

```asm
CFG_REGISTER AVGPOOL_P_1, ...
CFG_REGISTER AVGPOOL_P, ...
AVGPOOL R1, R2
```

或：

```asm
CFG_REGISTER MAXPOOL_P_1, ...
CFG_REGISTER MAXPOOL_P,   ...
MAXPOOL R1, R2
```

## 5. operator 调用

`generate_instr.py` 不直接拼具体寄存器配置，而是调用：

```text
operator/conv/conv.py
operator/dsmp/dsmp.py
operator/relu/relu.py
operator/avgpool/avgpool.py
operator/maxpool/maxpool.py
operator/full/full.py
```

每个 operator 接收当前 group 的执行描述，完成合法性检查和汇编片段生成。conv 的 `start_position` 来自 `CONV_MOVE_BY_INDEX`；conv 会写入 RCONV1/RCONV2 四段配置，avgpool/maxpool 会根据 IR 中解析得到的 stride 分别写入 RAVGPOOL/RMAXPOOL 的 step 字段，映射均为 `stride=1 -> step=0`、`stride=2 -> step=1`。

## 6. 指令大小校验

生成汇编后，脚本调用 `assembler.py` 转成 32-bit 指令 word，并检查：

```text
generated_instruction_bytes == memory_plan["tensors"]["instr"]["size_bytes"]
```

如果不一致，说明 memory plan 中的指令空间估计和实际生成的指令数量不一致，需要修正规划或生成逻辑。

## 7. FULL 指令生成

`linear` stage 会展开为多条 `FULL` 指令：`out_features` 有多少个输出，就生成多少次 FULL。每次 FULL 只计算一个 signed int8 输出元素。

`generate_instr.py` 从 `intr_move.json` 的 `FULL_MOVE_BY_INDEX` 读取截位 move，编号是当前 IR 中第 N 个 linear，从 1 开始：

```json
{
  "FULL_MOVE_BY_INDEX": {
    "1": 512
  }
}
```

生成指令时，同一个 linear 的所有输出共用输入地址和 `input_words`；权重地址按每个输出的参数块递增；输出地址按 byte 递增。
