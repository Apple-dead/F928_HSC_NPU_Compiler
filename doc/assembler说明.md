# assembler.py 说明

## 1. 总体功能

`python/assembler.py` 负责把 `data/instr.asm` 中的 NPU 汇编指令编码为 32-bit 机器码，并输出到：

```text
data/instr.txt
```

`generate_instr.py` 会直接调用其中的：

```python
assemble_lines(...)
write_instr_txt(...)
```

## 2. 支持的汇编指令

当前支持：

```text
CFG_REGISTER
CONV
DSMP
RELU
MADD
AVGPOOL
MAXPOOL
END
```

其中 `DSMP` 已按指令手册添加，opcode 为：

```text
000100
```

## 3. CFG_REGISTER

通用寄存器地址配置格式：

```asm
CFG_REGISTER R1, LOW,  0x0000
CFG_REGISTER R1, HIGH, 0x0000
```

特殊寄存器配置格式：

```asm
CFG_REGISTER CONV_P_1, ...
CFG_REGISTER CONV_P_2, ...
CFG_REGISTER CONV_P_3, ...
CFG_REGISTER CONV_P_4, ...
CFG_REGISTER DSMP_P_1, ...
CFG_REGISTER DSMP_P,   ...
CFG_REGISTER RELU_P_1, ...
CFG_REGISTER RELU_P_2, ...
CFG_REGISTER AVGPOOL_P_1, ...
CFG_REGISTER AVGPOOL_P, ...
CFG_REGISTER MAXPOOL_P_1, ...
CFG_REGISTER MAXPOOL_P, ...
CFG_REGISTER FULL_P_1, ...
CFG_REGISTER FULL_P_2, ...
CFG_REGISTER MADD_P, ...
CFG_REGISTER MMUL_P, ...
```

当前 special register code 与 `subop` 一致，按最新版指令手册配置如下：

| 汇编别名 | subop | 含义 |
| --- | --- | --- |
| `CONV_P_1` / `RCONV1_LOW` | `00010` / `0x02` | RCONV1 低 16 位 |
| `CONV_P_2` / `RCONV1_HIGH` | `00011` / `0x03` | RCONV1 高 16 位 |
| `CONV_P_3` / `RCONV2_LOW` | `00100` / `0x04` | RCONV2 低 16 位 |
| `CONV_P_4` / `RCONV2_HIGH` | `00101` / `0x05` | RCONV2 高 16 位 |
| `DSMP_P_1` / `RDSMP_LOW` | `00110` / `0x06` | RDSMP 低 16 位 |
| `DSMP_P` / `DSMP_P_2` / `RDSMP_HIGH` | `00111` / `0x07` | RDSMP 高 16 位 |
| `RELU_P_1` / `RRELU_LOW` | `01000` / `0x08` | RRELU 低 16 位 |
| `RELU_P_2` / `RRELU_HIGH` | `01001` / `0x09` | RRELU 高 16 位 |
| `AVGPOOL_P_1` / `RAVGPOOL_LOW` | `01010` / `0x0A` | RAVGPOOL 低 16 位 |
| `AVGPOOL_P` / `AVGPOOL_P_2` / `RAVGPOOL_HIGH` | `01011` / `0x0B` | RAVGPOOL 高 16 位 |
| `MAXPOOL_P_1` / `RMAXPOOL_LOW` | `01100` / `0x0C` | RMAXPOOL 低 16 位 |
| `MAXPOOL_P` / `MAXPOOL_P_2` / `RMAXPOOL_HIGH` | `01101` / `0x0D` | RMAXPOOL 高 16 位 |
| `FULL_P_1` / `RFULL_LOW` | `01110` / `0x0E` | RFULL 低 16 位 |
| `FULL_P_2` / `RFULL_HIGH` | `01111` / `0x0F` | RFULL 高 16 位 |
| `MADD_P` / `RMADD_HIGH` | `10011` / `0x13` | RMADD 高 16 位 |
| `MMUL_P` / `RMMUL_HIGH` | `10101` / `0x15` | RMMUL 高 16 位 |

## 4. 计算指令编码

计算指令统一使用：

```text
opcode + rs0 + rs1 + rs2 + dtype + reserve
```

当前 dtype 固定为 INT8。

## 5. CONV bias 语义

CONV 计算指令的操作数仍写作：

```asm
CONV R1, R2, R3
```

语义为：

```text
R1 = 输入数据地址
R2 = 输出数据地址
R3 = 卷积核地址
```

是否自动执行 bias 相加由 `CFG_REGISTER CONV_P_1/CONV_P_2` 写入的 RCONV1 `condition_bias` bit 决定；`CFG_REGISTER CONV_P_3/CONV_P_4` 用于写入 RCONV2。`assembler.py` 只负责按特殊寄存器编号编码 CFG 指令，不解析 RCONV 位域。

## 6. DSMP 操作数语义

DSMP 汇编格式为：

```asm
DSMP R1, R2
```

语义为：

```text
R1 = 输入数据地址
R2 = 输出数据地址
```

第三个寄存器字段在编码时置 0。

RDSMP 配置由 `DSMP_P_1` 写入低 16 位、`DSMP_P` 写入高 16 位。`DSMP_P_1` 的 special register code 为 `0x06`，即 subop `00110`；`DSMP_P` 的 special register code 为 `0x07`，即 subop `00111`。

## 7. FULL 支持

`assembler.py` 支持全连接相关汇编：

```asm
CFG_REGISTER FULL_P_1, 0x....
CFG_REGISTER FULL_P_2, 0x....
FULL R1, R2, R3
```

`FULL_P_1` / `FULL_P_2` 的 special register code 分别为 `0x0E` / `0x0F`。`FULL` 的 opcode 为 `001010`，操作数语义为：

```text
R1 = 输入数据起始地址
R2 = signed int8 输出写回地址
R3 = 全连接权重起始地址
```

输出地址允许 byte 粒度递增，由 NPU 保证 FULL 输出为 signed int8。
