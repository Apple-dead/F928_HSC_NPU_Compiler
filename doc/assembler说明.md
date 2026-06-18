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
CFG_REGISTER DSMP_P,   ...
CFG_REGISTER RELU_P_1, ...
CFG_REGISTER RELU_P_2, ...
CFG_REGISTER MADD_P,   ...
```

## 4. 计算指令编码

计算指令统一使用：

```text
opcode + rs0 + rs1 + rs2 + dtype + reserve
```

当前 dtype 固定为 INT8。

## 5. MADD 操作数语义

汇编仍写作：

```asm
MADD R1, R2, R3
```

当前语义为：

```text
R1 = 输入数据地址
R2 = 输出数据地址
R3 = 第二输入/bias 数据地址
```

`assembler.py` 不关心寄存器的业务语义，只按操作数顺序编码 `rs0/rs1/rs2`。因此 MADD 语义变化主要体现在 `operator/madd/madd.py` 的寄存器分配中。

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

