# Conv + LeakyReLU 独立仿真例子

两例使用完全相同的随机输入、权重和 bias，区域地址及对齐填充不同。
固定随机种子：20261020；全部随机数取 signed int8 完整范围 [-128,127]。
以下参考结果已用 NumPy 向量计算和独立逐项整数计算交叉校验，尚未执行 NPU RTL 仿真。

## 仿真入口（全部为字节地址）

| 例子 | 输入地址 | 参数/权重地址 | bias 地址 | 指令入口地址 | 输出地址 | 输出字节数 |
|---|---|---|---|---|---|---|
| aligned64 | 0x00000040 | 0x00001040 | 0x000014C0 | **0x00001500** | 0x00001538 | 2048 |
| unaligned64 | 0x00000008 | 0x00001008 | 0x00001488 | **0x000014A8** | 0x000014E0 | 2048 |

aligned64 的**输入区、参数区（weights+bias）、指令区起始地址均按 64 字节对齐**。
其参数区结束于 0x14E0，之后补 32 字节零，指令区从 0x00001500 开始。
unaligned64 保持紧密排列：输入从 0x08 开始，不强制各区域按 64 字节对齐。
0 本身也是 64 字节对齐，因此本例特意使用 0x08 覆盖非对齐访问。
两份 all.coe 均从 BRAM **字节地址 0** 初始化，前置填充已经包含在文件中。
初始化内容到最后一条 END 指令结束，仅在末尾补零凑满 512 位；不导出整块输出缓冲区。
aligned64 末尾补 8 字节零，unaligned64 补 32 字节零。
输出区紧接 END 后按 8 字节对齐，因此这少量补零可能覆盖输出区开头的一部分，属于最后一个初始化存储单元的补齐。

## 固定计算配置

- 输入 HWC=16×16×16；输出 HWC=16×16×8；单次 CONV，不拆输出通道。
- kernel=3×3，padding=1（四周补零），stride=1；使用深度学习中的互相关，不翻转卷积核。
- condition_bias=1 表示启用 bias，并非所有 bias 数值均为 1。
- bias 随机值属于 signed int8 范围，按现有接口**符号扩展成 int32，每项占 4 字节**。
- 起始 bit=11；融合激活选择 ReLU（activation=001），RRELU.tan=3，对应 LeakyReLU slope=1/8。
- RCONV1=0x901E079E，RCONV2=0x00002000，RRELU=0x03C07030。
- 每例 14 条 32 位指令（56 字节），包含 1 条 CONV 和 1 条 END；无独立 RELU 计算指令。

## 对照计算约定

```text
acc = sum(input * weight) + bias                  # 有符号整数累加
q = (((acc >> 11) + 128) % 256) - 128                     # 取 acc[7:0]，解释为 signed int8
y = q if q >= 0 else (q >> 3)                    # 负数算术右移，向负无穷舍入
```

先截位再激活；无饱和、无四舍五入。例如 -1 >> 3 = -1，-9 >> 3 = -2。
随机数据会触发溢出截位，这是起始 bit=11 的预期行为。
本组 acc 范围 [-244122, 229129]，
截位范围 [-128, 127]，
最终输出范围 [-16, 127]。

## 数据排列及端序

- input：通道组 → h → w → 组内8通道，先排ch1～ch8的所有像素，再排ch9～ch16。
- weights：输出通道 oc → kh → kw → 全部 16 个输入通道，每项 1 字节。
- params：全部 8 个输出通道的权重（1152 字节），随后是 8 项 int32 bias（32 字节）。
- output：h → w → 全部 8 个输出通道，每项 1 字节。
- 每个 512 位 COE 项占 64 字节。最低字节地址位于数值的 [7:0]，即文本最右侧的两位十六进制。
- 每条 32 位指令按现有编译器合并规则以 little-endian 字节写入 BRAM；最低地址存最低字节。
- 字节地址 A 对应 `BRAM[A//64][8*(A%64)+:8]`。
- **指令入口是字节地址，不是 512 位 BRAM 索引。** aligned64 指令入口位于 512 位字的开头；unaligned64 仍需保留字内偏移。
- 输入和输出通道数已经是 8 的倍数，此处不增加虚拟通道。

## 文件用途

| 每例文件 | 用途 |
|---|---|
| all.coe | 包含前置填充、输入、权重、bias、指令，仅在 END 后补零到完整 512 位项，不包含整块输出缓冲区 |
| all.mem | 与 all.coe 相同内容，每行 128 个十六进制字符，可用于 `$readmemh` |
| all.bin | 同一内存映像，按字节地址递增排列 |
| memory_map.txt / memory_map.json | 各区域地址、长度、BRAM 索引和字内偏移 |
| instr.asm / instr.txt / instructions.csv | 汇编、32 位机器码、每条指令的字节地址 |
| input_hwc.csv / weights_ohwi.csv / bias.csv | 原始 signed 数值，可人工核对 |
| reference.csv | 每个输出坐标的卷积和、bias、累加值、截位值、最终 signed/hex 结果及输出地址 |
| expected_hwc.csv | 最终输出矩阵，256 行，每行一个像素的 8 个通道 |
| expected_output.mem | 2048 行，每行一个预期输出字节，无 COE 文件头 |
| expected_output.bin | 2048 字节预期输出，HWC 顺序 |
| expected_output.coe | 32 个 512 位预期输出项；仅包含输出，相对偏移从 0 开始 |
| input.coe / params.coe / instr.coe | 32 位中间 COE，仅用于复核，512 位 BRAM 请加载 all.coe |
| model_ir.json / memory_plan.json / intr_move.json | 生成例子使用的 IR、内存规划、截位配置 |

根目录 **reference.txt** 是两例共用的人眼可读参考：先按顺序列出 16 个输入通道的 16×16 矩阵，
再列出 8 个输出通道的 16×16 矩阵，最后按输出通道、输入通道列出 128 个 3×3 权重矩阵，
并在每个输出通道的权重标题处注明 bias。TXT 中通道、行列编号均从 1 开始，数值为有符号十进制。

输出区由 NPU 在运行时写入，无需在 COE 中完整初始化。BRAM 的实际深度仍必须足够覆盖输出区。
expected_output.coe/.mem 是输出的相对布局；输出实际起始地址见上表。

## 建议仿真步骤

1. 为每例独立初始化 BRAM，加载对应 all.coe（或用 `$readmemh` 加载 all.mem）。
2. 宽度设为 512 位，深度至少为 aligned64: 117、unaligned64: 116；两例共用时可统一设为 128。
   all.coe 实际仅含 aligned64: 85 项、unaligned64: 84 项；不能据此缩小运行时 BRAM 深度。
3. 把对应的**指令入口字节地址**设置给 NPU，然后启动执行。
4. 等待执行结束后，从输出起始地址读取 2048 字节，依次对照 expected_output.mem。
5. 若数值不符，用 reference.csv 区分乘加、bias、截位、激活或数据排列的问题。

两例 expected_output.bin 必须完全一致；SHA256：
`cb28eb0f81048f71d9f619460827ce62a3113f281a8b2bfa1043426fbd7ab4a8`

所有 CSV 坐标从 0 开始，reference.txt 从 1 开始编号。未调用或修改 test/ 内的内容。
在工程根目录重新生成：`python -B python/generate_conv_relu_examples.py`。

## Input layout and scale update

Input bytes: channel group -> row -> column -> eight channels. Channels 1..8 for all pixels precede channels 9..16. Little-endian storage is unchanged.

Conv scale=11; accumulator range=[-244122, 229129]; shifted range=[-120, 111]. Apply this arithmetic shift before taking 8 bits and applying LeakyReLU. Parameter bytes, addresses, weights, bias, seed and all other operator settings are unchanged.
